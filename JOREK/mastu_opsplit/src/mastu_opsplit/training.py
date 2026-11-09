"""Explicit synthetic smoke training and identity-preserving checkpoint resume.

Synthetic fixtures exercise plumbing only and are never scientific gate evidence.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from .common import config_hash, write_json
from .learning import build_family, parameter_counts


def study_source_hashes():
    """Pin local semantics as well as external architecture/configuration."""
    root = Path(__file__).resolve().parents[2]
    names = [f"src/mastu_opsplit/{name}.py" for name in
             ("learning", "training", "common", "data", "numerical", "sampling", "evaluation")]
    names.append("scripts/train.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def checkpoint(path, model, optimizer, scheduler, *, identity, step, split):
    """Save complete state exclusively; accepted checkpoints are immutable."""
    state = {"identity": identity, "step": step, "split": split,
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "python_rng": random.getstate(),
             "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    with Path(path).open("xb") as stream:
        torch.save(state, stream)


def restore(path, model, optimizer, scheduler, *, identity, split):
    """Restore a trusted local checkpoint only after identity/split matching."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["identity"] != identity or state["split"] != split:
        raise ValueError("Checkpoint configuration, dependency, or data split mismatch")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("Checkpoint requires CUDA RNG restoration on a CUDA host")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"]


def smoke_train(config, model_config, output, *, resume=None):
    """Overfit a fixed analytic tiny batch; this does not simulate MAST-U."""
    steps = int(config.get("steps", 12))
    if steps < 1 or steps > 200:
        raise ValueError("Synthetic smoke steps must be between 1 and 200")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    tiny = {**model_config, "architecture": {**model_config["architecture"],
            "grid_shape": [8, 8], "modes": 2, "width": 4, "n_layers": 1}}
    family = config.get("family", "whole_rhs")
    model = build_family(family, 2, [[1., 2.], [-1., 1.]], tiny)
    coords = torch.stack([torch.linspace(1.05, 1.95, 6).repeat_interleave(6),
                          torch.linspace(-.95, .95, 6).repeat(6)], dim=-1)
    fields = torch.randn(2, coords.shape[0], 2)
    parameters = torch.tensor([[.5], [1.5]])
    target = torch.stack([.3 * fields[..., 0] + .1 * fields[..., 1],
                          -.2 * fields[..., 0] + .4 * fields[..., 1]], dim=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", .01)))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=.5)
    split = {"train": ["synthetic-parent-0", "synthetic-parent-1"], "validation": [], "test": []}
    provenances = [module.provenance for module in model.modules() if hasattr(module, "provenance")]
    source_hashes = study_source_hashes()
    identity = config_hash({"config": config, "model_config": tiny, "dependency": provenances,
                            "study_source_sha256": source_hashes,
                            "fixture": "analytic-smoke-v1", "split": split})
    start = restore(resume, model, optimizer, scheduler, identity=identity, split=split) if resume else 0
    if start >= steps:
        raise ValueError("Checkpoint already reached requested step count")
    losses = []
    for step in range(start, steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(fields, coords, parameters)
        loss = (prediction - target).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite synthetic training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite synthetic training gradient")
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 5 == 0 and step + 1 < steps:
            checkpoint(output / f"checkpoint-step-{step+1}.pt", model, optimizer, scheduler,
                       identity=identity, step=step+1, split=split)
    with torch.no_grad():
        final_loss = float((model(fields, coords, parameters) - target).square().mean())
    if final_loss >= losses[0]:
        raise RuntimeError("Synthetic tiny-batch loss did not decrease")
    checkpoint(output / "checkpoint.pt", model, optimizer, scheduler,
               identity=identity, step=steps, split=split)
    report = {"synthetic": True, "scientific_gate_evidence": False, "family": family,
              "seed": seed, "steps": steps, "resumed_at": start, "identity": identity,
              "initial_loss": losses[0], "final_loss": final_loss, "losses": losses,
              "parameter_counts": parameter_counts(model), "dependency": provenances,
              "study_source_sha256": source_hashes,
              "safeguards": [m.safeguard_diagnostics(coords) for m in model.modules()
                             if hasattr(m, "safeguard_diagnostics")]}
    write_json(output / "smoke.json", report)
    return report


def load_term_trajectory(path, *, expected_split, diagnostics_only=False):
    """Read one validated native trajectory without fitting normalization scales.

    Provisional real exchanges are available only to labelled diagnostics, never
    the production trainer. All current exchange files remain provisional.
    """
    from .data import geometry_hash, read_trajectory
    geometry, arrays, metadata = read_trajectory(path)
    if metadata["split"] != expected_split:
        raise ValueError("Trajectory split does not match requested training/validation split")
    if not diagnostics_only and (metadata["synthetic"] or metadata.get("mapping_status") != "verified_native"):
        raise ValueError("Scientific training rejects synthetic or provisional native mappings")
    parameters = metadata["parameters"]
    scale = float(parameters["s_perp"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("s_perp must be finite and positive")
    coords = geometry.quadrature_rz.reshape(-1, 2)
    time_count = len(arrays["time_seconds"])
    return {"fields": torch.from_numpy(arrays["state_samples"]),
            "labels": torch.from_numpy(arrays["pointwise_terms"].reshape(time_count, -1, 2)),
            "source_coords": torch.from_numpy(coords), "query_coords": torch.from_numpy(coords),
            "volume_weights": torch.from_numpy(geometry.volume_weights.reshape(-1)),
            "bounds": torch.from_numpy(geometry.bounds),
            "parameters": torch.full((time_count, 1), scale, dtype=torch.float64),
            "time_seconds": torch.from_numpy(arrays["time_seconds"]),
            "native_state": torch.from_numpy(arrays["native_state"]),
            "geometry_hash": geometry_hash(geometry), "metadata": metadata,
            "scientific_gate_evidence": False}


def iter_term_batches(paths, *, expected_split, batch_size, diagnostics_only=False):
    """Validate family splits up front, then hold at most one trajectory at a time."""
    import json
    import h5py
    from .sampling import validate_family_splits
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("A positive integer time-batch size is required")
    records = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            records.append(json.loads(handle.attrs["metadata_json"]))
    validate_family_splits(records)
    if not records or any(row["split"] != expected_split for row in records):
        raise ValueError("Nonempty trajectories in exactly the requested split are required")
    geometry_hash = None
    for path in paths:
        trajectory = load_term_trajectory(path, expected_split=expected_split, diagnostics_only=diagnostics_only)
        if geometry_hash is not None and trajectory["geometry_hash"] != geometry_hash:
            raise ValueError("Primary training requires one shared fixed geometry")
        geometry_hash = trajectory["geometry_hash"]
        for start in range(0, len(trajectory["fields"]), batch_size):
            yield {**trajectory, **{name: trajectory[name][start:start+batch_size]
                                   for name in ("fields", "labels", "parameters", "time_seconds", "native_state")}}


def term_supervised_loss(prediction, labels, volume_weights, label_scales):
    """Channel-scaled volume-weighted term loss; scales must come from training."""
    if prediction.shape != labels.shape or prediction.ndim != 3:
        raise ValueError("Prediction and labels require matching [B,N,C] layouts")
    weights = torch.as_tensor(volume_weights, dtype=prediction.dtype, device=prediction.device)
    scales = torch.as_tensor(label_scales, dtype=prediction.dtype, device=prediction.device)
    if (weights.shape != (prediction.shape[1],) or scales.shape != (prediction.shape[2],)
            or not torch.isfinite(weights).all() or not torch.isfinite(scales).all()
            or not (weights > 0).all() or not (scales > 0).all()):
        raise ValueError("Positive finite physical weights and training channel scales required")
    return (((prediction - labels) / scales).square() * weights[None, :, None]).sum() / (weights.sum() * prediction.shape[0] * prediction.shape[2])


def pretrain_term_epoch(model, batches, optimizer, label_scales):
    """One direct-label epoch; native rollout fine-tuning is a separate gated step."""
    model.train()
    losses = []
    for batch in batches:
        parameter = next(model.parameters())
        cast = lambda value: value.to(dtype=parameter.dtype, device=parameter.device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(cast(batch["fields"]), cast(batch["source_coords"]),
                           cast(batch["parameters"]), cast(batch["query_coords"]))
        loss = term_supervised_loss(prediction, cast(batch["labels"]), batch["volume_weights"], label_scales)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite term-supervision loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite term-supervision gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
    if not losses:
        raise ValueError("Nonempty term batches required")
    return losses
