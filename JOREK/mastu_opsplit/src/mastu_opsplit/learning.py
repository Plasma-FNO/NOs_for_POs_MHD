"""Current-field adapter for the pinned upstream 2-D L-SplineFNO.

This module predicts pointwise terms; native FE assembly remains separate.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn

PINNED_COMMIT = "a3ca59bba100ca56e535fc35c32967bdeb04d8d9"


def dependency_provenance(dependency: dict) -> dict:
    """Verify identity and record hashes of actual imported upstream sources."""
    root = Path(dependency["path"]).resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != dependency.get("commit", PINNED_COMMIT):
        raise ValueError(f"L-SplineFNO commit mismatch: {commit}")
    for relative, expected in dependency.get("sha256", {}).items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"L-SplineFNO source hash mismatch: {relative}")
    loaded = sys.modules.get("Models")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Another repository's Models package is already imported")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    backbone = importlib.import_module("Models.learnable_spline_fno")
    film = importlib.import_module("Models.conditioning.film")
    paths = sorted({Path(m.__file__).resolve() for name, m in list(sys.modules.items())
                    if name.startswith("Models") and getattr(m, "__file__", None)
                    and Path(m.__file__).resolve().is_relative_to(root)})
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    patch = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", "Models"])
    return {"root": str(root), "commit": commit, "imported_sha256": hashes,
            "backbone_path": backbone.__file__, "film_path": film.__file__,
            "models_patch": patch.decode(), "models_patch_sha256": hashlib.sha256(patch).hexdigest(),
            "dirty_status": subprocess.check_output(["git", "-C", str(root), "status", "--short"], text=True),
            "license_path": str(root / "LICENSE")}


class FieldSplineOperator(nn.Module):
    """Convert [B,N,C] physical fields to upstream layouts using fixed bounds."""

    def __init__(self, state_channels: int, output_channels: int, bounds, model_config: dict,
                 *, condition_transport: bool):
        super().__init__()
        if state_channels < 1 or output_channels < 1:
            raise ValueError("Field and output channel counts must be positive")
        self.provenance = dependency_provenance(model_config["dependency"])
        bounds = torch.as_tensor(bounds, dtype=torch.float64)
        if bounds.shape != (2, 2) or not torch.isfinite(bounds).all() or not (bounds[:, 1] > bounds[:, 0]).all():
            raise ValueError("Fixed physical bounds must be finite [2,2] axis min/max pairs")
        self.register_buffer("physical_bounds", bounds)
        self.state_channels = state_channels
        backbone = importlib.import_module("Models.learnable_spline_fno").LearnableSplineFNO
        film = importlib.import_module("Models.conditioning.film").FiLMConditioning
        architecture = dict(model_config["architecture"])
        if architecture.get("scatter_scales", [1]) != [1] or architecture.get("encode_refinement_steps", 0) != 0:
            raise ValueError("Primary study uses one scatter scale and zero encode refinement")
        self.backbone = backbone(in_channels=state_channels, out_channels=output_channels,
                                 space_dim=2, grid_bounds=[(0., 1.), (0., 1.)], **architecture)
        self.conditioning_parameters = int(model_config.get("conditioning_parameters", 1))
        if self.conditioning_parameters not in (1, 2):
            raise ValueError("Expected one transport parameter or transport and gamma")
        self.film = film(n_params=self.conditioning_parameters, field_channels=state_channels) if condition_transport else None

    def normalized_coordinates(self, coords: torch.Tensor, batch: int) -> torch.Tensor:
        if coords.ndim == 2:
            coords = coords.unsqueeze(0).expand(batch, -1, -1)
        if coords.ndim != 3 or coords.shape[0] != batch or coords.shape[2] != 2 or coords.shape[1] == 0:
            raise ValueError("Coordinates must have shape [N,2] or [B,N,2], with nonempty points")
        if not torch.isfinite(coords).all():
            raise ValueError("Coordinates must be finite")
        bounds = self.physical_bounds.to(coords)
        normalized = (coords - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        if ((normalized < 0) | (normalized > 1)).any():
            raise ValueError("Source and query points must remain inside the fixed physical bounds")
        return normalized.transpose(1, 2)

    def forward(self, fields, source_coords, parameters=None, query_coords=None):
        if fields.ndim != 3 or fields.shape[2] != self.state_channels or fields.shape[0] == 0:
            raise ValueError("Fields must have shape [B,N,C_state]")
        if not torch.isfinite(fields).all():
            raise ValueError("Fields must be finite")
        source = self.normalized_coordinates(source_coords, fields.shape[0])
        if source.shape[2] != fields.shape[1]:
            raise ValueError("Field and source point counts differ")
        source = source.to(fields)
        query = source if query_coords is None else self.normalized_coordinates(query_coords, fields.shape[0]).to(fields)
        if self.film is not None:
            if parameters is None or parameters.shape != (fields.shape[0], self.conditioning_parameters) or not torch.isfinite(parameters).all() or not (parameters > 0).all():
                raise ValueError("Conditioning parameters must be finite, positive and match the configured width")
            fields = self.film(fields, parameters)
        return self.backbone(fields.transpose(1, 2), source, query_coords=query).transpose(1, 2)

    @torch.no_grad()
    def safeguard_diagnostics(self, source_coords, batch=1):
        """Count activation using upstream factored stencils; never form N-by-G."""
        module = importlib.import_module("Models.learnable_spline_fno")
        source = self.normalized_coordinates(source_coords, batch)
        axes = self.backbone._get_grid_axes()
        row_floor = 0
        row_count = 0
        for d, (axis, kernel) in enumerate(zip(axes, self.backbone.scatter_kernels)):
            t = (source[:, d] - axis[0]) / (axis[1] - axis[0])
            knots = torch.floor(t).long().unsqueeze(-1) + torch.arange(-1, 3, device=t.device)
            raw = kernel(t.unsqueeze(-1) - knots) * ((knots >= 0) & (knots < axis.numel()))
            row_floor += int((raw.sum(-1).abs() < self.backbone.pou_floor).sum())
            row_count += raw.shape[0] * raw.shape[1]
        stencil = self.backbone._stencil(source, self.backbone.scatter_kernels, pou_floor=self.backbone.pou_floor)
        _, den, absolute = module._scatter_add_fused(
            source.new_zeros(batch, source.shape[2], 1), stencil, self.backbone._grid_nodes(),
            max_temp_bytes=self.backbone.scatter_max_temp_bytes)
        floor = (absolute * self.backbone.density_floor_frac).clamp_min(self.backbone.density_eps)
        return {"point_axis_rows": row_count, "pou_floor_rows": row_floor,
                "grid_nodes": den.numel(), "uncovered_nodes": int((absolute == 0).sum()),
                "covered_density_floor_nodes": int(((absolute > 0) & (den.abs() < floor)).sum()),
                "covered_absolute_epsilon_nodes": int(((absolute > 0) & (den.abs() < self.backbone.density_eps)).sum()),
                "signed_cancellation_floor_nodes": int(((absolute > 0) & (den.abs() < absolute * self.backbone.density_floor_frac)).sum())}


class OpsSplitOperators(nn.Module):
    """Independent density and pressure pointwise advection/compression predictors."""
    def __init__(self, state_channels, bounds, model_config):
        super().__init__()
        self.density = FieldSplineOperator(state_channels, 1, bounds, model_config, condition_transport=False)
        self.pressure = FieldSplineOperator(state_channels, 1, bounds, model_config, condition_transport=False)

    def forward(self, fields, source_coords, parameters=None, query_coords=None):
        return torch.cat([self.density(fields, source_coords, query_coords=query_coords),
                          self.pressure(fields, source_coords, query_coords=query_coords)], dim=-1)


def build_family(family, state_channels, bounds, model_config, *, output_channels=None):
    if family == "opsplit_full":
        if state_channels != 7 or output_channels not in (None, 15):
            raise ValueError("Full OpsSplit requires seven state and fifteen weak output channels")
        from .full_rollout import FullOpsSplitOperators
        return FullOpsSplitOperators(bounds, model_config)
    if family == "opsplit":
        if output_channels not in (None, 2):
            raise ValueError("OpsSplit has two scalar output blocks")
        return OpsSplitOperators(state_channels, bounds, model_config)
    if family not in ("whole_rhs", "autoregressive"):
        raise ValueError(f"Unknown neural family: {family}")
    return FieldSplineOperator(state_channels, output_channels or state_channels, bounds,
                               model_config, condition_transport=True)


def parameter_counts(model):
    """Disjoint trainable counts for budget matching; unmatched sizes stay visible."""
    result = {"spline": 0, "film": 0, "fno": 0, "other": 0}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = "spline" if "scatter_kernels" in name else "film" if ".film." in f".{name}." else "fno" if "fno_blocks" in name else "other"
            result[group] += parameter.numel()
    result["total"] = sum(result.values())
    return result
