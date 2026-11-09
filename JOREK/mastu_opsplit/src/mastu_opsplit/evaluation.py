"""Physical-volume metrics and parent-family uncertainty, including failed runs."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def weighted_errors(reference, prediction, weights, scales):
    """Return per-time/channel relative and training-scale-normalized L2 errors."""
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    weights, scales = np.asarray(weights), np.asarray(scales)
    if reference.ndim != 3 or prediction.shape != reference.shape:
        raise ValueError("fields must have equal [time,point,channel] shapes")
    if weights.shape != reference.shape[1:2] or scales.shape != reference.shape[2:]:
        raise ValueError("weights/scales must match points/channels")
    if not all(np.isfinite(x).all() for x in (reference, prediction, weights, scales)):
        raise ValueError("metric inputs must be finite; report failed rollouts separately")
    if (weights < 0).any() or weights.sum() <= 0 or (scales <= 0).any():
        raise ValueError("nonnegative volume weights and positive training scales required")
    error = np.sqrt(np.einsum("tnc,n->tc", (prediction-reference)**2, weights)/weights.sum())
    norm = np.sqrt(np.einsum("tnc,n->tc", reference**2, weights)/weights.sum())
    return {"relative_l2": error/np.maximum(norm, 1e-8*scales),
            "scaled_l2": error/scales, "absolute_l2": error}


def fit_training_scales(records):
    """Fit RMS perturbation scales with equal weight per training trajectory."""
    means = []
    for record in records:
        if record["split"] != "train":
            raise ValueError("normalization may use training records only")
        values, weights = np.asarray(record["perturbation"]), np.asarray(record["weights"])
        if values.ndim != 3 or weights.shape != values.shape[1:2]:
            raise ValueError("invalid training perturbation or weights")
        if not np.isfinite(values).all() or not np.isfinite(weights).all():
            raise ValueError("nonfinite training data")
        if (weights < 0).any() or weights.sum() <= 0 or len(values) == 0:
            raise ValueError("invalid training weights or empty trajectory")
        means.append(np.einsum("tnc,n->tc", values**2, weights).mean(axis=0)/weights.sum())
    if not means:
        raise ValueError("training records required")
    scales = np.sqrt(np.mean(means, axis=0))
    if (scales <= 0).any():
        raise ValueError("zero training scale: declare a physical scale for inactive channels")
    return scales


def summarize(records):
    """Accuracy is explicitly conditional; every failure enters completion rates."""
    groups = defaultdict(list)
    seen = set()
    for row in records:
        key = (row["model"], row["seed"], row["case_id"])
        if key in seen:
            raise ValueError(f"duplicate trajectory result: {key}")
        seen.add(key)
        if not row.get("parent_family_id") or not isinstance(row["completed"], bool):
            raise ValueError("family ID and boolean completion required")
        completed = row["completed"]
        error = row.get("perturbation_error")
        if completed and (error is None or not np.isfinite(error) or error < 0):
            raise ValueError("completed trajectory requires finite nonnegative error")
        if not completed and not row.get("failure_reason"):
            raise ValueError("failed trajectory requires failure_reason")
        groups[(row["model"], row["seed"], row["regime"])].append(row)
    result = []
    for (model, seed, regime), rows in sorted(groups.items()):
        errors = [row["perturbation_error"] for row in rows if row["completed"]]
        result.append({"model": model, "seed": seed, "regime": regime,
                       "trajectories": len(rows), "completed": len(errors),
                       "completion_fraction": len(errors)/len(rows),
                       "median_error_completed_only": float(np.median(errors)) if errors else None,
                       "p95_error_completed_only": float(np.percentile(errors, 95)) if errors else None,
                       "failed_case_ids": [row["case_id"] for row in rows if not row["completed"]]})
    return result


def paired_family_interval(left, right, *, seed=7411, samples=1000, confidence=0.95):
    """Bootstrap paired differences of family means, never individual frames.

    Inputs map case ID to (parent-family ID, scalar metric). Caller must select
    one regime/time/seed and declare any completed-only metric selection.
    """
    if set(left) != set(right) or not left:
        raise ValueError("paired case sets must match and be nonempty")
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    families = defaultdict(list)
    for case in left:
        family, a = left[case]
        other_family, b = right[case]
        if not family or family != other_family or not np.isfinite([a, b]).all():
            raise ValueError("paired family IDs and finite metrics required")
        families[family].append(a-b)
    means = np.array([np.mean(families[family]) for family in sorted(families)])
    rng = np.random.default_rng(seed)
    estimates = np.array([rng.choice(means, len(means), replace=True).mean() for _ in range(samples)])
    alpha = (1-confidence)/2
    return {"difference_mean": float(means.mean()), "interval": np.quantile(estimates, [alpha, 1-alpha]).tolist(),
            "parent_families": len(means), "paired_cases": len(left), "confidence": confidence}


def resource_totals(attempts):
    """Account actual measured attempts, including failed and retried work."""
    totals = dict(cpu_core_hours=0.0, gpu_hours=0.0, raw_bytes=0, export_bytes=0, peak_memory_bytes=0)
    seen = set()
    for attempt in attempts:
        ident = attempt["attempt_id"]
        if ident in seen:
            raise ValueError("duplicate attempt identity")
        seen.add(ident)
        names = ("elapsed_seconds", "cpu_cores", "gpus", "raw_bytes", "export_bytes", "peak_memory_bytes")
        if any(not np.isfinite(attempt[k]) or attempt[k] < 0 for k in names):
            raise ValueError("resource measurements must be finite and nonnegative")
        totals["cpu_core_hours"] += attempt["elapsed_seconds"]*attempt["cpu_cores"]/3600
        totals["gpu_hours"] += attempt["elapsed_seconds"]*attempt["gpus"]/3600
        for name in ("raw_bytes", "export_bytes"):
            totals[name] += attempt[name]
        totals["peak_memory_bytes"] = max(totals["peak_memory_bytes"], attempt["peak_memory_bytes"])
    return totals
