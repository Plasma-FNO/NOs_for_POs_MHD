"""Candidate stratified parameters; physical centers/widths remain pilot-gated."""
from __future__ import annotations

import numpy as np
from itertools import product


def validate_family_splits(records):
    families, cases = {}, set()
    for row in records:
        family, split, case = row["parent_family_id"], row["split"], row["case_id"]
        if not family or not split or not case or case in cases:
            raise ValueError("nonempty family/split/case IDs and unique cases required")
        cases.add(case)
        if family in families and families[family] != split:
            raise ValueError(f"parent family leaks across splits: {family}")
        families[family] = split


def candidate_parameters(config):
    """Deterministic bounded Latin-hypercube candidates, explicitly not run inputs."""
    rng = np.random.default_rng(config["seed"])
    allowed = {"train", "validation", "id_test", "strong_test", "low_transport_test",
               "high_transport_test", "combined_test"}
    if set(config["splits"]) - allowed:
        raise ValueError("unknown sampling split")
    for key in ("train_amplitude", "strong_amplitude", "train_transport", "low_transport", "high_transport"):
        bounds = config[key]
        if len(bounds) != 2 or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
            raise ValueError("parameter bounds must be finite positive ordered pairs")
    if not (config["strong_amplitude"][0] > config["train_amplitude"][1]
            and config["low_transport"][1] < config["train_transport"][0]
            and config["high_transport"][0] > config["train_transport"][1]):
        raise ValueError("claimed held-out parameter intervals must not overlap the training interval")
    records = []
    for split, count in config["splits"].items():
        if not isinstance(count, int) or count < 1:
            raise ValueError("each split needs a positive trajectory count")
        amplitude = config["strong_amplitude"] if split in {"strong_test", "combined_test"} else config["train_amplitude"]
        transport = config["train_transport"]
        if split == "low_transport_test":
            transport = config["low_transport"]
        elif split == "high_transport_test":
            transport = config["high_transport"]
        # Combined cases alternate held-out low/high transport strata.
        strata = np.stack([(rng.permutation(count)+rng.random(count))/count for _ in range(3)], axis=1)
        for index, unit in enumerate(strata):
            interval = config["low_transport"] if split == "combined_test" and index % 2 == 0 else (
                config["high_transport"] if split == "combined_test" else transport)
            scale = lambda u, bounds: float(bounds[0]+u*(bounds[1]-bounds[0]))
            case = f"{split}-{index:04d}"
            records.append({"case_id": case, "parent_family_id": case, "split": split,
                            "A_rho": scale(unit[0], amplitude), "A_T": scale(unit[1], amplitude),
                            "s_perp": scale(unit[2], interval), "status": "candidate_without_geometry",
                            "center_m": None, "width_m": None})
    validate_family_splits(records)
    return records


def training_envelope_pilot(config):
    """Eight training-range corners and matched baselines; development data only.

    Passing this pilot would qualify only the stated training interval, fixed
    center/width and horizon. It cannot qualify the proposed OOD intervals.
    """
    candidate_parameters({**config, 'splits': {'train': 1}})
    records = []
    for index, scale in enumerate(config['train_transport']):
        baseline = f'pilot-baseline-{index}'
        records.append({'case_id': baseline, 'parent_family_id': baseline,
                        'split': 'development', 'A_rho': 0., 'A_T': 0.,
                        's_perp': float(scale), 'baseline_id': baseline})
        for rho, temperature in product(config['train_amplitude'], repeat=2):
            case = f'pilot-s{index}-rho{rho:g}-T{temperature:g}'
            records.append({'case_id': case, 'parent_family_id': case,
                            'split': 'development', 'A_rho': float(rho),
                            'A_T': float(temperature), 's_perp': float(scale),
                            'baseline_id': baseline})
    validate_family_splits(records)
    return records


def transport_profile_overrides(profiles, scale):
    """Scale native D/ZK profile amplitudes (entries 1 and 6), preserving shape."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('transport multiplier must be finite and positive')
    if set(profiles) != {'D_perp', 'ZK_perp'}:
        raise ValueError('both native perpendicular transport profiles are required')
    result = {}
    for name, profile in profiles.items():
        profile = np.asarray(profile, dtype=float)
        if profile.shape != (10,) or not np.isfinite(profile).all():
            raise ValueError('native transport profile requires ten finite entries')
        for index in (1, 6):
            result[f'{name}({index})'] = float(profile[index-1]*scale)
    return result
