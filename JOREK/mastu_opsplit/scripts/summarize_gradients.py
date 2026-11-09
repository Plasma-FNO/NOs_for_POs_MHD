#!/usr/bin/env python3
"""Compare matched native/learned continuity gradients on held-out trajectories."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mastu_opsplit.common import config_hash, sha256_file, write_json


SEEDS = (17, 18, 19)
BRANCHES = {'learned': 'lspline_continuity', 'native': 'native_continuity'}
FIELDS = ('psi', 'u', 'zj', 'w', 'rho', 'T', 'Vpar')


def horizon_records(evaluation, horizon, substeps):
    rows = [row for result in evaluation['trajectories'] for row in result['records']
            if row['horizon'] == horizon and row['substeps'] == substeps]
    records = {(row['case_id'], row['start_frame']): row for row in rows}
    if len(records) != len(rows) or any(start != 0 for _, start in records):
        raise ValueError('Require unique held-out trajectories starting at frame zero')
    aggregates = [row for row in evaluation['aggregates']
                  if row['horizon'] == horizon and row['substeps'] == substeps]
    if len(aggregates) != 1:
        raise ValueError('Require exactly one aggregate per horizon/substep')
    aggregate = aggregates[0]
    complete = bool(aggregate['complete'] and aggregate['expected_trajectories'] == 20
                    and aggregate['trajectories'] == len(records) == 20
                    and all(row['audit_passed'] and np.isfinite(row['normalized_volume_mse'])
                            and np.isfinite([row['per_field_normalized_mse'][field] for field in FIELDS]).all()
                            for row in rows))
    if complete:
        np.testing.assert_allclose(aggregate['mean_normalized_volume_mse'],
                                   np.mean([row['normalized_volume_mse'] for row in rows]), rtol=1e-12, atol=0)
    return records, complete


def summarize(root):
    root = Path(root)
    required = [root/f'seed{seed}_{variant}'/name for seed in SEEDS for variant in BRANCHES
                for name in ('provenance.json', 'normalization.npz', 'opsplit/metrics.json', 'opsplit/sampled_windows.json')]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        return dict(status='incomplete', missing=missing, comparison=[])
    results, matching = {}, []
    for seed in SEEDS:
        signatures = []
        for variant, branch in BRANCHES.items():
            folder = root/f'seed{seed}_{variant}'
            provenance = json.loads((folder/'provenance.json').read_text())
            config = provenance['config']
            metrics = json.loads((folder/'opsplit/metrics.json').read_text())
            if config['seed'] != seed or config['linear_branch'] != branch or config['steps'] != 2000 or \
                    config['horizon'] != 4 or config['stride'] != 1 or config.get('diagnostic_only') or \
                    config.get('evaluation_cases') is not None or config['evaluation_horizons'] != [16, 32, 96] or \
                    config['evaluation_substeps'] != [1, 2, 4] or metrics['selected_step'] != 2000:
                raise ValueError('Require the prescribed full-budget gradient experiment')
            schedule = folder/'opsplit/sampled_windows.json'
            if len(json.loads(schedule.read_text())) != config['steps']:
                raise ValueError('Incomplete training-window schedule')
            canonical = copy.deepcopy(provenance)
            del canonical['config']['linear_branch']
            with np.load(folder/'normalization.npz', allow_pickle=False) as saved:
                normalization_hash = config_hash({name: dict(shape=saved[name].shape, dtype=str(saved[name].dtype),
                    sha256=hashlib.sha256(saved[name].tobytes()).hexdigest()) for name in saved.files})
            signatures.append((config_hash(canonical), metrics['initial_neural_sha256'], metrics['parameters'],
                               sha256_file(schedule), normalization_hash))
            results[seed, variant] = metrics['evaluation']
        if len(set(signatures)) != 1:
            raise ValueError(f'Native/learned runs are not matched for seed {seed}')
        matching.append(dict(seed=seed, canonical_provenance_sha256=signatures[0][0],
                             initial_neural_sha256=signatures[0][1], parameters=signatures[0][2],
                             sampled_windows_sha256=signatures[0][3], normalization_content_sha256=signatures[0][4]))
    comparison = []
    for horizon in (16, 32, 96):
        for substeps in (1, 2, 4):
            paired = []
            for seed in SEEDS:
                records, valid = {}, {}
                for variant in BRANCHES:
                    records[variant], valid[variant] = horizon_records(results[seed, variant], horizon, substeps)
                complete = all(valid.values())
                if complete and records['learned'].keys() != records['native'].keys():
                    raise ValueError('Native/learned held-out trajectory membership differs')
                scores = {variant: float(np.mean([row['normalized_volume_mse'] for row in records[variant].values()]))
                          if valid[variant] else None for variant in BRANCHES}
                delta = [records['native'][key]['normalized_volume_mse']-records['learned'][key]['normalized_volume_mse']
                         for key in sorted(records['learned'])] if complete else []
                field_scores = {variant: {field: float(np.mean([row['per_field_normalized_mse'][field]
                                for row in records[variant].values()])) for field in FIELDS}
                                if valid[variant] else {} for variant in BRANCHES}
                paired.append(dict(seed=seed, complete=complete, trajectories={key: len(value) for key, value in records.items()},
                    mean_normalized_volume_mse=scores, per_field_normalized_mse=field_scores,
                    native_minus_learned=float(np.mean(delta)) if complete else None,
                    paired_trajectory_std=float(np.std(delta, ddof=1)) if complete else None,
                    trajectories_with_lower_native_mse=sum(value < 0 for value in delta) if complete else None))
            complete = all(row['complete'] for row in paired)
            summary = {}
            for variant in BRANCHES:
                scores = [row['mean_normalized_volume_mse'][variant] for row in paired]
                summary[variant] = dict(seed_means=scores,
                    mean=float(np.mean(scores)) if complete else None,
                    seed_std=float(np.std(scores, ddof=1)) if complete else None,
                    per_field_normalized_mse={field: float(np.mean([row['per_field_normalized_mse'][variant][field]
                        for row in paired])) for field in FIELDS} if complete else {})
            deltas = [row['native_minus_learned'] for row in paired]
            baseline = summary['learned']['mean']
            comparison.append(dict(horizon=horizon, substeps=substeps, complete=complete, variants=summary,
                native_minus_learned=float(np.mean(deltas)) if complete else None,
                paired_seed_std=float(np.std(deltas, ddof=1)) if complete else None,
                native_change_percent=100*(summary['native']['mean']/baseline-1) if complete and baseline > 0 else None,
                paired_seeds=paired))
    passed = all(value['audits_passed'] for value in results.values()) and all(row['complete'] for row in comparison)
    return dict(status='complete' if passed else 'failed_audits', seeds=list(SEEDS), matching=matching,
        comparison=comparison, all_audits_passed=passed,
        numerical_failures=sum(value['numerical_failures'] for value in results.values()),
        checkpoint_selection='final fixed-budget update; no selection using held-out scores',
        uncertainty_unit='matched seed means and paired held-out trajectories; overlapping horizons are not independent samples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        report = summarize(args.root)
    except (ValueError, KeyError, AssertionError) as error:
        report = dict(status='invalid_comparison', error=str(error), comparison=[])
    write_json(args.output, report)
    if report['status'] != 'complete':
        raise SystemExit(1)
