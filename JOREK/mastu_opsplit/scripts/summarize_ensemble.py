#!/usr/bin/env python3
"""Check matching and report seed and trajectory variation without checkpoint selection."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from mastu_opsplit.common import config_hash, sha256_file, write_json


def summarize(root):
    root = Path(root)
    results = {}
    for seed in (17, 18, 19):
        matching = []
        for family in ('ar', 'node', 'opsplit'):
            folder = root/f'seed{seed}_{family}'
            provenance = json.loads((folder/'provenance.json').read_text())
            result = json.loads((folder/family/'metrics.json').read_text())
            matching.append((config_hash(provenance), result['initial_neural_sha256'], result['parameters'],
                             result['selected_step'], sha256_file(folder/family/'sampled_windows.json')))
            if provenance['config']['seed'] != seed or result['selected_step'] != provenance['config']['steps']:
                raise ValueError('Seed or fixed-budget checkpoint mismatch')
            results[seed, family] = result
        if len(set(matching)) != 1:
            raise ValueError(f'Models are not matched for seed {seed}')
    summaries = []
    for family in ('ar', 'node', 'opsplit'):
        for aggregate in results[17, family]['evaluation']['aggregates']:
            rows = [next(row for row in results[seed, family]['evaluation']['aggregates']
                         if row['horizon'] == aggregate['horizon'] and row['substeps'] == aggregate['substeps'])
                    for seed in (17, 18, 19)]
            # Legacy reports only have a global audit flag; fail closed for those.
            valid = [row.get('complete', results[seed, family]['evaluation']['audits_passed'])
                     and row['trajectories'] == 20 and row['mean_normalized_volume_mse'] is not None
                     and np.isfinite(row['mean_normalized_volume_mse']) for seed, row in zip((17, 18, 19), rows)]
            scores = [row['mean_normalized_volume_mse'] if passed else None for row, passed in zip(rows, valid)]
            complete = all(valid)
            summaries.append(dict(family=family, horizon=aggregate['horizon'], substeps=aggregate['substeps'],
                seed_means=scores, mean=float(np.mean(scores)) if complete else None,
                seed_std=float(np.std(scores, ddof=1)) if complete else None,
                within_seed_trajectory_std=[row['trajectory_std'] if passed else None for row, passed in zip(rows, valid)],
                complete=complete))
    return dict(seeds=[17, 18, 19], comparison=summaries,
                all_audits_passed=all(result['evaluation']['audits_passed'] for result in results.values()),
                numerical_failures=sum(result['evaluation']['numerical_failures'] for result in results.values()),
                uncertainty_unit='trajectory and matched training seed, never overlapping windows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    write_json(args.output, summarize(args.root))
