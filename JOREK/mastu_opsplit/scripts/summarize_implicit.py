#!/usr/bin/env python3
"""Matched test and fixed-conditioning OOD comparison: implicit OpsSplit arms beside AR/NODE/OpsSplit.

Reads only completed evaluation reports; incomplete or failed-audit horizons are reported as such,
never averaged. Paired comparisons use identical test cases (or OOD cases) within a seed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FIELDS = ('psi', 'u', 'zj', 'w', 'rho', 'T', 'Vpar')


def records(folder):
    """{(case, horizon, substeps): record} from per-trajectory evaluation files of one model."""
    out = {}
    for path in sorted(Path(folder).glob('evaluation_*_substeps*.json')):
        report = json.loads(path.read_text())
        for row in report['records']:
            out[row['case_id'], row['horizon'], row['substeps']] = row
    return out


def aggregate(rows, expected):
    complete = len(rows) == expected and all(row['audit_passed'] for row in rows)
    scores = [row['normalized_volume_mse'] for row in rows]
    return dict(trajectories=len(rows), expected=expected, complete=complete,
                mean=float(np.mean(scores)) if complete else None,
                std=float(np.std(scores, ddof=1)) if complete and len(scores) > 1 else None,
                per_field={f: float(np.mean([row['per_field_normalized_mse'][f] for row in rows])) for f in FIELDS}
                if complete else None,
                dynamic_relative_rms=float(np.mean([np.mean(list(row['dynamic_relative_rms'].values())) for row in rows]))
                if complete else None)


def summarize_model(folder, expected):
    data = records(folder)
    keys = sorted({(h, s) for _, h, s in data})
    return {f'{h}/{s}': aggregate([data[k] for k in data if k[1:] == (h, s)], expected) for h, s in keys}, data


def paired(left, right, horizon, substeps):
    cases = sorted({c for c, h, s in left if (h, s) == (horizon, substeps)} & {c for c, h, s in right if (h, s) == (horizon, substeps)})
    if not cases:
        return None
    a = np.array([left[c, horizon, substeps]['normalized_volume_mse'] for c in cases])
    b = np.array([right[c, horizon, substeps]['normalized_volume_mse'] for c in cases])
    return dict(cases=len(cases), left_better=int((a < b).sum()), median_ratio=float(np.median(a/b)),
                mean_ratio=float(a.mean()/b.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--ensemble', type=Path, required=True)
    parser.add_argument('--gradient', type=Path, help='learned-gradient OpsSplit retraining run (seeds 18/19)')
    parser.add_argument('--ood', type=Path, required=True)
    parser.add_argument('--coefficient', type=Path, help='coefficient-physics OOD evaluations of the implicit arms')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    models = {}
    for seed in (17, 18, 19):
        for arm in ('implicit', 'implicit_rollout', 'physics'):
            folder = args.campaign/f'seed{seed}_{arm}'/arm
            if folder.exists():
                models[arm, seed] = folder
        for family in ('ar', 'node', 'opsplit'):
            folder = args.ensemble/f'seed{seed}_{family}'/family
            if folder.exists() and (folder.parent/'results.json').exists():
                models[family, seed] = folder
        if args.gradient is not None and ('opsplit', seed) not in models:
            folder = args.gradient/f'seed{seed}_learned'/'opsplit'
            if folder.exists():
                models['opsplit', seed] = folder
    test, test_records = {}, {}
    for (name, seed), folder in models.items():
        test[f'{name}/seed{seed}'], test_records[name, seed] = summarize_model(folder, 20)
    def ood_tables(root):
        table, recs = {}, {}
        for folder in sorted(root.glob('seed*_*')) if root is not None else []:
            if folder.name.startswith('smoke') or not (folder/'provenance.json').exists():
                continue
            seed, name = folder.name.split('_', 1)
            table[f'{name}/{seed}'], recs[name, int(seed[4:])] = summarize_model(folder, 10)
        return table, recs
    ood, ood_records = ood_tables(args.ood)
    ood_coefficient, coefficient_records = ood_tables(args.coefficient)
    comparisons = {}
    for split, recs in (('test', test_records), ('ood', ood_records), ('ood_coefficient', coefficient_records)):
        reference = ood_records if split == 'ood_coefficient' else recs   # AR/NODE: fixed-conditioning OOD
        for (name, seed), data in recs.items():
            if name in ('implicit', 'implicit_rollout', 'physics') and ('node', 17) in reference:
                for horizon in (16, 32, 96):
                    comparisons[f'{split}/{name}/seed{seed}_vs_node/seed17/h{horizon}'] = paired(data, reference['node', 17], horizon, 1)
    seed_means = {}
    for name in ('implicit', 'implicit_rollout', 'opsplit', 'node', 'ar'):
        for key in ('16/1', '32/1', '96/1'):
            values = [test[f'{name}/seed{s}'][key]['mean'] for s in (17, 18, 19)
                      if f'{name}/seed{s}' in test and key in test[f'{name}/seed{s}'] and test[f'{name}/seed{s}'][key]['complete']]
            if values:
                seed_means[f'{name}/{key}'] = dict(seeds=len(values), mean=float(np.mean(values)),
                                                    std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
    result = dict(test=test, ood=ood, ood_coefficient=ood_coefficient, paired=comparisons, seed_means=seed_means,
                  notes=['OOD networks receive conditioning 1 (training value); the D_perp multiplier is withheld.',
                         'ood_coefficient: same networks and conditioning 1, but the implicit physics uses the true D_perp(1) '
                         '(exact affine dependence of the native RHS); paired against fixed-conditioning AR/NODE.',
                         'Implicit arms add native linear physics and exact-label pretraining; AR/NODE/OpsSplit do not.',
                         'Incomplete or failed-audit horizons have mean None and are never averaged.'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    lines = ['| model | 16 | 32 | 96 |', '|---|---:|---:|---:|']
    for label, table in (('test', test), ('OOD', ood), ('OOD+D', ood_coefficient)):
        for key in sorted(table):
            cells = [table[key].get(f'{h}/1', {}).get('mean') for h in (16, 32, 96)]
            lines.append(f'| {label} {key} | ' + ' | '.join('n/a' if c is None else f'{c:.4g}' for c in cells) + ' |')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
