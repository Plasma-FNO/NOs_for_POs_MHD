#!/usr/bin/env python3
"""Summarize the larger-network / learned-Delta-L implicit study against the small-network GPU references.

Runs: --root/seed{S}_{variant}_{arm}/{arm} (test), --root/ood/..., --root/ood_coefficient/...
The converged-Picard Delta L rerun (`*_conv`) is read from --conv-root when given; the truncated Delta L arms of
the first campaign stay labelled `dl/rollout` and `large_dl/rollout`.
References: --reference/seed{S}_{arm}/{arm} for the small-network arms and physics-only (same protocol),
with their OOD folders under --reference/ood and --reference/ood_coefficient.
Aggregation and pairing follow summarize_implicit.py: incomplete or failed-audit horizons are never averaged;
paired ratios use identical cases within a seed (physics-only pairs with every seed).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_implicit import FIELDS, paired, summarize_model  # noqa: E402

SEEDS = (17, 18, 19)
STUDY = {'large/rollout': ('implicit_large', 'implicit_rollout'), 'large/labels': ('implicit_large', 'implicit'),
         'dl/rollout': ('implicit_dl', 'implicit_rollout'), 'large_dl/rollout': ('implicit_large_dl', 'implicit_rollout')}
CONV = {'dl_conv/rollout': ('implicit_dl_conv', 'implicit_rollout'),
        'large_dl_conv/rollout': ('implicit_large_dl_conv', 'implicit_rollout')}
BASELINES = {'dl_conv/rollout': 'small/rollout', 'large_dl_conv/rollout': 'large/rollout',
             'dl/rollout': 'small/rollout', 'large_dl/rollout': 'large/rollout'}   # same network without Delta L
REFERENCE = {'small/rollout': 'implicit_rollout', 'small/labels': 'implicit', 'physics': 'physics'}
SPLITS = {'test': 20, 'ood': 10, 'ood_coefficient': 10}


def folder(root, split, name, arm, seed):
    return root/f'seed{seed}_{name}'/arm if split == 'test' else root/split/f'seed{seed}_{name}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path('/pitagora_scratch/userexternal/vgopakum/mastu_gpu_onestep_20261109'))
    parser.add_argument('--conv-root', type=Path, default=None, help='converged-Picard Delta L rerun')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = {label: (args.root, f'{variant}_{arm}', arm) for label, (variant, arm) in STUDY.items()}
    if args.conv_root is not None:
        sources.update({label: (args.conv_root, f'{variant}_{arm}', arm) for label, (variant, arm) in CONV.items()})
    sources.update({label: (args.reference, arm, arm) for label, arm in REFERENCE.items()})
    runs, records = {}, {}
    for label, (root, name, arm) in sources.items():
        for seed in SEEDS:
            run = root/f'seed{seed}_{name}'
            if not run.exists():
                continue
            entry = runs.setdefault(label, {})[str(seed)] = {}
            metrics = run/arm/'metrics.json'
            if metrics.exists():
                m = json.loads(metrics.read_text())
                entry.update({k: m.get(k) for k in ('parameters', 'correction_parameters', 'correction', 'picard_training',
                                                    'picard_evaluation', 'picard_settings', 'pretraining', 'evaluation_seconds', 'peak_device_gib')})
                history = m.get('rollout_history') or []
                entry['rollout_seconds'] = float(sum(r['training_seconds'] for r in history)) if history else None
                entry['final_rollout_loss'] = float(np.mean([r['training_loss'] for r in history[-200:]])) if history else None
            for split, expected in SPLITS.items():
                path = folder(root, split, name, arm, seed)
                if path.exists() and any(path.glob('evaluation_*_substeps*.json')):
                    entry[split], records[split, label, seed] = summarize_model(path, expected)
    seed_means = {}
    for split in SPLITS:
        for label in sources:
            for h in (16, 32, 96):
                cells = [runs[label][str(s)][split][f'{h}/1'] for s in SEEDS
                         if str(s) in runs.get(label, {}) and f'{h}/1' in runs[label][str(s)].get(split, {})]
                done = [c for c in cells if c['complete']]
                if done:
                    seed_means.setdefault(split, {}).setdefault(label, {})[str(h)] = dict(
                        seeds=len(done), mean=float(np.mean([c['mean'] for c in done])),
                        min=float(min(c['mean'] for c in done)), max=float(max(c['mean'] for c in done)),
                        per_field={f: float(np.mean([c['per_field'][f] for c in done])) for f in FIELDS})
    comparisons = {}
    for (split, label, seed), data in records.items():
        if label in REFERENCE:
            continue
        for reference in dict.fromkeys(('small/rollout', 'physics', BASELINES.get(label, 'physics'))):
            ref = records.get((split, reference, 17 if reference == 'physics' else seed))
            if ref is None:
                continue
            for h in (16, 32, 96):
                result = paired(data, ref, h, 1)
                if result is not None:
                    comparisons[f'{split}/{label}/seed{seed}_vs_{reference}/h{h}'] = result
    result = dict(root=str(args.root.resolve()), reference=str(args.reference.resolve()),
                  conv_root=None if args.conv_root is None else str(args.conv_root.resolve()), runs=runs,
                  seed_means=seed_means, paired=comparisons,
                  notes=['ood: network conditioning 1, D_perp withheld from the physics; ood_coefficient: true D_perp in '
                         'F*, L* (Delta L kept as trained).', 'Incomplete or failed-audit horizons are never averaged.'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    lines = ['| split | model | seeds | 16 | 32 | 96 |', '|---|---|---|---:|---:|---:|']
    for split, table in seed_means.items():
        for label, rows in table.items():
            cells = [rows.get(str(h), {}).get('mean') for h in (16, 32, 96)]
            seeds = max(r['seeds'] for r in rows.values())
            lines.append(f'| {split} | {label} | {seeds} | ' + ' | '.join('n/a' if c is None else f'{c:.4g}' for c in cells) + ' |')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
