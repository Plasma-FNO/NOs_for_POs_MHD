#!/usr/bin/env python3
"""Summarize the GPU one-step campaign: AR, NODE, explicit OpsSplit, conditional and implicit arms.

Layout under --root (slurm/gpu_onestep.sbatch):
  seed{S}_{model}/            training run; test evaluations in {model}/ (implicit arms: {arm}/)
  ood/seed{S}_{model}/        fixed-conditioning D_perp OOD (network conditioning 1)
  ood_film/seed{S}_conditional/      conditional with the true D_perp/D0 in FiLM
  ood_coefficient/seed{S}_{arm}/     implicit arms with the true D_perp in the frozen physics
Aggregation and pairing are summarize_implicit.py's: a horizon is averaged only when every
trajectory is present and passed its audits; paired ratios use identical cases within a seed.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_implicit import FIELDS, paired, summarize_model  # noqa: E402

MODELS = ('ar', 'node', 'opsplit', 'conditional', 'implicit', 'implicit_rollout', 'physics')
IMPLICIT = ('implicit', 'implicit_rollout', 'physics')
SEEDS = (17, 18, 19)
SPLITS = dict(test=('', 20), ood=('ood', 10), ood_film=('ood_film', 10), ood_coefficient=('ood_coefficient', 10))


def training_seconds(run, model):
    """Summed optimizer-update wall time per stage; None unless the stage finished its full budget."""
    out = {}
    stages = dict(rollout=run/model/'rollout'/'history.json', pretrain=run/model/'pretrain'/'history.json') \
        if model in IMPLICIT else dict(rollout=run/model/'history.json')
    provenance = json.loads((run/'provenance.json').read_text()) if (run/'provenance.json').exists() else {}
    budget = dict(rollout=provenance.get('config', {}).get('steps'), pretrain=provenance.get('config', {}).get('pretrain_steps'))
    for stage, path in stages.items():
        if path.exists():
            history = json.loads(path.read_text())
            if len(history) == budget[stage]:
                out[stage] = float(sum(row.get('training_seconds', row.get('seconds', 0.)) for row in history))
    return out


def evaluation_folder(root, split, model, seed):
    sub, _ = SPLITS[split]
    if split == 'test':
        return root/f'seed{seed}_{model}'/model
    return root/sub/f'seed{seed}_{model}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs, records = {}, {}
    for model in MODELS:
        for seed in SEEDS:
            run = args.root/f'seed{seed}_{model}'
            if not run.exists():
                continue
            entry = runs.setdefault(model, {})[str(seed)] = {}
            metrics = run/model/'metrics.json'
            if metrics.exists():
                m = json.loads(metrics.read_text())
                entry.update(parameters=m.get('parameters'), device_name=m.get('device_name'),
                             evaluation_seconds=m.get('evaluation_seconds'), peak_device_gib=m.get('peak_device_gib'))
            entry['training_seconds'] = training_seconds(run, model)
            for split, (_, expected) in SPLITS.items():
                folder = evaluation_folder(args.root, split, model, seed)
                if folder.exists() and any(folder.glob('evaluation_*_substeps*.json')):
                    entry[split], records[split, model, seed] = summarize_model(folder, expected)
    seed_means = {}
    for split in SPLITS:
        for model in MODELS:
            for key in ('16/1', '32/1', '96/1', '16/2', '32/2', '96/2', '16/4', '32/4', '96/4'):
                values = [runs[model][str(s)][split][key]['mean'] for s in SEEDS
                          if str(s) in runs.get(model, {}) and key in runs[model][str(s)].get(split, {})
                          and runs[model][str(s)][split][key]['complete']]
                if values:
                    expected_seeds = 1 if model == 'physics' else len(SEEDS)
                    fields = [runs[model][str(s)][split][key]['per_field'] for s in SEEDS
                              if str(s) in runs.get(model, {}) and key in runs[model][str(s)].get(split, {})
                              and runs[model][str(s)][split][key]['complete']]
                    seed_means.setdefault(split, {}).setdefault(model, {})[key] = dict(
                        seeds=len(values), expected_seeds=expected_seeds, mean=float(np.mean(values)),
                        min=float(np.min(values)), max=float(np.max(values)),
                        std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        per_field={f: float(np.mean([row[f] for row in fields])) for f in FIELDS})
    comparisons = {}
    for (split, model, seed), data in records.items():
        if model == 'node':
            continue
        reference_split = 'ood' if split in ('ood_film', 'ood_coefficient') else split   # NODE: conditioning 1
        for seed_ref in ([seed] if model != 'physics' else SEEDS):   # physics-only is seed-free
            reference = records.get((reference_split, 'node', seed_ref))
            if reference is None:
                continue
            for horizon in (16, 32, 96):
                result = paired(data, reference, horizon, 1)
                if result is not None:
                    comparisons[f'{split}/{model}/seed{seed}_vs_node/seed{seed_ref}/h{horizon}'] = result
    training = {}
    for model, seeds in runs.items():
        for stage in ('rollout', 'pretrain'):
            values = [entry['training_seconds'][stage] for entry in seeds.values() if stage in entry['training_seconds']]
            if values:
                training.setdefault(model, {})[stage] = dict(seeds=len(values), mean_seconds=float(np.mean(values)))
    result = dict(root=str(args.root.resolve()), runs=runs, seed_means=seed_means, paired=comparisons, training=training,
                  notes=['All models take one saved state to the next (h = one interval) and train on the same four-step '
                         'rollout windows for 2,000 updates; implicit labels + rollout adds 2,000 exact-label updates.',
                         'ood: network conditioning held at its training value 1; ood_film: conditional with the true '
                         'D_perp/D0 in FiLM (extrapolation); ood_coefficient: implicit physics with the true D_perp.',
                         'Incomplete or failed-audit horizons have mean None and are never averaged.'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    lines = ['| split | model | seeds | 16 | 32 | 96 |', '|---|---|---|---:|---:|---:|']
    for split, table in seed_means.items():
        for model, rows in table.items():
            cells = [rows.get(f'{h}/1', {}).get('mean') for h in (16, 32, 96)]
            seeds = max(row['seeds'] for row in rows.values())
            lines.append(f'| {split} | {model} | {seeds} | ' + ' | '.join('n/a' if c is None else f'{c:.4g}' for c in cells) + ' |')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
