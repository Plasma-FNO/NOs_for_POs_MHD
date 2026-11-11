#!/usr/bin/env python3
"""Compare completed conditional and rollout runs; withhold incomplete three-seed means."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mastu_opsplit.common import config_hash, write_json

FIELDS = ('psi', 'u', 'zj', 'w', 'rho', 'T', 'Vpar')
HORIZONS = (16, 32, 96)
SEEDS = (17, 18, 19)


def identity(provenance):
    config, normalization = provenance['config'], provenance['normalization']
    return dict(manifest=config['manifest_sha256'], splits=config['split_counts'],
                steps=config['steps'], horizon=config['horizon'], stride=config['stride'],
                mean=normalization['mean_sha256'], scale=normalization['scale_sha256'],
                training_ids=sorted(normalization['training_case_ids']))


def read_run(run, family, reference, expected_cases):
    metrics_path = run/family/'metrics.json'
    if not (run/'results.json').exists() or not metrics_path.exists():
        return dict(status='incomplete_or_missing', path=str(run)), {}
    provenance = json.loads((run/'provenance.json').read_text())
    if identity(provenance) != identity(reference):
        return dict(status='incompatible_data_or_budget', path=str(run)), {}
    metrics = json.loads(metrics_path.read_text())
    if metrics.get('selected_step') != provenance['config']['steps']:
        return dict(status='not_final_budget_checkpoint', path=str(run)), {}
    records = {}
    for path in sorted((run/family).glob('evaluation_*_substeps1.json')):
        result = json.loads(path.read_text())
        if result['provenance_hash'] != config_hash(provenance):
            return dict(status='evaluation_provenance_mismatch', path=str(run)), {}
        for row in result['records']:
            key = (row['case_id'], row['horizon'])
            if key in records or row['substeps'] != 1 or row['start_frame'] != 0:
                return dict(status='invalid_evaluation_identity', path=str(run)), {}
            records[key] = row
    scores = {}
    for horizon in HORIZONS:
        rows = [row for (case, h), row in records.items() if h == horizon]
        complete = ({row['case_id'] for row in rows} == set(expected_cases) and
                    all(row['audit_passed'] and np.isfinite(row['normalized_volume_mse']) and
                        all(np.isfinite(row['per_field_normalized_mse'][field]) for field in FIELDS)
                        for row in rows))
        if family == 'conditional':
            complete = complete and all(np.isfinite(row['raw_normalized_volume_mse']) for row in rows)
        scores[str(horizon)] = dict(complete=complete, trajectories=len(rows), expected=len(expected_cases),
            mean=float(np.mean([row['normalized_volume_mse'] for row in rows])) if complete else None,
            per_field={field: float(np.mean([row['per_field_normalized_mse'][field] for row in rows]))
                       for field in FIELDS} if complete else None)
        if complete:
            scores[str(horizon)].update(
                mean_prediction_and_audit_seconds=float(np.mean([row['elapsed_seconds'] for row in rows])),
                limited_field_queries=sum(row['guard_statistics']['limited_field_steps'] for row in rows),
                minimum_retained_fraction=min(row['guard_statistics']['minimum_alpha'] for row in rows))
            if family == 'conditional':
                scores[str(horizon)]['raw_mean'] = float(np.mean([row['raw_normalized_volume_mse'] for row in rows]))
                scores[str(horizon)]['time_regime'] = sorted({row['query_time_regime'] for row in rows})
    history = metrics.get('history', metrics.get('rollout_history', []))
    return dict(status='evaluated', path=str(run), scores=scores, parameters=metrics['parameters'],
                training_seconds=sum(row['training_seconds'] for row in history),
                overall_audits_passed=metrics['evaluation']['audits_passed']), records


def summarize(root, ensemble, gradient, implicit):
    reference_paths = sorted(root.glob('seed*_conditional*/provenance.json'))
    if not reference_paths:
        raise ValueError('No conditional training provenance available')
    reference = json.loads(reference_paths[0].read_text())
    manifest = json.loads(Path(reference['config']['manifest']).read_text())
    expected = [case['case_id'] for case in manifest['cases'] if case['split'] == 'test']
    models, records, schedules = {}, {}, {}
    names = ('conditional', 'conditional_matched', 'ar', 'node', 'opsplit', 'implicit', 'implicit_rollout')
    for seed in SEEDS:
        for name in names:
            family = 'conditional' if name.startswith('conditional') else name
            if name.startswith('conditional'):
                run = root/f'seed{seed}_{name}'
            elif name.startswith('implicit'):
                run = implicit/f'seed{seed}_{name}'
            else:
                run = ensemble/f'seed{seed}_{name}'
                if name == 'opsplit' and not (run/'results.json').exists():
                    run = gradient/f'seed{seed}_learned'
            key = f'{name}/seed{seed}'
            models[key], records[key] = read_run(run, family, reference, expected)
            schedule = run/family/('rollout/sampled_windows.json' if name.startswith('implicit') else 'sampled_windows.json')
            if schedule.exists():
                schedules[key] = [{k: row[k] for k in ('index', 'case_id', 'start_frame')}
                                  for row in json.loads(schedule.read_text())]
    comparisons = []
    for seed in SEEDS:
        for arm in ('conditional', 'conditional_matched'):
            left_key = f'{arm}/seed{seed}'
            for baseline in ('ar', 'node', 'opsplit', 'implicit', 'implicit_rollout'):
                right_key = f'{baseline}/seed{seed}'
                matched = (left_key in schedules and right_key in schedules and schedules[left_key] == schedules[right_key])
                for horizon in HORIZONS:
                    complete = all(models[key].get('scores', {}).get(str(horizon), {}).get('complete', False)
                                   for key in (left_key, right_key))
                    if arm == 'conditional_matched' and not matched:
                        complete = False
                    differences = [records[left_key][case, horizon]['normalized_volume_mse'] -
                                   records[right_key][case, horizon]['normalized_volume_mse'] for case in expected] if complete else []
                    comparisons.append(dict(arm=arm, baseline=baseline, seed=seed, horizon=horizon,
                        complete=complete, sampled_windows_match=matched,
                        mean_paired_mse_difference=float(np.mean(differences)) if complete else None,
                        paired_trajectory_std=float(np.std(differences, ddof=1)) if complete and len(differences)>1 else None))
    seed_summary = []
    for name in names:
        for horizon in HORIZONS:
            values = [models[f'{name}/seed{seed}'].get('scores', {}).get(str(horizon), {}) for seed in SEEDS]
            complete = all(row.get('complete', False) for row in values)
            seed_summary.append(dict(model=name, horizon=horizon, complete=complete,
                completed_seeds=sum(row.get('complete', False) for row in values),
                mean=float(np.mean([row['mean'] for row in values])) if complete else None,
                seed_std=float(np.std([row['mean'] for row in values], ddof=1)) if complete else None))
    return dict(models=models, seed_summary=seed_summary, paired=comparisons, notes=[
        'Primary three-seed means require seeds 17, 18 and 19; missing/failed runs remain explicit.',
        'Scores average future frames through each horizon; all comparisons use one solver substep.',
        'Conditional uses q0 and targets through 96 frames; conditional_matched uses the baseline four-step windows.',
        'Matched direct queries beyond four intervals extrapolate in lead time; no autoregressive feedback is used.',
        'Time normalization differs between the two conditional arms (96 versus 4 intervals).',
        'Implicit includes extra exact-label pretraining; implicit_rollout does not.',
        'Prediction-and-audit timings include physical checks; they are not pure neural inference timings.',
        'PDE coefficients are fixed in training; these runs do not establish coefficient generalization.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'ensemble', 'gradient', 'implicit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.ensemble, args.gradient, args.implicit)
    write_json(args.output, result)
    print(json.dumps(result['seed_summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
