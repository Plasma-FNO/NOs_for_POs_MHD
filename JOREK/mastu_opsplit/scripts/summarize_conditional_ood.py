#!/usr/bin/env python3
"""Summarize coefficient extrapolation only when every seed/case passes its audits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ('conditional', 'conditional_matched')
MODES = ('fixed', 'coefficient')
SEEDS = (17, 18, 19)
HORIZONS = (16, 32, 96)


def read_run(folder, training):
    if not (folder/'metrics.json').exists():
        return dict(status='incomplete_or_missing', path=str(folder))
    provenance = json.loads((folder/'provenance.json').read_text())
    metrics = json.loads((folder/'metrics.json').read_text())
    if provenance['config'].get('diagnostic_only') or provenance['normalization_refitted']:
        raise ValueError('Require full evaluation with unchanged training normalization')
    expected_hash = hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    # Use the provenance hash stored by the frozen evaluator; the shared hash uses this canonical encoding.
    cases = {case['case_id']: case for case in provenance['cases']}
    if len(cases) != 10 or len({case['matched_id_case'] for case in cases.values()}) != 10:
        raise ValueError('Require ten distinct OOD cases and matched held-out initial conditions')
    records = {}
    for trajectory in metrics['trajectories']:
        if trajectory['provenance_hash'] != expected_hash:
            raise ValueError('Evaluation provenance hash mismatch')
        for row in trajectory['records']:
            key = (row['case_id'], row['horizon'])
            if key in records or row['substeps'] != 1 or row['start_frame'] != 0:
                raise ValueError('Duplicate or incompatible evaluation record')
            records[key] = row
    id_rows = {}
    for path in (training/'conditional').glob('evaluation_*_substeps1.json'):
        for row in json.loads(path.read_text())['records']:
            id_rows[row['case_id'], row['horizon']] = row
    scores = {}
    for horizon in HORIZONS:
        rows = [row for (case, h), row in records.items() if h == horizon]
        complete = (metrics['audits_passed'] and not metrics['numerical_failures'] and
                    {row['case_id'] for row in rows} == set(cases) and all(
                        row['audit_passed'] and math.isfinite(row['normalized_volume_mse']) and
                        math.isfinite(row['raw_normalized_volume_mse']) and
                        all(math.isfinite(v) for v in row['per_field_normalized_mse'].values()) and
                        all(math.isfinite(v) for v in row['raw_per_field_normalized_mse'].values()) for row in rows))
        score = dict(complete=complete, trajectories=len(rows))
        if complete:
            matched = [id_rows[cases[row['case_id']]['matched_id_case'], horizon] for row in rows]
            if not all(row['audit_passed'] and math.isfinite(row['normalized_volume_mse']) for row in matched):
                raise ValueError('Matched in-distribution evaluation failed')
            score.update(mean=statistics.mean(row['normalized_volume_mse'] for row in rows),
                raw_mean=statistics.mean(row['raw_normalized_volume_mse'] for row in rows),
                matched_id_mean=statistics.mean(row['normalized_volume_mse'] for row in matched),
                limited_field_queries=sum(row['guard_statistics']['limited_field_steps'] for row in rows),
                minimum_retained_fraction=min(row['guard_statistics']['minimum_alpha'] for row in rows),
                per_coefficient=[dict(cases[row['case_id']], mse=row['normalized_volume_mse'],
                    raw_mse=row['raw_normalized_volume_mse'], matched_id_mse=id_row['normalized_volume_mse'])
                    for row, id_row in zip(rows, matched)])
        scores[str(horizon)] = score
    return dict(status='evaluated', path=str(folder), scores=scores,
                ood_manifest_sha256=provenance['ood_manifest_sha256'],
                normalization_sha256=provenance['normalization_sha256'])


def summarize(root, training_root):
    runs, summary = {}, []
    for arm in ARMS:
        for mode in MODES:
            for seed in SEEDS:
                key = f'seed{seed}_{arm}_{mode}'
                runs[key] = read_run(root/key, training_root/f'seed{seed}_{arm}')
            for horizon in HORIZONS:
                scores = [runs[f'seed{seed}_{arm}_{mode}'].get('scores', {}).get(str(horizon), {}) for seed in SEEDS]
                complete = all(score.get('complete', False) for score in scores)
                row = dict(arm=arm, conditioning=mode, horizon=horizon, complete=complete,
                           completed_seeds=sum(score.get('complete', False) for score in scores))
                for metric in ('mean', 'raw_mean', 'matched_id_mean'):
                    row[metric] = statistics.mean(score[metric] for score in scores) if complete else None
                row['seed_std'] = statistics.stdev(score['mean'] for score in scores) if complete else None
                summary.append(row)
    for key in ('ood_manifest_sha256', 'normalization_sha256'):
        if len({run[key] for run in runs.values() if key in run}) > 1:
            raise ValueError('Runs differ in OOD dataset or training normalization')
    return dict(runs=runs, seed_summary=summary, notes=[
        'All checkpoints trained at D/D0=1; true-coefficient inputs test extrapolation, not parameter-varied training.',
        'Fixed conditioning withholds the coefficient; true-coefficient conditioning supplies D/D0 in [2,10].',
        'Ten coefficient values have ten different held-out initial conditions; coefficient and initial-condition effects are confounded.',
        'Matched ID errors use exactly the corresponding ten held-out initial conditions at baseline diffusivity.',
        'Scores average future frames through each horizon; mean and sample standard deviation use three training seeds.',
        'The matched four-step arm also extrapolates in lead time; the full-time arm stays within its trained time range.',
        'Raw predictions and guard interventions are retained; physical audit success does not imply prediction accuracy.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'training-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.training_root)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)
    print(json.dumps(result['seed_summary'], indent=2))


if __name__ == '__main__':
    main()
