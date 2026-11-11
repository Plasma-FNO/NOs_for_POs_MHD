#!/usr/bin/env python3
"""Verify benchmark provenance, assemble matched video inputs, and summarize timings.

No model execution. Missing/failed runs remain explicit; complete primary videos
require all requested trajectories and every recorded physical audit to pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'src'))
from mastu_opsplit.common import sha256_file

LABELS = {'ground_truth': 'Ground truth · JOREK', 'ar': 'Autoregressive',
          'node': 'Neural ODE', 'implicit': 'Implicit Neural OpsSplit', 'physics': 'Native physics only'}
PRIMARY = {'ar': 1, 'node': 1, 'implicit': 2}
METRIC = ('Cumulative volume-weighted normalized seven-field MSE over frames 1:H; '
          'normalization scales are fitted to 80 training trajectories. Mean over cases, '
          'not a relative percentage error. Ground truth exists only through frame 96.')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_run(path, audit, *, verify_arrays=False):
    report = json.loads((path / 'report.json').read_text())
    if report['status'] not in ('complete', 'numerical_failure'):
        return report, None
    method = report['method']
    audited = next(row for row in audit['methods'] if row['method'] == ('implicit' if method == 'physics' else method))
    for name in ('checkpoint', 'provenance'):
        item = report[name]
        if item['sha256'] != audited[f'{name}_sha256'] or sha256_file(item['path']) != item['sha256']:
            raise ValueError(f'{path}: changed or unaudited {name}')
    if report['config']['manifest_sha256'] != audited['manifest_sha256']:
        raise ValueError(f'{path}: changed data release')
    if report['training_updates'] != (0 if method == 'physics' else audited['training_updates']):
        raise ValueError(f'{path}: wrong training budget')
    for name, expected in report['source_sha256'].items():
        if sha256_file(STUDY / 'src/mastu_opsplit' / name) != expected:
            raise ValueError(f'{path}: runtime source changed: {name}')
    if report['checkpoint']['step'] != audited['training_updates']:
        raise ValueError(f'{path}: checkpoint step mismatch')
    if not report['timing_trials'] or any(not np.isfinite(row['rollout_seconds']) or row['rollout_seconds'] <= 0
                                         for row in report['timing_trials']):
        raise ValueError(f'{path}: missing or invalid measured timing')
    if report['status'] == 'complete' and any(row['completed_intervals'] != report['horizon'] or row['failure'] is not None
                                            for row in report['timing_trials']):
        raise ValueError(f'{path}: complete status disagrees with trial completion')
    arrays = None
    if verify_arrays:
        source = path / 'trajectory.npz'
        if sha256_file(source) != report['trajectory_sha256']:
            raise ValueError(f'{source}: trajectory hash mismatch')
        with np.load(source, allow_pickle=False) as stream:
            arrays = {key: stream[key] for key in stream.files}
        for name in ('mean', 'scale'):
            if hashlib.sha256(arrays[name].tobytes()).hexdigest() != report['normalization'][f'{name}_sha256']:
                raise ValueError(f'{source}: normalization {name} mismatch')
        for field, mask in (('prediction', 'valid_prediction'), ('ground_truth', 'valid_ground_truth')):
            if len(arrays[field]) != report['horizon'] + 1 or arrays[mask].shape != (report['horizon'] + 1,):
                raise ValueError(f'{source}: wrong frame count')
            if not np.isfinite(arrays[field][arrays[mask]]).all():
                raise ValueError(f'{source}: nonfinite valid {field}')
            if not np.isnan(arrays[field][~arrays[mask]]).all():
                raise ValueError(f'{source}: invalid frames must be NaN, not filled predictions')
        np.testing.assert_array_equal(arrays['prediction'][0], arrays['ground_truth'][0])
    return report, arrays


def brief(report):
    trials = report.get('timing_trials', [])
    scores = report.get('scores', {})
    h96 = scores.get('horizons', {}).get('96', {})
    return dict(method=report['method'], label=LABELS[report['method']], case_id=report['case_id'],
                horizon=report['horizon'], microsteps=report['actual_microsteps'], status=report['status'],
                complete=report['status'] == 'complete',
                audits_passed=scores.get('all_completed_frame_audits_passed', False),
                rollout_seconds=report.get('median_rollout_seconds'),
                timing_samples=[row['rollout_seconds'] for row in trials],
                completed_intervals=[row['completed_intervals'] for row in trials],
                training_updates=report['training_updates'], normalized_volume_mse=h96.get('normalized_volume_mse'),
                per_field_normalized_mse=h96.get('per_field_normalized_mse', {}),
                horizons=scores.get('horizons', {}),
                failure=[row['failure'] for row in trials if row.get('failure')],
                guard_statistics=[row['guard_statistics'] for row in trials],
                stage_statistics=[row.get('stage_statistics') for row in trials],
                reference_frames=report['reference_frames'],
                ground_truth_validated_through=min(96, report['horizon']),
                long_horizon_scope=('Survival and physical audits only beyond frame 96; accuracy unmeasured'
                                    if report['horizon'] > 96 else 'Ground truth available throughout'),
                setup_seconds=report['setup_seconds'],
                cold_first_frame_seconds=report.get('cold_first_frame_seconds'),
                timing_scope=report['timing_scope'])


def assemble_case(root, case_id, reports, audit, *, horizon=96):
    loaded = {}
    for method in PRIMARY:
        path, report = reports[method]
        _, arrays = load_run(path, audit, verify_arrays=True)
        if horizon == 96 and not report['scores']['all_completed_frame_audits_passed']:
            raise ValueError(f'{path}: primary trajectory has failed physical audits')
        if horizon > 96:
            prefix_path = root / 'runs' / f'{case_id}_{method}_m{PRIMARY[method]}_h96'
            prefix_report = json.loads((prefix_path / 'report.json').read_text())
            if report['source_sha256'] != prefix_report['source_sha256'] or report['checkpoint'] != prefix_report['checkpoint']:
                raise ValueError(f'{path}: long rollout source/checkpoint differs from primary')
            with np.load(prefix_path / 'trajectory.npz', allow_pickle=False) as prefix:
                valid = arrays['valid_prediction'][:97]
                np.testing.assert_array_equal(arrays['prediction'][:97][valid], prefix['prediction'][valid],
                                               err_msg=f'{path}: long-rollout prefix differs from primary')
        loaded[method] = arrays
    first = loaded['ar']
    manifest_path = Path(reports['ar'][1]['config']['manifest'])
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(manifest_path) != reports['ar'][1]['config']['manifest_sha256']:
        raise ValueError('Changed native dataset manifest')
    case = next(row for row in manifest['cases'] if row['case_id'] == case_id)
    restart = Path(case['initial'])
    if not restart.is_absolute():
        restart = manifest_path.parent / restart
    if sha256_file(restart) != case['initial_check']['sha256']:
        raise ValueError(f'{case_id}: initial restart hash mismatch')
    native_path = manifest_path.parent / case['result']['data']['path']
    if sha256_file(native_path) != case['result']['data']['sha256']:
        raise ValueError(f'{case_id}: native ground-truth file changed')
    with h5py.File(native_path) as native:
        reference_frames = min(horizon + 1, len(native['time_seconds']))
        np.testing.assert_array_equal(first['valid_ground_truth'], np.arange(horizon + 1) < reference_frames)
        np.testing.assert_allclose(first['time_seconds'][:reference_frames], native['time_seconds'][:reference_frames],
                                   rtol=0, atol=1e-18)
        for frame in range(reference_frames):
            np.testing.assert_allclose(first['ground_truth'][frame], native['native_state'][frame],
                                       rtol=2e-14, atol=2e-15,
                                       err_msg=f'{case_id}: decoded ground truth differs from native frame {frame}')
    arrays = {'time_seconds': first['time_seconds'], 'ground_truth': first['ground_truth'],
              'valid_ground_truth': first['valid_ground_truth']}
    native_timing = next(row for row in audit['native_ground_truth'] if row['case_id'] == case_id)
    methods = {'ground_truth': dict(label=LABELS['ground_truth'], rollout_seconds=native_timing['wall_seconds'],
                                   timing_label=('native · 4 CPU MPI ranks' if horizon == 96
                                                 else '4 CPU · H96 reference only'), training_updates=0,
                                   timing_horizon=96,
                                   timing_scope=native_timing['timing_scope'])}
    rows = []
    for method, values in loaded.items():
        for field in ('time_seconds', 'mean', 'scale', 'ground_truth', 'valid_ground_truth'):
            np.testing.assert_array_equal(first[field], values[field], err_msg=f'{case_id}: mismatched {field}')
        arrays[method] = values['prediction']
        arrays[f'valid_{method}'] = values['valid_prediction']
        row = brief(reports[method][1])
        rows.append(row)
        timing_label = 'GPU rollout · H100' if horizon == 96 else f'H{horizon} GPU rollout'
        if not row['complete']:
            timing_label = f"GPU prefix · {row['completed_intervals'][0]}/{horizon} intervals"
        methods[method] = {**row, 'timing_label': timing_label}
    folder = root / 'cases' / (case_id if horizon == 96 else f'{case_id}_h{horizon}')
    folder.mkdir(parents=True, exist_ok=True)
    trajectory = folder / 'trajectory.npz'
    # Reuse only when exact source files and previous merged artifact still match.
    sources = {method: sha256_file(path / 'report.json') for method, (path, _) in reports.items()}
    previous_path = folder / 'metadata.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if not (previous.get('source_report_sha256') == sources and trajectory.exists()
            and previous.get('trajectory_sha256') == sha256_file(trajectory)):
        temporary = folder / 'trajectory.tmp.npz'
        np.savez_compressed(temporary, **arrays)
        temporary.replace(trajectory)
    metadata = dict(case_id=case_id, restart_path=str(restart.resolve()), restart_sha256=sha256_file(restart),
                    normalization_scale=first['scale'].tolist(), methods=methods,
                    source_report_sha256=sources, trajectory_sha256=sha256_file(trajectory),
                    metric_definition=METRIC, fields=['psi', 'u', 'zj', 'w', 'rho', 'T', 'Vpar'],
                    selected_intervals=horizon, ground_truth_intervals=reference_frames - 1,
                    reference_duration_seconds=reports['ar'][1]['reference_duration_seconds'],
                    caveats=([] if horizon == 96 else [
                        'Ground truth stops at frame 96; later accuracy is unmeasured.',
                        f'Native CPU timing covers H96 only and cannot be compared directly to these H{horizon} GPU timings.',
                        'Stopped models have NaN/invalid frames; available forecasts beyond H96 show numerical survival only.']))
    write_json(previous_path, metadata)
    return dict(case_id=case_id, metadata_path=str(previous_path), trajectory_path=str(trajectory), methods=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit', type=Path, default=STUDY / 'reports/video_baseline_audit_20261111.json')
    parser.add_argument('--summary-only', action='store_true', help='Skip large array validation/merging; preserve existing video paths')
    parser.add_argument('--long-horizon', type=int, help='Also assemble the protocol stress case at this horizon (e.g. 384)')
    args = parser.parse_args()
    if args.long_horizon is not None and args.long_horizon <= 96:
        parser.error('--long-horizon must exceed the 96-interval reference')
    audit = json.loads(args.audit.read_text())
    if audit['status'] != 'passed':
        raise ValueError('Require a passed independent baseline audit')
    protocol = json.loads((args.root / 'protocol.json').read_text())
    summary = dict(cases=[], methods=[], native_ground_truth=audit['native_ground_truth'], extras=[], long_cases=[],
                   missing=[], failures=[], pending=[], metric_definition=METRIC,
                   baseline_audit_path=str(args.audit.resolve()), baseline_audit_sha256=sha256_file(args.audit),
                   protocol_path=str(args.root / 'protocol.json'), protocol_sha256=sha256_file(args.root / 'protocol.json'),
                   assembler_sha256=sha256_file(__file__), seed=17, training_horizon=4,
                   reference_intervals=96, comparison_case_selection=protocol['case_selection'],
                   caveats=audit['caveats'], array_validation_performed=not args.summary_only)
    followups = args.root / 'protocol_followups.json'
    if followups.exists():
        summary['followup_protocol_path'] = str(followups)
        summary['followup_protocol_sha256'] = sha256_file(followups)
        summary['followup_protocol'] = json.loads(followups.read_text())
    primary_paths = set()
    for case_id in protocol['cases']:
        reports = {}
        for method, microsteps in PRIMARY.items():
            path = args.root / 'runs' / f'{case_id}_{method}_m{microsteps}_h96'
            primary_paths.add(path)
            if not (path / 'report.json').exists():
                summary['missing'].append(str(path)); continue
            report, _ = load_run(path, audit)
            if (report['method'], report['case_id'], report['horizon'], report['actual_microsteps']) != (method, case_id, 96, microsteps):
                raise ValueError(f'{path}: primary identity differs from protocol')
            if report['status'] not in ('complete', 'numerical_failure'):
                summary['pending'].append(str(path)); continue
            row = brief(report)
            if row['complete'] and len(row['timing_samples']) != protocol['primary']['repeats']:
                raise ValueError(f'{path}: wrong number of timing repeats')
            if not row['complete'] or not row['audits_passed']:
                summary['failures'].append({'path': str(path), **row}); continue
            reports[method] = (path, report)
        if len(reports) == len(PRIMARY):
            if args.summary_only:
                folder = args.root / 'cases' / case_id
                summary['cases'].append(dict(case_id=case_id, metadata_path=str(folder / 'metadata.json'),
                    trajectory_path=str(folder / 'trajectory.npz'), methods=[brief(reports[m][1]) for m in PRIMARY]))
            else:
                summary['cases'].append(assemble_case(args.root, case_id, reports, audit))
    for method in PRIMARY:
        rows = [row for case in summary['cases'] for row in case['methods'] if row['method'] == method]
        history = next(row for row in audit['methods'] if row['method'] == method)
        complete = len(rows) == len(protocol['cases'])
        times = [sample for row in rows for sample in row['timing_samples']]
        summary['methods'].append(dict(method=method, label=LABELS[method], complete=complete,
            cases=len(rows), expected_cases=len(protocol['cases']), microsteps=PRIMARY[method],
            mean_rollout_seconds=float(np.mean([row['rollout_seconds'] for row in rows])) if complete else None,
            rollout_seconds_range=[min(times), max(times)] if complete else None,
            mean_normalized_volume_mse=float(np.mean([row['normalized_volume_mse'] for row in rows])) if complete else None,
            training_updates=history['training_updates'], historical_training_update_seconds=history['training_update_seconds'],
            timing_aggregation='Arithmetic mean of two case medians, each from three repeated warm CUDA rollouts'))
    for report_path in sorted((args.root / 'runs').glob('*/report.json')):
        if report_path.parent in primary_paths:
            continue
        report, _ = load_run(report_path.parent, audit)
        if report['status'] not in ('complete', 'numerical_failure'):
            summary['pending'].append(str(report_path.parent)); continue
        row = {'path': str(report_path.parent), **brief(report)}
        summary['extras'].append(row)
        if not row['complete'] or not row['audits_passed']:
            summary['failures'].append(row)
    for case_id, method, microsteps, horizon, _ in protocol['jobs']:
        path = args.root / 'runs' / f'{case_id}_{method}_m{microsteps}_h{horizon}'
        if not (path / 'report.json').exists() and str(path) not in summary['missing']:
            summary['missing'].append(str(path))
    if args.long_horizon is not None:
        case_id, horizon = protocol['stress']['case'], args.long_horizon
        reports = {}
        for method, microsteps in PRIMARY.items():
            path = args.root / 'runs' / f'{case_id}_{method}_m{microsteps}_h{horizon}'
            if not (path / 'report.json').exists():
                if str(path) not in summary['missing']:
                    summary['missing'].append(str(path))
                continue
            report, _ = load_run(path, audit)
            if (report['method'], report['case_id'], report['horizon'], report['actual_microsteps']) != (method, case_id, horizon, microsteps):
                raise ValueError(f'{path}: long-run identity differs from selection')
            if report['status'] in ('complete', 'numerical_failure'):
                reports[method] = (path, report)
        if len(reports) == len(PRIMARY):
            if args.summary_only:
                folder = args.root / 'cases' / f'{case_id}_h{horizon}'
                summary['long_cases'].append(dict(case_id=case_id, metadata_path=str(folder / 'metadata.json'),
                    trajectory_path=str(folder / 'trajectory.npz'), methods=[brief(reports[m][1]) for m in PRIMARY]))
            else:
                summary['long_cases'].append(assemble_case(args.root, case_id, reports, audit, horizon=horizon))
    summary['primary_complete'] = all(row['complete'] for row in summary['methods'])
    write_json(args.root / 'summary.json', summary)
    print(json.dumps({key: summary[key] for key in ('primary_complete', 'methods', 'missing', 'pending')}, indent=2))


if __name__ == '__main__':
    main()
