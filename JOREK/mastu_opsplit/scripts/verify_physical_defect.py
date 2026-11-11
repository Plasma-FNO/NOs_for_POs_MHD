#!/usr/bin/env python3
"""CUDA parity gate for the autonomous physical remainder against native TRAINING labels.

Reads only explicitly selected entries from the immutable pooled training label
plan. Axis/X-point normalization is recomputed from each state, without native
trajectory diagnostics. No simulation, factorization or training is performed.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mastu_opsplit.common import sha256_file, write_json
from mastu_opsplit.implicit_opsplit import ACTIVE_CHANNELS, load_frozen_tangent
from mastu_opsplit.physical_rhs import PhysicalRemainder
from mastu_opsplit.restart import read_restart


def compare_labels(actual, expected, weights, training_scale):
    """Channel errors scaled by fixed training RMS; near-zero labels stay diagnostic."""
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[1] != 13:
        raise ValueError('Expected matching pointwise [points,13] arrays')
    if not all(np.isfinite(x).all() for x in (actual, expected, weights, training_scale)):
        raise ValueError('Nonfinite parity inputs')
    if weights.shape != actual.shape[:1] or training_scale.shape != (13,) or (weights <= 0).any() or (training_scale <= 0).any():
        raise ValueError('Invalid native weights/training scale')
    weight = weights[:, None]/weights.sum()
    error = actual-expected
    rms = np.sqrt((weight*error**2).sum(0))
    label_rms = np.sqrt((weight*expected**2).sum(0))
    return dict(channel_rms_error=rms.tolist(), channel_native_rms=label_rms.tolist(),
                channel_training_scaled_rms=(rms/training_scale).tolist(),
                channel_training_scaled_max=(np.abs(error).max(0)/training_scale).tolist(),
                channel_relative_rms_with_floor=(rms/np.maximum(label_rms, 1e-6*training_scale)).tolist(),
                maximum_absolute_error=float(np.abs(error).max()))


def pinned(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256_file(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--profile-manifest', type=Path, required=True)
    parser.add_argument('--reference-restart', type=Path, required=True)
    parser.add_argument('--normalization', type=Path, required=True)
    parser.add_argument('--states', nargs='+', default=['mastu_000:0', 'mastu_000:48', 'mastu_000:96', 'mastu_060:96'])
    parser.add_argument('--rms-tolerance', type=float, default=1e-5)
    parser.add_argument('--max-tolerance', type=float, default=1e-3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite evidence: {args.output}')
    if not torch.cuda.is_available():
        raise RuntimeError('Native physical parity requires CUDA')
    if not np.isfinite([args.rms_tolerance, args.max_tolerance]).all() or min(args.rms_tolerance, args.max_tolerance) <= 0:
        raise ValueError('Parity tolerances must be positive and finite')
    torch.set_num_threads(1)
    started = time.monotonic()
    plan = json.loads(args.plan.read_text())
    summary_path = args.plan.parent/'summary.json'
    summary = json.loads(summary_path.read_text())
    if summary['plan_sha256'] != sha256_file(args.plan) or tuple(plan['active_channels']) != ACTIVE_CHANNELS:
        raise ValueError('Label plan/summary identity or active-channel mismatch')
    entries = {(entry['case_id'], entry['frame']): entry for entry in plan['entries']}
    selected = []
    for spec in args.states:
        case, frame = spec.split(':')
        if case not in plan['cases'] or (case, int(frame)) not in entries:
            raise ValueError(f'Only planned pooled training states may be checked: {spec}')
        selected.append(entries[case, int(frame)])
    if len(set(args.states)) != len(selected):
        raise ValueError('Duplicate selected training states')
    geometry, _, _ = read_restart(args.reference_restart)
    frozen = load_frozen_tangent(plan['tangent'], geometry)
    if frozen.sha256 != plan['tangent_sha256']:
        raise ValueError('Label tangent identity mismatch')
    with np.load(args.normalization, allow_pickle=False) as data:
        mean, scale = data['mean'], data['scale']
        if set(data['training_case_ids'].tolist()) != set(plan['cases']):
            raise ValueError('Normalization training cases differ from native label plan')
    config = dict(profile_manifest=pinned(args.profile_manifest), reference_restart=pinned(args.reference_restart))
    rhs = PhysicalRemainder.from_config(config, geometry, frozen, mean, scale).to('cuda')
    report = dict(schema='mastu_physical_remainder_parity_v1', passed=False, device='cuda',
                  device_name=torch.cuda.get_device_name(), dtype='float64',
                  training_only=True, trajectory_diagnostics_used=False, states=[],
                  active_channels=list(ACTIVE_CHANNELS),
                  tolerances=dict(training_scaled_rms=args.rms_tolerance, training_scaled_max=args.max_tolerance),
                  relative_rms_floor_fraction=1e-6,
                  provenance=dict(**config, label_plan=pinned(args.plan), label_summary=pinned(summary_path),
                                  normalization=pinned(args.normalization), frozen_tangent=pinned(plan['tangent'])))
    package = Path(__file__).resolve().parents[1]/'src/mastu_opsplit'
    report['source_sha256'] = {str(path): sha256_file(path) for path in [Path(__file__).resolve()] +
                              [package/name for name in ('physical_rhs.py', 'native_critical.py', 'restart.py',
                                                        'implicit_opsplit.py', 'numerical.py')]}
    weights = geometry.volume_weights.reshape(-1)
    training_scale = np.array(summary['label_scale'])
    try:
        with torch.no_grad():
            for entry in selected:
                case, frame = entry['case_id'], entry['frame']
                label_path = args.plan.parent/'states'/f'{case}_{frame:03d}.npy'
                metadata_path = label_path.with_suffix('.json')
                metadata = json.loads(metadata_path.read_text())
                if any(metadata[key] != entry[key] for key in ('case_id', 'frame', 'data', 'restart')):
                    raise ValueError(f'Label metadata mismatch: {label_path}')
                if sha256_file(label_path) != metadata['label_sha256']:
                    raise ValueError(f'Label hash mismatch: {label_path}')
                with h5py.File(entry['data']) as stream:
                    physical = stream['native_state'][frame]
                # Verify this is precisely the native restart used to export the label.
                if sha256_file(entry['restart']) != metadata['restart_sha256']:
                    raise ValueError(f'Native label restart hash mismatch: {entry["restart"]}')
                _, restart_state, _ = read_restart(entry['restart'])
                if not np.array_equal(physical, restart_state):
                    raise ValueError(f'Label restart differs from trajectory state: {case}:{frame}')
                state = torch.tensor((physical-mean)/scale, dtype=torch.float64, device='cuda')[None]
                prediction = rhs.pointwise(state)[0].cpu().numpy()
                diagnostics = rhs.profiles.critical(state*rhs.scale+rhs.mean)
                row = dict(state=f'{case}:{frame}', time_seconds=entry['time_seconds'],
                           label=pinned(label_path), label_metadata=pinned(metadata_path),
                           restart=pinned(entry['restart']), data_path=entry['data'],
                           diagnostics={key: value.cpu().tolist() for key, value in diagnostics.items()},
                           critical_solver=rhs.profiles.critical.last_diagnostics,
                           **compare_labels(prediction, np.load(label_path).astype(np.float64), weights, training_scale))
                report['states'].append(row)
                print(json.dumps(dict(state=row['state'], rms=max(row['channel_training_scaled_rms']),
                                      maximum=max(row['channel_training_scaled_max']))), flush=True)
        report['worst_training_scaled_rms'] = max(max(row['channel_training_scaled_rms']) for row in report['states'])
        report['worst_training_scaled_max'] = max(max(row['channel_training_scaled_max']) for row in report['states'])
        report['passed'] = (report['worst_training_scaled_rms'] <= args.rms_tolerance and
                            report['worst_training_scaled_max'] <= args.max_tolerance)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    report['seconds'] = time.monotonic()-started
    report['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated()
    write_json(args.output, report)
    print(json.dumps(dict(passed=report['passed'], output=str(args.output), error=report.get('error'))), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
