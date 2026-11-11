#!/usr/bin/env python3
"""Export one autonomous CUDA rollout and benchmark it without plotting/audit work.

One method/case per process keeps work below the interactive host's CPU limit.
Checkpoints are frozen before comparison. All timing trials restart at the same
initial state; setup and lazy first-call cost are reported separately. No training.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY/'src'))
from mastu_opsplit.common import load_config, sha256_file
from mastu_opsplit.ensemble_training import atomic_json, prepare_data
from mastu_opsplit.euler import EulerSplineModel
from mastu_opsplit.gpu_native import configure_cuda_numerics
from mastu_opsplit.implicit_training import LabelStore, build_model, build_system
from diagnose_physical_defect import copy_setup_cache

PILOT = Path('/pitagora_scratch/userexternal/vgopakum/mastu_physical_defect_20261111')
BASELINES = Path('/pitagora_scratch/userexternal/vgopakum/mastu_gpu_onestep_20261109')
FIELDS = ('psi', 'u', 'zj', 'w', 'rho', 'T', 'Vpar')


def score(states, truth, space, guard, dense, mean, scale):
    """Post-timing volume errors and physical audits; no future inputs to rollout."""
    weights = space.weights/space.weights.sum()
    fixed = guard.fixed_boundary_values.cpu().numpy()
    initial = states[0]*scale+mean
    errors, audits = [], []
    for frame in range(1, len(states)):
        physical = states[frame]*scale+mean
        minima = (dense.basis@physical[:, 4:6]).min(0)
        defects = []
        for pair, potential, auxiliary in ((guard.current, 0, 2), (guard.vorticity, 1, 3)):
            left, right = pair.stiff@physical[:, potential], pair.mass@physical[:, auxiliary]
            defects.append(float(np.linalg.norm(left+right)/max(np.linalg.norm(left)+np.linalg.norm(right), 1e-30)))
        drift = float(np.max(np.abs((physical[fixed, :2]-initial[fixed, :2])/scale[:2])))
        audits.append(dict(frame=frame, q12_minima=minima.tolist(), constraint_defects=defects,
                           boundary_drift=drift, passed=bool((minima > 0).all() and max(defects) < 1e-8 and drift < 1e-7)))
        if frame < len(truth):
            difference = space.basis@(states[frame]-truth[frame])
            errors.append(np.einsum('pc,p->c', difference**2, weights))
    errors = np.array(errors)
    horizons = {}
    for h in (4, 16, 32, 64, 96):
        if len(errors) >= h:
            field = errors[:h].mean(0)
            horizons[str(h)] = dict(normalized_volume_mse=float(field.mean()),
                                    per_field_normalized_mse=dict(zip(FIELDS, field.tolist())),
                                    audit_passed=all(a['passed'] for a in audits[:h]))
    return dict(horizons=horizons, frame_normalized_volume_mse=errors.mean(1).tolist() if len(errors) else [],
                frame_per_field_normalized_mse=errors.tolist(), physical_audits=audits,
                all_completed_frame_audits_passed=all(a['passed'] for a in audits))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', required=True, choices=('ar', 'node', 'implicit', 'physics'))
    p.add_argument('--case', required=True)
    p.add_argument('--microsteps', type=int, default=1)
    p.add_argument('--horizon', type=int, default=96)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if min(args.microsteps, args.horizon, args.repeats) < 1 or (args.method == 'ar' and args.microsteps != 1):
        p.error('Positive counts required; AR only supports one update per saved interval')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback is forbidden')
    args.output.mkdir(parents=True, exist_ok=False)
    configure_cuda_numerics()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)
    started = time.perf_counter()
    config = load_config(PILOT/'learning_pilot64.yaml')
    config['prepared_data_cache'] = str(PILOT/'prepared_data_cache')
    config['physical_defect']['profile_manifest']['path'] = str(STUDY/'reports/physical_profiles_20261111.json')
    # Keep data/cache configuration identical; the inference-only substep change follows setup.
    data, space, guard, physical, dense, verification = prepare_data(config)
    windows = data.windows('test', 96, stride=1)
    index = next(i for i, c in enumerate(windows.cases) if c['case_id'] == args.case)
    sample = windows[index*windows.starts]
    case = windows.cases[index]
    mean, scale = data.mean, data.scale
    if args.method in ('implicit', 'physics'):
        old = copy_setup_cache(PILOT/'pilot64_learned', args.output, config)
        for key in ('training_case_ids', 'mean_sha256', 'scale_sha256'):
            if old['normalization'][key] != verification['normalization'][key]:
                raise ValueError('Implicit normalization mismatch')
        labels = LabelStore(config['labels'], tangent_sha256=config['tangent_sha256'],
                            training_case_ids=[c['case_id'] for c in data.cases if c['split'] == 'train'])
        system, boundary = build_system(config, data, space, verification, args.output, resume=False, labels=labels)
        original = json.loads((PILOT/'pilot64_learned/provenance.json').read_text())
        if boundary != original['dirichlet']:
            raise ValueError('Changed boundary configuration')
        config['physical_defect']['internal_substeps'] = args.microsteps
        model = build_model(config, space, system, original['model_config'], labels.scale,
                            physical_guard=guard, physics_only=args.method == 'physics').to('cuda').eval()
        checkpoint = PILOT/'pilot64_learned/implicit_rollout/rollout/selected.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['step'] != 64:
            raise ValueError('Require frozen 64-update checkpoint')
        if args.method == 'implicit':
            model.operator.load_state_dict(saved['model'], strict=True)
        substeps = 1
        updates = 64 if args.method == 'implicit' else 0
        provenance_path = PILOT/'pilot64_learned/provenance.json'
    else:
        run = BASELINES/f'seed17_{args.method}'
        original = json.loads((run/'provenance.json').read_text())
        with np.load(run/'normalization.npz') as normal:
            np.testing.assert_array_equal(normal['mean'], mean)
            np.testing.assert_array_equal(normal['scale'], scale)
        if original['manifest_sha256'] != config['manifest_sha256']:
            raise ValueError('Baseline data release differs')
        model = EulerSplineModel(args.method, space, original['model_config'],
                                 linear_branch=original['config']['linear_branch'],
                                 physical_guard=guard, physical_linear=physical).to('cuda').eval()
        checkpoint = run/args.method/'selected.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['step'] != 2000:
            raise ValueError('Require frozen final-budget baseline checkpoint')
        model.operator.load_state_dict(saved['model'], strict=True)
        substeps, updates = args.microsteps, 2000
        provenance_path = run/'provenance.json'
    parameters = sample['parameters'][None].to('cuda')
    initial = sample['state'][:1].to('cuda')
    torch.cuda.synchronize()
    report = dict(status='benchmarking', method=args.method, case_id=args.case, horizon=args.horizon,
                  actual_microsteps=args.microsteps, training_updates=updates, seed=17,
                  device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  setup_seconds=time.perf_counter()-started, normalization=verification['normalization'],
                  reference_frames=len(sample['state']), reference_duration_seconds=float(sample['time_seconds'][-1]-sample['time_seconds'][0]),
                  observation_interval_seconds=float(sample['time_seconds'][1]-sample['time_seconds'][0]),
                  checkpoint=dict(path=str(checkpoint), sha256=sha256_file(checkpoint), step=saved['step']),
                  provenance=dict(path=str(provenance_path), sha256=sha256_file(provenance_path)),
                  source_sha256={p.name:sha256_file(p) for p in sorted((STUDY/'src/mastu_opsplit').glob('*.py'))},
                  benchmark_script_sha256=sha256_file(__file__), timing_trials=[],
                  timing_scope='Synchronized wall time of autonomous CUDA inference with deployed guards; no plotting, scoring, file IO or device-to-host frame export. CPU orchestration and synchronizations inside model included. One warmup frame; cached factorizations. Batch one. GPU trials run sequentially.',
                  restart_path=str(case['initial']), config=config)
    def save():
        report['elapsed_seconds'] = time.perf_counter()-started
        report['cpu_seconds'] = time.process_time()
        atomic_json(args.output/'report.json', report)
    save()
    with torch.no_grad():
        cold = time.perf_counter()
        model(initial, parameters, substeps=substeps)
        torch.cuda.synchronize()
        report['cold_first_frame_seconds'] = time.perf_counter()-cold
        reference_states = None
        for repeat in range(args.repeats):
            guard.reset_statistics()
            if hasattr(model, 'reset_stage_statistics'):
                model.reset_stage_statistics()
            state = initial
            tensors = [initial[0]]
            failure = None
            torch.cuda.synchronize()
            begin = time.perf_counter()
            for step in range(1, args.horizon+1):
                try:
                    state = model(state, parameters, substeps=substeps)
                    if not torch.isfinite(state).all():
                        raise FloatingPointError('Nonfinite predicted native coefficients')
                    tensors.append(state[0])
                except (FloatingPointError, ValueError, RuntimeError) as error:
                    failure = dict(frame=step, type=type(error).__name__, message=str(error))
                    break
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-begin
            # Export and validation deliberately outside timed region.
            states = torch.stack(tensors).cpu().numpy()
            trial = dict(repeat=repeat, rollout_seconds=elapsed, completed_intervals=len(states)-1,
                         failure=failure, guard_statistics=dict(guard.statistics))
            if hasattr(model, 'stage_statistics'):
                trial['stage_statistics'] = dict(model.stage_statistics)
            if reference_states is None:
                reference_states = states
            else:
                if states.shape != reference_states.shape:
                    raise ValueError('Repeated trajectory changed length')
                trial['maximum_repeat_coefficient_difference'] = float(np.abs(states-reference_states).max())
                if not np.allclose(states, reference_states, rtol=1e-10, atol=1e-10):
                    raise ValueError('Timing repeats do not reproduce the same trajectory')
            report['timing_trials'].append(trial)
            save()
            print(json.dumps(dict(method=args.method, case_id=args.case, repeat=repeat,
                                  seconds=elapsed, completed=len(states)-1, failure=failure)), flush=True)
            if failure:
                break
    truth = sample['state'].numpy()
    report['scores'] = score(reference_states, truth, space, guard, dense, mean, scale)
    report['status'] = 'complete' if len(reference_states) == args.horizon+1 else 'numerical_failure'
    report['median_rollout_seconds'] = float(np.median([t['rollout_seconds'] for t in report['timing_trials']]))
    report['rollout_seconds_range'] = [min(t['rollout_seconds'] for t in report['timing_trials']),
                                       max(t['rollout_seconds'] for t in report['timing_trials'])]
    times = float(sample['time_seconds'][0]) + np.arange(args.horizon+1)*report['observation_interval_seconds']
    physical_states = np.full((args.horizon+1, *initial.shape[1:]), np.nan)
    physical_states[:len(reference_states)] = reference_states*scale+mean
    ground_truth = np.full_like(physical_states, np.nan)
    ground_truth[:min(len(truth), args.horizon+1)] = truth[:args.horizon+1]*scale+mean
    np.savez_compressed(args.output/'trajectory.npz', prediction=physical_states, ground_truth=ground_truth,
                        time_seconds=times, valid_prediction=np.arange(args.horizon+1)<len(reference_states),
                        valid_ground_truth=np.arange(args.horizon+1)<len(truth), mean=mean, scale=scale)
    report['trajectory_sha256'] = sha256_file(args.output/'trajectory.npz')
    save()
    print(json.dumps(dict(status=report['status'], scores=report['scores']['horizons'],
                          median_seconds=report['median_rollout_seconds'])), flush=True)


if __name__ == '__main__':
    main()
