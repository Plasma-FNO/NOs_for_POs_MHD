"""Matched, resumable trajectory-supervised ensemble experiments on native DOFs."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import random
import resource
import sys
import time

import h5py
import numpy as np
import scipy
import torch

from .common import config_hash, sha256_file, write_json
from .continuity import FrozenContinuity
from .euler import EulerSplineModel, NativeSplineSpace, field_loss
from .euler_dataset import EulerDataset
from .euler_training import FIELDS
from .physical_guard import PhysicalGuard, QuadratureSampler
from .restart import read_restart


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_checkpoint(path, value):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def neural_hash(model):
    digest = hashlib.sha256()
    for name, value in model.operator.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def observation_scaling(data, stride):
    reference = None
    initial_path = None
    fixed_names = ('x', 'size', 'index', 'vertex', 'boundary', 'n_sons', 'constrained')
    for case in data.cases:
        path = data.verified_path({'path': case['initial'], 'sha256': case['initial_check']['sha256']})
        with h5py.File(path) as stream:
            unit = float(np.asarray(stream['sqrt_mu0_rho0']).item())
            fixed = {name: stream[name][()] for name in fixed_names}
        if not np.isfinite(unit) or unit <= 0:
            raise ValueError('Invalid native time normalization')
        if reference is None:
            reference, time_unit, initial_path = fixed, unit, path
        elif unit != time_unit or any(not np.array_equal(fixed[name], reference[name]) for name in fixed_names):
            raise ValueError('Initial mesh or time normalization differs across cases')
    interval = data.windows('train', 1, stride=stride).observation_interval_seconds
    return initial_path, dict(time_unit_seconds=time_unit, observation_interval_seconds=interval,
                             code_interval=interval/time_unit)


def merge_training_split(data, space):
    original_count = sum(case['split'] == 'train' for case in data.cases)*len(data.time_seconds)
    added = [case for case in data.cases if case['split'] == 'validation']
    added_count = len(added)*len(data.time_seconds)
    added_mean = np.zeros_like(data.mean)
    for case in added:
        with h5py.File(case['resolved_path']) as stream:
            for start in range(0, len(data.time_seconds), 8):
                added_mean += stream['native_state'][start:start+8].sum(0)/added_count
    variance = np.zeros(7)
    weights = space.weights/space.weights.sum()
    for case in added:
        with h5py.File(case['resolved_path']) as stream:
            for frame in stream['native_state']:
                values = space.basis@(frame-added_mean)
                variance += np.einsum('pc,p->c', values**2, weights)/added_count
    fraction = added_count/(original_count+added_count)
    difference = space.basis@(added_mean-data.mean)
    combined_variance = ((1-fraction)*data.scale**2 + fraction*variance
                         + fraction*(1-fraction)*np.einsum('pc,p->c', difference**2, weights))
    data.mean = (1-fraction)*data.mean + fraction*added_mean
    data.scale = np.sqrt(combined_variance)
    if not np.isfinite(data.mean).all() or not np.all(np.isfinite(data.scale) & (data.scale > 0)):
        raise ValueError('Invalid pooled training-only normalization')
    data.cases = [{**case, 'split': 'train' if case['split'] == 'validation' else case['split']} for case in data.cases]
    return dict(method='pooled volume moments: original training statistics plus former validation trajectories',
                training_case_ids=[case['case_id'] for case in data.cases if case['split'] == 'train'],
                fitted_frames=original_count+added_count,
                mean_sha256=hashlib.sha256(data.mean.tobytes()).hexdigest(),
                scale_sha256=hashlib.sha256(data.scale.tobytes()).hexdigest())


def prepare_data(config):
    started = time.monotonic()
    manifest = Path(config['manifest'])
    if sha256_file(manifest) != config['manifest_sha256']:
        raise ValueError('Manifest hash differs from the pinned release')
    completion = json.loads((manifest.parent/'completion.json').read_text())
    if completion['manifest_sha256'] != config['manifest_sha256'] or completion['status'] != 'dataset_ready':
        raise ValueError('Dataset completion record mismatch')
    data = EulerDataset(manifest)
    if data.manifest['split_counts'] != config['source_split_counts']:
        raise ValueError('Trajectory split counts differ from the experiment')
    data.windows('train', config['horizon'], stride=config['stride'])
    data.windows('test', max(config['evaluation_horizons']), stride=config['stride'])
    initial, timing = observation_scaling(data, config['stride'])
    geometry, _, _ = read_restart(initial)
    for item in fields(type(geometry)):
        np.testing.assert_array_equal(getattr(geometry, item.name), getattr(data.geometry, item.name))
    space = NativeSplineSpace(geometry)
    normalization = merge_training_split(data, space)
    split_counts = {split: sum(case['split'] == split for case in data.cases) for split in ('train', 'validation', 'test')}
    if split_counts != config['split_counts']:
        raise ValueError('Merged trajectory splits differ from the experiment')
    q8, _, _ = read_restart(initial, quadrature_order=8)
    guard = PhysicalGuard(geometry, q8, data.mean, data.scale, positivity_basis='bernstein_subdivided')
    del q8
    q12, _, _ = read_restart(initial, quadrature_order=12)
    dense = QuadratureSampler(q12)
    del q12
    boundary = guard.fixed_boundary_values.numpy()
    maximum_drift = 0.
    for case in data.cases:
        with h5py.File(case['resolved_path']) as stream:
            first = stream['native_state'][0][boundary, :2]
            for start in range(0, len(data.time_seconds), 8):
                states = stream['native_state'][start:start+8]
                if not np.isfinite(states).all():
                    raise ValueError('Nonfinite trajectory coefficients')
                maximum_drift = max(maximum_drift, float(np.max(np.abs(
                    (states[:, boundary, :2]-first)/data.scale[:2]))))
    if maximum_drift > 1e-7:
        raise ValueError(f'Potential boundary nodal values evolve: {maximum_drift}')
    probe = data.windows('train', 1)[0]['state'][:1]
    projection_error = float(field_loss(space.project(space.sample(probe)), probe, space))
    if projection_error > 1e-12:
        raise ValueError('Native mass projection roundtrip failed')
    physical = FrozenContinuity(geometry, data.mean, data.scale, timing['code_interval'], config['gamma'])
    report = dict(timing, split_counts=split_counts, normalization=normalization, trajectories=len(data.cases),
                  frames_per_trajectory=len(data.time_seconds), native_dofs=len(geometry.dof_ids),
                  windows={split: len(data.windows(split, config['horizon'], stride=config['stride']))
                           for split in ('train', 'validation', 'test')},
                  training_mean_lower_bounds=(guard.sampler.basis@data.mean[:, 4:6]).min(0).tolist(),
                  boundary_max_normalized_drift=maximum_drift, projection_roundtrip_mse=projection_error,
                  all_trajectory_and_initial_hashes_verified=True, all_coefficients_finite=True,
                  verification_seconds=time.monotonic()-started)
    return data, space, guard, physical, dense, report


def check_config(config):
    if config.get('coefficient_split') not in (None, 'density_diffusion', 'density_diffusion_gamma'):
        raise ValueError('Unknown coefficient-aware physical split')
    for name in ('steps', 'horizon', 'stride', 'threads'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if config['split_counts'] != {'train': 80, 'validation': 0, 'test': 20}:
        raise ValueError('This experiment requires 80 training, no validation and 20 test trajectories')
    if config.get('coefficient_split') and config['linear_branch'] != 'lspline_continuity':
        raise ValueError('Coefficient-aware experiments require learned spline derivatives')
    if config['batch_size'] != 1 or config['linear_branch'] not in ('lspline_continuity', 'native_continuity') or \
            config['physical_guard'] is not True or config['positivity_basis'] != 'bernstein_subdivided':
        raise ValueError('Require batch one, physical continuity and whole-cell guards')
    for name in ('learning_rate', 'gamma'):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f'Invalid {name}')
    if not np.isfinite(config['positivity_penalty']) or config['positivity_penalty'] < 0:
        raise ValueError('Invalid positivity penalty')
    for name in ('evaluation_horizons', 'evaluation_substeps'):
        if not config[name] or any(type(value) is not int or value < 1 for value in config[name]):
            raise ValueError(f'Invalid {name}')
    if sorted(set(config['evaluation_substeps'])) != [1, 2, 4]:
        raise ValueError('Require fixed-weight 1/2/4-substep comparison')
    if config.get('evaluation_cases') is not None and (
            not config.get('diagnostic_only') or type(config['evaluation_cases']) is not int or config['evaluation_cases'] < 1):
        raise ValueError('Case caps are permitted only in labeled smoke diagnostics')


def train_family(model, data, space, config, folder, provenance_hash, *, resume=False, stop_after=None):
    windows = data.windows('train', config['horizon'], stride=config['stride'])
    sampler = np.random.default_rng(config['seed'])
    schedule = sampler.integers(len(windows), size=config['steps'])
    schedule_records = [dict(index=int(index), case_id=windows.cases[index//windows.starts]['case_id'],
                             start_frame=int(index % windows.starts)*config['stride']) for index in schedule]
    schedule_path = folder/'sampled_windows.json'
    if schedule_path.exists():
        if json.loads(schedule_path.read_text()) != schedule_records:
            raise ValueError('Saved sampling schedule changed')
    else:
        write_json(schedule_path, schedule_records)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    checkpoint_path = folder/'last.pt'
    history, completed = [], 0
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if saved['provenance_hash'] != provenance_hash or saved['family'] != model.family or \
                saved['schedule_sha256'] != sha256_file(schedule_path):
            raise ValueError('Checkpoint provenance or sampling mismatch')
        model.operator.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        sampler.bit_generator.state = saved['sampler_rng']
        history, completed = saved['history'], saved['step']
    for index in range(completed, config['steps']):
        if stop_after is not None and index >= stop_after:
            return None
        started = time.monotonic()
        sample = windows[int(schedule[index])]
        model.train()
        model.physical_guard.reset_statistics()
        optimizer.zero_grad(set_to_none=True)
        state = sample['state'][:1]
        loss = state.new_zeros(())
        for target in sample['state'][1:]:
            state = model(state, sample['parameters'][None])
            loss = loss + field_loss(state, target[None], space) + config['positivity_penalty']*model.physical_guard.last_penalty
        loss = loss/config['horizon']
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        row = dict(step=index+1, training_loss=float(loss.detach()), gradient_norm=float(gradient_norm),
                   training_seconds=time.monotonic()-started, guard_statistics=dict(model.physical_guard.statistics))
        history.append(row)
        atomic_checkpoint(checkpoint_path, dict(model=model.operator.state_dict(), optimizer=optimizer.state_dict(),
            step=index+1, history=history, family=model.family,
            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            sampler_rng=sampler.bit_generator.state, schedule_sha256=sha256_file(schedule_path),
            provenance_hash=provenance_hash))
        print(json.dumps(dict(family=model.family, **row)), flush=True)
    model.eval()
    atomic_json(folder/'history.json', history)
    atomic_checkpoint(folder/'selected.pt', dict(model=model.operator.state_dict(), step=config['steps'], family=model.family,
                                               selection='final fixed-budget update', provenance_hash=provenance_hash))
    return dict(selected_step=config['steps'], history=history)


@torch.no_grad()
def evaluate_trajectory(model, sample, space, guard, dense, horizons, substeps, *, records=None):
    """Append completed horizons so the caller can retain them after a later failure."""
    started = time.monotonic()
    guard.reset_statistics()
    state = sample['state'][:1]
    weights = space.weights/space.weights.sum()
    mean, scale = guard.mean.numpy(), guard.scale.numpy()
    vertices = np.unique(space.geometry.dof_map[:, ::4])
    initial_boundary = (state[0].numpy()*scale+mean)[guard.fixed_boundary_values.numpy(), :2]
    squared_error, decoded_norm = np.zeros(7), np.zeros(7)
    target_mean, target_moment = np.zeros((len(weights), 7)), np.zeros((len(weights), 7))
    bounds, dense_minimum, vertex_minimum, defects = np.full(2, np.inf), np.full(2, np.inf), np.full(2, np.inf), np.zeros(2)
    boundary_drift = 0.
    records = [] if records is None else records
    for step, target in enumerate(sample['state'][1:], 1):
        state = model(state, sample['parameters'][None], substeps=substeps)
        if not torch.isfinite(state).all():
            raise FloatingPointError(f'Nonfinite rollout at step {step}')
        predicted = space.basis@state[0].numpy()
        truth = space.basis@target.numpy()
        squared_error += np.einsum('pc,p->c', (predicted-truth)**2, weights)
        decoded_norm += np.einsum('pc,p->c', (truth*scale+space.basis@mean)**2, weights)
        delta = truth-target_mean
        target_mean += delta/step
        target_moment += delta*(truth-target_mean)
        physical = state[0].numpy()*scale+mean
        bounds = np.minimum(bounds, (guard.sampler.basis@physical[:, 4:6]).min(0))
        dense_minimum = np.minimum(dense_minimum, (dense.basis@physical[:, 4:6]).min(0))
        vertex_minimum = np.minimum(vertex_minimum, physical[vertices, 4:6].min(0))
        boundary_drift = max(boundary_drift, float(np.max(np.abs(
            (physical[guard.fixed_boundary_values.numpy(), :2]-initial_boundary)/scale[:2]))))
        for channel, (pair, potential, auxiliary) in enumerate(((guard.current, 0, 2), (guard.vorticity, 1, 3))):
            left, right = pair.stiff@physical[:, potential], pair.mass@physical[:, auxiliary]
            defects[channel] = max(defects[channel], float(np.linalg.norm(left+right)/max(np.linalg.norm(left)+np.linalg.norm(right), 1e-30)))
        if step in horizons:
            variation = np.einsum('pc,p->c', target_moment, weights)/step
            mse = squared_error/step
            passed = bool((bounds > 0).all() and (dense_minimum > 0).all() and (vertex_minimum > 0).all()
                          and (defects < 1e-8).all() and boundary_drift < 1e-7)
            records.append(dict(horizon=step, substeps=substeps, case_id=sample['case_id'], start_frame=sample['start_frame'],
                duration_seconds=float(sample['time_seconds'][step]-sample['time_seconds'][0]),
                normalized_volume_mse=float(mse.mean()), per_field_normalized_mse=dict(zip(FIELDS, mse.tolist())),
                decoded_absolute_rms=dict(zip(FIELDS, (np.sqrt(mse)*scale).tolist())),
                decoded_relative_rms=dict(zip(FIELDS, np.sqrt(mse*scale**2/np.maximum(decoded_norm/step, 1e-30)).tolist())),
                dynamic_relative_rms=dict(zip(FIELDS, np.sqrt(mse/np.maximum(variation, 1e-30)).tolist())),
                density_temperature_lower_bounds=bounds.tolist(), q12_minima=dense_minimum.tolist(),
                vertex_minima=vertex_minimum.tolist(), constraint_defects=defects.tolist(),
                boundary_max_normalized_drift=boundary_drift, guard_statistics=dict(guard.statistics),
                audit_passed=passed, elapsed_seconds=time.monotonic()-started))
    return records


def evaluate(model, data, space, guard, dense, config, folder, provenance_hash):
    horizons = sorted(set(config['evaluation_horizons']))
    windows = data.windows('test', max(horizons), stride=config['stride'])
    count = config.get('evaluation_cases') or len(windows.cases)
    expected = min(count, len(windows.cases))
    results = []
    for case_index in range(expected):
        sample = windows[case_index*windows.starts]
        for substeps in ([1] if model.family == 'ar' else config['evaluation_substeps']):
            path = folder/f'evaluation_{case_index:03d}_substeps{substeps}.json'
            if path.exists():
                result = json.loads(path.read_text())
                if result['provenance_hash'] != provenance_hash:
                    raise ValueError('Evaluation provenance mismatch')
            else:
                records = []
                try:
                    evaluate_trajectory(model, sample, space, guard, dense, horizons, substeps, records=records)
                    result = dict(records=records, failure=None, provenance_hash=provenance_hash)
                except (FloatingPointError, ValueError, RuntimeError) as error:
                    result = dict(records=records, failure=str(error), case_id=sample['case_id'], substeps=substeps,
                                  provenance_hash=provenance_hash)
                atomic_json(path, result)
            results.append(result)
    aggregates = []
    for horizon in horizons:
        for substeps in ([1] if model.family == 'ar' else config['evaluation_substeps']):
            rows = [row for result in results for row in result['records'] if row['horizon'] == horizon and row['substeps'] == substeps]
            scores = [row['normalized_volume_mse'] for row in rows]
            audit_failures = sum(not row['audit_passed'] for row in rows)
            complete = bool(expected > 0 and len(rows) == expected and not audit_failures
                            and np.isfinite(scores).all() and all(
                                np.isfinite([row['per_field_normalized_mse'][name] for name in FIELDS]).all()
                                for row in rows))
            aggregates.append(dict(horizon=horizon, substeps=substeps, trajectories=len(rows),
                expected_trajectories=expected, audit_failures=audit_failures, complete=complete,
                mean_normalized_volume_mse=float(np.mean(scores)) if complete else None,
                trajectory_std=float(np.std(scores, ddof=1)) if complete and len(scores) > 1 else None,
                per_field_normalized_mse={name: float(np.mean([row['per_field_normalized_mse'][name] for row in rows]))
                                          for name in FIELDS} if complete else {}))
    return dict(trajectories=results, aggregates=aggregates,
                numerical_failures=sum(result['failure'] is not None for result in results),
                audits_passed=bool(aggregates) and all(row['complete'] for row in aggregates)
                              and all(result['failure'] is None for result in results))


def run_ensemble(config, model_config, output, *, families=('ar', 'node', 'opsplit'), resume=False, stop_after=None):
    check_config(config)
    if not families or any(family not in ('ar', 'node', 'opsplit') for family in families):
        raise ValueError('Invalid model families')
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=resume)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    data, space, guard, physical, dense, verification = prepare_data(config)
    architecture = {**model_config, 'architecture': {**model_config['architecture'], **config['architecture']}}
    provenance = dict(config=config, model_config=architecture, manifest_sha256=sha256_file(config['manifest']),
        source_sha256={path.name: sha256_file(path) for path in Path(__file__).parent.glob('*.py')},
        python_version=sys.version, torch_version=str(torch.__version__), numpy_version=np.__version__,
        scipy_version=scipy.__version__, h5py_version=h5py.__version__,
        optimizer=dict(name='Adam', learning_rate=config['learning_rate'], betas=[.9, .999], eps=1e-8,
                       weight_decay=0, gradient_clip_norm=2.),
        checkpoint_selection='final fixed-budget update; no validation or test-based selection',
        evaluation_start_frame=0, training_substeps=1,
        time_basis='one effective observation interval is one pseudotime unit',
        physical_interval=verification['code_interval'], normalization=verification['normalization'],
        source_normalization=data.manifest['normalization'],
        metric_denominators=dict(dynamic='per-point temporal variance of normalized truth, future frames only',
                                 decoded='volume-weighted physical truth squared, future frames only', floor=1e-30),
        uncertainty_unit='independent initial-condition trajectory; seeds reported separately',
        independent_initial_conditions=True, independent_equilibria=False)
    provenance_hash = config_hash(provenance)
    provenance_path = output/'provenance.json'
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError('Immutable experiment provenance changed; use a fresh output directory')
    else:
        write_json(provenance_path, provenance)
    normalization_path = output/'normalization.npz'
    if normalization_path.exists():
        with np.load(normalization_path) as saved:
            np.testing.assert_array_equal(saved['mean'], data.mean)
            np.testing.assert_array_equal(saved['scale'], data.scale)
    else:
        np.savez(normalization_path, mean=data.mean, scale=data.scale, time_seconds=data.time_seconds,
                 training_case_ids=verification['normalization']['training_case_ids'])
    atomic_json(output/'verification.json', verification)
    print(json.dumps(dict(phase='verified', **verification)), flush=True)
    results = {}
    for family in families:
        folder = output/family
        folder.mkdir(exist_ok=resume)
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        if config.get('coefficient_split') in ('density_diffusion', 'density_diffusion_gamma'):
            from .coefficient_ood import DensityDiffusionEulerModel, GammaEulerModel
            model_class = GammaEulerModel if config['coefficient_split'] == 'density_diffusion_gamma' else DensityDiffusionEulerModel
            model = model_class(family, space, architecture, mean=data.mean, scale=data.scale,
                code_interval=verification['code_interval'], physical_guard=guard, physical_linear=physical)
        else:
            model = EulerSplineModel(family, space, architecture, linear_branch=config['linear_branch'],
                                     physical_guard=guard, physical_linear=physical)
        initial_hash = neural_hash(model)
        training = train_family(model, data, space, config, folder, provenance_hash, resume=resume, stop_after=stop_after)
        if training is None:
            return dict(interrupted=True)
        evaluation_started = time.monotonic()
        evaluation = evaluate(model, data, space, guard, dense, config, folder, provenance_hash)
        result = dict(training, evaluation=evaluation, initial_neural_sha256=initial_hash,
                      parameters=sum(parameter.numel() for parameter in model.parameters()),
                      evaluation_seconds=time.monotonic()-evaluation_started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(folder/'metrics.json', result)
        results[family] = result
        del model
    atomic_json(output/'results.json', dict(results=results, elapsed_seconds=time.monotonic()-started,
                diagnostic_only=config.get('diagnostic_only', False), all_audits_passed=all(
                    result['evaluation']['audits_passed'] for result in results.values())))
    return results
