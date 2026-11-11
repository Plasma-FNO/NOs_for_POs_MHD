"""Implicit native-tangent OpsSplit: label pretraining, rollout fine-tuning and matched evaluation.

Data, normalization, guards, rollout windows, loss, optimizer settings, budget and the
evaluator are the ensemble study's own functions, so scores are directly comparable
with its AR/NODE/OpsSplit runs. Arms:
  implicit          exact-label pretraining, then the ensemble's 4-step rollout fine-tuning
  implicit_rollout  rollout fine-tuning only, from the zero-output (physics-only) start
  physics           N = 0; no trained parameters (reference for what the network adds)
The selected checkpoint is always the final fixed-budget update. No test score is used.
"""
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
from .ensemble_training import (atomic_checkpoint, atomic_json, cpu_budget_expired, evaluate, model_device, neural_hash, prepare_data,
                                train_family)
from .gpu_native import configure_cuda_numerics
from .implicit_labels import sha256
from .implicit_opsplit import (ACTIVE_CHANNELS, BoundaryRelations, ImplicitNativeSystem, ImplicitOpsSplitModel,
                               TangentCorrection, fixed_dofs_from_trajectories, identify_boundary_rows,
                               load_frozen_tangent, remainder_loss)

ARMS = ('implicit', 'implicit_rollout', 'physics')


def check_implicit_config(config):
    for name in ('steps', 'pretrain_steps', 'horizon', 'stride', 'threads'):
        if type(config[name]) is not int or config[name] < (0 if name == 'pretrain_steps' else 1):
            raise ValueError(f'Invalid {name}')
    if config['split_counts'] != {'train': 80, 'validation': 0, 'test': 20}:
        raise ValueError('Matched study requires 80 training, no validation and 20 test trajectories')
    if config['batch_size'] != 1 or config['physical_guard'] is not True or config['positivity_basis'] != 'bernstein_subdivided':
        raise ValueError('Require batch one and the shared whole-cell guards')
    for name in ('learning_rate', 'pretrain_learning_rate', 'gamma', 'F0'):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f'Invalid {name}')
    if not .5 <= config['theta'] <= 1:
        raise ValueError('theta must lie in [1/2, 1]')
    if sorted(set(config['evaluation_substeps'])) != [1, 2, 4]:
        raise ValueError('Require fixed-weight 1/2/4-substep comparison')
    if config.get('evaluation_cases') is not None and not config.get('diagnostic_only'):
        raise ValueError('Case caps are permitted only in labeled smoke diagnostics')
    if config.get('device', 'cpu') not in ('cpu', 'cuda'):
        raise ValueError("device must be 'cpu' (SciPy/SuperLU) or 'cuda' (torch + cuDSS)")
    if config.get('cpu_budget_seconds') is not None and (
            not np.isfinite(config['cpu_budget_seconds']) or config['cpu_budget_seconds'] <= 0):
        raise ValueError('cpu_budget_seconds must be finite and positive')
    physical = config.get('physical_defect')
    if physical is not None:
        if not isinstance(physical, dict) or physical.get('scheme', 'be') not in ('be', 'ars222'):
            raise ValueError('physical_defect requires a be or ars222 scheme')
        if config['theta'] != 1 or config['pretrain_steps'] != 0 or config.get('tangent_correction'):
            raise ValueError('Physical defects require theta=1, no remainder-label pretraining, and no learned tangent')
        if config.get('device') != 'cuda':
            raise ValueError('Physical-defect research training requires CUDA')
    correction = config.get('tangent_correction')
    if correction is not None:
        allowed = {'basis_shape', 'bound', 'share', 'train_iterations', 'max_iterations', 'tolerance',
                   'train_tolerance'}
        if not isinstance(correction, dict) or set(correction) - allowed:
            raise ValueError(f'tangent_correction keys must be among {sorted(allowed)}')


def build_model(config, space, system, architecture, label_scale, *, physical_guard, physics_only=False, mean=None,
                scale=None):
    """The implicit model of a configuration (shared by training and both OOD evaluators)."""
    physical = config.get('physical_defect')
    if physical is not None:
        from .physical_defect import PhysicalDefectModel
        from .physical_rhs import PhysicalRemainder
        rhs = PhysicalRemainder.from_config(physical, space.geometry, system.frozen,
                                           system.mean if mean is None else mean,
                                           system.scale if scale is None else scale)
        if rhs.profiles.physics['f0'] != config['F0'] or rhs.profiles.physics['gamma'] != config['gamma']:
            raise ValueError('Native volume profiles and temporal-system F0/gamma differ')
        return PhysicalDefectModel(space, system, architecture, label_scale, physical_guard=physical_guard,
                                   physical_rhs=rhs, reference_state=rhs.reference_coefficients,
                                   scheme=physical.get('scheme', 'be'), physics_only=physics_only,
                                   internal_substeps=physical.get('internal_substeps', 1),
                                   boundary_tolerance=physical.get('boundary_tolerance', 1e-10),
                                   boundary_max_iterations=physical.get('boundary_max_iterations', 12),
                                   held_dof_tolerance=physical.get('held_dof_tolerance'))
    options = config.get('tangent_correction')
    correction = picard = None
    if options is not None and not physics_only:
        correction = TangentCorrection(space.geometry, system.frozen, system.mean if mean is None else mean,
                                       system.scale if scale is None else scale,
                                       basis_shape=tuple(options.get('basis_shape', (6, 12))),
                                       bound=options.get('bound', .5), share=options.get('share', 'pair'))
        picard = {k: options[k] for k in ('train_iterations', 'max_iterations', 'tolerance', 'train_tolerance')
                  if k in options}
    return ImplicitOpsSplitModel(space, system, architecture, label_scale, physical_guard=physical_guard,
                                 physics_only=physics_only, correction=correction, picard=picard)


class LabelStore:
    """Finalized exact remainder labels for training states only."""
    def __init__(self, root, *, tangent_sha256, training_case_ids):
        self.root = Path(root)
        self.plan = json.loads((self.root/'plan.json').read_text())
        self.summary = json.loads((self.root/'summary.json').read_text())
        if self.summary['plan_sha256'] != sha256(self.root/'plan.json'):
            raise ValueError('Label plan changed after finalization')
        if self.plan['tangent_sha256'] != tangent_sha256:
            raise ValueError('Labels were computed against a different frozen tangent')
        if self.plan['active_channels'] != list(ACTIVE_CHANNELS):
            raise ValueError('Label channel layout differs from the model')
        cases = set(self.plan['cases'])
        if not cases <= set(training_case_ids):
            raise ValueError('Label store contains non-training trajectories')
        self.entries = [dict(entry, path=str(self.root/'states'/f'{entry["case_id"]}_{entry["frame"]:03d}.npy'))
                        for entry in self.plan['entries']]
        self.scale = np.asarray(self.summary['label_scale'], dtype=np.float64)
        self.identity = dict(plan_sha256=self.summary['plan_sha256'], summary_sha256=sha256(self.root/'summary.json'),
                             states=len(self.entries), cases=len(cases), frames=self.plan['frames'])


def pretrain(model, data, labels, space, config, folder, provenance_hash, *, resume=False, stop_after=None):
    """Fixed-budget supervised pretraining of N_theta on exact native labels (batch one state)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config['seed'] + 1000)
    schedule = rng.integers(len(labels.entries), size=config['pretrain_steps'])
    schedule_records = [dict(case_id=labels.entries[i]['case_id'], frame=labels.entries[i]['frame']) for i in schedule]
    schedule_path = folder/'sampled_states.json'
    if schedule_path.exists():
        if json.loads(schedule_path.read_text()) != schedule_records:
            raise ValueError('Saved pretraining schedule changed')
    else:
        write_json(schedule_path, schedule_records)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['pretrain_learning_rate'])
    device = model_device(model)
    weights = torch.as_tensor(space.weights, dtype=torch.float64, device=device)
    path = folder/'last.pt'
    history, completed = [], 0
    if resume and path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['provenance_hash'] != provenance_hash or saved['schedule_sha256'] != sha256_file(schedule_path):
            raise ValueError('Pretraining checkpoint provenance mismatch')
        model.operator.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        history, completed = saved['history'], saved['step']
    if completed >= config['pretrain_steps'] and (folder/'pretrained.pt').exists():
        model.operator.load_state_dict(torch.load(folder/'pretrained.pt', map_location='cpu', weights_only=False)['model'])
        return dict(steps=completed, history=history)
    for index in range(completed, config['pretrain_steps']):
        if (stop_after is not None and index >= stop_after) or cpu_budget_expired(config):
            return None
        started = time.monotonic()
        entry = labels.entries[int(schedule[index])]
        with h5py.File(entry['data']) as stream:
            native = stream['native_state'][entry['frame']]
        state = torch.from_numpy((native-data.mean)/data.scale)[None].to(device)
        target = torch.from_numpy(np.load(entry['path']).astype(np.float64))[None].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = remainder_loss(model.remainder(state, torch.ones(1, 1, device=device)), target, weights,
                              model.remainder_scale)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite pretraining loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(step=index+1, remainder_loss=float(loss.detach()), gradient_norm=float(norm),
                            case_id=entry['case_id'], frame=entry['frame'], seconds=time.monotonic()-started))
        atomic_checkpoint(path, dict(model=model.operator.state_dict(), optimizer=optimizer.state_dict(), step=index+1,
                                     history=history, torch_rng=torch.get_rng_state(),
                                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                                     schedule_sha256=sha256_file(schedule_path), provenance_hash=provenance_hash))
        print(json.dumps(dict(stage='pretrain', **history[-1])), flush=True)
    atomic_json(folder/'history.json', history)
    atomic_checkpoint(folder/'pretrained.pt', dict(model=model.operator.state_dict(), step=config['pretrain_steps'],
                                                  selection='final fixed-budget pretraining update',
                                                  provenance_hash=provenance_hash))
    return dict(steps=config['pretrain_steps'], history=history)


def _wall_samples(cases, stride=8):
    temperature, vpar = [], []
    for case in cases:
        with h5py.File(case['resolved_path']) as stream:
            states = stream['native_state'][::stride]
        temperature.append(states[..., 5])
        vpar.append(states[..., 6])
    return np.concatenate(temperature), np.concatenate(vpar)


def build_system(config, data, space, verification, output, *, resume, labels=None):
    """Frozen tangent system; Dirichlet set and Mach-1 wall rows identified on training data only (cached)."""
    frozen = load_frozen_tangent(config['tangent'], space.geometry)
    if frozen.sha256 != config['tangent_sha256']:
        raise ValueError('Frozen tangent hash differs from configuration')
    train = [case for case in data.cases if case['split'] == 'train']
    train_ids = [case['case_id'] for case in train]
    cache = output/'fixed_dofs.npz'
    if cache.exists():
        with np.load(cache) as saved:
            fixed, ids = saved['fixed'], saved['case_ids'].tolist()
            fixed_report = json.loads(str(saved['report']))
        if ids != train_ids:
            raise ValueError('Cached Dirichlet set came from different training trajectories')
    else:
        fixed, fixed_report = fixed_dofs_from_trajectories([case['resolved_path'] for case in train], data.scale,
                                                           space.geometry.boundary_labels,
                                                           tolerance=config['dirichlet_tolerance'])
        np.savez(cache, fixed=fixed, case_ids=train_ids, report=json.dumps(fixed_report))
    wall_cache = output/'wall_rows.npz'
    if wall_cache.exists():
        with np.load(wall_cache) as saved:
            if saved['case_ids'].tolist() != train_ids:
                raise ValueError('Cached wall rows came from different training trajectories')
            held = saved['held']
            relations = BoundaryRelations(saved['rows'], saved['kind'], saved['value_dofs'], saved['coefficients'])
            wall_report = json.loads(str(saved['report']))
    else:
        if labels is None:
            raise ValueError('Wall-row identification needs the exact training labels')
        candidates = [e for e in labels.entries if 4 <= e['frame'] <= 92]
        chosen = [candidates[i] for i in np.linspace(0, len(candidates)-1, 6).round().astype(int)]
        identity = []
        for entry in chosen:
            with h5py.File(entry['data']) as stream:
                window = stream['native_state'][entry['frame']-1:entry['frame']+2]
            identity.append((window[0], window[1], window[2], np.load(entry['path']).astype(np.float64)))
        held, relations, wall_report = identify_boundary_rows(
            space.geometry, frozen, fixed, identity, _wall_samples(train[0::2]), _wall_samples(train[1::2]),
            F0=config['F0'], code_interval=verification['code_interval'])
        wall_report['identity_states'] = [dict(case_id=e['case_id'], frame=e['frame']) for e in chosen]
        np.savez(wall_cache, held=held, rows=relations.rows, kind=relations.kind, value_dofs=relations.value_dofs,
                 coefficients=relations.coefficients, case_ids=train_ids, report=json.dumps(wall_report))
    device = torch.device(config.get('device', 'cpu'))
    system = ImplicitNativeSystem(space.geometry, frozen, data.mean, data.scale, fixed | held,
                                  code_interval=verification['code_interval'], theta=config['theta'], F0=config['F0'],
                                  relations=relations, backend='scipy' if device.type == 'cpu' else 'torch',
                                  device=device)
    system.frozen = frozen   # the tangent correction re-weights these couplings
    digest = lambda *arrays: hashlib.sha256(b''.join(np.ascontiguousarray(a).tobytes() for a in arrays)).hexdigest()
    return system, dict(fixed_report, fixed_sha256=digest(np.packbits(fixed)),
                        wall=dict(wall_report, sha256=digest(np.packbits(held), relations.rows, relations.coefficients)))


def run_implicit(config, model_config, output, *, arm, resume=False, stop_after=None):
    check_implicit_config(config)
    if arm not in ARMS:
        raise ValueError(f'Unknown arm {arm}')
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=resume)
    device = torch.device(config.get('device', 'cpu'))
    if device.type == 'cuda':
        configure_cuda_numerics()
        if not torch.cuda.is_available():
            raise RuntimeError('device: cuda requested but no CUDA device is visible')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    data, space, guard, _, dense, verification = prepare_data(config)
    training_ids = [case['case_id'] for case in data.cases if case['split'] == 'train']
    labels = LabelStore(config['labels'], tangent_sha256=config['tangent_sha256'], training_case_ids=training_ids)
    system, dirichlet = build_system(config, data, space, verification, output, resume=resume, labels=labels)
    architecture = {**model_config, 'architecture': {**model_config['architecture'], **config['architecture']}}
    provenance = dict(config=config, model_config=architecture, arm=arm, manifest_sha256=sha256_file(config['manifest']),
        source_sha256={path.name: sha256_file(path) for path in Path(__file__).parent.glob('*.py')},
        python_version=sys.version, torch_version=str(torch.__version__), numpy_version=np.__version__,
        scipy_version=scipy.__version__, h5py_version=h5py.__version__,
        labels=labels.identity, label_scale=labels.scale.tolist(), dirichlet=dirichlet,
        method=('(M* - theta h L*) q\' = (M* + (1-theta) h L*) q + h (F(q*) - L* q* + N_theta(q)); '
                'native frozen tangent and mass at q*; current/vorticity weak rows; Dirichlet DOFs from training data'),
        optimizer=dict(name='Adam', learning_rate=config['learning_rate'], pretrain_learning_rate=config['pretrain_learning_rate'],
                       betas=[.9, .999], eps=1e-8, weight_decay=0, gradient_clip_norm=2.),
        checkpoint_selection='final fixed-budget update; no validation or test-based selection',
        evaluation_start_frame=0, training_substeps=1,
        physical_interval=verification['code_interval'], normalization=verification['normalization'],
        source_normalization=data.manifest['normalization'],
        uncertainty_unit='independent initial-condition trajectory; seeds reported separately',
        independent_initial_conditions=True, independent_equilibria=False)
    provenance_hash = config_hash(provenance)
    if config.get('physical_defect') is not None:
        provenance.update(method='Native state-dependent volume RHS with frozen M* and implicit L*; '
                          'learned defect G(q)-G(q*); stage constraints and positivity guards',
                          reference_forcing_retained=True,
                          physical_internal_substeps=config['physical_defect'].get('internal_substeps', 1))
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
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    model = build_model(config, space, system, architecture, labels.scale, physical_guard=guard,
                        physics_only=(arm == 'physics')).to(device)
    initial_hash = neural_hash(model)
    folder = output/arm
    folder.mkdir(exist_ok=resume)
    result = dict(arm=arm, initial_neural_sha256=initial_hash,
                  parameters=0 if arm == 'physics' else sum(p.numel() for p in model.parameters()),
                  correction_parameters=0 if model.correction is None else model.correction.weight.numel())
    if arm == 'implicit' and config['pretrain_steps']:
        stage = pretrain(model, data, labels, space, config, folder/'pretrain', provenance_hash,
                         resume=resume, stop_after=stop_after)
        if stage is None:
            return dict(interrupted=True)
        result['pretraining'] = dict(steps=stage['steps'], final_loss=stage['history'][-1]['remainder_loss'] if stage['history'] else None)
        stop_after = None if stop_after is None else max(stop_after-config['pretrain_steps'], 0)
    if arm != 'physics':
        (folder/'rollout').mkdir(exist_ok=True)
        training = train_family(model, data, space, config, folder/'rollout', provenance_hash,
                                resume=resume, stop_after=stop_after)
        if training is None:
            return dict(interrupted=True)
        result.update(selected_step=training['selected_step'], rollout_history=training['history'])
        if model.correction is not None:
            result['picard_training'] = dict(model.picard_statistics)
    model.eval()
    model.reset_picard_statistics()
    system.release_adjoints()  # cuDSS: the A^T factorisation (~13 GB on the production system) is training-only
    evaluation_started = time.monotonic()
    result['evaluation'] = evaluate(model, data, space, guard, dense, config, folder, provenance_hash)
    if result['evaluation'] is None:
        return dict(interrupted=True)
    if hasattr(model, 'stage_statistics'):
        result['last_trajectory_stage_statistics'] = dict(model.stage_statistics)
        result['physics_provenance'] = getattr(model.physical_rhs, 'provenance', {})
    if model.correction is not None:
        result.update(picard_evaluation=dict(model.picard_statistics), correction=model.correction.summary(),
                      picard_settings=model.picard)
    result.update(evaluation_seconds=time.monotonic()-evaluation_started, system=system.report(),
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                  peak_device_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else None)
    atomic_json(folder/'metrics.json', result)
    atomic_json(output/'results.json', dict(results={arm: result}, elapsed_seconds=time.monotonic()-started,
                diagnostic_only=config.get('diagnostic_only', False), all_audits_passed=result['evaluation']['audits_passed']))
    return {arm: result}
