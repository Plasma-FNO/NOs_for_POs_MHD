#!/usr/bin/env python3
"""Compare learned/native gradients and fixed-weight rollouts on training trajectories."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch


def power(space, coefficients):
    sampled = space.sample(coefficients).detach().numpy()
    return np.einsum('bpc,p->c', sampled**2, space.weights/space.weights.sum())/len(sampled)


def increments(model, state, parameters):
    values = model.space.sample(state)
    neural = model.space.project(model.operator(values.float(), model.coordinates, parameters.float()).to(state.dtype))
    linear = model.linear_rhs(state, values)
    return neural, linear


def native_gradients(model, state):
    from mastu_opsplit.full_rollout import native_jets
    return native_jets(state, model.space.geometry)[1].flatten(1, 2)


@torch.no_grad()
def measure_state(model, sample):
    space, guard = model.space, model.physical_guard
    state = sample['state'][:1]
    neural, linear = increments(model, state, sample['parameters'][None])
    torch.testing.assert_close(neural+linear, model.rhs(state, sample['parameters'][None]), rtol=0, atol=0)
    guard.reset_statistics()
    full = guard(state, state+(neural+linear))
    full_statistics = dict(guard.statistics)
    guard.reset_statistics()
    without = guard(state, state+neural)
    without_statistics = dict(guard.statistics)
    exact_gradients = native_gradients(model, state)
    values = space.sample(state)
    native_linear = space.project(model.physical_linear(values, exact_gradients))
    source = model.operator.normalized_coordinates(model.coordinates, 1).to(values.float())
    coefficients = model.operator.backbone._encode(values.float().transpose(1, 2), source)
    learned_gradients = torch.stack((model.derivatives(coefficients, (1, 0)),
                                    model.derivatives(coefficients, (0, 1))), -1).double().numpy()
    weights = space.weights/space.weights.sum()
    gradient_error = np.einsum('bpcr,p->c', (learned_gradients-exact_gradients.numpy())**2, weights)
    gradient_norm = np.einsum('bpcr,p->c', exact_gradients.numpy()**2, weights)
    moments = {name: power(space, value).tolist() for name, value in dict(
        neural=neural, linear=linear, total=neural+linear,
        observed_increment=sample['state'][1:2]-state, guarded_linear_effect=full-without,
        native_gradient_linear=native_linear, linear_vs_native_error=linear-native_linear).items()}
    moments.update(gradient_error=gradient_error.tolist(), gradient_norm=gradient_norm.tolist())
    return dict(case_id=sample['case_id'], start_frame=sample['start_frame'], power=moments,
                guard_full=full_statistics, guard_without_linear=without_statistics)


@torch.no_grad()
def ablate_rollout(model, sample):
    predictions, statistics = {}, {}
    for branch in ('learned', 'native', 'without_linear'):
        state = sample['state'][:1]
        future = []
        model.physical_guard.reset_statistics()
        for _ in sample['state'][1:]:
            values = model.space.sample(state)
            neural = model.space.project(model.operator(values.float(), model.coordinates,
                                         sample['parameters'][None].float()).to(state.dtype))
            if branch == 'learned':
                linear = model.linear_rhs(state, values)
            elif branch == 'native':
                linear = model.space.project(model.physical_linear(values, native_gradients(model, state)))
            else:
                linear = 0.
            state = model.physical_guard(state, state+(neural+linear))
            future.append(state)
        predictions[branch] = torch.cat(future)
        statistics[branch] = dict(model.physical_guard.statistics)
    truth = sample['state'][1:]
    return dict(case_id=sample['case_id'], start_frame=sample['start_frame'], horizon=len(truth),
                full_per_field_mse=power(model.space, predictions['learned']-truth).tolist(),
                native_per_field_mse=power(model.space, predictions['native']-truth).tolist(),
                without_linear_per_field_mse=power(model.space, predictions['without_linear']-truth).tolist(),
                prediction_difference_per_field_mse=power(model.space, predictions['learned']-predictions['without_linear']).tolist(),
                native_prediction_difference_per_field_mse=power(model.space, predictions['learned']-predictions['native']).tolist(),
                guard_statistics=statistics)


def aggregate(records):
    moments = {name: np.mean([row['power'][name] for row in records], axis=0) for name in records[0]['power']}
    result = {name+'_rms': np.sqrt(values).tolist() for name, values in moments.items()
              if name not in ('gradient_error', 'gradient_norm')}
    result['linear_over_neural_rms'] = np.sqrt(moments['linear']/np.maximum(moments['neural'], 1e-30)).tolist()
    result['linear_over_observed_increment_rms'] = np.sqrt(moments['linear']/np.maximum(moments['observed_increment'], 1e-30)).tolist()
    result['guard_effect_over_linear_rms'] = np.sqrt(moments['guarded_linear_effect']/np.maximum(moments['linear'], 1e-30)).tolist()
    result['linear_vs_native_relative_rms'] = np.sqrt(moments['linear_vs_native_error']/np.maximum(moments['native_gradient_linear'], 1e-30)).tolist()
    result['learned_vs_native_gradient_relative_rms'] = np.sqrt(moments['gradient_error']/np.maximum(moments['gradient_norm'], 1e-30)).tolist()
    result['all_field_linear_over_neural_rms'] = float(np.sqrt(moments['linear'].sum()/moments['neural'].sum()))
    result['rho_T_linear_over_neural_rms'] = float(np.sqrt(moments['linear'][4:6].sum()/moments['neural'][4:6].sum()))
    result['linear_neural_cosine'] = ((moments['total']-moments['linear']-moments['neural']) /
                                    np.maximum(2*np.sqrt(moments['linear']*moments['neural']), 1e-30)).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', choices=('selected', 'last'), default='selected')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.run/'study/src'))
    from mastu_opsplit.common import config_hash, sha256_file
    from mastu_opsplit.euler import EulerSplineModel
    from mastu_opsplit.ensemble_training import prepare_data, atomic_json
    from mastu_opsplit.euler_training import FIELDS
    folder = args.run/'seed17_opsplit'
    copied = args.output/'opsplit_checkpoint.pt'
    shutil.copyfile(folder/f'opsplit/{args.checkpoint}.pt', copied)
    checkpoint = torch.load(copied, map_location='cpu', weights_only=False)
    provenance = json.loads((folder/'provenance.json').read_text())
    if checkpoint['provenance_hash'] != config_hash(provenance) or checkpoint['family'] != 'opsplit':
        raise ValueError('Checkpoint provenance mismatch')
    config = provenance['config']
    if args.checkpoint == 'selected' and (checkpoint['step'] != config['steps'] or
            checkpoint.get('selection') != 'final fixed-budget update'):
        raise ValueError('Require the final fixed-budget checkpoint')
    source = args.run/'study/src/mastu_opsplit'
    if provenance['source_sha256'] != {path.name: sha256_file(path) for path in source.glob('*.py')}:
        raise ValueError('Frozen training implementation changed')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    data, space, guard, physical, dense, verification = prepare_data(config)
    del dense
    with np.load(folder/'normalization.npz') as saved:
        np.testing.assert_array_equal(saved['mean'], data.mean)
        np.testing.assert_array_equal(saved['scale'], data.scale)
    windows = data.windows('train', config['horizon'], stride=config['stride'])
    indices = [int(case)*windows.starts+start for case in np.linspace(0, len(windows.cases)-1, 8, dtype=int)
               for start in (0, 60)]
    torch.manual_seed(config['seed'])
    model = EulerSplineModel('opsplit', space, provenance['model_config'], linear_branch='lspline_continuity',
                             physical_guard=guard, physical_linear=physical).eval()
    report = dict(checkpoint_step=checkpoint['step'], checkpoint_sha256=sha256_file(copied),
                  checkpoint_source=str(folder/f'opsplit/{args.checkpoint}.pt'), frozen_source_verified=True,
                  diagnostic_source_sha256=sha256_file(__file__), source_provenance_hash=config_hash(provenance),
                  fields=FIELDS, time_scaling=verification['code_interval'], training_only=True,
                  held_out_scores_used=False, window_indices=indices,
                  norm_definition='RMS of native volume samples, equally averaged over selected windows',
                  ablation_scope='Fixed trained N; learned gradients, native gradients, or no L, each advancing its own state; no retraining',
                  phases={})
    for phase in ('initialization', 'checkpoint'):
        if phase == 'checkpoint':
            model.operator.load_state_dict(checkpoint['model'])
        records, ablations = [], []
        for index in indices:
            sample = windows[index]
            records.append(measure_state(model, sample))
            if phase == 'checkpoint':
                ablations.append(ablate_rollout(model, sample))
            print(json.dumps(dict(phase=phase, case=sample['case_id'], frame=sample['start_frame'])), flush=True)
        phase_result = dict(samples=records, aggregate=aggregate(records), ablations=ablations)
        if ablations:
            phase_result['ablation_aggregate'] = {
                key: np.mean([row[key] for row in ablations], axis=0).tolist() for key in
                ('full_per_field_mse', 'native_per_field_mse', 'without_linear_per_field_mse',
                 'prediction_difference_per_field_mse', 'native_prediction_difference_per_field_mse')}
        report['phases'][phase] = phase_result
        atomic_json(args.output/'report.json', report)
    report['elapsed_seconds'] = time.monotonic()-started
    atomic_json(args.output/'report.json', report)


if __name__ == '__main__':
    main()
