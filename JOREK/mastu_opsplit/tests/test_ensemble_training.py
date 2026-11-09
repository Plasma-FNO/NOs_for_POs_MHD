from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mastu_opsplit.continuity import FrozenContinuity
from mastu_opsplit.coefficient_ood import DensityDiffusionEulerModel
from mastu_opsplit.euler import EulerSplineModel, NativeSplineSpace
from mastu_opsplit.ensemble_training import neural_hash, observation_scaling, train_family, evaluate, evaluate_trajectory, merge_training_split
from mastu_opsplit.physical_guard import PhysicalGuard, QuadratureSampler
from test_learning import configuration
from test_physical_guard import fixture


class Windows:
    def __init__(self, mean, split, horizon):
        self.cases = [{'case_id': split}]
        self.starts = 2
        self.horizon = horizon
        self.mean = mean

    def __len__(self):
        return self.starts

    def __getitem__(self, index):
        state = torch.zeros(self.horizon+1, len(self.mean), 7, dtype=torch.float64)
        state[:, :, 2:4] = .001*(index+1)
        return dict(state=state, parameters=torch.ones(1), case_id=self.cases[0]['case_id'],
                    start_frame=index, time_seconds=torch.arange(self.horizon+1)*1e-8)


@pytest.mark.parametrize('family,coefficient_aware,branch',
    [(family, aware, 'lspline_continuity') for family in ('ar', 'node', 'opsplit') for aware in (False, True)]
    + [('opsplit', False, 'native_continuity')])
def test_matched_training_exact_resume_and_dense_guards(tmp_path, family, coefficient_aware, branch):
    torch.set_num_threads(1)
    geometry, dense_geometry, mean = fixture(tmp_path)
    space = NativeSplineSpace(geometry)
    guard = PhysicalGuard(geometry, dense_geometry, mean, np.ones(7))
    physical = FrozenContinuity(geometry, mean, np.ones(7), .03125)
    data = SimpleNamespace(windows=lambda split, horizon, stride=1: Windows(mean, split, horizon))
    config = dict(seed=17, steps=3, horizon=2, stride=1,
                  learning_rate=.001, positivity_penalty=.001)
    def model():
        torch.manual_seed(17)
        if coefficient_aware:
            return DensityDiffusionEulerModel(family, space, configuration(), mean=mean, scale=np.ones(7),
                code_interval=.03125, physical_guard=guard, physical_linear=physical)
        return EulerSplineModel(family, space, configuration(), physical_guard=guard,
                                physical_linear=physical, linear_branch=branch)
    whole, interrupted = model(), model()
    assert neural_hash(whole) == neural_hash(interrupted)
    full = tmp_path/'full'
    partial = tmp_path/'partial'
    full.mkdir()
    partial.mkdir()
    train_family(whole, data, space, config, full, 'immutable')
    assert train_family(interrupted, data, space, config, partial, 'immutable', stop_after=1) is None
    resumed = model()
    train_family(resumed, data, space, config, partial, 'immutable', resume=True)
    assert (full/'sampled_windows.json').read_bytes() == (partial/'sampled_windows.json').read_bytes()
    for name, value in whole.operator.state_dict().items():
        torch.testing.assert_close(value, resumed.operator.state_dict()[name], rtol=0, atol=0)
    whole_checkpoint = torch.load(full/'last.pt', weights_only=False)
    resumed_checkpoint = torch.load(partial/'last.pt', weights_only=False)
    assert whole_checkpoint['step'] == resumed_checkpoint['step'] == 3
    assert whole_checkpoint['family'] == resumed_checkpoint['family'] == family
    for name, value in whole_checkpoint['model'].items():
        torch.testing.assert_close(value, resumed_checkpoint['model'][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match='provenance'):
        train_family(model(), data, space, config, partial, 'changed', resume=True)
    records = evaluate_trajectory(resumed, data.windows('test', 2)[0], space, guard,
                                  QuadratureSampler(dense_geometry), [1, 2], 1)
    assert all(record['audit_passed'] for record in records)
    assert records[-1]['guard_statistics']['steps'] == 2


def test_observation_stride_and_native_time_compatibility(tmp_path):
    import h5py
    from mastu_opsplit.common import sha256_file
    from mastu_opsplit.euler_dataset import EulerDataset
    from test_restart import rectangle
    rectangle(tmp_path/'initial.h5')
    case = dict(initial='initial.h5', initial_check={'sha256': sha256_file(tmp_path/'initial.h5')})
    data = SimpleNamespace(root=tmp_path, cases=[case], windows=lambda split, horizon, stride:
                           SimpleNamespace(observation_interval_seconds=stride*1e-8))
    data.verified_path = lambda record: EulerDataset.verified_path(data, record)
    _, first = observation_scaling(data, 1)
    _, second = observation_scaling(data, 2)
    assert second['code_interval'] == 2*first['code_interval']
    rectangle(tmp_path/'other.h5')
    with h5py.File(tmp_path/'other.h5', 'r+') as stream:
        stream['sqrt_mu0_rho0'][...] *= 2
    data.cases.append(dict(initial='other.h5', initial_check={'sha256': sha256_file(tmp_path/'other.h5')}))
    with pytest.raises(ValueError, match='time normalization'):
        observation_scaling(data, 1)


def test_pooled_normalization_excludes_test_and_preserves_trajectory_membership(tmp_path):
    import h5py
    from mastu_opsplit.euler_dataset import EulerDataset
    from test_euler_dataset import fixture_release
    fixture_release(tmp_path)
    data = EulerDataset(tmp_path/'manifest.json')
    space = NativeSplineSpace(data.geometry)
    originals = {case['case_id']: case['split'] for case in data.cases}
    report = merge_training_split(data, space)
    states = []
    for case in data.cases:
        if case['split'] == 'train':
            with h5py.File(case['resolved_path']) as stream:
                states.extend(stream['native_state'][()])
    states = np.stack(states)
    expected_mean = states.mean(0)
    variance = sum(np.einsum('pc,p->c', (space.basis@(state-expected_mean))**2,
                            space.weights/space.weights.sum()) for state in states)/len(states)
    np.testing.assert_allclose(data.mean, expected_mean, rtol=1e-14)
    np.testing.assert_allclose(data.scale, np.sqrt(variance), rtol=1e-14)
    assert report['fitted_frames'] == 9
    assert set(report['training_case_ids']) == {name for name, split in originals.items() if split != 'test'}
    assert {case['case_id'] for case in data.cases if case['split'] == 'test'} == {'case3'}
    assert not data.windows('validation', 1).cases
    assert all(case['split'] == originals[case['case_id']] for case in data.manifest['cases'])


def test_summary_rejects_unmatched_windows(tmp_path):
    import importlib.util
    from pathlib import Path
    from mastu_opsplit.common import write_json
    script = Path(__file__).resolve().parents[1]/'scripts/summarize_ensemble.py'
    spec = importlib.util.spec_from_file_location('ensemble_summary', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for seed in (17, 18, 19):
        for family in ('ar', 'node', 'opsplit'):
            folder = tmp_path/f'seed{seed}_{family}'
            write_json(folder/'provenance.json', {'config': {'seed': seed, 'steps': 2}})
            write_json(folder/family/'sampled_windows.json', [1, 2])
            write_json(folder/family/'metrics.json', dict(initial_neural_sha256='matched', parameters=10,
                selected_step=2, evaluation=dict(audits_passed=True, numerical_failures=0, aggregates=[
                    dict(horizon=16, substeps=1, trajectories=20, mean_normalized_volume_mse=seed-16,
                         trajectory_std=.1)])))
    summary = module.summarize(tmp_path)
    assert summary['all_audits_passed']
    assert all(row['mean'] == 2 and row['seed_std'] == 1 for row in summary['comparison'])
    (tmp_path/'seed18_node/node/sampled_windows.json').write_text('[2, 1]')
    with pytest.raises(ValueError, match='not matched'):
        module.summarize(tmp_path)


@pytest.mark.parametrize('problem', ['missing', 'audit', 'nonfinite', 'nonfinite_field', 'empty', 'no_cases'])
def test_evaluation_withholds_invalid_aggregate(tmp_path, problem):
    import json
    from mastu_opsplit.euler_training import FIELDS
    windows = Windows(np.zeros((16, 7)), 'test', 1)
    windows.cases *= 2
    if problem == 'no_cases':
        windows.cases = []
    for index in range(len(windows.cases)):
        row = dict(horizon=1, substeps=1, case_id=f'test{index}', audit_passed=True,
                   normalized_volume_mse=.01, per_field_normalized_mse=dict.fromkeys(FIELDS, .01))
        failure = None
        if index == 1 and problem == 'audit':
            row['audit_passed'] = False
        if index == 1 and problem == 'nonfinite':
            row['normalized_volume_mse'] = float('nan')
        if index == 1 and problem == 'nonfinite_field':
            row['per_field_normalized_mse'][FIELDS[0]] = float('nan')
        records = [row]
        if problem == 'empty' or (index == 1 and problem == 'missing'):
            records = []
            failure = 'Failed rollout' if problem == 'missing' else None
        # Include malformed legacy caches that the strict production writer rejects.
        (tmp_path/f'evaluation_{index:03d}_substeps1.json').write_text(json.dumps(
            dict(records=records, failure=failure, provenance_hash='fixed')))
    result = evaluate(SimpleNamespace(family='opsplit'), SimpleNamespace(windows=lambda *a, **kw: windows),
                      None, None, None, dict(evaluation_horizons=[1], evaluation_substeps=[1], stride=1),
                      tmp_path, 'fixed')
    row = result['aggregates'][0]
    assert row['mean_normalized_volume_mse'] is None
    assert row['trajectory_std'] is None
    assert row['per_field_normalized_mse'] == {}
    assert row['expected_trajectories'] == len(windows.cases)
    assert not row['complete']
    assert not result['audits_passed']


def test_evaluation_retains_completed_horizons_on_failure_and_resume(tmp_path):
    geometry, dense_geometry, mean = fixture(tmp_path)
    space = NativeSplineSpace(geometry)
    guard = PhysicalGuard(geometry, dense_geometry, mean, np.ones(7))
    windows = Windows(mean, 'test', 3)
    class FailsLate:
        family = 'node'
        calls = 0

        def __call__(self, state, parameters, substeps):
            self.calls += 1
            if self.calls == 3:
                raise FloatingPointError('Failure after two valid steps')
            return guard(state, state)

    model = FailsLate()
    config = dict(evaluation_horizons=[1, 2, 3], evaluation_substeps=[1], stride=1)
    args = (model, SimpleNamespace(windows=lambda *a, **kw: windows), space, guard,
            QuadratureSampler(dense_geometry), config, tmp_path, 'fixed')
    result = evaluate(*args)
    assert [row['horizon'] for row in result['trajectories'][0]['records']] == [1, 2]
    assert [row['complete'] for row in result['aggregates']] == [True, True, False]
    assert result['aggregates'][0]['mean_normalized_volume_mse'] is not None
    assert result['aggregates'][2]['mean_normalized_volume_mse'] is None
    assert result['numerical_failures'] == 1
    assert not result['audits_passed']
    assert evaluate(*args) == result
    assert model.calls == 3


@pytest.mark.parametrize('complete,global_passed,valid', [(None, False, False), (False, True, False), (True, False, True)])
def test_summary_requires_valid_horizon_audits(tmp_path, complete, global_passed, valid):
    import importlib.util
    from pathlib import Path
    from mastu_opsplit.common import write_json
    spec = importlib.util.spec_from_file_location('ensemble_summary', Path(__file__).resolve().parents[1]/'scripts/summarize_ensemble.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for seed in (17, 18, 19):
        for family in ('ar', 'node', 'opsplit'):
            folder = tmp_path/f'seed{seed}_{family}'
            write_json(folder/'provenance.json', {'config': {'seed': seed, 'steps': 2}})
            write_json(folder/family/'sampled_windows.json', [1, 2])
            row = dict(horizon=16, substeps=1, trajectories=20, mean_normalized_volume_mse=.01, trajectory_std=.1)
            if complete is not None:
                row['complete'] = complete
            write_json(folder/family/'metrics.json', dict(initial_neural_sha256='matched', parameters=10,
                selected_step=2, evaluation=dict(audits_passed=global_passed, numerical_failures=0, aggregates=[row])))
    result = module.summarize(tmp_path)
    assert all(row['complete'] == valid for row in result['comparison'])
    assert all(row['mean'] == (.01 if valid else None) for row in result['comparison'])
    if not valid:
        assert all(row['seed_means'] == [None]*3 for row in result['comparison'])
