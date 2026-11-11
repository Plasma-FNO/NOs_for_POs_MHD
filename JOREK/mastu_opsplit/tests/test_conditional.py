from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from mastu_opsplit.conditional import TimeConditionalSplineModel, initial_time_sample
from mastu_opsplit.ensemble_training import train_family, evaluate_trajectory, check_config
from mastu_opsplit.euler import NativeSplineSpace, field_loss
from mastu_opsplit.physical_guard import PhysicalGuard, QuadratureSampler
from test_learning import configuration
from test_physical_guard import fixture


def setup(tmp_path):
    torch.set_num_threads(1)
    geometry, dense_geometry, mean = fixture(tmp_path)
    space = NativeSplineSpace(geometry)
    guard = PhysicalGuard(geometry, dense_geometry, mean, np.ones(7))
    return space, guard, QuadratureSampler(dense_geometry), mean


def test_initial_identity_time_parameter_sensitivity_and_gradients(tmp_path):
    space, guard, _, _ = setup(tmp_path)
    model = TimeConditionalSplineModel(space, configuration(), 1e-6, physical_guard=guard)
    initial = torch.randn(2, 16, 7, dtype=torch.float64)*.001
    initial.requires_grad_()
    parameters = torch.ones(2, 1)
    torch.testing.assert_close(model(initial, parameters, elapsed_seconds=0), initial, rtol=0, atol=0)
    # FiLM starts at identity; activate its heads to test the conditioning pathway itself.
    with torch.no_grad():
        model.operator.film.beta_head.weight.fill_(.05)
    output = model(initial, parameters, elapsed_seconds=torch.tensor([2e-7, 5e-7], dtype=torch.float64))
    assert not torch.allclose(output, model(initial, parameters*2, elapsed_seconds=2e-7))
    assert not torch.allclose(output, model(initial, parameters, elapsed_seconds=4e-7))
    assert not torch.allclose(output, model(initial+.001, parameters, elapsed_seconds=2e-7))
    field_loss(output, initial, space).backward()
    assert torch.isfinite(initial.grad).all() and initial.grad.abs().sum() > 0
    for group in (model.operator.film, model.operator.backbone.fno_blocks,
                  model.operator.backbone.scatter_kernels):
        gradients = [p.grad for p in group.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    model.eval()
    with torch.no_grad():
        a = model(initial[:1], parameters[:1], elapsed_seconds=2e-7)
        model(initial[:1], parameters[:1], elapsed_seconds=9e-7)
        b = model(initial[:1], parameters[:1], elapsed_seconds=2e-7)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('elapsed', [-1., float('nan'), float('inf'), [0., 1.]])
def test_invalid_times_rejected(tmp_path, elapsed):
    space, guard, _, _ = setup(tmp_path)
    model = TimeConditionalSplineModel(space, configuration(), 1e-6, physical_guard=guard)
    with pytest.raises(ValueError, match='Elapsed seconds'):
        model(torch.zeros(1, 16, 7, dtype=torch.float64), torch.ones(1, 1), elapsed_seconds=elapsed)
    with pytest.raises(ValueError, match='positive'):
        model(torch.zeros(1, 16, 7, dtype=torch.float64), -torch.ones(1, 1), elapsed_seconds=0)


def data_fixture(tmp_path, mean):
    cases = []
    times = np.arange(5, dtype=np.float64)*1e-8 + 4e-6
    for split in ('train', 'test'):
        path = tmp_path/f'{split}.h5'
        state = np.broadcast_to(mean, (5, *mean.shape)).copy()
        state[:, :, 2:4] += np.arange(5)[:, None, None]*.001
        with h5py.File(path, 'w') as stream:
            stream['native_state'] = state
        cases.append(dict(case_id=split, split=split, resolved_path=path))
    data = SimpleNamespace(mean=mean, scale=np.ones(7), time_seconds=times, cases=cases)

    class Windows:
        def __init__(self, split, horizon):
            self.cases = [case for case in cases if case['split'] == split]
            self.horizon = horizon
            self.starts = len(times)-horizon

        def __len__(self):
            return len(self.cases)*self.starts

        def __getitem__(self, index):
            case, start = divmod(index, self.starts)
            return initial_time_sample(data, self.cases[case], np.arange(1, self.horizon+1),
                                       stride=1, start_frame=start)

    data.windows = lambda split, horizon, stride=1: Windows(split, horizon)
    return data


def test_sparse_queries_use_initial_state_physical_time_and_training_normalization(tmp_path):
    _, _, _, mean = setup(tmp_path)
    data = data_fixture(tmp_path, mean)
    sample = initial_time_sample(data, data.cases[0], [2, 1, 2], stride=2)
    np.testing.assert_array_equal(sample['time_seconds'], data.time_seconds[[0, 4, 2, 4]])
    np.testing.assert_array_equal(sample['state'][0], np.zeros_like(mean))
    np.testing.assert_allclose(sample['state'][1:, :, 2:4],
                               np.broadcast_to(np.array([.004, .002, .004])[:, None, None], (3, 16, 2)))
    assert sample['start_frame'] == 0
    with pytest.raises(ValueError, match='exceeds'):
        initial_time_sample(data, data.cases[0], [5], stride=1)
    shifted = initial_time_sample(data, data.cases[0], [1, 2], stride=1, start_frame=2)
    np.testing.assert_array_equal(shifted['time_seconds'], data.time_seconds[2:5])
    np.testing.assert_allclose(shifted['state'][0, :, 2:4], .002)
    assert shifted['start_frame'] == 2


def test_matched_queries_use_the_rollout_window_schedule_and_targets(tmp_path):
    import json
    space, guard, _, mean = setup(tmp_path)
    data = data_fixture(tmp_path, mean)
    config = dict(seed=17, steps=5, horizon=2, query_horizon=2, query_start='sampled_window',
                  stride=1, learning_rate=.001, positivity_penalty=.001)
    model = TimeConditionalSplineModel(space, configuration(), 2e-8, physical_guard=guard)
    folder = tmp_path/'matched'
    folder.mkdir()
    starts_seen = []
    hook = model.register_forward_pre_hook(lambda module, args: starts_seen.append(args[0].clone()))
    train_family(model, data, space, config, folder, 'matched')
    hook.remove()
    windows = data.windows('train', 2)
    expected = np.random.default_rng(17).integers(len(windows), size=5)
    schedule = json.loads((folder/'sampled_windows.json').read_text())
    assert [row['index'] for row in schedule] == expected.tolist()
    assert any(row['start_frame'] > 0 for row in schedule)
    for step, row in enumerate(schedule):
        assert row['query_frames'] == [1, 2]
        sample = windows[row['index']]
        assert row['start_frame'] == sample['start_frame']
        for initial in starts_seen[2*step:2*step+2]:
            torch.testing.assert_close(initial, sample['state'][:1], rtol=0, atol=0)


def test_direct_training_exact_resume_and_independent_evaluation(tmp_path):
    space, guard, dense, mean = setup(tmp_path)
    data = data_fixture(tmp_path, mean)
    config = dict(seed=17, steps=3, horizon=2, query_horizon=4, stride=1,
                  learning_rate=.001, positivity_penalty=.001)

    def model():
        torch.manual_seed(17)
        return TimeConditionalSplineModel(space, configuration(), 4e-8, physical_guard=guard)

    full, partial = tmp_path/'full', tmp_path/'partial'
    full.mkdir()
    partial.mkdir()
    whole = model()
    train_family(whole, data, space, config, full, 'fixed')
    assert train_family(model(), data, space, config, partial, 'fixed', stop_after=1) is None
    resumed = model()
    train_family(resumed, data, space, config, partial, 'fixed', resume=True)
    assert (full/'sampled_windows.json').read_bytes() == (partial/'sampled_windows.json').read_bytes()
    for name, value in whole.operator.state_dict().items():
        torch.testing.assert_close(value, resumed.operator.state_dict()[name], rtol=0, atol=0)
    sample = data.windows('test', 4)[0]
    # The evaluation must always pass q0, including queries after its first prediction.
    seen = []
    hook = resumed.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    records = evaluate_trajectory(resumed, sample, space, guard, dense, [2, 4], 1)
    hook.remove()
    assert len(seen) == 4
    for initial in seen:
        torch.testing.assert_close(initial, sample['state'][:1], rtol=0, atol=0)
    assert all(row['audit_passed'] for row in records)
    assert all(row['prediction_mode'] == 'initial_time_query' for row in records)
    assert records[-1]['guard_statistics']['steps'] == 4
    assert np.isfinite(records[-1]['raw_normalized_volume_mse'])
    assert len(records[-1]['raw_per_field_normalized_mse']) == 7
    with pytest.raises(ValueError, match='substep'):
        evaluate_trajectory(resumed, sample, space, guard, dense, [4], 2)


def test_direct_config_has_no_substeps_or_rhs_split():
    from pathlib import Path
    from mastu_opsplit.common import load_config
    config = load_config(Path(__file__).parents[1]/'configs/conditional_time.yaml')
    check_config(config)
    with pytest.raises(ValueError, match='substep'):
        check_config({**config, 'evaluation_substeps': [1, 2, 4]})
    with pytest.raises(ValueError, match='Query horizon'):
        check_config({**config, 'query_horizon': 1})
    with pytest.raises(ValueError, match='physical RHS split'):
        check_config({**config, 'coefficient_split': 'density_diffusion'})


def test_direct_runner_writes_provenance_checkpoint_and_raw_metrics(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from mastu_opsplit import ensemble_training as training
    from mastu_opsplit.common import load_config
    space, guard, dense, mean = setup(tmp_path)
    data = data_fixture(tmp_path, mean)
    data.manifest = {'normalization': {'fitted_split': 'train'}}
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    verification = dict(code_interval=.01, normalization={'training_case_ids': ['train']})
    monkeypatch.setattr(training, 'prepare_data', lambda config: (data, space, guard, None, dense, verification))
    config = load_config(Path(__file__).parents[1]/'configs/conditional_time.yaml')
    config.update(manifest=str(manifest), steps=1, horizon=2, query_horizon=4,
                  evaluation_horizons=[2, 4], threads=1)
    folder = tmp_path/'run'
    result = training.run_ensemble(config, configuration(), folder, families=('conditional',))
    assert result['conditional']['evaluation']['audits_passed']
    assert result['conditional']['selected_step'] == 1
    assert (folder/'conditional/selected.pt').exists()
    provenance = json.loads((folder/'provenance.json').read_text())
    assert provenance['time_scale_seconds'] == float(data.time_seconds[4]-data.time_seconds[0])
    assert 'broader temporal supervision' in provenance['comparison_caveat']
    for row in result['conditional']['evaluation']['aggregates']:
        assert np.isfinite(row['raw_mean_normalized_volume_mse'])
        assert len(row['raw_per_field_normalized_mse']) == 7
    resumed = training.run_ensemble(config, configuration(), folder, families=('conditional',), resume=True)
    assert resumed['conditional']['evaluation'] == result['conditional']['evaluation']
