"""One-step time-conditional family: same exposure as AR/NODE/OpsSplit, FiLM input [D/D0, (t - t0)/T]."""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from mastu_opsplit.common import load_config
from mastu_opsplit.ensemble_training import check_config, evaluate_trajectory, time_conditioning, train_family
from mastu_opsplit.euler import EulerSplineModel
from test_conditional import setup
from test_learning import configuration

TIME = dict(origin_seconds=4e-6, scale_seconds=4e-8, interval_seconds=1e-8)


def model(space, guard, family='conditional'):
    torch.manual_seed(17)
    return EulerSplineModel(family, space, configuration(), physical_guard=guard,
                            time_conditioning=TIME if family == 'conditional' else None)


def windows_fixture(tmp_path, mean):
    """Two four-frame-capable trajectories with the dataset's window sample layout (including start time)."""
    times = np.arange(6, dtype=np.float64)*1e-8 + 4e-6
    cases = []
    for split in ('train', 'test'):
        path = tmp_path/f'{split}.h5'
        state = np.broadcast_to(mean, (6, *mean.shape)).copy()
        state[:, :, 2:4] += np.arange(6)[:, None, None]*.001
        with h5py.File(path, 'w') as stream:
            stream['native_state'] = state
        cases.append(dict(case_id=split, split=split, resolved_path=path))
    data = SimpleNamespace(mean=mean, scale=np.ones(7), time_seconds=times, cases=cases)

    class Windows:
        def __init__(self, split, horizon):
            self.cases = [case for case in cases if case['split'] == split]
            self.horizon, self.starts = horizon, len(times)-horizon

        def __len__(self):
            return len(self.cases)*self.starts

        def __getitem__(self, index):
            case, start = divmod(index, self.starts)
            with h5py.File(self.cases[case]['resolved_path']) as stream:
                state = stream['native_state'][start:start+self.horizon+1]
            return dict(state=torch.from_numpy(state-mean), parameters=torch.ones(1), case_id=self.cases[case]['case_id'],
                        start_frame=start, time_seconds=torch.from_numpy(times[start:start+self.horizon+1].copy()))

    data.windows = lambda split, horizon, stride=1: Windows(split, horizon)
    return data


def test_time_is_required_only_by_the_conditional_family(tmp_path):
    space, guard, _, _ = setup(tmp_path)
    state, parameters = torch.zeros(1, 16, 7, dtype=torch.float64), torch.ones(1, 1)
    with pytest.raises(ValueError, match='state time'):
        model(space, guard)(state, parameters)
    with pytest.raises(ValueError, match='state time'):
        model(space, guard, 'node')(state, parameters, time_seconds=4e-6)
    with pytest.raises(ValueError, match='trajectory start'):
        model(space, guard)(state, parameters, time_seconds=3e-6)
    with pytest.raises(ValueError, match='time conditioning'):
        EulerSplineModel('conditional', space, configuration(), physical_guard=guard)
    with pytest.raises(ValueError, match='time conditioning'):
        EulerSplineModel('node', space, configuration(), physical_guard=guard, time_conditioning=TIME)
    conditional, node = model(space, guard), model(space, guard, 'node')
    assert conditional.operator.film.n_params == 2 and node.operator.film.n_params == 1


def test_state_time_reaches_film_and_substeps_advance_it(tmp_path):
    space, guard, _, _ = setup(tmp_path)
    network = model(space, guard)
    state = torch.randn(1, 16, 7, dtype=torch.float64)*.001
    parameters = torch.ones(1, 1)
    with torch.no_grad():
        network.operator.film.beta_head.weight.fill_(.05)   # FiLM starts at identity; open the pathway
        early = network(state, parameters, time_seconds=4e-6)
        late = network(state, parameters, time_seconds=4.02e-6)
        again = network(state, parameters, time_seconds=4e-6)
    assert not torch.allclose(early, late)
    torch.testing.assert_close(early, again, rtol=0, atol=0)
    seen = []
    original = network.rhs
    network.rhs = lambda state, parameters, tau=None: seen.append(float(tau)) or original(state, parameters, tau)
    with torch.no_grad():
        network(state, parameters, substeps=4, time_seconds=4.01e-6)
    np.testing.assert_allclose(seen, [.25, .3125, .375, .4375], rtol=1e-12)


def test_training_uses_rollout_windows_and_each_input_state_time(tmp_path):
    space, guard, dense, mean = setup(tmp_path)
    data = windows_fixture(tmp_path, mean)
    config = dict(seed=17, steps=4, horizon=2, stride=1, learning_rate=.001, positivity_penalty=.001)
    conditional, node = model(space, guard), model(space, guard, 'node')
    seen = []
    hook = conditional.register_forward_pre_hook(lambda module, args, kwargs: seen.append(
        (args[0].clone(), float(kwargs['time_seconds']))), with_kwargs=True)
    for name, network in (('conditional', conditional), ('node', node)):
        (tmp_path/name).mkdir()
        train_family(network, data, space, config, tmp_path/name, 'p')
    hook.remove()
    # Same sampled windows as the rollout baselines; the second query advances the model's own prediction.
    assert (tmp_path/'conditional/sampled_windows.json').read_bytes() == (tmp_path/'node/sampled_windows.json').read_bytes()
    windows = data.windows('train', 2)
    for step, row in enumerate(json.loads((tmp_path/'conditional/sampled_windows.json').read_text())):
        sample = windows[row['index']]
        (first, t0), (second, t1) = seen[2*step:2*step+2]
        torch.testing.assert_close(first, sample['state'][:1], rtol=0, atol=0)
        assert not torch.equal(second, sample['state'][1:2])
        assert (t0, t1) == (float(sample['time_seconds'][0]), float(sample['time_seconds'][1]))
    # Autoregressive evaluation from frame zero with 1/2/4 substeps, like NODE.
    sample = data.windows('test', 4)[0]
    for substeps in (1, 2, 4):
        records = evaluate_trajectory(conditional, sample, space, guard, dense, [2, 4], substeps)
        assert [row['horizon'] for row in records] == [2, 4] and all(row['audit_passed'] for row in records)
        assert 'raw_normalized_volume_mse' not in records[-1]


def test_runner_trains_conditional_with_the_ensemble_config(tmp_path, monkeypatch):
    from mastu_opsplit import ensemble_training as training
    space, guard, dense, mean = setup(tmp_path)
    data = windows_fixture(tmp_path, mean)
    data.manifest = {'normalization': {'fitted_split': 'train'}}
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    verification = dict(code_interval=.01, normalization={'training_case_ids': ['train']})
    monkeypatch.setattr(training, 'prepare_data', lambda config: (data, space, guard, None, dense, verification))
    config = load_config(Path(__file__).parents[1]/'configs/gpu_onestep.yaml')
    config.update(manifest=str(manifest), steps=2, horizon=2, evaluation_horizons=[2, 4], threads=1, device='cpu',
                  time_conditioning_intervals=4)
    result = training.run_ensemble(config, configuration(), tmp_path/'run', families=('conditional', 'node'))
    assert all(result[name]['evaluation']['audits_passed'] for name in ('conditional', 'node'))
    assert {row['substeps'] for row in result['conditional']['evaluation']['aggregates']} == {1, 2, 4}
    provenance = json.loads((tmp_path/'run/provenance.json').read_text())
    assert provenance['time_conditioning'] == time_conditioning(data, config) == dict(
        origin_seconds=4e-6, scale_seconds=float(data.time_seconds[4]-data.time_seconds[0]),
        interval_seconds=float(data.time_seconds[1]-data.time_seconds[0]))
    assert result['conditional']['parameters'] > result['node']['parameters']
    with pytest.raises(ValueError, match='time_conditioning_intervals'):
        training.run_ensemble({**config, 'time_conditioning_intervals': None}, configuration(), tmp_path/'other',
                              families=('conditional',))


def test_gpu_onestep_config_matches_the_ensemble_protocol():
    root = Path(__file__).parents[1]/'configs'
    config, reference = load_config(root/'gpu_onestep.yaml'), load_config(root/'ensemble_training.yaml')
    check_config(config)
    differences = {key for key in config.keys() | reference.keys() if config.get(key) != reference.get(key)}
    assert differences == {'device', 'time_conditioning_intervals', 'threads'}
    assert config['device'] == 'cuda' and config['time_conditioning_intervals'] == 96
    with pytest.raises(ValueError, match='device'):
        check_config({**config, 'device': 'tpu'})
    with pytest.raises(ValueError, match='time conditioning'):
        check_config({**config, 'time_conditioning_intervals': 0})
