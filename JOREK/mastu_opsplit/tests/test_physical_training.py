"""GPU interruption and evaluation reporting for native physical-defect experiments."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mastu_opsplit.common import load_config
from mastu_opsplit import ensemble_training
from mastu_opsplit.euler import NativeSplineSpace
from mastu_opsplit.implicit_training import check_implicit_config
from test_ensemble_training import Windows
from test_physical_guard import fixture


def test_autoregressive_physical_evaluation_records_raw_errors_and_stage_work(tmp_path):
    from mastu_opsplit.physical_guard import QuadratureSampler
    from test_physical_defect import make_model
    model = make_model(tmp_path, scheme='ars222', physics_only=True, guard=True)
    initial = model.reference_normalized.detach().cpu()
    sample = dict(state=torch.stack((initial, initial, initial)), parameters=torch.ones(1),
                  time_seconds=torch.arange(3, dtype=torch.float64)*1e-8, case_id='native', start_frame=0)
    dense = QuadratureSampler(model.space.geometry)
    with torch.no_grad():
        records = ensemble_training.evaluate_trajectory(model, sample, model.space, model.physical_guard,
                                                         dense, [1, 2], 1)
        again = ensemble_training.evaluate_trajectory(model, sample, model.space, model.physical_guard,
                                                       dense, [2], 1)
    assert records[-1]['stage_statistics']['stages'] == again[-1]['stage_statistics']['stages'] == 4
    assert np.isfinite(records[-1]['raw_normalized_volume_mse'])
    assert len(records[-1]['raw_per_field_normalized_mse']) == 7
    assert 'query_time_regime' not in records[-1]


def test_physical_training_rejects_cpu_and_remainder_pretraining():
    config = load_config(Path(__file__).parents[1]/'configs/implicit_opsplit.yaml')
    config.update(physical_defect={'scheme': 'be'}, pretrain_steps=0, device='cuda')
    check_implicit_config(config)
    with pytest.raises(ValueError, match='CUDA'):
        check_implicit_config({**config, 'device': 'cpu'})
    with pytest.raises(ValueError, match='pretraining'):
        check_implicit_config({**config, 'pretrain_steps': 1})
    with pytest.raises(ValueError, match='budget'):
        check_implicit_config({**config, 'cpu_budget_seconds': float('nan')})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='training check requires CUDA')
def test_worker_budget_resume_restores_cuda_randomness_and_optimizer(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    geometry, _, mean = fixture(tmp_path)
    space = NativeSplineSpace(geometry)
    data = SimpleNamespace(windows=lambda split, horizon, stride=1: Windows(mean, split, horizon))
    config = dict(seed=17, steps=3, horizon=2, stride=1, learning_rate=1e-3, positivity_penalty=0.)

    class StochasticStep(torch.nn.Module):
        family = 'opsplit'

        def __init__(self):
            super().__init__()
            self.operator = torch.nn.Linear(7, 7, dtype=torch.float64, device='cuda')
            self.physical_guard = SimpleNamespace(reset_statistics=lambda: None, last_penalty=0., statistics={})

        def forward(self, state, parameters):
            return state + .01*self.operator(state) + 1e-4*torch.randn_like(state)

    def fresh():
        torch.manual_seed(17)
        return StochasticStep()

    whole = fresh()
    full = tmp_path/'full'; full.mkdir()
    ensemble_training.train_family(whole, data, space, config, full, 'fixed')
    expected = torch.load(full/'last.pt', weights_only=False)
    partial = fresh()
    folder = tmp_path/'interrupted'; folder.mkdir()
    calls = iter((False, True))
    with monkeypatch.context() as patch:
        patch.setattr(ensemble_training, 'cpu_budget_expired', lambda config: next(calls))
        assert ensemble_training.train_family(partial, data, space, config, folder, 'fixed') is None
    saved = torch.load(folder/'last.pt', weights_only=False)
    assert saved['step'] == 1 and saved['cuda_rng'] is not None
    resumed = fresh()
    torch.rand(100, device='cuda')  # An unrelated worker's RNG use must not change the continuation.
    ensemble_training.train_family(resumed, data, space, config, folder, 'fixed', resume=True)
    actual = torch.load(folder/'last.pt', weights_only=False)
    for name, value in expected['model'].items():
        torch.testing.assert_close(value, actual['model'][name], rtol=0, atol=0)
    for index, state in expected['optimizer']['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(value, actual['optimizer']['state'][index][key], rtol=0, atol=0)
    assert all(torch.equal(a, b) for a, b in zip(expected['cuda_rng'], actual['cuda_rng']))
    np.testing.assert_array_equal([r['training_loss'] for r in expected['history']],
                                  [r['training_loss'] for r in actual['history']])
