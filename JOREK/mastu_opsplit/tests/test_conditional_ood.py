"""Coefficient inputs must reach direct queries without changing targets or q0."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mastu_opsplit.conditional import TimeConditionalSplineModel
from mastu_opsplit.ensemble_training import evaluate_trajectory
from test_conditional import setup
from test_learning import configuration

spec = importlib.util.spec_from_file_location('ood_evaluator',
    Path(__file__).parents[1]/'scripts/evaluate_ood_fixed_conditioning.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


@pytest.mark.parametrize('multiplier', [2., 10.])
def test_direct_ood_conditioning_control_preserves_initial_state_and_targets(tmp_path, multiplier):
    space, guard, dense, _ = setup(tmp_path)
    model = TimeConditionalSplineModel(space, configuration(), 4e-8, physical_guard=guard).eval()
    state = torch.zeros(5, 16, 7, dtype=torch.float64)
    sample = dict(state=state, parameters=torch.tensor([multiplier]), case_id='ood', start_frame=0,
                  time_seconds=torch.arange(5, dtype=torch.float64)*1e-8)

    class Windows:
        cases, starts = ['ood'], 1
        def __len__(self): return 1
        def __getitem__(self, index): return dict(sample)

    data = SimpleNamespace(windows=lambda *args, **kwargs: Windows())
    fixed = evaluator.FixedConditioning(data).windows('test', 4)[0]
    assert fixed['state'] is sample['state'] and fixed['time_seconds'] is sample['time_seconds']
    torch.testing.assert_close(fixed['parameters'], torch.ones(1))
    torch.testing.assert_close(fixed['withheld_coefficient_parameters'], sample['parameters'])
    assert sample['parameters'].item() == multiplier
    seen = []
    hook = model.register_forward_pre_hook(lambda module, args: seen.append((args[0].clone(), args[1].clone())))
    with torch.no_grad():
        for query in (sample, fixed):
            rows = evaluate_trajectory(model, query, space, guard, dense, [4], 1)
            assert len(rows) == 1 and rows[0]['audit_passed']
            assert np.isfinite(rows[0]['raw_normalized_volume_mse'])
    hook.remove()
    assert len(seen) == 8
    for index, (initial, parameters) in enumerate(seen):
        torch.testing.assert_close(initial, state[:1])
        assert parameters.item() == (multiplier if index < 4 else 1.)
