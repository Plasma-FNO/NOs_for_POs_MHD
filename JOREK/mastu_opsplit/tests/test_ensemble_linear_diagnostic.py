import importlib.util
from pathlib import Path

import numpy as np
import torch

from mastu_opsplit.continuity import FrozenContinuity
from mastu_opsplit.euler import EulerSplineModel, NativeSplineSpace, field_loss
from mastu_opsplit.euler_training import roll
from mastu_opsplit.ensemble_training import neural_hash
from mastu_opsplit.physical_guard import PhysicalGuard
from test_learning import configuration
from test_physical_guard import fixture


def test_branch_decomposition_and_fixed_weight_ablation(tmp_path):
    spec = importlib.util.spec_from_file_location('linear_diagnostic', Path(__file__).parents[1]/'scripts/diagnose_ensemble_linear.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    geometry, q8, mean = fixture(tmp_path)
    space = NativeSplineSpace(geometry)
    guard = PhysicalGuard(geometry, q8, mean, np.ones(7))
    physical = FrozenContinuity(geometry, mean, np.ones(7), .03125)
    model = EulerSplineModel('opsplit', space, configuration(), physical_guard=guard,
                            physical_linear=physical, linear_branch='lspline_continuity').eval()
    sample = dict(state=torch.randn(3, 16, 7, dtype=torch.float64)*.001, parameters=torch.ones(1),
                  case_id='training', start_frame=0)
    record = module.measure_state(model, sample)
    assert np.all(np.array(record['power']['linear'])[[0, 1, 2, 3, 6]] == 0)
    summary = module.aggregate([record, record])
    assert np.isfinite(summary['all_field_linear_over_neural_rms'])
    initial_hash = neural_hash(model)
    result = module.ablate_rollout(model, sample)
    predicted = roll(model, sample['state'], sample['parameters'][None])
    np.testing.assert_allclose(np.mean(result['full_per_field_mse']),
                               float(field_loss(predicted[1:], sample['state'][1:], space)), rtol=1e-12)
    # Independent native-gradient trajectory: differentiate the original bicubic
    # coefficients after every accepted step, using the same neural weights.
    native = sample['state'][:1]
    native_error = np.zeros(7)
    with torch.no_grad():
        for target in sample['state'][1:]:
            values = space.sample(native)
            gradient = np.einsum('eqlr,belc->beqcr', geometry.basis_grad,
                                 native.numpy()[:, geometry.dof_map]).reshape(1, -1, 7, 2)
            neural, _ = module.increments(model, native, sample['parameters'][None])
            linear = space.project(physical(values, torch.from_numpy(gradient)))
            native = guard(native, native+neural+linear)
            native_error += module.power(space, native-target[None])
    np.testing.assert_allclose(result['native_per_field_mse'], native_error/2, rtol=1e-12)
    assert all(stats['steps'] == 2 for stats in result['guard_statistics'].values())
    assert neural_hash(model) == initial_hash
    physical.code_interval = 0.
    result = module.ablate_rollout(model, sample)
    assert not np.any(result['prediction_difference_per_field_mse'])
    assert not np.any(result['native_prediction_difference_per_field_mse'])
    np.testing.assert_array_equal(result['native_per_field_mse'], result['full_per_field_mse'])
