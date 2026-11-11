"""Learned tangent correction Delta L: zero start, native consistency, implicit (Picard) solve, gradients, devices."""
import numpy as np
import pytest
import torch

from mastu_opsplit.euler import NativeSplineSpace
from mastu_opsplit.gpu_native import configure_cuda_numerics
from mastu_opsplit.implicit_opsplit import (ACTIVE_CHANNELS, FrozenTangent, ImplicitNativeSystem,
                                            ImplicitOpsSplitModel, TangentCorrection, assemble_tangent, spline_basis)
from mastu_opsplit.implicit_training import build_model, check_implicit_config
from mastu_opsplit.physical_guard import PhysicalGuard
from test_implicit_opsplit import dirichlet, frozen_fixture
from test_learning import configuration
from test_physical_guard import fixture

configure_cuda_numerics()
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def setup(tmp_path, *, weights=None, picard=None, scale=None):
    torch.manual_seed(0)
    geometry, q8, mean = fixture(tmp_path)
    frozen = frozen_fixture(geometry)
    frozen.tangent[..., 3:5, 6] = 0   # some couplings absent, as in model303
    scale = np.linspace(.5, 2., 7) if scale is None else scale
    space = NativeSplineSpace(geometry)
    system = ImplicitNativeSystem(geometry, frozen, mean, scale, dirichlet(geometry), code_interval=.03125)
    correction = TangentCorrection(geometry, frozen, mean, scale, basis_shape=(4, 4), bound=.5)
    if weights is not None:
        with torch.no_grad():
            correction.weight.normal_(0, weights, generator=torch.Generator().manual_seed(1))
    guard = PhysicalGuard(geometry, q8, mean, scale)
    model = ImplicitOpsSplitModel(space, system, configuration(), np.ones(13), physical_guard=guard,
                                  correction=correction, picard=picard)
    return geometry, frozen, mean, scale, space, system, model


def test_spline_basis_is_a_partition_of_unity():
    points = np.random.default_rng(0).uniform([1, -2], [2, 2], size=(200, 2))
    basis = spline_basis(points, np.array([[1, 2], [-2, 2]]), (5, 7))
    assert basis.shape == (200, 35) and (basis >= 0).all()
    np.testing.assert_allclose(basis.sum(1), 1, rtol=0, atol=1e-12)


def test_zero_weights_reproduce_the_frozen_tangent_exactly(tmp_path):
    geometry, _, mean, scale, space, system, model = setup(tmp_path)
    plain = ImplicitOpsSplitModel(space, system, configuration(), np.ones(13), physical_guard=model.physical_guard)
    plain.operator.load_state_dict(model.operator.network.state_dict())
    state = torch.randn(1, len(geometry.dof_ids), 7, dtype=torch.float64)*.01
    assert not model.correction(state).abs().max() > 0
    with torch.no_grad():
        model.eval()
        torch.testing.assert_close(model(state, torch.ones(1, 1)), plain(state, torch.ones(1, 1)), rtol=0, atol=0)
    assert set(model.operator.state_dict()) - set(f'network.{k}' for k in plain.operator.state_dict()) == \
        {'correction.weight'}   # large buffers are not checkpointed


def test_correction_is_the_reweighted_native_tangent(tmp_path):
    geometry, frozen, mean, scale, _, _, model = setup(tmp_path, weights=.7)
    correction = model.correction
    m = correction.multipliers().detach().numpy()
    assert 0 < np.abs(m).max() < .5
    reweighted = np.zeros_like(frozen.tangent).reshape(-1, 21, 15)
    for p, (j, o) in enumerate(correction.pairs):
        reweighted[:, j, o] = frozen.tangent.reshape(-1, 21, 15)[:, j, o]*m[:, p]
    blocks = assemble_tangent(geometry, FrozenTangent(reweighted.reshape(frozen.tangent.shape), frozen.forcing,
                                                      frozen.values, frozen.gradients, 'x', 'x'))
    n = len(geometry.dof_ids)
    a, b = (torch.randn(1, n, 7, dtype=torch.float64) for _ in range(2))
    with torch.no_grad():
        difference = (correction(a) - correction(b))[0].numpy()
    physical = ((a-b)[0]*torch.as_tensor(scale)).numpy()
    expected = np.stack([sum(blocks[k, f] @ physical[:, f] for f in range(7) if (k, f) in blocks) for k in range(5)], 1)
    np.testing.assert_allclose(difference, expected, rtol=1e-10, atol=1e-12*np.abs(expected).max())
    assert {o for _, o in correction.pairs} <= set(ACTIVE_CHANNELS)
    assert (3, 6) not in correction.pairs and (4, 6) not in correction.pairs


def test_evaluation_solves_the_corrected_implicit_system(tmp_path):
    # Small multipliers: the random fixture tangent is non-normal with growing modes, so contraction needs |m| << 1.
    geometry, _, _, _, _, system, model = setup(tmp_path, weights=.05, picard=dict(max_iterations=40, tolerance=1e-10))
    model.eval()
    state = torch.randn(1, len(geometry.dof_ids), 7, dtype=torch.float64)*.01
    with torch.no_grad():
        forcing = model.forcing(state, torch.ones(1, 1))
        new = model.advance(system, state, forcing, 1.)
        total = forcing + model.correction(new)
    residual = system.residual(state[0].numpy(), new[0].numpy(), total[0].numpy())
    assert np.abs(residual).max() < 1e-9
    assert model.picard_statistics['max_iterations'] > 1
    explicit = system.step(state, forcing + model.correction(state))
    assert not torch.allclose(explicit, new)   # Delta L is treated implicitly, not at q_n


def test_unconverged_evaluation_fails_loudly(tmp_path):
    geometry, _, _, _, _, system, model = setup(tmp_path, weights=.7,
                                                picard=dict(train_iterations=1, max_iterations=1, tolerance=1e-14))
    model.eval()
    with torch.no_grad(), pytest.raises(FloatingPointError, match='Picard'):
        model.advance(system, torch.randn(1, len(geometry.dof_ids), 7, dtype=torch.float64)*.01, None, 1.)


def test_rollout_gradient_reaches_correction_and_network(tmp_path):
    geometry, _, _, _, space, _, model = setup(tmp_path, weights=.3)
    with torch.no_grad():
        model.operator.network.backbone.project.weight.normal_(0, .01)
    state = torch.randn(1, len(geometry.dof_ids), 7, dtype=torch.float64)*.01
    out = model(state, torch.ones(1, 1), substeps=2)
    out.square().sum().backward()
    assert model.correction.weight.grad.abs().sum() > 0
    assert model.operator.network.backbone.project.weight.grad.abs().sum() > 0
    assert model.picard_statistics['iterations'] == 2*model.picard['train_iterations']


def test_converged_training_matches_evaluation_and_reaches_gradients(tmp_path):
    # Jacobian-free training: no-grad Picard to train_tolerance, then train_iterations differentiable iterations.
    picard = dict(train_iterations=2, max_iterations=40, tolerance=1e-10, train_tolerance=1e-10)
    geometry, _, _, _, _, system, model = setup(tmp_path, weights=.05, picard=picard)
    with torch.no_grad():
        model.operator.network.backbone.project.weight.normal_(0, .01)
    state = torch.randn(1, len(geometry.dof_ids), 7, dtype=torch.float64)*.01
    forcing = model.forcing(state, torch.ones(1, 1))
    trained = model.advance(system, state, forcing, 1.)
    stats = dict(model.picard_statistics)
    assert stats['max_warm_iterations'] > 1 and stats['unconverged_warm_steps'] == 0
    assert stats['iterations'] == picard['train_iterations'] and stats['max_final_change'] < 1e-9
    trained.square().sum().backward()
    assert model.correction.weight.grad.abs().sum() > 0
    assert model.operator.network.backbone.project.weight.grad.abs().sum() > 0
    model.reset_picard_statistics()
    with torch.no_grad():
        evaluated = model.advance(system, state, forcing.detach(), 1.)
    torch.testing.assert_close(trained.detach(), evaluated, rtol=0, atol=1e-9*float(evaluated.abs().max()))
    truncated = setup(tmp_path, weights=.05, picard=dict(train_iterations=1, max_iterations=40, tolerance=1e-10))[-1]
    truncated.operator.load_state_dict(model.operator.state_dict())
    one = truncated.advance(system, state, truncated.forcing(state, torch.ones(1, 1)), 1.)
    assert (one.detach() - evaluated).abs().max() > 1e3*(trained.detach() - evaluated).abs().max()


def test_unconverged_warm_start_is_counted_not_fatal(tmp_path):
    picard = dict(train_iterations=1, max_iterations=1, tolerance=1e-14, train_tolerance=1e-14)
    geometry, _, _, _, _, system, model = setup(tmp_path, weights=.3, picard=picard)
    model.advance(system, torch.randn(1, len(geometry.dof_ids), 7, dtype=torch.float64)*.01, None, 1.)
    assert model.picard_statistics['unconverged_warm_steps'] == 1
    with pytest.raises(ValueError, match='Picard'):
        setup(tmp_path, picard=dict(train_tolerance=0.))


def test_config_and_builder(tmp_path):
    geometry, q8, mean = fixture(tmp_path)
    frozen = frozen_fixture(geometry)
    system = ImplicitNativeSystem(geometry, frozen, mean, np.ones(7), dirichlet(geometry), code_interval=.03125)
    system.frozen = frozen
    space = NativeSplineSpace(geometry)
    guard = PhysicalGuard(geometry, q8, mean, np.ones(7))
    config = dict(tangent_correction=dict(basis_shape=[4, 4], bound=.3, train_iterations=2, train_tolerance=1e-6))
    model = build_model(config, space, system, configuration(), np.ones(13), physical_guard=guard)
    assert model.correction.bound == .3 and model.picard['train_iterations'] == 2
    assert model.picard['train_tolerance'] == 1e-6
    assert build_model(config, space, system, configuration(), np.ones(13), physical_guard=guard,
                       physics_only=True).correction is None
    assert build_model({}, space, system, configuration(), np.ones(13), physical_guard=guard).correction is None
    base = dict(steps=1, pretrain_steps=0, horizon=1, stride=1, threads=1, batch_size=1, physical_guard=True,
                positivity_basis='bernstein_subdivided', learning_rate=1., pretrain_learning_rate=1., gamma=1., F0=1.,
                theta=1., evaluation_substeps=[1, 2, 4], split_counts={'train': 80, 'validation': 0, 'test': 20})
    check_implicit_config({**base, **config})
    with pytest.raises(ValueError, match='tangent_correction'):
        check_implicit_config({**base, 'tangent_correction': dict(rank=3)})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA device')
def test_correction_on_device_matches_cpu(tmp_path):
    geometry, _, _, _, _, _, model = setup(tmp_path, weights=.7)
    state = torch.randn(2, len(geometry.dof_ids), 7, dtype=torch.float64)*.01
    weight = torch.randn(2, len(geometry.dof_ids), 5, dtype=torch.float64)
    results = []
    for device in ('cpu', 'cuda'):
        correction = model.correction.to(device)
        correction.weight.grad = None
        out = correction(state.to(device))
        (out*weight.to(device)).sum().backward()
        results.append((out.detach().cpu().clone(), correction.weight.grad.cpu().clone()))   # .to() moves .grad in place
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12*float(expected.abs().max()))


def test_sharing_ties_value_and_gradient_terms_of_one_field(tmp_path):
    geometry, q8, mean = fixture(tmp_path)
    frozen = frozen_fixture(geometry)
    counts = {}
    for share in ('pair', 'block_field', 'block'):
        correction = TangentCorrection(geometry, frozen, mean, np.ones(7), basis_shape=(4, 4), share=share)
        with torch.no_grad():
            correction.weight.normal_(0, 1., generator=torch.Generator().manual_seed(2))
        m = correction.multipliers()
        for p, (j, o) in enumerate(correction.pairs):
            for q, (k, r) in enumerate(correction.pairs):
                same = dict(pair=(j, o) == (k, r), block_field=(o//3, j % 7) == (r//3, k % 7), block=o//3 == r//3)[share]
                if same:
                    assert torch.equal(m[:, p], m[:, q])
        counts[share] = len(correction.groups)
    assert counts == {'pair': 13*21, 'block_field': 5*7, 'block': 5}
    with pytest.raises(ValueError, match='share'):
        TangentCorrection(geometry, frozen, mean, np.ones(7), basis_shape=(4, 4), share='field')
