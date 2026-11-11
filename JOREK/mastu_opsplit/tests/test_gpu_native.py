"""Device (torch / cuDSS) path of the implicit OpsSplit against the NumPy/SuperLU reference.

Every test runs on the CPU (torch kernels with CPU tensors, SuperLU solves) and, when a CUDA device
is visible, again on CUDA with cuDSS and a bitwise-repeatability check.
"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from mastu_opsplit.ensemble_training import evaluate_trajectory, train_family
from mastu_opsplit.euler import NativeSplineSpace, _FixedNative
from mastu_opsplit.gpu_native import DeviceCSR, direct_solver, configure_cuda_numerics
from mastu_opsplit.implicit_opsplit import ImplicitNativeSystem, ImplicitOpsSplitModel, WeakAssembler
from mastu_opsplit.implicit_training import LabelStore, pretrain
from mastu_opsplit.physical_guard import PhysicalGuard, QuadratureSampler
from test_ensemble_training import Windows
from test_implicit_opsplit import dirichlet, frozen_fixture, relation_fixture
from test_implicit_training import label_store
from test_learning import configuration
from test_physical_guard import fixture

configure_cuda_numerics()
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA device')


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize('device', DEVICES)
def test_device_csr_matches_scipy_including_empty_and_long_rows(device):
    rng = np.random.default_rng(0)
    M = sp.random(300, 200, density=.03, random_state=1, format='lil')
    M[5, :] = 0
    M[7, :] = rng.normal(size=200)  # one long row in its own bucket
    M = M.tocsr()
    D, DT = DeviceCSR(M, device), DeviceCSR(M.T, device)
    for x in (rng.normal(size=200), rng.normal(size=(200, 3))):
        out = D @ torch.as_tensor(x, device=device)
        np.testing.assert_allclose(out.cpu().numpy(), M @ x, rtol=1e-13, atol=1e-13)
        assert torch.equal(out, D @ torch.as_tensor(x, device=device))
    y = rng.normal(size=300)
    np.testing.assert_allclose((DT @ torch.as_tensor(y, device=device)).cpu().numpy(), M.T @ y, rtol=1e-13, atol=1e-13)
    assert (D @ torch.ones(200, dtype=torch.float64, device=device))[5] == 0


@pytest.mark.parametrize('device', DEVICES)
def test_direct_solver_matches_dense(device):
    rng = np.random.default_rng(1)
    A = (sp.diags(rng.uniform(2, 3, 80)) + .1*sp.random(80, 80, density=.1, random_state=2)).tocsr()
    b = rng.normal(size=(80, 2))
    solver = direct_solver(A, device)
    x = solver.solve(torch.as_tensor(b, device=device))
    np.testing.assert_allclose(x.cpu().numpy(), np.linalg.solve(A.toarray(), b), rtol=1e-12)
    if not solver.supports_transpose:
        transposed = direct_solver(A.T, device)
        np.testing.assert_allclose(transposed.solve(torch.as_tensor(b[:, 0], device=device)).cpu().numpy(),
                                   np.linalg.solve(A.toarray().T, b[:, 0]), rtol=1e-12)


def maps(tmp_path):
    geometry, q8, mean = fixture(tmp_path)
    space = NativeSplineSpace(geometry)
    guard = PhysicalGuard(geometry, q8, mean, np.ones(7))
    n, points = len(geometry.dof_ids), space.basis.shape[0]
    return geometry, guard, [(space, 'sample', (2, n, 3), (2, points, 3)),
                             (space, 'project', (2, points, 3), (2, n, 3)),
                             (guard.sampler, 'sample', (2, n, 2), (2, guard.sampler.basis.shape[0], 2)),
                             (WeakAssembler(geometry), 'assemble', (2, points, 13), (2, n, 5)),
                             (WeakAssembler(geometry, channels=(7, 8)), 'assemble', (2, points, 2), (2, n, 5)),
                             (guard.current, 'closure', (2, n, 2), (2, n, 2)),
                             (guard.vorticity, 'closure', (2, n, 2), (2, n, 2))]


@pytest.mark.parametrize('device', DEVICES)
def test_fixed_map_kernels_match_numpy_forward_and_transpose(tmp_path, device):
    _, _, operators = maps(tmp_path)
    rng = np.random.default_rng(4)
    for owner, operation, shape_in, shape_out in operators:
        x, g = rng.normal(size=shape_in), rng.normal(size=shape_out)
        forward = owner.apply_torch(torch.as_tensor(x, device=device), operation)
        transpose = owner.apply_torch(torch.as_tensor(g, device=device), operation, transpose=True)
        np.testing.assert_allclose(forward.cpu().numpy(), owner.apply(x, operation), rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(transpose.cpu().numpy(), owner.apply(g, operation, transpose=True), rtol=1e-11, atol=1e-12)


@cuda_only
def test_fixed_native_dispatches_to_device_kernels_with_exact_adjoint(tmp_path):
    _, _, operators = maps(tmp_path)
    rng = np.random.default_rng(5)
    for owner, operation, shape_in, shape_out in operators:
        x = torch.tensor(rng.normal(size=shape_in), device='cuda', requires_grad=True)
        weights = torch.tensor(rng.normal(size=shape_out), device='cuda')
        out = _FixedNative.apply(x, owner, operation)
        assert out.is_cuda
        gradient, = torch.autograd.grad((out*weights).sum(), x)
        np.testing.assert_allclose(out.detach().cpu().numpy(), owner.apply(x.detach().cpu().numpy(), operation),
                                   rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(gradient.cpu().numpy(), owner.apply(weights.cpu().numpy(), operation, transpose=True),
                                   rtol=1e-11, atol=1e-12)
        again, = torch.autograd.grad((_FixedNative.apply(x, owner, operation)*weights).sum(), x)
        assert torch.equal(gradient, again)


def systems(tmp_path, device, relations=False):
    geometry, _, _ = fixture(tmp_path)
    frozen = frozen_fixture(geometry)
    n = len(geometry.dof_ids)
    rng = np.random.default_rng(12)
    mean, scale = rng.normal(size=(n, 7)), rng.uniform(.5, 2, size=7)
    mean[:, 5] = 1 + rng.uniform(size=n)
    if relations:
        relation, fixed = relation_fixture(geometry)
    else:
        relation, fixed = None, dirichlet(geometry)
    make = lambda **kw: ImplicitNativeSystem(geometry, frozen, mean, scale, fixed, code_interval=.03125,
                                             relations=relation, **kw)
    return make(), make(backend='torch', device=device), fixed, n, rng


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('relations', [False, True])
def test_torch_backend_step_and_adjoint_match_scipy(tmp_path, device, relations):
    reference, system, fixed, n, rng = systems(tmp_path, device, relations)
    y0, f0 = rng.normal(size=(2, n, 7))*.05, rng.normal(size=(2, n, 5))
    weights = rng.normal(size=(2, n, 7))
    results = []
    for chosen, where in ((reference, 'cpu'), (system, device), (system, device)):
        y = torch.tensor(y0, device=where, requires_grad=True)
        forcing = torch.tensor(f0, device=where, requires_grad=True)
        out = chosen.step(y, forcing, .5)
        grads = torch.autograd.grad((out*torch.as_tensor(weights, device=where)).sum(), (y, forcing))
        results.append([t.detach().cpu() for t in (out,) + grads])
    for expected, actual in zip(results[0], results[1]):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9*float(expected.abs().max()))
    for first, second in zip(results[1], results[2]):
        assert torch.equal(first, second)  # bitwise repeatable
    np.testing.assert_array_equal(results[1][0][0].numpy()[fixed], y0[0][fixed])
    residual = system.residual(y0[0], results[1][0][0].numpy(), f0[0], .5)
    factor = system.factor(.5)
    assert np.abs(residual).max() <= 1e-15*factor.norms['N']*np.abs(results[1][0][0].numpy()).max()
    report = system.report()
    assert report['backend'] == 'torch' and report['factors']['0.5']['solves']['transpose']['solves'] == 2
    # Releasing adjoint memory is safe: the next adjoint rebuilds what it needs and gives the same gradient.
    system.release_adjoints()
    y = torch.tensor(y0, device=device, requires_grad=True)
    forcing = torch.tensor(f0, device=device, requires_grad=True)
    grads = torch.autograd.grad((system.step(y, forcing, .5)*torch.as_tensor(weights, device=device)).sum(), (y, forcing))
    for first, again in zip(results[1][1:], grads):
        assert torch.equal(first, again.detach().cpu())


@pytest.mark.parametrize('device', DEVICES[1:])
def test_model_with_guard_on_device_matches_cpu(tmp_path, device):
    geometry, q8, mean = fixture(tmp_path)
    space = NativeSplineSpace(geometry)
    frozen = frozen_fixture(geometry)
    n = len(geometry.dof_ids)
    torch.manual_seed(0)
    models = []
    for backend, where in (('scipy', 'cpu'), ('torch', device)):
        guard = PhysicalGuard(geometry, q8, mean, np.ones(7))
        system = ImplicitNativeSystem(geometry, frozen, mean, np.ones(7), dirichlet(geometry), code_interval=.03125,
                                      backend=backend, device=where)
        model = ImplicitOpsSplitModel(space, system, configuration(), np.ones(13), physical_guard=guard, zero_output=False)
        models.append(model.to(where))
    models[1].load_state_dict(copy.deepcopy(models[0].state_dict()))
    state = torch.randn(1, n, 7, dtype=torch.float64)*.01
    target = torch.randn(1, n, 7, dtype=torch.float64)*.01
    # Float64 physics (guard + implicit step, two substeps): value and state-gradient agree to roundoff.
    physics = []
    for model in models:
        where = next(model.parameters()).device
        model.physics_only = True
        y = state.to(where).requires_grad_(True)
        out = model(y, torch.ones(1, 1), substeps=2)
        gradient, = torch.autograd.grad((out-target.to(where)).square().sum(), y)
        physics.append([out.detach().cpu(), gradient.cpu()])
        model.physics_only = False
    for actual, expected in zip(physics[1], physics[0]):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9*float(expected.abs().max()))
    # The float32 remainder network: forcing and parameter gradients agree to float32 precision. (Its
    # roundoff is amplified by the ill-conditioned fixture solve, so full learned steps are not compared.)
    learned = []
    for model in models:
        where = next(model.parameters()).device
        model.zero_grad(set_to_none=True)
        forcing = model.forcing(state.to(where), torch.ones(1, 1))
        forcing.square().sum().backward()
        learned.append([forcing.detach().cpu()] + [p.grad.detach().cpu() for p in model.parameters() if p.grad is not None])
    for actual, expected in zip(learned[1], learned[0]):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6*float(expected.abs().max()))


@pytest.mark.parametrize('device', DEVICES)
def test_pretraining_and_rollout_resume_exactly_on_device(tmp_path, device):
    torch.set_num_threads(1)
    geometry, q8, mean = fixture(tmp_path)
    space = NativeSplineSpace(geometry)
    frozen = frozen_fixture(geometry)
    n = len(geometry.dof_ids)
    states = mean[None] + .01*np.random.default_rng(3).normal(size=(4, n, 7))
    store = LabelStore(label_store(tmp_path, geometry, states), tangent_sha256='t', training_case_ids=['case'])
    guard = PhysicalGuard(geometry, q8, mean, np.ones(7))
    system = ImplicitNativeSystem(geometry, frozen, mean, np.ones(7), dirichlet(geometry), code_interval=.03125,
                                  backend='torch', device=device)
    data = SimpleNamespace(mean=mean, scale=np.ones(7), windows=lambda split, horizon, stride=1: Windows(mean, split, horizon))
    config = dict(seed=17, steps=2, pretrain_steps=3, horizon=2, stride=1, learning_rate=.001,
                  pretrain_learning_rate=.001, positivity_penalty=.001)

    def model():
        torch.manual_seed(17)
        return ImplicitOpsSplitModel(space, system, configuration(), store.scale, physical_guard=guard).to(device)
    whole, partial = model(), model()
    pretrain(whole, data, store, space, config, tmp_path/'whole', 'p')
    assert pretrain(partial, data, store, space, config, tmp_path/'partial', 'p', stop_after=1) is None
    resumed = model()
    pretrain(resumed, data, store, space, config, tmp_path/'partial', 'p', resume=True)
    for name, value in whole.operator.state_dict().items():
        torch.testing.assert_close(value, resumed.operator.state_dict()[name], rtol=0, atol=0)
    for name in ('whole', 'resumed'):
        (tmp_path/name/'rollout').mkdir(parents=True)
    train_family(whole, data, space, config, tmp_path/'whole'/'rollout', 'p')
    assert train_family(resumed, data, space, config, tmp_path/'resumed'/'rollout', 'p', stop_after=1) is None
    train_family(resumed, data, space, config, tmp_path/'resumed'/'rollout', 'p', resume=True)
    for name, value in whole.operator.state_dict().items():
        torch.testing.assert_close(value, resumed.operator.state_dict()[name], rtol=0, atol=0)
    history = json.loads((tmp_path/'whole'/'rollout'/'history.json').read_text())
    assert len(history) == 2 and all(np.isfinite(row['training_loss']) for row in history)
    records = evaluate_trajectory(whole, data.windows('test', 2)[0], space, guard, QuadratureSampler(q8), [1, 2], 2)
    assert all(record['audit_passed'] for record in records)
