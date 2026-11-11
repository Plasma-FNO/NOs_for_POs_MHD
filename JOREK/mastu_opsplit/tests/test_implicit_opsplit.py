import h5py
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import torch

from mastu_opsplit.euler import NativeSplineSpace
from mastu_opsplit.implicit_opsplit import (ACTIVE_CHANNELS, BLOCK_ROWS, FrozenTangent, ImplicitNativeSystem,
                                            ImplicitOpsSplitModel, WeakAssembler, _Factor, assemble_mass,
                                            assemble_tangent, corr_neg_dens1, fixed_dofs_from_trajectories,
                                            load_frozen_tangent, remainder_loss)
from mastu_opsplit.physical_guard import PhysicalGuard
from test_learning import configuration
from test_physical_guard import fixture


def frozen_fixture(geometry, seed=3):
    rng = np.random.default_rng(seed)
    e, q = geometry.basis.shape[:2]
    tangent = rng.normal(size=(e, q, 21, 15))*.1
    tangent[..., [1, 2]] = 0
    forcing = rng.normal(size=(e, q, 5, 3))
    forcing[..., 0, 1:] = 0
    values = rng.normal(size=(e, q, 7))
    values[..., 4] = .005 + rng.uniform(size=(e, q))*.02  # includes the corrected low-density range
    values[..., 5] = 1 + rng.uniform(size=(e, q))
    gradients = rng.normal(size=(e, q, 7, 2))
    return FrozenTangent(tangent, forcing, values, gradients, 'test', 'memory')


def jets(state, geometry):
    local = state[geometry.dof_map]
    values = np.einsum('eql,elc->eqc', geometry.basis, local)
    gradients = np.einsum('eqlr,elc->eqcr', geometry.basis_grad, local)
    return values, gradients, np.concatenate((values, gradients[..., 0], gradients[..., 1]), axis=-1)


def dirichlet(geometry):
    fixed = np.zeros((len(geometry.dof_ids), 7), dtype=bool)
    boundary_values = (geometry.boundary_labels != 0) & np.isin(np.arange(len(geometry.dof_ids)), geometry.dof_map[:, ::4])
    fixed[boundary_values, :4] = True
    fixed[np.flatnonzero(boundary_values)[:1], 4:] = True
    return fixed


def test_corr_neg_dens1_matches_jorek_formula():
    rho = np.array([-1., 0., .005, .0099, .01, .5])
    expected = np.where(rho < .01, .005 + .005*np.exp((rho-.01)/.005), rho)
    np.testing.assert_allclose(corr_neg_dens1(rho), expected, rtol=0, atol=1e-15)


def test_sparse_tangent_and_mass_match_pointwise_native_forms(tmp_path):
    geometry, _, _ = fixture(tmp_path)
    frozen = frozen_fixture(geometry)
    rng = np.random.default_rng(5)
    state = rng.normal(size=(len(geometry.dof_ids), 7))
    _, gradients, jet = jets(state, geometry)
    pointwise = np.einsum('eqjo,eqj->eqo', frozen.tangent, jet)
    expected = WeakAssembler(geometry, channels=tuple(range(15))).numpy(pointwise.reshape(1, -1, 15))[0]
    blocks = assemble_tangent(geometry, frozen)
    for b in range(5):
        action = sum(blocks[b, f] @ state[:, f] for f in range(7) if (b, f) in blocks)
        np.testing.assert_allclose(action, expected[:, b], rtol=1e-11, atol=1e-11)
    # Native SBDF2 history mass action, with BigR*xjac -> volume weights.
    values = jets(state, geometry)[0]
    radius = geometry.quadrature_rz[..., 0]
    rho, T, vpar = frozen.values[..., 4], frozen.values[..., 5], frozen.values[..., 6]
    corrected = corr_neg_dens1(rho)
    F0 = .64
    action = np.zeros((*radius.shape, 5, 3))
    action[..., 0, 0] = values[..., 0]/radius**2
    action[..., 1, 1:] = -radius[..., None]**2*corrected[..., None]*gradients[..., 1, :]
    action[..., 2, 0] = values[..., 4]
    action[..., 3, 0] = T*values[..., 4] + corrected*values[..., 5]
    action[..., 4, 0] = (rho*vpar*np.einsum('eqr,eqr->eq', frozen.gradients[..., 0, :], gradients[..., 0, :])
                         + corrected*F0**2*values[..., 6])/radius**2
    expected = WeakAssembler(geometry, channels=tuple(range(15))).numpy(action.reshape(1, -1, 15))[0]
    mass = assemble_mass(geometry, frozen, F0=F0)
    for b in range(5):
        np.testing.assert_allclose(sum(mass[b, f] @ state[:, f] for f in range(7) if (b, f) in mass), expected[:, b],
                                   rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('theta', [1., .5])
def test_implicit_step_solves_requested_equations(tmp_path, theta):
    geometry, _, _ = fixture(tmp_path)
    frozen = frozen_fixture(geometry)
    n = len(geometry.dof_ids)
    rng = np.random.default_rng(11)
    mean, scale = rng.normal(size=(n, 7)), rng.uniform(.5, 2, size=7)
    fixed = dirichlet(geometry)
    system = ImplicitNativeSystem(geometry, frozen, mean, scale, fixed, code_interval=.03125, theta=theta)
    y = torch.tensor(rng.normal(size=(1, n, 7)))
    forcing = torch.tensor(rng.normal(size=(1, n, 5)))
    fraction = .5
    out = system.step(y, forcing, fraction)
    # Roundoff-level (backward-stable) residual; the fixture is ill-conditioned (|x| ~ 1e5 for O(1) data).
    factor = system.factor(fraction)
    residual = system.residual(y[0].numpy(), out[0].numpy(), forcing[0].numpy(), fraction)
    assert np.abs(residual).max() <= 1e-15*factor.norms['N']*np.abs(system.vector(out[0].numpy())).max()
    # Same equations in physical units, independently of normalization and row scaling.
    q, z = mean + scale*y[0].numpy(), mean + scale*out[0].numpy()
    h = .03125*fraction
    L, M = system.L, system.M
    pointwise_star = frozen.forcing.reshape(*frozen.forcing.shape[:2], 15) - np.einsum('eqjo,eqj->eqo', frozen.tangent, frozen.jets)
    reference = WeakAssembler(geometry, channels=tuple(range(15))).numpy(pointwise_star.reshape(1, -1, 15))[0]
    for b, row in enumerate(BLOCK_ROWS):
        lhs = sum(M[b, f] @ (z[:, f]-q[:, f]) for f in range(7) if (b, f) in M)
        mixed = theta*z + (1-theta)*q
        rhs = h*(reference[:, b] + sum(L[b, f] @ mixed[:, f] for f in range(7) if (b, f) in L) + forcing[0, :, b].numpy())
        free = ~fixed[:, row]
        np.testing.assert_allclose(lhs[free], rhs[free], rtol=1e-9, atol=1e-10)
    for name, potential, auxiliary in (('current', 0, 2), ('vorticity', 1, 3)):
        stiff, mass = system.constraints[name]
        free = ~fixed[:, auxiliary]
        np.testing.assert_allclose((stiff@z[:, potential] + mass@z[:, auxiliary])[free], 0, atol=1e-10)
    np.testing.assert_array_equal(out[0].numpy()[fixed], y[0].numpy()[fixed])
    solves = system.report()['factors'][str(fraction)]['solves']['forward']
    assert solves['solves'] == 1 and solves['worst_backward_error'] <= system.refinement_tolerance


def test_implicit_step_adjoint_is_exact_transpose_solve(tmp_path):
    # Random fixtures are ill-conditioned for finite differences; compare with the dense A^-1 B instead.
    geometry, _, _ = fixture(tmp_path)
    frozen = frozen_fixture(geometry)
    n = len(geometry.dof_ids)
    rng = np.random.default_rng(2)
    system = ImplicitNativeSystem(geometry, frozen, rng.normal(size=(n, 7)), rng.uniform(.5, 2, size=7),
                                  dirichlet(geometry), code_interval=.03125)
    factor = system.factor(.25)
    A, B = factor.A.toarray(), factor.B.toarray()
    E = np.zeros((7*n, 5*n))
    E[factor.forcing_rows, system.forcing_index] = factor.forcing_scale
    exact_state, exact_forcing = np.linalg.solve(A, B), np.linalg.solve(A, E)
    y = torch.tensor(rng.normal(size=(2, n, 7)), requires_grad=True)
    forcing = torch.tensor(rng.normal(size=(2, n, 5)), requires_grad=True)
    out = system.step(y, forcing, .25)
    weights = torch.tensor(rng.normal(size=(2, n, 7)))
    grad_y, grad_f = torch.autograd.grad((out*weights).sum(), (y, forcing))
    for i in range(2):
        w = weights[i].numpy().T.ravel()
        np.testing.assert_allclose(grad_y[i].numpy().T.ravel(), exact_state.T @ w, rtol=1e-9, atol=1e-9*np.abs(exact_state).max())
        np.testing.assert_allclose(grad_f[i].numpy().T.ravel(), exact_forcing.T @ w, rtol=1e-9, atol=1e-9*np.abs(exact_forcing).max())
        forward = np.linalg.solve(A, B @ y[i].detach().numpy().T.ravel() + factor.constant + E @ forcing[i].detach().numpy().T.ravel())
        np.testing.assert_allclose(out[i].detach().numpy().T.ravel(), forward, rtol=1e-9, atol=1e-9*np.abs(forward).max())


def small_factor(refinement=0, n=60, seed=5):
    rng = np.random.default_rng(seed)
    A = (sp.diags(rng.uniform(2, 3, n)) + .05*sp.random(n, n, density=.1, random_state=seed)).tocsc()
    factor = _Factor(A, sp.identity(n), np.zeros(n), np.zeros(0, dtype=int), np.zeros(0), 1., refinement)
    return factor, A.toarray(), rng.normal(size=n)


def reset(factor):
    for stats in factor.statistics.values():
        stats.update(solves=0, lu_solves=0, stalled=0, worst_backward_error=0.)


def test_refinement_skips_exact_solves_and_converges_inexact_ones():
    factor, A, b = small_factor()
    for transpose, matrix in ((False, A), (True, A.T)):
        np.testing.assert_allclose(factor.solve(b, transpose), np.linalg.solve(matrix, b), rtol=1e-12)
    assert factor.statistics['N']['lu_solves'] == 1 and factor.statistics['T']['lu_solves'] == 1
    assert max(s['worst_backward_error'] for s in factor.statistics.values()) <= factor.tolerance
    # Factors of a nearby matrix: each correction contracts by ~1e-6, so refinement must converge.
    factor.lu = splu(sp.csc_matrix(A + 1e-6*np.eye(len(A))))
    reset(factor)
    x = factor.solve(b)
    stats = factor.statistics['N']
    assert 2 <= stats['lu_solves'] <= 1 + factor.max_refinement and stats['stalled'] == 0
    assert stats['worst_backward_error'] <= factor.tolerance
    np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-12)


def test_refinement_stops_at_stagnation_and_keeps_the_better_iterate():
    factor, A, b = small_factor()
    factor.lu = splu(sp.csc_matrix(A + 20*np.eye(len(A))))  # corrections contract by ~0.9: never halve
    first = factor.lu.solve(b)
    x = factor.solve(b)
    stats = factor.statistics['N']
    assert stats['lu_solves'] == 2 and stats['stalled'] == 1
    berr = lambda y: float(factor.backward_error(b - factor.A @ y, y, b, 'N'))
    assert berr(x) < berr(first)


def test_minimum_refinement_is_always_applied():
    factor, A, b = small_factor(refinement=2)
    np.testing.assert_allclose(factor.solve(b, True), np.linalg.solve(A.T, b), rtol=1e-12)
    assert factor.statistics['T']['lu_solves'] == 3


def test_nested_dissection_and_colamd_orderings_agree(tmp_path):
    geometry, _, _ = fixture(tmp_path)
    frozen = frozen_fixture(geometry)
    n = len(geometry.dof_ids)
    rng = np.random.default_rng(8)
    mean, scale = rng.normal(size=(n, 7)), rng.uniform(.5, 2, size=7)
    y = torch.tensor(rng.normal(size=(1, n, 7)), requires_grad=True)
    forcing = torch.tensor(rng.normal(size=(1, n, 5)))
    outputs, gradients = [], []
    for ordering in ('metis', 'colamd'):
        system = ImplicitNativeSystem(geometry, frozen, mean, scale, dirichlet(geometry), code_interval=.03125,
                                      ordering=ordering)
        out = system.step(y, forcing, 1.)
        outputs.append(out.detach())
        gradients.append(torch.autograd.grad(out.square().sum(), y)[0])
        assert system.report()['ordering'] == ordering
    assert system.permutation is None
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-8, atol=1e-8)


def relation_fixture(geometry):
    from mastu_opsplit.implicit_opsplit import BoundaryRelations, dof_nodes
    node, kind = dof_nodes(geometry)
    fixed = dirichlet(geometry)
    candidates = [d for d in np.flatnonzero(geometry.boundary_labels != 0) if not fixed[d, 6] and kind[d] in (0, 1)][:4]
    return BoundaryRelations(np.array(candidates), (kind[candidates] != 0).astype(np.int64), node[candidates],
                             np.array([[.3, .7], [-.2, .5], [.1, .4], [.6, -.3]])[:len(candidates)]), fixed


def test_mach_relation_rows_are_enforced_and_differentiated(tmp_path):
    geometry, _, _ = fixture(tmp_path)
    frozen = frozen_fixture(geometry)
    n = len(geometry.dof_ids)
    rng = np.random.default_rng(12)
    mean, scale = rng.normal(size=(n, 7)), rng.uniform(.5, 2, size=7)
    mean[:, 5] = 1 + rng.uniform(size=n)
    relations, fixed = relation_fixture(geometry)
    assert len(relations.rows) >= 2 and set(relations.kind) == {0, 1}
    system = ImplicitNativeSystem(geometry, frozen, mean, scale, fixed, code_interval=.03125, relations=relations)
    y = torch.tensor(rng.normal(size=(1, n, 7))*.05, requires_grad=True)
    forcing = torch.tensor(rng.normal(size=(1, n, 5)))
    out = system.step(y, forcing, .5)
    q, z = mean + scale*y[0].detach().numpy(), mean + scale*out[0].detach().numpy()
    g_value, g_own = system.relation_gradient
    lhs = z[relations.rows, 6] - g_value*z[relations.value_dofs, 5] - g_own*z[relations.rows, 5]
    rhs = relations.evaluate(q[None, :, 5])[0] - g_value*q[relations.value_dofs, 5] - g_own*q[relations.rows, 5]
    np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(system.residual(y[0].detach().numpy(), out[0].detach().numpy(), forcing[0].numpy(), .5), 0, atol=1e-10)
    weights, direction = torch.tensor(rng.normal(size=(1, n, 7))), torch.tensor(rng.normal(size=(1, n, 7)))
    gradient, = torch.autograd.grad((out*weights).sum(), y)
    eps = 1e-6
    with torch.no_grad():
        plus = (system.step(y + eps*direction, forcing, .5)*weights).sum()
        minus = (system.step(y - eps*direction, forcing, .5)*weights).sum()
    assert float((gradient*direction).sum()) == pytest.approx(float((plus-minus)/(2*eps)), rel=1e-5)
    # Linearization is exact for the relation when T stays at the training mean.
    at_mean = system.step(torch.zeros(1, n, 7, dtype=torch.float64), None, .5)[0].numpy()
    expected = relations.evaluate(mean[None, :, 5])[0]
    zt = mean + scale*at_mean
    np.testing.assert_allclose(zt[relations.rows, 6] - g_value*(zt[relations.value_dofs, 5]-mean[relations.value_dofs, 5])
                               - g_own*(zt[relations.rows, 5]-mean[relations.rows, 5]), expected, rtol=1e-9, atol=1e-10)


def test_mach_fit_recovers_jorek_forms():
    from mastu_opsplit.implicit_opsplit import mach_fit
    rng = np.random.default_rng(0)
    T = 1 + rng.uniform(size=(30, 5))
    vpar = np.zeros_like(T)
    vpar[:, 2] = .4 + 1.5*np.sqrt(T[:, 0])
    vpar[:, 3] = -.7*T[:, 3]/np.sqrt(T[:, 0]) + .2*np.sqrt(T[:, 0])
    np.testing.assert_allclose(mach_fit(T, vpar, 2, 0, 0)[0], [.4, 1.5], atol=1e-12)
    np.testing.assert_allclose(mach_fit(T, vpar, 3, 0, 1)[0], [-.7, .2], atol=1e-12)


def test_weak_assembler_adjoint(tmp_path):
    geometry, _, _ = fixture(tmp_path)
    assembler = WeakAssembler(geometry)
    x = torch.randn(2, geometry.basis.shape[0]*geometry.basis.shape[1], len(ACTIVE_CHANNELS), dtype=torch.float64,
                    requires_grad=True)
    assert torch.autograd.gradcheck(assembler, (x,), eps=1e-6, atol=1e-8)


def test_zero_initialized_remainder_reproduces_physics_only_and_trains(tmp_path):
    torch.manual_seed(0)
    geometry, q8, mean = fixture(tmp_path)
    space = NativeSplineSpace(geometry)
    frozen = frozen_fixture(geometry)
    n = len(geometry.dof_ids)
    scale = np.ones(7)
    guard = PhysicalGuard(geometry, q8, mean, scale)
    system = ImplicitNativeSystem(geometry, frozen, mean, scale, dirichlet(geometry), code_interval=.03125)
    learned = ImplicitOpsSplitModel(space, system, configuration(), np.ones(13), physical_guard=guard)
    physics = ImplicitOpsSplitModel(space, system, configuration(), np.ones(13), physical_guard=guard, physics_only=True)
    state = torch.zeros(1, n, 7, dtype=torch.float64)
    parameters = torch.ones(1, 1)
    torch.testing.assert_close(learned(state, parameters, substeps=2), physics(state, parameters, substeps=2))
    target = torch.randn(1, n, 7, dtype=torch.float64)*.01
    (learned(state, parameters)-target).square().mean().backward()
    assert learned.operator.backbone.project.weight.grad.abs().sum() > 0
    assert len(system.report()['factors']) == 2


def test_remainder_loss_scaling():
    weights = torch.tensor([1., 3.], dtype=torch.float64)
    scale = torch.tensor([2., 1.], dtype=torch.float64)
    prediction = torch.tensor([[[2., 1.], [0., 0.]]], dtype=torch.float64)
    target = torch.zeros_like(prediction)
    assert float(remainder_loss(prediction, target, weights, scale)) == pytest.approx(((.25*1)+(.25*1))/2)


def test_fixed_dofs_from_trajectories_and_tangent_loader(tmp_path):
    geometry, _, _ = fixture(tmp_path)
    n = len(geometry.dof_ids)
    states = np.random.default_rng(1).normal(size=(5, n, 7))
    states[:, 0, :] = states[0, 0, :]
    states[1:, 3, 2] = states[0, 3, 2] + 1e-12
    states[1:, 5, 1] = states[0, 5, 1] + 1e-13  # interior DOF: never Dirichlet, however still
    with h5py.File(tmp_path/'a.h5', 'w') as stream:
        stream['native_state'] = states
    labels = np.ones(n, dtype=int)
    labels[5] = 0
    fixed, report = fixed_dofs_from_trajectories([tmp_path/'a.h5'], np.ones(7), labels)
    assert fixed[0].all() and fixed[3, 2] and not fixed[5].any() and fixed.sum() == 8
    assert report['largest_fixed_change'] == pytest.approx(1e-12, rel=1e-3, abs=0)
    states[1:, 7, 4] = states[0, 7, 4] + 1e-11  # boundary continuum without a clear gap is rejected
    states[1:, 9, 4] = states[0, 9, 4] + 2e-10
    with h5py.File(tmp_path/'b.h5', 'w') as stream:
        stream['native_state'] = states
    with pytest.raises(ValueError, match='Ambiguous'):
        fixed_dofs_from_trajectories([tmp_path/'b.h5'], np.ones(7), labels, tolerance=1e-10)
    frozen = frozen_fixture(geometry)
    with h5py.File(tmp_path/'terms.h5', 'w') as stream:
        stream['coordinates'] = geometry.quadrature_rz
        stream['fixed_coefficient_tangent'] = frozen.tangent
        stream['baseline_forcing'] = frozen.forcing
        stream['values'] = frozen.values
        stream['gradients'] = frozen.gradients
        stream.attrs['measure'] = '2*pi*R*dR*dZ'
    loaded = load_frozen_tangent(tmp_path/'terms.h5', geometry)
    np.testing.assert_array_equal(loaded.tangent, frozen.tangent)
    with h5py.File(tmp_path/'terms.h5', 'r+') as stream:
        stream['fixed_coefficient_tangent'][0, 0, 0, 1] = 1.
    with pytest.raises(ValueError, match='Induction flux'):
        load_frozen_tangent(tmp_path/'terms.h5', geometry)
