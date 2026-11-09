import numpy as np
import torch

from mastu_opsplit.euler import NativeSplineSpace
from mastu_opsplit.implicit_coefficient import (JET_COLUMNS, PSI_R, PSI_Z, RHO_FLUX_CHANNELS, RHO_R, RHO_Z,
                                                CoefficientImplicitModel, DensityDiffusionRemainder,
                                                coefficient_tangent, diffusion_direction, diffusion_jacobian)
from mastu_opsplit.implicit_opsplit import ImplicitNativeSystem, ImplicitOpsSplitModel
from mastu_opsplit.physical_guard import PhysicalGuard
from test_implicit_opsplit import dirichlet, frozen_fixture, jets
from test_learning import configuration
from test_physical_guard import fixture


def native_density_flux(psi_r, psi_z, rho_r, rho_z, R, F, D, Dpar):
    """Transcription of opsplit_physical: a(3:5) and c(2:3,3)."""
    b1, b2 = psi_z/R, -psi_r/R
    B2 = (F**2 + psi_r**2 + psi_z**2)/R**2
    a3, a4, a5 = D + (Dpar-D)*b1**2/B2, (Dpar-D)*b1*b2/B2, D + (Dpar-D)*b2**2/B2
    return -a3*rho_r - a4*rho_z, -a4*rho_r - a5*rho_z


def test_diffusion_direction_is_exact_D_derivative_of_native_form():
    rng = np.random.default_rng(0)
    psi_r, psi_z, rho_r, rho_z, R = rng.normal(size=(5, 1000))
    R = 1 + abs(R)
    for Dpar in (0., 3e-3):
        base = np.stack(native_density_flux(psi_r, psi_z, rho_r, rho_z, R, .64, 1e-5, Dpar))
        for D in (2e-5, 1e-4, 1e-3):
            change = np.stack(native_density_flux(psi_r, psi_z, rho_r, rho_z, R, .64, D, Dpar)) - base
            np.testing.assert_allclose(change, (D-1e-5)*np.stack(diffusion_direction(psi_r, psi_z, rho_r, rho_z, .64)),
                                       rtol=1e-12, atol=1e-17)


def test_diffusion_jacobian_matches_autograd():
    rng = np.random.default_rng(1)
    jet = rng.normal(size=(50, 21))
    x = torch.tensor(jet[:, list(JET_COLUMNS)], requires_grad=True)
    expected = np.stack([torch.autograd.grad(diffusion_direction(*x.unbind(-1), .64)[k].sum(), x)[0].numpy()
                         for k in range(2)], -1)
    np.testing.assert_allclose(diffusion_jacobian(jet, .64), expected, rtol=1e-13, atol=1e-14)


def test_coefficient_tangent_changes_only_density_flux_and_is_identity_at_one(tmp_path):
    geometry, _, _ = fixture(tmp_path)
    frozen = frozen_fixture(geometry)
    same = coefficient_tangent(frozen, 1.)
    np.testing.assert_array_equal(same.tangent, frozen.tangent)
    np.testing.assert_array_equal(same.forcing, frozen.forcing)
    moved = coefficient_tangent(frozen, 7.)
    other = [k for k in range(15) if k not in RHO_FLUX_CHANNELS]
    np.testing.assert_array_equal(moved.tangent[..., other], frozen.tangent[..., other])
    flat = lambda f: f.forcing.reshape(*f.forcing.shape[:2], 15)
    np.testing.assert_array_equal(flat(moved)[..., other], flat(frozen)[..., other])
    assert moved.sha256 != frozen.sha256 and np.abs(moved.tangent-frozen.tangent).max() > 0


def test_frozen_increment_plus_explicit_remainder_is_exact_native_increment(tmp_path):
    """dF* + dL*(jets - jets*) + explicit = dD g(q) 1(rho >= 0) pointwise, for any state."""
    geometry, _, _ = fixture(tmp_path)
    frozen = frozen_fixture(geometry)
    n = len(geometry.dof_ids)
    rng = np.random.default_rng(4)
    mean, scale = rng.normal(size=(n, 7)), rng.uniform(.5, 2, size=7)
    mean[:, 4] = .02
    y = rng.normal(size=(2, n, 7))*.01
    y[1, :5, 4] = -10   # negative density at some points: native override keeps D = D_prof_neg there
    multiplier = 6.5
    delta = (multiplier-1)*1e-5
    moved = coefficient_tangent(frozen, multiplier)
    remainder = DensityDiffusionRemainder(geometry, frozen, mean, scale)
    explicit = remainder.pointwise(torch.tensor(y), delta).numpy()
    for i in range(2):
        values, _, jet = jets(mean + scale*y[i], geometry)
        offset = jet - frozen.jets
        increment = (moved.forcing - frozen.forcing).reshape(*offset.shape[:2], 15) + \
            np.einsum('eqjo,eqj->eqo', moved.tangent - frozen.tangent, offset)
        g = np.stack(diffusion_direction(jet[..., PSI_R], jet[..., PSI_Z], jet[..., RHO_R], jet[..., RHO_Z], .64), -1)
        exact = delta*g*(values[..., 4] >= 0)[..., None]
        total = increment[..., list(RHO_FLUX_CHANNELS)].reshape(-1, 2) + explicit[i]
        np.testing.assert_allclose(total, exact.reshape(-1, 2), rtol=1e-10, atol=1e-13)
        assert (values[..., 4] < 0).any() == (i == 1)
    assert remainder(torch.tensor(y), 0.) is None


def test_coefficient_model_is_base_model_at_one_and_uses_exact_physics_otherwise(tmp_path):
    torch.manual_seed(0)
    geometry, q8, mean = fixture(tmp_path)
    space = NativeSplineSpace(geometry)
    frozen = frozen_fixture(geometry)
    n = len(geometry.dof_ids)
    scale = np.ones(7)
    guard = PhysicalGuard(geometry, q8, mean, scale)
    fixed = dirichlet(geometry)
    make = lambda f: ImplicitNativeSystem(geometry, f, mean, scale, fixed, code_interval=.03125)
    base = ImplicitOpsSplitModel(space, make(frozen), configuration(), np.ones(13), physical_guard=guard,
                                 zero_output=False)
    built = []
    factory = lambda m: built.append(m) or make(coefficient_tangent(frozen, m))
    correction = DensityDiffusionRemainder(geometry, frozen, mean, scale)
    model = CoefficientImplicitModel(base, factory, correction)
    state = torch.randn(1, n, 7, dtype=torch.float64)*1e-3
    with torch.no_grad():
        torch.testing.assert_close(model(state, torch.ones(1, 1), substeps=2), base(state, torch.ones(1, 1), substeps=2),
                                   rtol=0, atol=0)
        assert built == []
        out = model(state, torch.full((1, 1), 4.), substeps=1)
        # Network still conditioned on 1; physics at D = 4 D0 with the exact explicit remainder.
        system = model.system
        forcing = base.forcing(state, torch.ones(1, 1)) + correction(state, 3e-5)
        torch.testing.assert_close(out, guard(state, system.step(state, forcing, 1.)), rtol=0, atol=0)
        model(state, torch.full((1, 1), 4.))
        assert built == [4.]
        model(state, torch.full((1, 1), 2.))
        assert built == [4., 2.] and model.system is not system
        assert model(state, torch.ones(1, 1)) is not None and model.system is base.system
    try:
        model(state.repeat(2, 1, 1), torch.tensor([[2.], [3.]]))
    except ValueError:
        pass
    else:
        raise AssertionError('Mixed multipliers in one batch must be rejected')


def test_float32_parameters_resolve_to_exact_manifest_multipliers():
    exact = 2.5964926980013585
    model = CoefficientImplicitModel.__new__(CoefficientImplicitModel)
    torch.nn.Module.__init__(model)
    model.known = {float(np.float32(m)): float(m) for m in [exact, 10., 1.]}
    passed = float(torch.tensor([exact], dtype=torch.float32).double()[0])
    assert passed != exact and model.resolve(passed) == exact and model.resolve(1.) == 1.
    try:
        model.resolve(3.)
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown multipliers must be rejected')
