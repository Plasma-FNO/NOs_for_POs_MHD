import pytest
import torch

from equations import JorekTerms


@pytest.fixture
def grid():
    r = torch.linspace(9.5, 10.5, 41, dtype=torch.float64)
    z = torch.linspace(-.5, .5, 41, dtype=torch.float64)
    zz, rr = torch.meshgrid(z, r, indexing='ij')
    return JorekTerms(r, z), rr, zz


def test_native_constraints_distinguish_laplacians(grid):
    op, r, z = grid
    result = op.constraints(r*r + z*z, r*r + z*z)
    torch.testing.assert_close(result['omega'], torch.full_like(result['omega'], 6), atol=1e-9, rtol=0)
    torch.testing.assert_close(result['j'], torch.full_like(result['j'], 2), atol=1e-9, rtol=0)


def test_compression_and_pressure_curvature(grid):
    op, r, z = grid
    # u=Z gives v_R=-R, div(v)=-2. Constant rho and p still change.
    one = torch.ones_like(r)
    result = op.fluid_advection(one, 3*one, z, gamma=5/3)
    torch.testing.assert_close(result['rho'], 2*op.interior(one))
    torch.testing.assert_close(result['p'], 10*op.interior(one))
    result = op.fluid_advection(one, z, 0*z, gamma=5/3)
    torch.testing.assert_close(result['pressure_force'], -2*op.interior(r))


def test_variable_diffusion_is_flux_divergence(grid):
    op, r, z = grid
    # D=R and rho=R: div(D grad rho)=2, whereas D*Laplacian(rho)=1.
    result = op.scalar_diffusion(r, r*r+z*z, r, 3)
    torch.testing.assert_close(result['rho'], torch.full_like(result['rho'], 2))
    torch.testing.assert_close(result['p'], torch.full_like(result['p'], 18), atol=1e-8, rtol=0)


def test_magnetic_signs_and_independent_terms(grid):
    op, r, z = grid
    # psi=R gives compatible j=-1/R and zero [psi,j].
    result = op.magnetic(z, r, -1/r, eta=.2)
    torch.testing.assert_close(result['induction'], op.interior(r))
    torch.testing.assert_close(result['resistivity'], op.interior(-.2/r))
    torch.testing.assert_close(result['lorentz_force'], torch.zeros_like(result['lorentz_force']), atol=1e-11, rtol=0)
    # Independent bracket sign check for a manufactured load.
    torch.testing.assert_close(op.interior(op.bracket(r, z)), torch.ones_like(result['lorentz_force']))


def test_zero_parallel_diffusion_does_not_mean_zero_correction(grid):
    op, r, z = grid
    # psi=R^2/2 gives vertical B. No toroidal B: perpendicular diffusion
    # must exactly cancel for scalar Z^2, which varies along the field.
    scalar = z*z
    base = op.scalar_diffusion(scalar, scalar, 1, 1)['rho']
    correction = op.anisotropic_correction(scalar, r*r/2, f0=0, parallel=0, perpendicular=1)
    torch.testing.assert_close(base+correction, torch.zeros_like(base), atol=1e-10, rtol=0)
    toroidal = op.anisotropic_correction(scalar, 0*r, f0=10, parallel=0, perpendicular=1)
    torch.testing.assert_close(toroidal, torch.zeros_like(toroidal))


def test_interior_has_no_wrapping_and_supports_autograd(grid):
    op, r, z = grid
    u = (r*r+z*z).clone().requires_grad_()
    result = op.fluid_advection(r, z, u, gamma=5/3)['rho']
    result.square().mean().backward()
    assert torch.isfinite(u.grad).all() and u.grad.abs().sum() > 0
    changed = u.detach().clone()
    changed[..., -1] += 1e6
    other = op.fluid_advection(r, z, changed, gamma=5/3)['rho']
    torch.testing.assert_close(other, result.detach())


def test_bad_grid_rejected():
    with pytest.raises(ValueError, match='increasing'):
        JorekTerms(torch.arange(5.).flip(0), torch.arange(5.))
