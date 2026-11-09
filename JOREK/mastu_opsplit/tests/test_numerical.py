"""Manufactured local FE checks, not evidence of JOREK trajectory agreement."""
import numpy as np
import pytest
from mastu_opsplit.numerical import (NativeGeometry, assemble_strong, assemble_flux,
                                    evaluate_native, residual_identity, require_native_backend)


def geometry_fixture():
    # Two linear elements share a DOF, exercising duplicate global assembly.
    return NativeGeometry(
        dof_map=np.array([[0, 1], [1, 2]]),
        basis=np.array([[[.75, .25], [.25, .75]], [[.75, .25], [.25, .75]]]),
        basis_grad=np.broadcast_to(np.array([[[-1., 0.], [1., 0.]]]), (2, 2, 2, 2)).copy(),
        quadrature_rz=np.array([[[1.25, 0.], [1.75, 0.]], [[2.25, 0.], [2.75, 0.]]]),
        poloidal_weights=np.full((2, 2), .5), bounds=np.array([[1., 3.], [-.5, .5]]),
        dof_ids=np.arange(3), element_ids=np.arange(2), patch_ids=np.array([0, 1]),
        boundary_labels=np.array([1, 0, 1]), source_ids=np.arange(4), query_ids=np.arange(4))


def test_cylindrical_integral_evaluation_and_shared_dofs():
    geometry = geometry_fixture()
    coefficients = np.array([1., 2., 3.])[:, None]
    values = evaluate_native(coefficients, geometry)
    np.testing.assert_allclose(values[..., 0], geometry.quadrature_rz[..., 0])
    assembled = assemble_strong(np.ones((2, 2, 1)), geometry)
    assert assembled.sum() == pytest.approx(8*np.pi)  # toroidal volume, R=1..3, unit Z
    assert assembled[1, 0] > assembled[0, 0]
    np.testing.assert_allclose(assemble_flux(np.ones((2, 2, 1, 2)), geometry).sum(), 0., atol=1e-14)


def test_derivative_dofs_not_assumed_to_partition_unity():
    geometry = geometry_fixture()
    geometry.basis[:, :, 1] *= -1  # generic signed derivative basis values are legal
    assert geometry.validate() is geometry
    assert np.isfinite(assemble_strong(np.ones((2, 2, 1)), geometry)).all()


def test_residual_cancellation_and_mismatch():
    total = np.zeros(3)
    selected = np.stack((np.ones(3), -np.ones(3)), axis=-1)
    assert residual_identity(total, total, selected)['cancellation']
    with pytest.raises(ValueError, match='reconstruction'):
        residual_identity(np.ones(3), total, selected)
    with pytest.raises(ValueError, match='nonfinite'):
        residual_identity(np.array([np.nan]), np.zeros(1), np.zeros((1, 2)))


def test_geometry_rejects_invalid_indices_weights_and_backend():
    geometry = geometry_fixture()
    geometry.dof_map[0, 0] = -1
    with pytest.raises(ValueError, match='out of range'):
        geometry.validate()
    with pytest.raises(NotImplementedError, match='temporal/constraint'):
        require_native_backend()


def test_model303_density_pressure_cylindrical_compression():
    from mastu_opsplit.numerical import model303_advection_blocks
    state = np.ones((2, 7))
    state[:, 4] = 2
    state[:, 5] = 3
    gradients = np.zeros((2, 7, 2))
    gradients[:, 1, 1] = .25  # u_Z: cylindrical compression survives constant rho,T
    actual = model303_advection_blocks(state, gradients, np.array([1., 2.]), 5/3)
    np.testing.assert_allclose(actual, np.array([[1., 5.], [1., 5.]]))
    # Parallel compression survives axisymmetry when magnetic poloidal gradients exist.
    gradients[:, 0, 1] = 4
    gradients[:, 6, 0] = .5
    actual = model303_advection_blocks(state, gradients, np.array([1., 2.]), 5/3)
    np.testing.assert_allclose(actual, [[-3., -15.], [-1., -5.]])


def test_differentiable_assembly_and_state_dependent_solve():
    torch = pytest.importorskip('torch')
    from mastu_opsplit.numerical import torch_assemble_strong
    geometry = geometry_fixture()
    labels = torch.full((2, 2, 1), .2, dtype=torch.float64, requires_grad=True)
    np.testing.assert_allclose(torch_assemble_strong(labels, geometry).detach().numpy(),
                               assemble_strong(labels.detach().numpy(), geometry))
    def assembled_solve(values):
        residual = torch_assemble_strong(values, geometry)
        mass = torch.diag_embed(1 + residual[:, 0].square())
        return torch.linalg.solve(mass, residual)
    assert torch.autograd.gradcheck(assembled_solve, (labels,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_native_physical_derivatives():
    from mastu_opsplit.numerical import evaluate_native_grad
    geometry = geometry_fixture()
    coefficients = np.array([1., 2., 3.])[:, None]
    gradient = evaluate_native_grad(coefficients, geometry)
    np.testing.assert_allclose(gradient[..., 0, 0], 1.)
    np.testing.assert_allclose(gradient[..., 0, 1], 0.)
