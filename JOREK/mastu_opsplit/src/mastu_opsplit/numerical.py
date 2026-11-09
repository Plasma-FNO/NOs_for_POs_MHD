"""Native element evaluation and weak assembly; no model-303 time solver is implied."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NativeGeometry:
    """Local basis arrays retain derivative DOFs and avoid dense global transfer matrices."""
    dof_map: np.ndarray  # [element, local_dof], zero-based global IDs
    basis: np.ndarray  # [element, quadrature, local_dof]
    basis_grad: np.ndarray  # [element, quadrature, local_dof, R/Z]
    quadrature_rz: np.ndarray  # [element, quadrature, R/Z], metres
    poloidal_weights: np.ndarray  # [element, quadrature], |Jac|*Gauss weights, m^2
    bounds: np.ndarray  # [R/Z, min/max], fixed mesh bounds in metres
    dof_ids: np.ndarray
    element_ids: np.ndarray
    patch_ids: np.ndarray
    boundary_labels: np.ndarray  # native DOF labels, including derivative DOFs
    source_ids: np.ndarray
    query_ids: np.ndarray

    def validate(self):
        e, q, local = self.basis.shape
        d = len(self.dof_ids)
        shapes = {
            'dof_map': (e, local), 'basis_grad': (e, q, local, 2),
            'quadrature_rz': (e, q, 2), 'poloidal_weights': (e, q),
            'bounds': (2, 2), 'element_ids': (e,), 'patch_ids': (e,),
            'boundary_labels': (d,), 'source_ids': (e*q,), 'query_ids': (e*q,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'invalid geometry {name}: expected finite {shape}')
        if not np.isfinite(self.basis).all() or not e or not q or not d:
            raise ValueError('empty or nonfinite basis')
        if not np.issubdtype(self.dof_map.dtype, np.integer):
            raise ValueError('dof_map must contain integer indices')
        if self.dof_map.min() < 0 or self.dof_map.max() >= d:
            raise ValueError('dof_map index out of range')
        for name in ('dof_ids', 'element_ids', 'source_ids', 'query_ids'):
            ids = np.asarray(getattr(self, name))
            if not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
                raise ValueError(f'{name} must contain unique integers')
        if (self.poloidal_weights <= 0).any() or (self.quadrature_rz[..., 0] <= 0).any():
            raise ValueError('axisymmetric quadrature needs positive radius and weights')
        if (self.bounds[:, 1] <= self.bounds[:, 0]).any():
            raise ValueError('coordinate bounds must have positive span')
        if ((self.quadrature_rz < self.bounds[:, 0]) |
                (self.quadrature_rz > self.bounds[:, 1])).any():
            raise ValueError('quadrature lies outside fixed mesh coordinate bounds')
        return self

    @property
    def volume_weights(self):
        return 2 * np.pi * self.quadrature_rz[..., 0] * self.poloidal_weights


def evaluate_native(coefficients, geometry):
    """Evaluate [..., global_dof, channel] to [..., element, quadrature, channel]."""
    geometry.validate()
    values = np.asarray(coefficients)
    if values.shape[-2] != len(geometry.dof_ids) or not np.isfinite(values).all():
        raise ValueError('invalid native coefficients')
    return np.einsum('eql,...elc->...eqc', geometry.basis, values[..., geometry.dof_map, :])


def assemble_strong(values, geometry):
    """Assemble integral(test * strong_term * dV); no implicit boundary-row edits."""
    geometry.validate()
    values = np.asarray(values)
    if values.shape[-3:-1] != geometry.basis.shape[:2] or not np.isfinite(values).all():
        raise ValueError('pointwise labels must have [..., element, quadrature, channel] shape')
    local = np.einsum('eql,...eqc,eq->...elc', geometry.basis, values, geometry.volume_weights)
    result = np.zeros((*local.shape[:-3], len(geometry.dof_ids), local.shape[-1]), dtype=local.dtype)
    for element, indices in enumerate(geometry.dof_map):
        for local_index, global_index in enumerate(indices):
            result[..., global_index, :] += local[..., element, local_index, :]
    return result


def assemble_flux(flux, geometry):
    """Assemble -integral(grad(test).flux*dV); boundary flux must be added explicitly."""
    geometry.validate()
    flux = np.asarray(flux)
    if flux.shape[-4:-2] != geometry.basis.shape[:2] or flux.shape[-1] != 2 or not np.isfinite(flux).all():
        raise ValueError('flux requires [..., element, quadrature, channel, R/Z]')
    local = -np.einsum('eqlr,...eqcr,eq->...elc', geometry.basis_grad, flux, geometry.volume_weights)
    result = np.zeros((*local.shape[:-3], len(geometry.dof_ids), local.shape[-1]), dtype=local.dtype)
    for element, indices in enumerate(geometry.dof_map):
        for local_index, global_index in enumerate(indices):
            result[..., global_index, :] += local[..., element, local_index, :]
    return result


def residual_identity(total, retained, selected, rtol=1e-8, atol=1e-12):
    """Check same-unit residuals and report cancellation separately from relative error."""
    total, retained, selected = map(np.asarray, (total, retained, selected))
    if selected.shape[:-1] != total.shape or retained.shape != total.shape:
        raise ValueError('selected residuals must have total.shape + [n_blocks]')
    if not all(np.isfinite(x).all() for x in (total, retained, selected)):
        raise ValueError('nonfinite spatial residual')
    error = np.linalg.norm((total - retained - selected.sum(axis=-1)).ravel())
    total_norm = np.linalg.norm(total.ravel())
    component_scale = np.linalg.norm(retained.ravel()) + sum(
        np.linalg.norm(selected[..., i].ravel()) for i in range(selected.shape[-1]))
    cancellation = bool(total_norm <= max(atol, rtol*component_scale))
    # Cancellation is assessed by an absolute scale tolerance, never a tiny total denominator.
    tolerance = atol + rtol*total_norm if not cancellation else atol + 64*np.finfo(float).eps*component_scale
    if error > tolerance:
        raise ValueError(f'residual reconstruction error {error:.3e} exceeds {tolerance:.3e}')
    return {'absolute_error': float(error), 'total_norm': float(total_norm),
            'component_scale': float(component_scale), 'cancellation': cancellation,
            'relative_error': None if cancellation else float(error/total_norm)}


def require_native_backend():
    raise NotImplementedError(
        'This legacy in-process API has no differentiable temporal/constraint solve. '
        'The full native subprocess rollout is implemented by scripts/run_full_split.py; '
        'its scientific validation and dataset gates remain mandatory before training.')


def model303_constraint_blocks(state, geometry):
    """Interior current/vorticity weak blocks from pinned model303 rows 3/4.

    Physical volume contributes a common 2*pi factor. Native boundary/axis rows
    and their elimination are absent; this diagnostic cannot approve Gate B.
    """
    values = evaluate_native(state, geometry)
    gradient = evaluate_native_grad(state, geometry)
    radius2 = geometry.quadrature_rz[..., 0]**2
    mass = assemble_strong(np.stack((values[..., 2]/radius2, values[..., 3]), axis=-1), geometry)
    flux = np.stack((gradient[..., 0, :]/radius2[..., None], gradient[..., 1, :]), axis=-2)
    gradient_block = -assemble_flux(flux, geometry)
    return mass, gradient_block


def model303_advection_blocks(state, state_grad, radius, gamma):
    """Candidate n=0 strong residual blocks from model303 lines 890–941, in code units.

    Energy output belongs to the pressure equation, not to dT/dt. The density
    diamagnetic pressure term and all numerical stabilization remain retained.
    This local algebra does not provide the missing native evaluator or mass solve.
    """
    state, state_grad, radius = map(np.asarray, (state, state_grad, radius))
    if state.shape[-1] != 7 or state_grad.shape != state.shape + (2,):
        raise ValueError('expected seven current fields and their physical R/Z gradients')
    if radius.shape != state.shape[:-1] or (radius <= 0).any() or not np.isfinite(gamma) or gamma <= 1:
        raise ValueError('invalid model303 radius or gamma')
    if not all(np.isfinite(x).all() for x in (state, state_grad, radius)):
        raise ValueError('nonfinite current state or gradient')
    rho, temperature, parallel = state[..., 4], state[..., 5], state[..., 6]
    def bracket(a, b):
        return state_grad[..., a, 0]*state_grad[..., b, 1] - state_grad[..., a, 1]*state_grad[..., b, 0]
    density = (radius*bracket(4, 1) + 2*rho*state_grad[..., 1, 1]
               - (parallel*bracket(4, 0) + rho*bracket(6, 0))/radius)
    pressure = (radius*(rho*bracket(5, 1) + temperature*bracket(4, 1))
                + 2*gamma*rho*temperature*state_grad[..., 1, 1]
                - parallel*(rho*bracket(5, 0)+temperature*bracket(4, 0))/radius
                - gamma*rho*temperature*bracket(6, 0)/radius)
    return np.stack((density, pressure), axis=-1)


def torch_assemble_strong(values, geometry):
    """Differentiable strong-label assembly using immutable native geometry."""
    import torch
    geometry.validate()
    if tuple(values.shape[-3:-1]) != geometry.basis.shape[:2]:
        raise ValueError('pointwise tensor shape does not match geometry')
    basis = torch.as_tensor(geometry.basis, dtype=values.dtype, device=values.device)
    weights = torch.as_tensor(geometry.volume_weights, dtype=values.dtype, device=values.device)
    local = torch.einsum('eql,...eqc,eq->...elc', basis, values, weights)
    indices = torch.as_tensor(geometry.dof_map.ravel(), dtype=torch.long, device=values.device)
    result = values.new_zeros((*local.shape[:-3], len(geometry.dof_ids), local.shape[-1]))
    return result.index_add(-2, indices, local.flatten(-3, -2))


def evaluate_native_grad(coefficients, geometry):
    """Evaluate physical derivatives to [..., element, quadrature, channel, R/Z]."""
    geometry.validate()
    coefficients = np.asarray(coefficients)
    if coefficients.shape[-2] != len(geometry.dof_ids) or not np.isfinite(coefficients).all():
        raise ValueError('invalid native coefficients')
    return np.einsum('eqlr,...elc->...eqcr', geometry.basis_grad,
                     coefficients[..., geometry.dof_map, :])
