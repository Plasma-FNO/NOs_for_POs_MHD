"""Pinned n=0 model303 temporal/constraint volume action, before boundary edits."""
import numpy as np

from .numerical import assemble_flux, assemble_strong, evaluate_native, evaluate_native_grad


def corrected_density(values, *, rho_1, rho_min_neg, coefficients):
    """Match corr_neg_dens1, including its positive-low-density correction."""
    values = np.asarray(values, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (2,) or not np.isfinite(coefficients).all() or (coefficients <= 0).any():
        raise ValueError('two positive finite density-correction coefficients required')
    if not np.isfinite([rho_1, rho_min_neg]).all() or rho_1 <= 0 or not np.isfinite(values).all():
        raise ValueError('invalid native density correction parameters')
    low, width = (rho_min_neg if rho_min_neg >= 0 else rho_1)*coefficients
    result = values.copy()
    active = values < low+width
    if width == 0:
        if active.any():
            raise ValueError('native zero-width density correction is singular')
    else:
        result[active] = low+width*np.exp((values[active]-low-width)/width)
    return result


def temporal_constraint_action(state, increment, geometry, *, F0, rho_1, rho_min_neg, density_coefficients):
    """Apply native M plus current/vorticity constraint Jacobians at zeta=0.

    Units use a common 2*pi factor relative to native n=0 volume assembly.
    This is a volume action, not a boundary-complete native time solver.
    """
    state, increment = np.asarray(state), np.asarray(increment)
    if state.shape != increment.shape or state.shape != (len(geometry.dof_ids), 7):
        raise ValueError('current and increment require canonical native seven-channel coefficients')
    if not np.isfinite(F0) or F0 == 0:
        raise ValueError('finite nonzero toroidal field coefficient required')
    values, delta = evaluate_native(state, geometry), evaluate_native(increment, geometry)
    gradient, delta_gradient = evaluate_native_grad(state, geometry), evaluate_native_grad(increment, geometry)
    rho = values[..., 4]
    corrected = corrected_density(rho, rho_1=rho_1, rho_min_neg=rho_min_neg, coefficients=density_coefficients)
    radius2 = geometry.quadrature_rz[..., 0]**2
    strong = np.zeros_like(delta)
    strong[..., 0] = delta[..., 0]/radius2
    strong[..., 2] = delta[..., 2]/radius2
    strong[..., 3] = delta[..., 3]
    strong[..., 4] = delta[..., 4]
    strong[..., 5] = values[..., 5]*delta[..., 4] + corrected*delta[..., 5]
    strong[..., 6] = (rho*values[..., 6]*np.sum(gradient[..., 0, :]*delta_gradient[..., 0, :], axis=-1)
                       + corrected*F0**2*delta[..., 6])/radius2
    flux = np.zeros((*delta.shape, 2))
    flux[..., 1, :] = radius2[..., None]*corrected[..., None]*delta_gradient[..., 1, :]
    flux[..., 2, :] = -delta_gradient[..., 0, :]/radius2[..., None]
    flux[..., 3, :] = -delta_gradient[..., 1, :]
    return assemble_strong(strong, geometry) + assemble_flux(flux, geometry)
