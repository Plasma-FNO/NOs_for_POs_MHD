"""Exact D_perp(1) dependence of the implicit OpsSplit operator (coefficient-OOD physics).

In this MAST-U case D_perp(2) = D_par = 0, so JOREK's get_dperp gives D_prof = D_perp(1) at
every quadrature point (models/diffusivities.f90), apart from the override D_prof = D_prof_neg
(1e-5) where rho < D_prof_neg_thresh (0). D_prof enters the closed-form native coefficients
(native/full_split_procedures.f90, opsplit_physical) only through the density diffusion tensor

    a(3:5) = D (I - b b^T/B2) + D_par b b^T/B2,     [c(2,3), c(3,3)] = -a . grad(rho),

with b = (psi_Z, -psi_R)/R and B2 = (F0^2 + |grad psi|^2)/R^2. The volume RHS is therefore
affine in D:  c(q; D) = c(q; D0) + (D - D0) g(q),  g = -(I - b b^T/B2) grad(rho),  exactly.

For a multiplier m (D = m D0) the implicit split F(q) = F(q*) + L*(q - q*) + N(q) changes by

    F(q*) += dD g(q*),   L* += dD dg/djets(q*),   N += dD [g(q) 1(rho >= 0) - g(q*) - dg*(jets - jets*)],

dD = (m - 1) D0. The first two enter the implicit matrix (one factorisation per coefficient);
the last is evaluated explicitly in closed form. N_theta keeps the conditioning value it was
trained with (1): the coefficient reaches the model only through this exact physics.
"""
import dataclasses
import gc
import hashlib

import numpy as np
import torch
from torch import nn

from .implicit_opsplit import WeakAssembler

RHO_FLUX_CHANNELS = (7, 8)          # c(2,3), c(3,3) in the [block][strong, test_R, test_Z] layout
PSI_R, PSI_Z, RHO_R, RHO_Z = 7, 14, 11, 18   # jets = [values(7), d_R(7), d_Z(7)]
JET_COLUMNS = (PSI_R, PSI_Z, RHO_R, RHO_Z)
BASELINE_D_PERP = 1e-5


def diffusion_direction(psi_r, psi_z, rho_r, rho_z, F0):
    """g = dc/dD for the density flux channels (test_R, test_Z). NumPy or Torch, real or complex."""
    s = F0**2 + psi_r**2 + psi_z**2
    cross = (psi_z*rho_r - psi_r*rho_z)/s
    return -rho_r + psi_z*cross, -rho_z - psi_r*cross


def diffusion_jacobian(jets, F0):
    """[..., 4, 2]: d g / d (psi_R, psi_Z, rho_R, rho_Z) by complex step (exact to roundoff)."""
    base = np.asarray(jets, dtype=np.float64)[..., list(JET_COLUMNS)]
    out = np.empty(base.shape[:-1] + (4, 2))
    for column in range(4):
        z = base.astype(np.complex128)
        z[..., column] += 1e-30j
        g_r, g_z = diffusion_direction(z[..., 0], z[..., 1], z[..., 2], z[..., 3], F0)
        out[..., column, 0], out[..., column, 1] = g_r.imag/1e-30, g_z.imag/1e-30
    return out


def coefficient_tangent(frozen, multiplier, *, F0=0.64, baseline=BASELINE_D_PERP):
    """Frozen F(q*) and L* at D_perp(1) = multiplier*baseline, from the baseline export (exact)."""
    multiplier = float(multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('D_perp multiplier must be positive and finite')
    if (frozen.values[..., 4] <= 0).any():
        raise ValueError('Frozen baseline density must be positive (D_prof_neg override inactive)')
    delta = (multiplier-1.)*baseline
    jets = frozen.jets
    g_r, g_z = diffusion_direction(jets[..., PSI_R], jets[..., PSI_Z], jets[..., RHO_R], jets[..., RHO_Z], F0)
    forcing = frozen.forcing.copy()
    forcing[..., 2, 1] += delta*g_r
    forcing[..., 2, 2] += delta*g_z
    tangent = frozen.tangent.copy()
    jacobian = diffusion_jacobian(jets, F0)
    for i, column in enumerate(JET_COLUMNS):
        tangent[..., column, RHO_FLUX_CHANNELS[0]] += delta*jacobian[..., i, 0]
        tangent[..., column, RHO_FLUX_CHANNELS[1]] += delta*jacobian[..., i, 1]
    digest = hashlib.sha256(f'{frozen.sha256}|D_perp(1)={multiplier!r}*{baseline!r}|F0={F0!r}'.encode()).hexdigest()
    return dataclasses.replace(frozen, tangent=tangent, forcing=forcing, sha256=digest,
                               path=f'{frozen.path}#D_perp(1)x{multiplier!r}')


class DensityDiffusionRemainder(nn.Module):
    """Explicit dD [g(q) 1(rho >= 0) - g(q*) - dg*(jets - jets*)], assembled weakly ([batch, ndof, 5])."""
    def __init__(self, geometry, frozen, mean, scale, *, F0=0.64, negative_threshold=0.):
        super().__init__()
        e, q, _ = geometry.basis.shape
        jets = frozen.jets.reshape(e*q, 21)
        g_r, g_z = diffusion_direction(jets[:, PSI_R], jets[:, PSI_Z], jets[:, RHO_R], jets[:, RHO_Z], F0)
        as_tensor = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)
        self.register_buffer('basis', as_tensor(geometry.basis))
        self.register_buffer('basis_grad', as_tensor(geometry.basis_grad))
        self.register_buffer('dof_map', torch.as_tensor(geometry.dof_map, dtype=torch.long))
        self.register_buffer('mean', as_tensor(np.asarray(mean)[:, [0, 4]]))
        self.register_buffer('scale', as_tensor(np.asarray(scale)[[0, 4]]))
        self.register_buffer('jets_star', as_tensor(jets[:, list(JET_COLUMNS)]))
        self.register_buffer('g_star', as_tensor(np.stack((g_r, g_z), -1)))
        self.register_buffer('jacobian_star', as_tensor(diffusion_jacobian(jets, F0)))
        self.F0, self.negative_threshold = float(F0), float(negative_threshold)
        self.assembler = WeakAssembler(geometry, channels=RHO_FLUX_CHANNELS)

    def pointwise(self, state, delta):
        """[batch, E*Q, 2] explicit remainder increment in native code units."""
        physical = state[..., [0, 4]]*self.scale.to(state) + self.mean.to(state)      # psi, rho coefficients
        local = physical[:, self.dof_map]                                              # [B,E,16,2]
        rho = torch.einsum('eql,bel->beq', self.basis.to(state), local[..., 1]).flatten(1)
        gradient = torch.einsum('eqlr,belc->beqcr', self.basis_grad.to(state), local).flatten(1, 2)
        jets = torch.stack((gradient[..., 0, 0], gradient[..., 0, 1], gradient[..., 1, 0], gradient[..., 1, 1]), -1)
        g = torch.stack(diffusion_direction(*jets.unbind(-1), self.F0), -1)
        g = g*(rho >= self.negative_threshold).to(g)[..., None]                       # D_prof_neg override
        linear = self.g_star.to(state) + torch.einsum('pjk,bpj->bpk', self.jacobian_star.to(state),
                                                      jets - self.jets_star.to(state))
        return delta*(g - linear)

    def forward(self, state, delta):
        if delta == 0:
            return None
        return self.assembler(self.pointwise(state, delta))


class CoefficientImplicitModel(nn.Module):
    """Implicit OpsSplit whose physics uses the true D_perp(1); the network always receives 1.

    `parameters` carry the D_perp(1) multiplier (the OOD windows supply it). The implicit system for
    that coefficient is built by `system_factory(multiplier)` (one LU per step size) and replaced when
    the coefficient changes. With multiplier 1 this is exactly the trained base model.
    """
    family = 'opsplit'
    variant = 'implicit_native_tangent_coefficient_physics'

    def __init__(self, base, system_factory, correction, *, baseline=BASELINE_D_PERP, known_multipliers=None):
        super().__init__()
        self.base, self.correction, self.baseline = base, correction, float(baseline)
        self.system_factory = system_factory
        self.multiplier, self.system = 1., base.system
        self.systems = []
        # Loaders pass float32 parameters; map them back to the exact manifest coefficients.
        self.known = None if known_multipliers is None else {float(np.float32(m)): float(m) for m in
                                                             list(known_multipliers) + [1.]}
        if self.known is not None and len(self.known) != len(set(known_multipliers) | {1.}):
            raise ValueError('Known multipliers collide in float32')

    def resolve(self, value):
        if self.known is None:
            return value
        if value not in self.known:
            raise ValueError(f'Unknown D_perp(1) multiplier {value!r}')
        return self.known[value]

    def system_for(self, multiplier):
        if multiplier == 1.:
            self.multiplier, self.system = 1., self.base.system
        elif multiplier != self.multiplier:
            self.system = None   # release the previous coefficient's factors before building new ones
            gc.collect()
            self.system = self.system_factory(multiplier)
            self.multiplier = multiplier
            self.systems.append(multiplier)
        return self.system

    def forward(self, state, parameters, *, substeps=1, interval=1.):
        if type(substeps) is not int or substeps < 1 or not 0 < interval <= 1:
            raise ValueError('Require positive substeps and 0 < frame interval <= 1')
        values = parameters.detach().double().flatten()
        if parameters.shape != (state.shape[0], 1) or not torch.all(torch.isfinite(values) & (values > 0)) or \
                not torch.all(values == values[0]):
            raise ValueError('Expected one positive D_perp(1) multiplier per batch, shape [B,1]')
        multiplier = self.resolve(float(values[0]))
        system = self.system_for(multiplier)
        delta = (multiplier-1.)*self.baseline
        conditioning = torch.ones_like(parameters)
        for _ in range(substeps):
            forcing = self.base.forcing(state, conditioning)
            extra = self.correction(state, delta)
            if extra is not None:
                forcing = extra if forcing is None else forcing + extra
            proposal = self.base.advance(system, state, forcing, interval/substeps)   # Delta L kept as trained
            guard = self.base.physical_guard
            state = guard(state, proposal) if guard is not None else proposal
        return state
