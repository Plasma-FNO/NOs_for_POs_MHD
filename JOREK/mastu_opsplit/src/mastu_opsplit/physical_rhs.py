"""Differentiable native model303 volume coefficients and exact split remainder.

The formulas follow ``native/full_split_procedures.f90:opsplit_physical``.
They exclude temporal and boundary operators: those remain the responsibility
of the coupled native step. Transport/source profiles must match the native
case, including their state dependence; a frozen tangent does not supply that
dependence on its own.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .implicit_opsplit import ACTIVE_CHANNELS, WeakAssembler


NATIVE_PROFILE_KEYS = frozenset('''f0 gamma eta visco eta_ohmic d_perp zk_perp d_par zk_par zk_par_max
    visco_par eta_t_dependent visco_t_dependent zkpar_t_dependent t_0 t_1 t_min_neg corr_neg_temp_coef
    t_max_eta t_max_eta_ohm t_min t_min_zkpar rho_0 rho_1 rho_coef t_coef ff_0 ff_1 ff_coef
    particlesource particlesource_psin particlesource_sig heatsource heatsource_psin heatsource_sig
    normalized_velocity_profile'''.split())


def _pedestal(psi_n, delta, z, zx, high, low, coefficients, z_width):
    """Native analytic density/temperature profile and its declared d/dpsi.

    Native profile derivatives are evaluated at clamped psi_n even outside
    [0,2]; reproducing that convention matters to the imposed current source.
    """
    c = coefficients
    polynomial = (high-low)*(1+c[0]*psi_n+c[1]*psi_n**2+c[2]*psi_n**3)
    derivative = (high-low)*(c[0]+2*c[1]*psi_n+3*c[2]*psi_n**2)/delta
    argument = ((psi_n-c[4])/c[3]).clamp(-40, 40)
    edge = .5-.5*torch.tanh(argument)
    edge_derivative = -.5/torch.cosh(argument)**2/c[3]/delta
    lower = .5-.5*torch.tanh(((zx[..., 0:1]-z)/z_width).clamp(-40, 40))
    upper = .5-.5*torch.tanh(((z-zx[..., 1:2])/z_width).clamp(-40, 40))
    return polynomial*edge*lower*upper+low, (derivative*edge+polynomial*edge_derivative)*lower*upper


class NativeProfiles(nn.Module):
    """Pinned analytic, double-null model303 profiles, evaluated on the state.

    Axis/X-point diagnostics come from the predicted native coefficients.
    Supported case: n=0, one temperature, analytic density/T/FF profiles,
    constant perpendicular diffusivities, zero velocity/rotation source and
    zero edge/Gaussian sources. The manifest factory verifies those branches.
    """
    def __init__(self, geometry, physics, critical):
        super().__init__()
        if set(physics) != NATIVE_PROFILE_KEYS:
            raise ValueError(f'Native profile keys mismatch; missing={sorted(NATIVE_PROFILE_KEYS-set(physics))}, '
                             f'unknown={sorted(set(physics)-NATIVE_PROFILE_KEYS)}')
        self.physics = dict(physics)
        self.critical = critical
        self.register_buffer('coordinates', torch.tensor(geometry.quadrature_rz.reshape(-1, 2), dtype=torch.float64),
                             persistent=False)
        for key, size in (('rho_coef', 5), ('t_coef', 5), ('ff_coef', 9), ('corr_neg_temp_coef', 2)):
            if len(physics[key]) != size or not np.isfinite(physics[key]).all():
                raise ValueError(f'Native profile {key} must have {size} finite entries')
        for key in NATIVE_PROFILE_KEYS - {'rho_coef', 't_coef', 'ff_coef', 'corr_neg_temp_coef'}:
            if not np.isfinite(physics[key]):
                raise ValueError(f'Nonfinite native profile {key}')
        positive = ('f0', 'gamma', 't_0', 't_1', 't_max_eta', 't_max_eta_ohm', 't_min',
                    'zk_par_max', 'particlesource_sig', 'heatsource_sig')
        if any(physics[key] <= 0 for key in positive) or any(x <= 0 for x in physics['corr_neg_temp_coef']):
            raise ValueError('Native temperature/transport scales and source widths must be positive')
        if physics['t_min_zkpar'] < 0:
            raise ValueError('Resolve native post-input T_min_ZKpar to its nonnegative runtime value')
        if physics['rho_coef'][3] <= 0 or physics['t_coef'][3] <= 0 or physics['ff_coef'][3] <= 0 or physics['ff_coef'][7] <= 0:
            raise ValueError('Native analytic pedestal widths must be positive')

    def from_diagnostics(self, jets, diagnostics):
        """Local formulas with supplied critical values; useful for native parity tests."""
        p, z = self.physics, self.coordinates[..., 1]
        axis = diagnostics.get('source_psi_axis', diagnostics['psi_axis'])[..., None]
        boundary = diagnostics['psi_bnd'][..., None]
        zx = diagnostics['Z_xpoint']
        delta = boundary-axis
        if not torch.isfinite(delta).all() or (delta <= 0).any():
            raise FloatingPointError('Native profiles require finite psi_bnd > psi_axis')
        psi_n = (jets[..., 0]-axis)/delta
        clipped = psi_n.clamp(0, 2)
        density, density_psi = _pedestal(clipped, delta, z, zx, p['rho_0'], p['rho_1'], p['rho_coef'], .1)
        temperature, temperature_psi = _pedestal(clipped, delta, z, zx, p['t_0'], p['t_1'], p['t_coef'], .05)
        ff = p['ff_coef']
        perturbation = ff[5]/torch.cosh((clipped-ff[6])/ff[7])**2/(2*ff[7])/delta
        if ff[8] == 1:
            perturbation = perturbation*delta
        polynomial = (p['ff_0']-p['ff_1'])*(1+ff[0]*clipped+ff[1]*clipped**2+ff[2]*clipped**3)+perturbation
        edge = .5-.5*torch.tanh(((clipped-ff[4])/ff[3]).clamp(-40, 40))
        lower = .5-.5*torch.tanh(((zx[..., 0:1]-z)/.1).clamp(-40, 40))
        upper = .5-.5*torch.tanh(((z-zx[..., 1:2])/.1).clamp(-40, 40))
        ffprime = polynomial*edge*lower*upper+p['ff_1']
        current = ffprime-self.coordinates[..., 0]**2*(density*temperature_psi+density_psi*temperature)
        source_psi = torch.where((z < zx[..., 0:1]) & (psi_n < 1), 2-psi_n, psi_n)
        particle = p['particlesource']*(.5-.5*torch.tanh((source_psi-p['particlesource_psin'])/p['particlesource_sig']))
        heat = p['heatsource']*(.5-.5*torch.tanh((source_psi-p['heatsource_psin'])/p['heatsource_sig']))
        raw_T = jets[..., 5]
        correction_scale = p['t_min_neg'] if p['t_min_neg'] >= 0 else p['t_1']
        low, width = (correction_scale*x for x in p['corr_neg_temp_coef'])
        if width <= 0:
            raise ValueError('Native temperature correction width must be positive')
        corrected = torch.where(raw_T < low+width, low+width*torch.exp(((raw_T-low-width)/width).clamp_max(0)), raw_T)
        eta = p['eta']+torch.zeros_like(raw_T)
        etah = p['eta_ohmic']+torch.zeros_like(raw_T)
        if p['eta_t_dependent']:
            eta = p['eta']*(corrected.clamp_max(p['t_max_eta'])/p['t_0'])**-1.5
            eta = torch.where(raw_T < p['t_min'], p['eta']*(p['t_min']/p['t_0'])**-1.5, eta)
            etah = p['eta_ohmic']*(corrected.clamp_max(p['t_max_eta_ohm'])/p['t_0'])**-1.5
        mu = p['visco']+torch.zeros_like(raw_T)
        if p['visco_t_dependent']:
            mu = p['visco']*(corrected/p['t_0'])**-1.5
            mu = torch.where(raw_T < p['t_min'], p['visco']*(p['t_min']/p['t_0'])**-1.5, mu)
        kpar = p['zk_par']+torch.zeros_like(raw_T)
        if p['zkpar_t_dependent']:
            kpar = (p['zk_par']*(corrected/p['t_0'])**2.5).clamp_max(p['zk_par_max'])
            kpar = torch.where(raw_T < p['t_min_zkpar'], p['zk_par']*(p['t_min_zkpar']/p['t_0'])**2.5, kpar)
        zero = torch.zeros_like(jets[..., :2])
        return dict(F0=p['f0'], gamma=p['gamma'], eta=eta, mu=mu, etah=etah, D=p['d_perp'],
                    K=p['zk_perp'], Dpar=p['d_par'], Kpar=kpar, jsource=current, Sp=particle, Sh=heat,
                    muv=p['visco_par'], normalized=p['normalized_velocity_profile'], vsource=zero, omegasource=zero)

    def forward(self, jets, *, physical_state):
        return self.from_diagnostics(jets, self.critical(physical_state))


def physical_coefficients(jets, radius, profiles):
    """Native [strong,test_R,test_Z] coefficients, flattened to [...,15].

    ``jets`` has [...,21] entries [q,d_R q,d_Z q], q=(psi,u,zj,w,rho,T,Vpar).
    Profile tensors broadcast over the leading dimensions. ``normalized`` is
    the native normalized_velocity_profile flag, not field normalization.
    This low-level function also accepts complex jets for derivative checks.
    """
    if jets.shape[-1] != 21:
        raise ValueError('Native RHS requires 21 value/gradient channels')
    q, qr, qz = jets[..., :7], jets[..., 7:14], jets[..., 14:]
    R, p = radius, profiles
    F, gamma = p['F0'], p['gamma']
    eta, mu, etah = p['eta'], p['mu'], p['etah']
    D, K, Dpar, Kpar = p['D'], p['K'], p['Dpar'], p['Kpar']
    Sp, Sh, muv = p['Sp'], p['Sh'], p['muv']
    br, bz = qz[..., 0]/R, -qr[..., 0]/R
    B2 = (F**2 + qr[..., 0]**2 + qz[..., 0]**2)/R**2
    gp_r = q[..., 5]*qr[..., 4] + q[..., 4]*qr[..., 5]
    gp_z = q[..., 5]*qz[..., 4] + q[..., 4]*qz[..., 5]
    gu2 = qr[..., 1]**2 + qz[..., 1]**2
    rhou = qr[..., 4]*qz[..., 1] - qz[..., 4]*qr[..., 1]
    Tu = qr[..., 5]*qz[..., 1] - qz[..., 5]*qr[..., 1]
    rhopsi = qr[..., 4]*qz[..., 0] - qz[..., 4]*qr[..., 0]
    Tpsi = qr[..., 5]*qz[..., 0] - qz[..., 5]*qr[..., 0]
    Vpsi = qr[..., 6]*qz[..., 0] - qz[..., 6]*qr[..., 0]
    ppsi = gp_r*qz[..., 0] - gp_z*qr[..., 0]
    pj = qr[..., 0]*qz[..., 2] - qz[..., 0]*qr[..., 2]
    drr, drz, dzz = D+(Dpar-D)*br**2/B2, (Dpar-D)*br*bz/B2, D+(Dpar-D)*bz**2/B2
    krr, krz, kzz = K+(Kpar-K)*br**2/B2, (Kpar-K)*br*bz/B2, K+(Kpar-K)*bz**2/B2
    if p['normalized']:
        a9, a10 = muv, 0.
        vr_source, vz_source = muv*p['vsource'][..., 0], muv*p['vsource'][..., 1]
    else:
        a9, a10 = muv*F**2/R**2, 2*muv*F**2/R**3
        vr_source = muv*2*torch.pi*F*p['omegasource'][..., 0]
        vz_source = muv*2*torch.pi*F*p['omegasource'][..., 1]
    zero = torch.zeros_like(q[..., 0])
    return torch.stack((
        eta*(q[..., 2]-p['jsource'])/R**2 + (qr[..., 0]*qz[..., 1]-qz[..., 0]*qr[..., 1])/R,
        zero, zero,
        pj/R,
        -mu*qr[..., 3] - .5*R**3*gu2*qz[..., 4] - R**3*q[..., 4]*q[..., 3]*qz[..., 1]
        + R*gp_z + R**2*Sp*qr[..., 1],
        -mu*qz[..., 3] + .5*R*gu2*(2*R*q[..., 4]+R**2*qr[..., 4])
        + R**3*q[..., 4]*q[..., 3]*qr[..., 1] - R*gp_r + R**2*Sp*qz[..., 1],
        Sp + R*rhou + 2*q[..., 4]*qz[..., 1] - (q[..., 6]*rhopsi+q[..., 4]*Vpsi)/R,
        -drr*qr[..., 4]-drz*qz[..., 4],
        -drz*qr[..., 4]-dzz*qz[..., 4],
        Sh + R*(q[..., 4]*Tu+q[..., 5]*rhou) + 2*gamma*q[..., 4]*q[..., 5]*qz[..., 1]
        - q[..., 6]*(q[..., 4]*Tpsi+q[..., 5]*rhopsi)/R - gamma*q[..., 4]*q[..., 5]*Vpsi/R
        + (gamma-1)*etah*(q[..., 2]/R)**2,
        -krr*qr[..., 5]-krz*qz[..., 5],
        -krz*qr[..., 5]-kzz*qz[..., 5],
        -ppsi/R - Sp*q[..., 6]*B2 + .5*q[..., 6]**2*B2*rhopsi/R,
        -a9*qr[..., 6]+a10*q[..., 6]+.5*q[..., 4]*q[..., 6]**2*B2*br+vr_source,
        -a9*qz[..., 6]+.5*q[..., 4]*q[..., 6]**2*B2*bz+vz_source,
    ), dim=-1)


class PhysicalRemainder(nn.Module):
    """Exact native volume remainder around the exported fixed-profile tangent.

    ``profiles`` is an nn.Module mapping physical jets [B,E*Q,21] to the
    profile dictionary consumed by physical_coefficients. All arrays and
    profiles move with this module. The reference forcing is retained exactly;
    the warm reference is not assumed to be a stationary equilibrium.
    """
    def __init__(self, geometry, frozen, mean, scale, profiles):
        super().__init__()
        if not isinstance(profiles, nn.Module):
            raise TypeError('profiles must be an nn.Module supplying native transport/source values')
        self.profiles = profiles
        arrays = dict(basis=geometry.basis, basis_grad=geometry.basis_grad, mean=mean, scale=scale,
                      radius=geometry.quadrature_rz[..., 0].reshape(-1),
                      reference_jets=frozen.jets.reshape(-1, 21),
                      reference_forcing=frozen.forcing.reshape(-1, 15), tangent=frozen.tangent.reshape(-1, 21, 15))
        for name, value in arrays.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64).clone(), persistent=False)
        self.register_buffer('dof_map', torch.as_tensor(geometry.dof_map, dtype=torch.long), persistent=False)
        if self.mean.shape != (len(geometry.dof_ids), 7) or self.scale.shape != (7,) or not (self.scale > 0).all():
            raise ValueError('Native normalization shape/scale mismatch')
        self.assembler = WeakAssembler(geometry)

    @classmethod
    def from_config(cls, config, geometry, frozen, mean, scale):
        """Build only from explicit, SHA256-pinned profiles and q* restart.

        config requires profile_manifest/reference_restart dictionaries, each
        with path and sha256. The manifest records the supported native branch
        audit and exact input/default values; no profile defaults are inferred.
        """
        def verified(record):
            if set(record) != {'path', 'sha256'}:
                raise ValueError('Physical input requires an explicit path and sha256')
            path, digest = Path(record['path']), hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    digest.update(block)
            if digest.hexdigest() != record['sha256']:
                raise ValueError(f'Physical input SHA256 mismatch: {path}')
            return path

        manifest_path = verified(config['profile_manifest'])
        reference_path = verified(config['reference_restart'])
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('schema') != 'mastu_native_profiles_v1' or manifest.get('frozen_sha256') != frozen.sha256:
            raise ValueError('Physical manifest schema or frozen-tangent SHA256 mismatch')
        if manifest.get('supported_branches') != 'n0_model303_analytic_double_null_v1':
            raise ValueError('Physical manifest lacks the supported native branch audit')
        from .native_critical import NativeCritical
        from .restart import read_restart
        reference_geometry, reference, _ = read_restart(reference_path)
        if not np.array_equal(reference_geometry.dof_ids, geometry.dof_ids) or not np.array_equal(reference_geometry.dof_map, geometry.dof_map):
            raise ValueError('Physical reference restart native mesh differs')
        del reference_geometry
        profiles = NativeProfiles(geometry, manifest['physics'], NativeCritical(geometry, reference_path))
        result = cls(geometry, frozen, mean, scale, profiles)
        result.register_buffer('reference_coefficients', torch.tensor(reference, dtype=torch.float64), persistent=False)
        with torch.no_grad():
            expected = result.reference_jets
            actual = result.jets(((result.reference_coefficients-result.mean)/result.scale)[None])[0]
            error = (actual-expected).abs().amax(0)/(expected.abs().amax(0)+1e-30)
            if (error > 1e-7).any():
                raise ValueError(f'Physical q* differs from exported baseline jets: {error.max().item():.3g}')
        result.provenance = dict(profile_manifest=dict(config['profile_manifest']),
                                 reference_restart=dict(config['reference_restart']), frozen_sha256=frozen.sha256)
        return result

    def jets(self, state):
        if state.ndim != 3 or state.shape[1:] != self.mean.shape:
            raise ValueError('Expected normalized [batch,ndof,7] state')
        if state.device != self.mean.device or state.dtype != self.mean.dtype:
            raise ValueError('Native RHS and state must share device and float64 dtype')
        local = (state*self.scale+self.mean)[:, self.dof_map]
        values = torch.einsum('eql,belc->beqc', self.basis, local).flatten(1, 2)
        gradients = torch.einsum('eqlr,belc->beqcr', self.basis_grad, local).flatten(1, 2)
        jets = torch.cat((values, gradients[..., 0], gradients[..., 1]), -1)
        if not torch.isfinite(jets).all() or (values[..., 4:6] <= 0).any():
            raise FloatingPointError('Native RHS requires finite jets and positive density/temperature')
        return jets

    @staticmethod
    def _baseline_parameters(parameters):
        if parameters is not None and not torch.all(parameters == 1):
            raise ValueError('This native profile object supports only its configured physical coefficient')

    def linear_pointwise(self, jets):
        return self.reference_forcing + torch.einsum('pjk,bpj->bpk', self.tangent, jets-self.reference_jets)

    def pointwise(self, state, parameters=None):
        self._baseline_parameters(parameters)
        jets = self.jets(state)
        physical = physical_coefficients(jets, self.radius,
                                         self.profiles(jets, physical_state=state*self.scale+self.mean))
        return (physical-self.linear_pointwise(jets))[..., list(ACTIVE_CHANNELS)]

    remainder = pointwise

    def linear_forcing(self, state):
        return self.assembler(self.linear_pointwise(self.jets(state))[..., list(ACTIVE_CHANNELS)])

    def forward(self, state, parameters=None):
        return self.assembler(self.pointwise(state, parameters))
