"""Autonomous local magnetic axis/X-point tracking on the pinned bicubic mesh.

Restricted to the qualified double-null, fixed-boundary model303 neighborhood.
Reference diagnostics supply starting locations only; returned locations and
fluxes depend on current coefficients. Missing or distant roots fail.
"""
import h5py
import numpy as np
import torch
from torch import nn

# Native cubic basis in powers of s/t; derivative DOFs include factor three.
_ONE = np.array([[1., 0., -3., 2.], [0., 3., -6., 3.],
                 [0., 0., 3., -2.], [0., 0., 3., -3.]])
_BASIS = np.stack([np.outer(_ONE[2*i+d], _ONE[2*j+e])
                   for i, j in ((0, 0), (1, 0), (1, 1), (0, 1))
                   for d, e in ((0, 0), (1, 0), (0, 1), (1, 1))])


def polynomial(coefficients, s, t):
    """Value, local gradient and Hessian of a bicubic polynomial."""
    one, zero = torch.ones_like(s), torch.zeros_like(s)
    sp = torch.stack((one, s, s*s, s*s*s), -1)
    tp = torch.stack((one, t, t*t, t*t*t), -1)
    ds = torch.stack((zero, one, 2*s, 3*s*s), -1)
    dt = torch.stack((zero, one, 2*t, 3*t*t), -1)
    dds = torch.stack((zero, zero, 2*one, 6*s), -1)
    ddt = torch.stack((zero, zero, 2*one, 6*t), -1)
    def value(a, b):
        return (coefficients*a[..., :, None]*b[..., None, :]).sum((-1, -2))
    return value(sp, tp), value(ds, tp), value(sp, dt), value(dds, tp), value(ds, dt), value(sp, ddt)


class NativeCritical(nn.Module):
    """Track one minimum axis and two saddle roots from physical native psi.

    The immutable reference_restart must use geometry. Supports axisymmetric
    bicubic model303 with two retained X-points and no limiter transition.
    The caller must reject free-boundary or other topology configurations.
    Candidate-cell selection is discrete; selected root values/locations remain
    differentiable. No trajectory history enters this module.
    """
    def __init__(self, geometry, reference_restart, *, candidates=128, iterations=20,
                 tolerance=1e-9, search_radius=.08):
        super().__init__()
        if not isinstance(candidates, int) or candidates < 1 or not isinstance(iterations, int) or iterations < 1:
            raise ValueError('Critical-point candidates and iterations must be positive integers')
        if not np.isfinite([tolerance, search_radius]).all() or min(tolerance, search_radius) <= 0:
            raise ValueError('Critical-point tolerance/radius must be positive and finite')
        with h5py.File(reference_restart) as f:
            for key, value in dict(jorek_model=303, n_tor=1, n_order=3, n_degrees=4, n_var=7).items():
                if np.asarray(f[key]).item() != value:
                    raise ValueError('Critical-point tracker requires axisymmetric bicubic model303')
            if np.any(f['n_sons'][()] != 0) or np.any(f['constrained'][()] != b'F'):
                raise ValueError('Critical-point tracker does not support refined/constrained meshes')
            vertex = f['vertex'][()].T-1
            sizes = f['size'][()].transpose(2, 1, 0).reshape(-1, 16)
            xy = f['x'][()][:, :, 0, :].transpose(2, 1, 0)[vertex].reshape(-1, 16, 2)
            indices = f['index'][()].T
            local_indices = indices[vertex].reshape(-1, 16)-1
            if not np.array_equal(local_indices, geometry.dof_map):
                raise ValueError('Reference restart and runtime mesh have different native DOF maps')
            anchors = np.array([[f['R_axis_t'][()][-1], f['Z_axis_t'][()][-1]],
                                [f['R_xpoint_t'][()][0, -1], f['Z_xpoint_t'][()][0, -1]],
                                [f['R_xpoint_t'][()][1, -1], f['Z_xpoint_t'][()][1, -1]]])
            if not np.isfinite(anchors).all() or anchors[1, 1] >= anchors[0, 1] or anchors[2, 1] <= anchors[0, 1]:
                raise ValueError('Reference must have lower/upper X-points around its axis')
            if f['psi_axis_t'][()][-1] >= f['psi_bnd_t'][()][-1]:
                raise ValueError('This scoped tracker requires a minimum-flux magnetic axis')
        centers = geometry.quadrature_rz
        distances = np.linalg.norm(centers[None]-anchors[:, None, None], axis=-1)
        count = min(candidates, len(vertex))
        selected = np.argsort(distances.min(-1), axis=-1)[:, :count]
        q = int(round(centers.shape[1]**.5))
        if q*q != centers.shape[1]:
            raise ValueError('Expected tensor-product native quadrature')
        gauss = (np.polynomial.legendre.leggauss(q)[0]+1)/2
        mapping = sizes[..., None, None]*_BASIS[None]
        coordinate_polynomials = np.einsum('elij,elc->ecij', mapping, xy)
        arrays = dict(dof_map=local_indices[selected], mapping=mapping[selected],
                      coordinate_polynomials=coordinate_polynomials[selected],
                      quadrature_grad=geometry.basis_grad[selected],
                      quadrature_rz=centers[selected],
                      gauss=gauss, anchors=anchors)
        for key, value in arrays.items():
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.long if key == 'dof_map' else torch.float64),
                                 persistent=False)
        self.ndof = len(geometry.dof_ids)
        self.iterations, self.tolerance, self.search_radius = iterations, tolerance, search_radius
        self.last_diagnostics = None

    def _source_axis_flux(self, coefficients, gradient):
        """Native mnewtax axis choice for sources, after strict topology validation.

        Native find_axis accepts a stationary saddle as well as a minimum. Its
        flux normalizes current/heat source profiles. Reproduce its stopping and
        Q4 priority within our qualified reference neighborhood; no history or
        native failed-search fallback enters autonomous evolution.
        """
        distance = (self.quadrature_rz[0]-self.anchors[0]).square().sum(-1)
        priority, seed = torch.where(distance < self.search_radius**2,
                                     gradient[:, 0].square().sum(-1), float('inf')).min(-1)
        s, t = self.gauss[seed // len(self.gauss)], self.gauss[seed % len(self.gauss)]
        coefficients = coefficients[:, 0]
        accepted = torch.zeros_like(s, dtype=torch.bool)
        for _ in range(20):  # Pinned native mnewtax uses twenty iterations.
            _, gs, gt, hss, hst, htt = polynomial(coefficients, s, t)
            accepted = accepted | (gs.abs()+gt.abs() <= 1e-16)
            determinant = hss*htt-hst*hst
            nonzero = determinant != 0
            safe = torch.where(nonzero, determinant, torch.ones_like(determinant))
            ds = torch.where(nonzero, (-htt*gs+hst*gt)/safe, -gs)
            dt = torch.where(nonzero, (-hss*gt+hst*gs)/safe, -gt)
            s = torch.where(accepted, s, (s+ds.clamp(-.25, .25)).clamp(0., 1.))
            t = torch.where(accepted, t, (t+dt.clamp(-.25, .25)).clamp(0., 1.))
            accepted = accepted | (ds.abs()+dt.abs() <= 1e-8)
        accepted = accepted & torch.isfinite(priority)
        if not accepted.any(-1).all():
            raise FloatingPointError('Native source-axis search failed in its qualified neighborhood')
        selected = torch.where(accepted, priority, float('inf')).argmin(-1)
        flux = polynomial(coefficients, s, t)[0].gather(-1, selected[:, None]).squeeze(-1)
        if not torch.isfinite(flux).all():
            raise FloatingPointError('Nonfinite native source-axis flux')
        return flux

    def forward(self, physical_state):
        psi = physical_state[..., 0] if physical_state.ndim == 3 else physical_state
        if psi.ndim != 2 or psi.shape[-1] != self.ndof or psi.dtype != torch.float64 or psi.device != self.mapping.device:
            raise ValueError('Expected physical float64 psi [batch,ndof] on the tracker device')
        if not torch.isfinite(psi).all():
            raise FloatingPointError('Nonfinite magnetic flux')
        coefficients = torch.einsum('bpkl,pklij->bpkij', psi[:, self.dof_map], self.mapping)
        local = psi[:, self.dof_map]
        gradient = torch.einsum('pknld,bpkl->bpknd', self.quadrature_grad, local)
        seed_norm = gradient.square().sum(-1)
        priority, seed = seed_norm.min(-1)
        s = self.gauss[seed // len(self.gauss)]
        t = self.gauss[seed % len(self.gauss)]
        for _ in range(self.iterations):
            _, gs, gt, hss, hst, htt = polynomial(coefficients, s, t)
            determinant = hss*htt-hst*hst
            safe = torch.where(determinant.abs() > 1e-40, determinant, torch.ones_like(determinant))
            ds, dt = (htt*gs-hst*gt)/safe, (hss*gt-hst*gs)/safe
            s = (s-ds.clamp(-.25, .25)).clamp(0., 1.)
            t = (t-dt.clamp(-.25, .25)).clamp(0., 1.)
        flux, gs, gt, hss, hst, htt = polynomial(coefficients, s, t)
        R, Rs, Rt, *_ = polynomial(self.coordinate_polynomials[..., 0, :, :], s, t)
        Z, Zs, Zt, *_ = polynomial(self.coordinate_polynomials[..., 1, :, :], s, t)
        jac = Rs*Zt-Rt*Zs
        safe_jac = torch.where(jac.abs() > 1e-30, jac, torch.ones_like(jac))
        residual = torch.sqrt(((gs*Zt-gt*Zs)/safe_jac)**2 + ((gt*Rs-gs*Rt)/safe_jac)**2)
        distance = torch.sqrt((R-self.anchors[:, 0, None])**2+(Z-self.anchors[:, 1, None])**2)
        determinant = hss*htt-hst*hst
        classified = torch.cat(((determinant[:, :1] > 0) & (hss[:, :1] > 0), determinant[:, 1:] < 0), 1)
        valid = ((s >= -1e-8) & (s <= 1+1e-8) & (t >= -1e-8) & (t <= 1+1e-8)
                 & (residual <= self.tolerance) & (distance <= self.search_radius)
                 & (jac.abs() > 1e-30) & classified & torch.isfinite(flux))
        if not valid.any(-1).all():
            counts = valid.sum(-1).detach().cpu().tolist()
            raise FloatingPointError(f'Native axis/X-point tracking failed in its qualified neighborhood: {counts}')
        # Native axis search tries the smallest Q4 gradient first. The same
        # priority avoids selecting a nearby secondary root merely by distance.
        selected = torch.where(valid, priority, torch.full_like(priority, float('inf'))).argmin(-1)
        gather = lambda a: a.gather(-1, selected[..., None]).squeeze(-1)
        flux, R, Z, residual = map(gather, (flux, R, Z, residual))
        if not (flux[:, 1:] > flux[:, :1]).all():
            raise FloatingPointError('X-point flux no longer bounds the minimum-flux axis')
        boundary = flux[:, 1:].min(-1).values
        source_axis = self._source_axis_flux(coefficients, gradient)
        if not (boundary > source_axis).all():
            raise FloatingPointError('Native source-axis flux is not bounded by the X-points')
        self.last_diagnostics = dict(max_gradient=float(residual.detach().max()),
                                     candidate_counts=valid.sum(-1).detach().cpu().tolist())
        return dict(psi_axis=flux[:, 0], source_psi_axis=source_axis,
                    psi_bnd=boundary, Z_xpoint=Z[:, 1:],
                    R_axis=R[:, 0], Z_axis=Z[:, 0], R_xpoint=R[:, 1:], psi_xpoint=flux[:, 1:])
