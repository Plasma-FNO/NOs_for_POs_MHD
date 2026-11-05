"""Source-based common-native adapter and constrained coarse-grid reference.

This is a gridded reference discretization, not the generating JOREK FEM revision.
Potential conversion follows spamela/neural_parareal norm_JOREK at 5e4203a.
ES coefficient/time rescaling follows the release input comment.
"""
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from equations import JorekTerms

N0 = 1e20
MU0 = 1.257e-6
EV = 1.602176487e-19
TAU = (3.32e-27*N0*MU0)**.5
F0 = 10.


def read_input(path):
    result = {}
    for line in Path(path).read_text().lower().splitlines():
        match = re.fullmatch(r'\s*([\w]+(?:\(\d+\))?)\s*=\s*([^!]+?)\s*(?:!.*)?', line)
        if match:
            key, value = match.groups()
            if value.strip() in ('.t.', '.true.', '.f.', '.false.'):
                result[key] = value.strip() in ('.t.', '.true.')
            else:
                try:
                    result[key] = float(value.replace('d', 'e'))
                except ValueError:
                    pass
    return result


def common_coefficients(inputs, case):
    if inputs.get('central_density', 1.) != 1. or inputs.get('f0', F0) != F0:
        raise ValueError('Release adapter only validated for n0=1e20 and F0=10')
    factor = 100. if case == 'es' else 1.
    return dict(d=inputs['d_perp']/factor, k=inputs['zk_perp']/factor,
                mu=inputs['visco']/factor, mu_hyper=inputs.get('visco_num', 0.)/factor,
                d_hyper=inputs.get('d_perp_num', 0.)/factor, k_hyper=inputs.get('zk_perp_num', 0.)/factor,
                eta=inputs.get('eta', 0.) if case == 'em' else 0.,
                eta_hyper=inputs.get('eta_num', 0.) if case == 'em' else 0.,
                gamma=5/3, f0=inputs.get('f0', F0),
                rho_threshold=inputs.get('d_prof_neg_thresh', 0.), d_negative=inputs.get('d_prof_neg', 1e-5)/factor,
                T_threshold=inputs.get('zk_prof_neg_thresh', 0.)/(1e4 if case == 'es' else 1.),
                k_negative=inputs.get('zk_prof_neg', 1e-5)/factor, parallel_k=inputs.get('zk_par', 0.)/factor,
                parallel_d=inputs.get('d_par', 0.)/factor,
                viscosity='old' if case == 'es' or inputs.get('visco_old_setup', False) else 'new',
                conservative=case == 'em',
                nominal_frame_dt=inputs['tstep']*factor*10.,
                rho_wall=inputs['rho_0'], T_wall=inputs['t_0']/(1e4 if case == 'es' else 1.))


def decode_release(raw):
    """Raw release order [rho,T,potential,omega,(psi,j)] -> primary native q."""
    rho, temp, potential = raw[..., 0, :, :]/N0, raw[..., 1, :, :]*(N0*MU0*EV), raw[..., 2, :, :]
    q = [rho, rho*temp, -potential*TAU/F0]
    if raw.shape[-3] == 6:
        q.append(raw[..., 4, :, :])
    return torch.stack(q, dim=-3)


def encode_release(q, omega, current=None):
    rho, p, u = q.unbind(-3)[:3]
    if torch.any(rho <= 0):
        raise ValueError('Nonpositive density in native state')
    fields = [rho*N0, p/rho/(N0*MU0*EV), -u*F0/TAU, omega]
    if q.shape[-3] == 4:
        fields += [q[..., 3, :, :], current]
    return torch.stack(fields, -3)


class MomentumSolve:
    """Finite-volume -div(a grad x)=rhs, zero wall increment, SPD integrated form.

    Wall coordinates enclose every sampled point. No sampled edge is clamped.
    Jacobi-preconditioned CG is differentiable through executed iterations.
    """
    def __init__(self, r, z, bounds=(9.5, 10.5, -.5, .5)):
        self.r, self.z = r, z
        self.re = torch.cat((r.new_tensor([bounds[0]]), r, r.new_tensor([bounds[1]])))
        self.ze = torch.cat((z.new_tensor([bounds[2]]), z, z.new_tensor([bounds[3]])))
        if torch.any(torch.diff(self.re) <= 0) or torch.any(torch.diff(self.ze) <= 0):
            raise ValueError('Wall coordinates must strictly enclose samples')
        self.wr, self.wz = (self.re[2:]-self.re[:-2])/2, (self.ze[2:]-self.ze[:-2])/2
        self.volume = self.wz[:, None]*self.wr[None, :]

    def edges(self, a):
        if not torch.isfinite(a).all() or torch.any(a <= 0):
            raise ValueError('Positive finite momentum coefficient required')
        ap = F.pad(a.unsqueeze(-3), (1, 1, 1, 1), mode='replicate').squeeze(-3)
        cr = (ap[..., 1:-1, 1:]+ap[..., 1:-1, :-1])/2 / torch.diff(self.re)*self.wz[:, None]
        cz = (ap[..., 1:, 1:-1]+ap[..., :-1, 1:-1])/2 / torch.diff(self.ze)[:, None]*self.wr
        return cr, cz

    @staticmethod
    def apply(x, cr, cz):
        xp = F.pad(x, (1, 1, 1, 1))
        return ((cr[..., 1:]+cr[..., :-1]+cz[..., 1:, :]+cz[..., :-1, :])*x
                - cr[..., 1:]*xp[..., 1:-1, 2:]-cr[..., :-1]*xp[..., 1:-1, :-2]
                - cz[..., 1:, :]*xp[..., 2:, 1:-1]-cz[..., :-1, :]*xp[..., :-2, 1:-1])

    def solve(self, a, rhs, tolerance=1e-7, max_iterations=1000):
        cr, cz = self.edges(a)
        diagonal = cr[..., 1:]+cr[..., :-1]+cz[..., 1:, :]+cz[..., :-1, :]
        b = rhs*self.volume
        x, residual = torch.zeros_like(b), b
        norm = b.square().sum((-2, -1), keepdim=True).sqrt()
        if torch.all(norm == 0):
            return b*0
        z = residual/diagonal
        p, rz = z, (residual*z).sum((-2, -1), keepdim=True)
        tiny = torch.finfo(b.dtype).tiny
        for _ in range(max_iterations):
            ap = self.apply(p, cr, cz)
            alpha = rz/(p*ap).sum((-2, -1), keepdim=True).clamp_min(tiny)
            x = x+alpha*p
            residual = residual-alpha*ap
            if torch.all(residual.square().sum((-2, -1), keepdim=True).sqrt() <= tolerance*norm.clamp_min(tiny)):
                # Verify the true residual, not just the CG recurrence.
                actual = b-self.apply(x, cr, cz)
                if torch.all(actual.square().sum((-2, -1), keepdim=True).sqrt() <= 5*tolerance*norm.clamp_min(tiny)):
                    return x
            z = residual/diagonal
            new_rz = (residual*z).sum((-2, -1), keepdim=True)
            p = z+(new_rz/rz.clamp_min(tiny))*p
            rz = new_rz
        raise RuntimeError('Momentum CG failed the residual tolerance')


class NativeReference:
    """Constrained A/D/C reference on interior samples, static physical wall trace.

    The wall trace is supplied from initial data/inputs and kept independent of
    future snapshots. This is a continuum candidate with declared FD wall closure,
    hyperdiffusion and threshold transport; it does not duplicate the FEM scheme.
    """
    def __init__(self, r, z, wall, coefficients, bounds=(9.5, 10.5, -.5, .5)):
        self.mass = MomentumSolve(r, z, bounds)
        self.op = JorekTerms(self.mass.re, self.mass.ze)
        self.wall, self.c = wall, coefficients
        self.r = self.mass.re

    def extend(self, q):
        return F.pad(q, (1, 1, 1, 1)) + self.wall

    def div(self, fr, fz):
        return self.op.grad(fr)[0]+self.op.grad(fz)[1]

    @staticmethod
    def second(x, coordinate, axis):
        y = x.movedim(axis, -1)
        slopes = torch.diff(y, dim=-1)/torch.diff(coordinate)
        inner = 2*torch.diff(slopes, dim=-1)/(coordinate[2:]-coordinate[:-2])
        return F.pad(inner, (1, 1), mode='constant').movedim(-1, axis)

    def laplacian(self, x):
        return self.second(x, self.mass.re, -1)+self.op.grad(x)[0]/self.r+self.second(x, self.mass.ze, -2)

    def constraints(self, q):
        x = self.extend(q)
        u = x[..., 2, :, :]
        ur, uz = self.op.grad(u)
        omega = self.second(u, self.mass.re, -1)+ur/self.r+self.second(u, self.mass.ze, -2)
        mask = F.pad(torch.ones_like(omega[..., 1:-1, 1:-1]), (1, 1, 1, 1))
        omega = omega*mask
        if q.shape[-3] == 4:
            pr, pz = self.op.grad(x[..., 3, :, :])
            psi = x[..., 3, :, :]
            j = self.second(psi, self.mass.re, -1)-pr/self.r+self.second(psi, self.mass.ze, -2)
            j = F.pad(j[..., 1:-1, 1:-1].unsqueeze(-3), (1, 1, 1, 1), mode='replicate').squeeze(-3)
        else:
            j = torch.zeros_like(u)
        return omega, j

    def terms(self, q, imex=False):
        if torch.any(q[...,0,:,:]<=0):
            raise ValueError('Nonpositive density; refusing floor/clipping')
        return self._terms(q,imex)

    def _terms(self, q, imex=False):
        x = self.extend(q)
        rho, p, u = x.unbind(-3)[:3]
        T = p/rho
        c, op, r = self.c, self.op, self.r
        ur, uz = op.grad(u)
        rhohat = rho*r**2
        omega, j = self.constraints(q)
        rr,rz=op.grad(rho);pr,pz=op.grad(p)
        A = torch.stack((r*(rr*uz-rz*ur)+2*rho*uz,
                         r*(pr*uz-pz*ur)+2*c['gamma']*p*uz), -3)
        inertia = .5*op.bracket(r**2*(ur**2+uz**2), rhohat)+op.bracket(rhohat*r**2*omega, u)
        pressure = -2*r*pz
        tr, tz = op.grad(T)
        d = torch.where(rho < c['rho_threshold'], rho.new_tensor(c['d_negative']), rho.new_tensor(c['d']))
        k = torch.where(T < c['T_threshold'], T.new_tensor(c['k_negative']), T.new_tensor(c['k']))
        D = torch.stack((op.divergence(d*rr, d*rz), op.divergence(k*tr, k*tz)), -3)
        isotropic = D
        numerical = torch.stack((-c['d_hyper']*self.laplacian(self.laplacian(rho)),
                                 -c['k_hyper']*self.laplacian(self.laplacian(T))), -3)
        visc,hyper_visc=u*0,u*0
        if not imex:
            wr, wz = op.grad(omega)
            power = 1 if c['viscosity'] == 'old' else 3
            visc = self.div(c['mu']*r**power*wr, c['mu']*r**power*wz)
            lap_w = self.laplacian(omega)
            hyper_visc = -c['mu_hyper']*(self.second(lap_w, self.mass.re, -1)-op.grad(lap_w)[0]/r
                                         +lap_w/r**2+self.second(lap_w, self.mass.ze, -2))
            if c['viscosity'] == 'new':
                visc += op.grad(2*c['mu']*r**2*omega)[0]
        induction, lorentz, resistivity = torch.zeros_like(u), torch.zeros_like(u), torch.zeros_like(u)
        if q.shape[-3] == 4:
            psi = x[..., 3, :, :]
            pr, pz = op.grad(psi)
            br, bz, bt = -pz/r, pr/r, c['f0']/r
            b2 = br**2+bz**2+bt**2
            for i, (sr, sz, par, perp) in enumerate(((rr, rz, torch.where(rho<c['rho_threshold'], d, rho.new_tensor(c['parallel_d'])), d),
                                                   (tr, tz, c['parallel_k'], k))):
                flux = (par-perp)*(br*sr+bz*sz)/b2
                # Out-of-place update preserves autograd.
                correction = op.divergence(flux*br, flux*bz)
                D = D + torch.stack((correction, correction*0) if i == 0 else (correction*0, correction), -3)
            induction = r*(pr*uz-pz*ur)
            jr,jz=op.grad(j)
            lorentz=pr*jz-pz*jr
            if not imex:resistivity = c['eta']*j-c['eta_hyper']*r*self.div(jr,jz)
        # Conservative EM momentum: extra density diffusion contribution.
        conservative = -self.div(r**3*(D+numerical)[..., 0, :, :]*ur, r**3*(D+numerical)[..., 0, :, :]*uz) if c['conservative'] else u*0
        return {key: value[..., 1:-1, 1:-1] for key, value in dict(
            scalar_advection=A, fluid_momentum=inertia+pressure, scalar_diffusion=D, isotropic_diffusion=isotropic,
            numerical_diffusion=numerical, viscosity=visc+hyper_visc, conservative=conservative, induction=induction,
            lorentz=lorentz, resistivity=resistivity).items()}

    def rhs(self, q):
        terms = self.terms(q)
        rho = q[..., 0, :, :]
        load = terms['fluid_momentum']+terms['viscosity']+terms['lorentz']+terms['conservative']
        # M is div(rho R^3 grad); SPD solver represents its negative.
        du = self.mass.solve(rho*self.mass.r**3, -load)
        scalar = terms['scalar_advection']+terms['scalar_diffusion']+terms['numerical_diffusion']
        result = [scalar[..., 0, :, :], scalar[..., 1, :, :], du]
        if q.shape[-3] == 4:
            result += [terms['induction']+terms['resistivity']]
        return torch.stack(result, -3)


def wall_trace(q0, r, z, coefficients, bounds=(9.5, 10.5, -.5, .5)):
    """Constant rho/T and zero stream function at the actual wall.

    Magnetic trace uses linear extrapolation of INITIAL psi through the 1 mm
    unsampled strip; this is a declared closure approximation, not observed BC.
    """
    wall = q0.new_zeros((*q0.shape[:-2], len(z)+2, len(r)+2))
    wall[..., 0, :, :] = coefficients['rho_wall']
    wall[..., 1, :, :] = coefficients['rho_wall']*coefficients['T_wall']
    if q0.shape[-3] == 4:
        psi = q0[..., 3, :, :]
        ext = F.pad(psi, (1, 1, 1, 1))
        ext[..., 1:-1, 0] = psi[..., :, 0]+(bounds[0]-r[0])*(psi[..., :, 1]-psi[..., :, 0])/(r[1]-r[0])
        ext[..., 1:-1, -1] = psi[..., :, -1]+(bounds[1]-r[-1])*(psi[..., :, -1]-psi[..., :, -2])/(r[-1]-r[-2])
        ext[..., 0, :] = ext[..., 1, :]+(bounds[2]-z[0])*(ext[..., 2, :]-ext[..., 1, :])/(z[1]-z[0])
        ext[..., -1, :] = ext[..., -2, :]+(bounds[3]-z[-1])*(ext[..., -2, :]-ext[..., -3, :])/(z[-1]-z[-2])
        wall[..., 3, :, :] = ext
    wall[..., 1:-1, 1:-1] = 0
    return wall
