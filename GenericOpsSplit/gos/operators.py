"""Implicit linear operators L(mu) = O0 + mu_t O1 for the generic step

    (M - theta dt L) q' = (M + (1-theta) dt L) q + dt * load,      load = M r  (r: explicit rate)

Modes
  known  : the exact discrete operator of the problem (native tangent analogue)
  struct : learned, L = J - D with J skew and D symmetric positive semidefinite in the M inner product,
           for every mu_t >= 0. Then x^T L x <= 0, so the theta >= 1/2 step is non-expansive in the
           M-norm for every dt (unconditionally stable linear part).
  free   : learned with the same sparsity/locality, no sign structure (ablation).

Parameters are dimensionless: dt-scaled eigenvalues h*lambda are exp/sinh of the raw parameters.
"""
import math

import torch
from torch import nn

from .problems import FourierGrid, FEM1D, hermitian


class FourierLinear(nn.Module):
    """Translation-invariant operator as an rfft multiplier (M = I)."""

    def __init__(self, problem, mode, device):
        super().__init__()
        self.problem, self.mode, self.h = problem, mode, problem.h
        self.grid = FourierGrid(problem.n, problem.length, problem.dim, device, torch.float32)
        shape = (2,) + self.grid.spec_shape
        if mode == 'struct':
            self.re = nn.Parameter(torch.full(shape, math.log(1e-2)))
        elif mode == 'free':
            self.re = nn.Parameter(torch.zeros(shape))
        if mode in ('struct', 'free'):
            self.im = nn.Parameter(torch.zeros(shape))

    def components(self):
        """(O0, O1) symbols, complex64 [*spec]."""
        if self.mode == 'known':
            zero = torch.zeros(1, dtype=torch.float64, device=self.grid.k2.device)
            o0 = self.problem.symbol(zero, self.grid)[0]
            o1 = self.problem.symbol(zero + self.problem.mu_ref, self.grid)[0] - o0
            return o0.to(torch.complex64), o1.to(torch.complex64)
        re = -torch.exp(self.re)/self.h if self.mode == 'struct' else torch.sinh(self.re)/self.h
        lam = torch.complex(re, torch.sinh(self.im)/self.h)
        return lam[0], lam[1]

    def symbol(self, mu_t):
        if self.mode == 'known':
            mu = mu_t.double()*self.problem.mu_ref
            return self.problem.symbol(mu, self.grid).to(torch.complex64)
        o0, o1 = self.components()
        mu_t = mu_t.view(-1, *([1]*self.grid.dim))
        return hermitian(o0[None] + mu_t*o1[None], self.grid.dim)

    def apply(self, q, mu_t):
        return self.grid.ifft(self.symbol(mu_t).unsqueeze(1)*self.grid.fft(q))

    def mass(self, r):
        return r

    def mass_solve(self, load):
        return load

    def solve(self, q, load, mu_t, dt, theta):
        s = self.symbol(mu_t).unsqueeze(1)
        qh, fh = self.grid.fft(q), self.grid.fft(load)
        return self.grid.ifft(((1 + (1-theta)*dt*s)*qh + dt*fh)/(1 - theta*dt*s))

    def shifted_solve(self, rhs, mu_t, shift):
        """Solve (M - shift L) x = rhs."""
        return self.grid.ifft(self.grid.fft(rhs)/(1 - shift*self.symbol(mu_t).unsqueeze(1)))

    @torch.no_grad()
    def init_from(self, true):
        """Warm start: set (O0, O1) to the true operator (projected onto Re <= 0 for struct)."""
        h = self.h
        for i, o in enumerate(true.components()):
            re, im = o.real.float()*h, o.imag.float()*h
            self.re[i] = torch.log((-re).clamp_min(1e-8)) if self.mode == 'struct' else torch.asinh(re)
            self.im[i] = torch.asinh(im)


class FEMLinear(nn.Module):
    """Tridiagonal operator on a 1D P1 mesh with consistent mass matrix M (SPD)."""

    def __init__(self, problem, mode, device):
        super().__init__()
        self.problem, self.mode, self.h = problem, mode, problem.h
        self.grid = FEM1D(problem.elements, device, torch.float32)
        n, E = self.grid.n, self.grid.elements
        self.unit = self.grid.dx/self.h   # entries of this size give h*M^-1 L ~ O(1)
        if mode == 'struct':
            self.conductance = nn.Parameter(torch.full((2, E), math.log(1e-2)))
            self.skew = nn.Parameter(torch.zeros(2, E-2))
            self.reaction = nn.Parameter(torch.full((2, n), math.log(1e-3)))
        elif mode == 'free':
            self.bands_raw = nn.Parameter(torch.zeros(3, 2, n))
        M = tuple(b.float() for b in self.grid.mass_bands)
        self.register_buffer('mass_dense', FEM1D.dense(M))
        for i, b in enumerate(M):
            self.register_buffer(f'mass_band{i}', b)
        if mode == 'known':
            diffusion, advection = problem.operator_parts(self.grid)
            for i, (d, a) in enumerate(zip(diffusion, advection)):
                self.register_buffer(f'diff{i}', d.float()*problem.mu_ref)
                self.register_buffer(f'adv{i}', a.float())

    @property
    def mass_bands(self):
        return (self.mass_band0, self.mass_band1, self.mass_band2)

    def components(self):
        """(O0, O1) as band triples [lower, diag, upper], each [n]."""
        if self.mode == 'known':
            return (self.adv0, self.adv1, self.adv2), (self.diff0, self.diff1, self.diff2)
        if self.mode == 'struct':
            out = []
            for i in range(2):
                lower, diag, upper = self.grid.bands_from_elements(torch.exp(self.conductance[i])*self.unit,
                                                                   torch.sinh(self.skew[i])*self.unit)
                out.append((lower, diag - torch.exp(self.reaction[i])*self.unit, upper))
            return tuple(out)
        raw = torch.sinh(self.bands_raw)*self.unit
        mask_lower = torch.ones_like(raw[0, 0])
        mask_lower[0] = 0
        mask_upper = torch.flip(mask_lower, (0,))
        return tuple((raw[0, i]*mask_lower, raw[1, i], raw[2, i]*mask_upper) for i in range(2))

    def bands(self, mu_t):
        o0, o1 = self.components()
        mu_t = mu_t.view(-1, 1)
        return tuple(a[None] + mu_t*b[None] for a, b in zip(o0, o1))

    def apply(self, q, mu_t):
        return FEM1D.apply(self.bands(mu_t), q.squeeze(1)).unsqueeze(1)

    def mass(self, r):
        return FEM1D.apply(self.mass_bands, r)

    def mass_solve(self, load):
        return torch.linalg.solve(self.mass_dense, load.squeeze(1).transpose(0, 1)).transpose(0, 1).unsqueeze(1)

    def solve(self, q, load, mu_t, dt, theta):
        L = FEM1D.dense(self.bands(mu_t))
        A = self.mass_dense[None] - theta*dt*L
        x = q.squeeze(1)
        rhs = (self.mass_dense[None] @ x[..., None]).squeeze(-1) + (1-theta)*dt*(L @ x[..., None]).squeeze(-1) \
            + dt*load.squeeze(1)
        return torch.linalg.solve(A, rhs).unsqueeze(1)

    def shifted_solve(self, rhs, mu_t, shift):
        A = self.mass_dense[None] - shift*FEM1D.dense(self.bands(mu_t))
        return torch.linalg.solve(A, rhs.squeeze(1)).unsqueeze(1)

    @torch.no_grad()
    def init_from(self, true):
        """Warm start from the true operator: O0 = skew advection, O1 = diffusion at mu_ref."""
        p, g = self.problem, self.grid
        conductance = p.mu_ref*p.kappa_shape(g.xe.double())/g.dx
        skew = -p.velocity(g.x[:-1].double() + g.dx/2)/2
        if self.mode == 'struct':
            self.conductance[0].fill_(math.log(1e-8))
            self.conductance[1].copy_(torch.log((conductance/self.unit).clamp_min(1e-8)).float())
            self.skew[0].copy_(torch.asinh(skew/self.unit).float())
            self.skew[1].zero_()
            self.reaction.fill_(math.log(1e-8))
        else:
            for i, bands in enumerate(true.components()):
                for j, band in enumerate(bands):
                    self.bands_raw[j, i].copy_(torch.asinh(band.float()/self.unit))


def gmres(A, b, Minv, iterations, eps=1e-30):
    """Batched right-preconditioned GMRES from x0 = 0 (differentiable, fixed iteration count).
    Returns (x, relative residual of the preconditioned least-squares problem)."""
    B, shape = b.shape[0], b.shape
    flat = lambda z: z.reshape(B, -1)
    beta = flat(b).norm(dim=1)
    V = [flat(b)/beta[:, None].clamp_min(eps)]
    columns = []
    for j in range(iterations):
        w = flat(A(Minv(V[j].view(shape))))
        h = []
        for i in range(j+1):  # modified Gram-Schmidt
            hij = (w*V[i]).sum(1)
            w = w - hij[:, None]*V[i]
            h.append(hij)
        hn = w.norm(dim=1)
        h.append(hn)
        V.append(w/hn[:, None].clamp_min(eps))
        columns.append(torch.stack(h + [torch.zeros_like(hn)]*(iterations-j-1), 1))
    H = torch.stack(columns, 2)                      # [B, m+1, m]
    e1 = torch.zeros(B, iterations+1, 1, device=b.device, dtype=b.dtype)
    e1[:, 0, 0] = beta
    Q, R = torch.linalg.qr(H)
    # Exact Krylov breakdown leaves unused zero pivots (including an entirely zero RHS).
    diagonal = R.diagonal(dim1=-2, dim2=-1)
    floor = torch.finfo(b.dtype).eps*H.abs().amax((1, 2)).clamp_min(1.)[:, None]
    safe = torch.where(diagonal.abs() < floor, torch.where(diagonal < 0, -floor, floor), diagonal)
    R = R + torch.diag_embed(safe-diagonal)
    y = torch.linalg.solve_triangular(R, Q.transpose(1, 2) @ e1, upper=True)
    residual = (e1 - H @ y).squeeze(-1).norm(dim=1)/beta.clamp_min(eps)
    x = sum(y[:, j]*V[j] for j in range(iterations))
    return Minv(x.view(shape)), residual


class _OseenSolve(torch.autograd.Function):
    """Differentiate the linear system, avoiding gradients through Arnoldi and QR."""

    @staticmethod
    def forward(q, load, mu_t, dt, operator):
        return operator._solve(q, load, mu_t, dt)

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, load, mu_t, dt, operator = inputs
        ctx.dt, ctx.operator = dt, operator
        ctx.save_for_backward(q, output, mu_t)
        ctx.save_for_forward(q, output, mu_t)

    @staticmethod
    def backward(ctx, grad):
        q, x, mu_t = ctx.saved_tensors
        op, dt = ctx.operator, ctx.dt
        A, Minv = op.system(q, mu_t, dt, transpose=True)
        adjoint, _ = gmres(A, grad, Minv, op.iterations)
        g = op.grid
        ph = g.ifft(g.fft(adjoint)*g.mask)
        xh = g.fft(x)
        dx, dy = g.ifft(1j*g.ks[1]*xh), g.ifft(1j*g.ks[0]*xh)
        # velocity() is skew-adjoint in each component on the periodic grid.
        uq, _ = op.velocity(ph*dx)
        _, wq = op.velocity(ph*dy)
        grad_q = adjoint + dt*(uq + wq)
        grad_mu = -dt*op.problem.mu_ref*(adjoint*g.ifft(g.k2*xh)).flatten(1).sum(1)
        return grad_q, dt*adjoint, grad_mu, None, None

    @staticmethod
    def jvp(ctx, dq, dload, dmu, _dt, _operator):
        q, x, mu_t = ctx.saved_tensors
        op, dt = ctx.operator, ctx.dt
        rhs = torch.zeros_like(q)
        if dq is not None:
            u, w = op.velocity(dq)
            rhs = rhs + dq - dt*op.advect(u, w, x)
        if dload is not None:
            rhs = rhs + dt*dload
        if dmu is not None:
            dnu = (dmu*op.problem.mu_ref).view(-1, 1, 1, 1)
            rhs = rhs - dt*dnu*op.grid.ifft(op.grid.k2*op.grid.fft(x))
        A, Minv = op.system(q, mu_t, dt)
        tangent, _ = gmres(A, rhs, Minv, op.iterations)
        return tangent


class OseenNS(nn.Module):
    """2D vorticity: L(q_n) = nu lap - u(q_n).grad, re-linearised every step (Picard/Oseen), so the stiff
    advection is implicit. Solved by GMRES right-preconditioned with (I - dt nu lap)^-1. M = I, theta = 1."""

    def __init__(self, problem, mode, device, iterations=40):
        super().__init__()
        if problem.name != 'ns2d' or mode != 'known':
            raise ValueError('Oseen operator is implemented for the known ns2d operator only')
        self.problem, self.mode, self.h, self.iterations = problem, mode, problem.h, iterations
        self.grid = FourierGrid(problem.n, problem.length, problem.dim, device, torch.float32)
        self.scale, self.last_residual = 1.0, None
        self.implicit_gradient = False

    def velocity(self, q):
        g = self.grid
        k2 = g.k2.clone()
        k2[0, 0] = 1.
        psi = g.fft(q*self.scale)/k2
        return g.ifft(1j*g.ks[0]*psi), g.ifft(-1j*g.ks[1]*psi)

    def advect(self, u, w, x):
        g = self.grid
        xh = g.fft(x)
        return g.ifft(g.fft(u*g.ifft(1j*g.ks[1]*xh) + w*g.ifft(1j*g.ks[0]*xh))*g.mask)

    def apply(self, q, mu_t):
        nu = (mu_t*self.problem.mu_ref).view(-1, 1, 1, 1)
        u, w = self.velocity(q)
        return self.grid.ifft(-nu*self.grid.k2*self.grid.fft(q)) - self.advect(u, w, q)

    def mass(self, r):
        return r

    def mass_solve(self, load):
        return load

    def solve(self, q, load, mu_t, dt, theta):
        if theta != 1:
            raise ValueError('Oseen step is backward Euler only')
        if self.implicit_gradient:
            return _OseenSolve.apply(q, load, mu_t, dt, self)
        return self._solve(q, load, mu_t, dt)

    def system(self, q, mu_t, dt, transpose=False):
        g = self.grid
        nu = (mu_t*self.problem.mu_ref).view(-1, 1, 1, 1)
        u, w = self.velocity(q)
        diag = 1 + dt*nu*g.k2

        def A(x):
            diffusion = g.ifft(nu*g.k2*g.fft(x))
            if transpose:
                projected = g.ifft(g.fft(x)*g.mask)
                advection = -g.ifft(1j*g.ks[1]*g.fft(u*projected) + 1j*g.ks[0]*g.fft(w*projected))
            else:
                advection = self.advect(u, w, x)
            return x + dt*(diffusion + advection)

        return A, lambda r: g.ifft(g.fft(r)/diag)

    def _solve(self, q, load, mu_t, dt):
        A, Minv = self.system(q, mu_t, dt)
        x, self.last_residual = gmres(A, q + dt*load, Minv, self.iterations)
        return x


def make_linear(problem, mode, device):
    if mode == 'oseen':
        return OseenNS(problem, 'known', device)
    return (FourierLinear if problem.space == 'fourier' else FEMLinear)(problem, mode, device)


def true_components(problem, device):
    """Exact (O0, O1) of the problem in the same layout as the learned operators, for recovery metrics."""
    return make_linear(problem, 'known', device)
