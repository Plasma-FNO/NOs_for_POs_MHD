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


def make_linear(problem, mode, device):
    return (FourierLinear if problem.space == 'fourier' else FEMLinear)(problem, mode, device)


def true_components(problem, device):
    """Exact (O0, O1) of the problem in the same layout as the learned operators, for recovery metrics."""
    return make_linear(problem, 'known', device)
