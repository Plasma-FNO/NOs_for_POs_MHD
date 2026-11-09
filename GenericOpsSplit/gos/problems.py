"""Benchmark PDEs written as semi-discrete systems  M dq/dt = L(mu) q + N(q; mu).

Every problem provides
  * the exact linear operator L(mu) on the model grid (Fourier symbol or FEM matrix), affine in mu,
  * the exact nonlinear remainder N (used only by the untrained `imex_exact` reference arm),
  * a reference generator on a finer grid (ETDRK4 or Crank-Nicolson/AB2 FEM), restricted to the model grid.

Data are generated finer than the model grid, so a model that knows L exactly still has a closure to
learn on top of the time-discretisation error, as in the native MHD case.
"""
import math

import torch

TWO_PI = 2*math.pi


# ----------------------------------------------------------------------------------------------- grids
class FourierGrid:
    """Periodic grid on [0, length)^dim with rfft layout; axis order (..., y, x)."""
    kind = 'fourier'

    def __init__(self, n, length, dim, device='cpu', dtype=torch.float64):
        self.n, self.length, self.dim = n, length, dim
        self.dims = tuple(range(-dim, 0))
        self.shape = (n,)*dim
        j_half = torch.fft.rfftfreq(n, d=1/n).to(device)
        j_full = torch.fft.fftfreq(n, d=1/n).to(device)
        scale = TWO_PI/length
        if dim == 1:
            self.ks = ((scale*j_half).to(dtype),)
            self.mask = j_half.abs() < n/3
        else:
            self.ks = ((scale*j_full)[:, None].to(dtype).expand(n, n//2+1), (scale*j_half)[None, :].to(dtype).expand(n, n//2+1))
            self.mask = (j_full.abs()[:, None] < n/3) & (j_half.abs()[None, :] < n/3)
        self.k2 = sum(k**2 for k in self.ks)
        self.spec_shape = tuple(self.k2.shape)
        self.coords = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=dtype, device=device)/n]*dim, indexing='ij'))

    def fft(self, u):
        return torch.fft.rfftn(u, dim=self.dims)

    def ifft(self, v):
        return torch.fft.irfftn(v, s=self.shape, dim=self.dims)


def hermitian(symbol, dim):
    """Project an rfft-layout multiplier onto symbols of real operators (self-conjugate columns)."""
    if dim == 1:
        out = symbol.clone()
        out[..., 0] = out[..., 0].real.to(out.dtype)
        if symbol.shape[-1] > 1:
            out[..., -1] = out[..., -1].real.to(out.dtype)
        return out
    n = symbol.shape[-2]
    flip = (-torch.arange(n, device=symbol.device)) % n
    out = symbol.clone()
    for c in (0, symbol.shape[-1]-1):
        col = symbol[..., :, c]
        out[..., :, c] = 0.5*(col + col[..., flip].conj())
    return out


def restrict_fourier(u, fine, coarse):
    """Spectral truncation of real fields from a fine to a coarse periodic grid."""
    v = fine.fft(u)
    m = coarse.n
    if fine.dim == 1:
        w = v[..., :m//2+1]
    else:
        w = torch.cat([v[..., :m//2, :m//2+1], v[..., -m//2:, :m//2+1]], dim=-2)
    w = w*(coarse.n/fine.n)**fine.dim
    return coarse.ifft(w)


# --------------------------------------------------------------------------------------- ETDRK4 core
def etdrk4_coefficients(symbol, dt, contour=32):
    """Kassam-Trefethen ETDRK4 weights for a (possibly complex) diagonal symbol, any batch shape."""
    r = torch.exp(1j*math.pi*(torch.arange(contour, device=symbol.device, dtype=torch.float64)+0.5)*2/contour)
    LR = dt*symbol[..., None] + r
    mean = lambda z: z.mean(-1)
    E, E2 = torch.exp(dt*symbol), torch.exp(dt*symbol/2)
    Q = dt*mean((torch.exp(LR/2)-1)/LR)
    f1 = dt*mean((-4-LR+torch.exp(LR)*(4-3*LR+LR**2))/LR**3)
    f2 = dt*mean((2+LR+torch.exp(LR)*(-2+LR))/LR**3)
    f3 = dt*mean((-4-3*LR-LR**2+torch.exp(LR)*(4-LR))/LR**3)
    if not torch.is_complex(symbol) or symbol.imag.abs().max() == 0:
        Q, f1, f2, f3 = (z.real.to(torch.complex128) for z in (Q, f1, f2, f3))
    return E, E2, Q, f1, f2, f3


def etdrk4(v, nonlinear, coefficients, steps):
    E, E2, Q, f1, f2, f3 = coefficients
    for _ in range(steps):
        Nv = nonlinear(v)
        a = E2*v + Q*Nv
        Na = nonlinear(a)
        b = E2*v + Q*Na
        Nb = nonlinear(b)
        c = E2*a + Q*(2*Nb-Nv)
        Nc = nonlinear(c)
        v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
    return v


# ------------------------------------------------------------------------------------------ problems
class Problem:
    """Base class. Coefficient mu is per trajectory; mu_tilde = mu/mu_ref enters L affinely."""
    name = None
    channels = 1
    needs_coords = False
    mu_name = 'mu'

    def __init__(self, device='cpu'):
        self.device = device

    def sample_mu(self, count, generator, ood=None):
        if ood is not None:
            return torch.full((count,), float(ood), dtype=torch.float64)
        lo, hi = self.mu_train
        u = torch.rand(count, generator=generator, dtype=torch.float64)
        return torch.exp(math.log(lo) + u*(math.log(hi)-math.log(lo)))

    def generate(self, count, generator, ood=None, batch=64):
        mus = self.sample_mu(count, generator, ood)
        frames = []
        for s in range(0, count, batch):
            frames.append(self._generate(mus[s:s+batch].to(self.device), generator).float().cpu())
        return torch.cat(frames), mus.float()


class FourierProblem(Problem):
    space = 'fourier'

    def __init__(self, device='cpu'):
        super().__init__(device)
        self.grid = FourierGrid(self.n, self.length, self.dim, device)
        self.fine = FourierGrid(self.n_ref, self.length, self.dim, device)

    def symbol(self, mu, grid=None):
        """Exact L(mu) symbol, complex128 [B, *spec]."""
        raise NotImplementedError

    def nonlinear_hat(self, v, mu, grid):
        raise NotImplementedError

    def nonlinear(self, q, mu):
        """Exact N(q) in physical space on the model grid (rate form)."""
        g = self.grid
        return g.ifft(self.nonlinear_hat(g.fft(q.double()), mu.double(), g)).to(q.dtype)

    def _generate(self, mu, generator):
        g = self.fine
        u0 = self.initial(len(mu), generator).to(self.device)
        v = g.fft(u0)*g.mask
        coeffs = etdrk4_coefficients(self.symbol(mu, g), self.dt_ref)
        nonlinear = lambda w: self.nonlinear_hat(w, mu, g)
        per_frame = round(self.h/self.dt_ref)
        if abs(per_frame*self.dt_ref - self.h) > 1e-12:
            raise ValueError('frame interval must be a multiple of the reference step')
        v = etdrk4(v, nonlinear, coeffs, round(getattr(self, 'burn_in', 0.)/self.dt_ref))
        out = []
        for i in range(self.frames):
            if i:
                v = etdrk4(v, nonlinear, coeffs, per_frame)
            out.append(restrict_fourier(g.ifft(v), g, self.grid))
        frames = torch.stack(out, 1)[:, :, None]
        if not torch.isfinite(frames).all():
            raise FloatingPointError(f'{self.name}: non-finite reference trajectory')
        return frames

    def _random_modes(self, count, generator, modes, wave, norm_range):
        x = self.fine.coords[0]*self.length
        u = torch.zeros(count, self.n_ref, dtype=torch.float64)
        for j in range(1, modes+1):
            a, b = torch.randn(2, count, 1, generator=generator, dtype=torch.float64)
            u = u + (a*torch.cos(wave*j*x.cpu()) + b*torch.sin(wave*j*x.cpu()))/j
        lo, hi = norm_range
        amp = lo + (hi-lo)*torch.rand(count, 1, generator=generator, dtype=torch.float64)
        return amp*u/u.abs().amax(-1, keepdim=True)


def _burgers_flux_hat(v, grid):
    u = grid.ifft(v)
    return -0.5j*grid.ks[0]*grid.fft(u*u)*grid.mask


class Burgers(FourierProblem):
    """u_t = nu u_xx - (u^2/2)_x on [0, 2pi): dissipative L, energy-conserving N (shocks)."""
    name, dim, n, n_ref, length = 'burgers', 1, 256, 1024, TWO_PI
    h, frames, dt_ref = 0.05, 201, 1e-3
    mu_name, mu_train, mu_ref, mu_ood = 'nu', (0.02, 0.06), 0.035, (0.012, 0.1)
    counts = dict(train=256, val=32, test=64, ood=32)

    def symbol(self, mu, grid=None):
        grid = grid or self.grid
        return (-mu.double()[:, None]*grid.k2[None]).to(torch.complex128)

    def nonlinear_hat(self, v, mu, grid):
        return _burgers_flux_hat(v, grid)

    def initial(self, count, generator):
        return self._random_modes(count, generator, 4, 1.0, (0.5, 1.0))


class KdV(FourierProblem):
    """u_t = -mu u_xxx - (u^2/2)_x on [0, 2) (Zabusky-Kruskal, mu = delta^2): purely skew L."""
    name, dim, n, n_ref, length = 'kdv', 1, 256, 512, 2.0
    h, frames, dt_ref = 0.02, 201, 2.5e-4
    mu_name, mu_train, mu_ref, mu_ood = 'delta2', (4.8e-4, 1.2e-3), 7.6e-4, (2.6e-4, 2.0e-3)
    counts = dict(train=256, val=32, test=64, ood=32)

    def symbol(self, mu, grid=None):
        grid = grid or self.grid
        return 1j*mu.double()[:, None]*grid.ks[0][None]**3

    def nonlinear_hat(self, v, mu, grid):
        return _burgers_flux_hat(v, grid)

    def initial(self, count, generator):
        return self._random_modes(count, generator, 3, math.pi, (0.5, 1.0))


class KuramotoSivashinsky(FourierProblem):
    """u_t = -u_xx - u_xxxx - (u^2/2)_x on [0, 32pi): L has a physically unstable band 0 < k < 1."""
    name, dim, n, n_ref, length = 'ks', 1, 128, 256, 32*math.pi
    h, frames, dt_ref, burn_in = 1.0, 201, 0.05, 150.
    mu_name, mu_train, mu_ref, mu_ood = 'unit', (1.0, 1.0), 1.0, ()
    counts = dict(train=256, val=32, test=64, ood=0)

    def symbol(self, mu, grid=None):
        grid = grid or self.grid
        k2 = grid.k2[None]
        return (mu.double()[:, None]*0 + k2 - k2**2).to(torch.complex128)

    def nonlinear_hat(self, v, mu, grid):
        return _burgers_flux_hat(v, grid)

    def initial(self, count, generator):
        return self._random_modes(count, generator, 6, TWO_PI/self.length*3, (0.5, 1.5))

    def sample_mu(self, count, generator, ood=None):
        return torch.ones(count, dtype=torch.float64)


class NavierStokes2D(FourierProblem):
    """Vorticity form on the unit torus with the Li et al. forcing: w_t = nu lap w - u.grad w + f."""
    name, dim, n, n_ref, length = 'ns2d', 2, 64, 128, 1.0
    h, frames, dt_ref, burn_in = 0.5, 81, 4e-3, 5.0  # skip the fast decay of the GRF initial condition
    mu_name, mu_train, mu_ref, mu_ood = 'nu', (1e-3, 4e-3), 2e-3, (6e-4, 7e-3)
    counts = dict(train=128, val=16, test=32, ood=16)

    def symbol(self, mu, grid=None):
        grid = grid or self.grid
        return (-mu.double()[:, None, None]*grid.k2[None]).to(torch.complex128)

    def forcing_hat(self, grid):
        cache = self.__dict__.setdefault('_forcing', {})
        if grid.n not in cache:
            y, x = grid.coords
            cache[grid.n] = grid.fft(0.1*(torch.sin(TWO_PI*(x+y)) + torch.cos(TWO_PI*(x+y))))
        return cache[grid.n]

    def nonlinear_hat(self, v, mu, grid):
        ky, kx = grid.ks
        k2 = grid.k2.clone()
        k2[0, 0] = 1.
        psi = v/k2
        u, w = grid.ifft(1j*ky*psi), grid.ifft(-1j*kx*psi)
        wx, wy = grid.ifft(1j*kx*v), grid.ifft(1j*ky*v)
        return -grid.fft(u*wx + w*wy)*grid.mask + self.forcing_hat(grid)

    def initial(self, count, generator):
        g = self.fine
        noise = torch.randn(count, g.n, g.n, generator=generator, dtype=torch.float64).to(self.device)
        sqrt_eig = 7**1.5*(g.k2 + 49)**(-1.25)  # Li et al.: N(0, 7^1.5 (-lap + 49)^-2.5)
        sqrt_eig[0, 0] = 0.
        w = g.ifft(g.fft(noise)*sqrt_eig*g.mask)
        return w/w.std(dim=(-2, -1), keepdim=True)


# -------------------------------------------------------------------------------------- 1D P1 FEM
def tridiag_pcr(a, b, c, d):
    """Batched parallel cyclic reduction for tridiagonal systems; a[...,0] = c[...,-1] = 0."""
    n = b.shape[-1]
    shift_down = lambda z, s, fill: torch.cat([torch.full_like(z[..., :s], fill), z[..., :-s]], -1)
    shift_up = lambda z, s, fill: torch.cat([z[..., s:], torch.full_like(z[..., :s], fill)], -1)
    s = 1
    while s < n:
        bm, bp = shift_down(b, s, 1.), shift_up(b, s, 1.)
        alpha, gamma = -a/bm, -c/bp
        a_new = alpha*shift_down(a, s, 0.)
        c_new = gamma*shift_up(c, s, 0.)
        b = b + alpha*shift_down(c, s, 0.) + gamma*shift_up(a, s, 0.)
        d = d + alpha*shift_down(d, s, 0.) + gamma*shift_up(d, s, 0.)
        a, c = a_new, c_new
        s *= 2
    return d/b


class FEM1D:
    """P1 elements on [0,1] with homogeneous Dirichlet ends; unknowns are the interior nodal values."""
    kind = 'fem'

    def __init__(self, elements, device='cpu', dtype=torch.float64):
        self.elements, self.n, self.dx = elements, elements-1, 1/elements
        self.x = torch.arange(1, elements, dtype=dtype, device=device)*self.dx
        self.xe = (torch.arange(elements, dtype=dtype, device=device)+0.5)*self.dx
        self.shape, self.dim = (self.n,), 1
        self.coords = self.x[None]
        self.mass_bands = (torch.full((self.n,), self.dx/6, dtype=dtype, device=device),
                           torch.full((self.n,), 2*self.dx/3, dtype=dtype, device=device),
                           torch.full((self.n,), self.dx/6, dtype=dtype, device=device))
        self.mass_bands[0][0] = 0.
        self.mass_bands[2][-1] = 0.

    def bands_from_elements(self, conductance, skew):
        """L = S - G^T diag(conductance) G as (sub, diag, super); conductance [.., E], skew [.., E-2]."""
        k = conductance
        diag = -(k[..., :-1] + k[..., 1:])
        upper = torch.cat([k[..., 1:-1] + skew, torch.zeros_like(k[..., :1])], -1)
        lower = torch.cat([torch.zeros_like(k[..., :1]), k[..., 1:-1] - skew], -1)
        return lower, diag, upper

    @staticmethod
    def dense(bands):
        lower, diag, upper = bands
        return torch.diag_embed(diag) + torch.diag_embed(upper[..., :-1], 1) + torch.diag_embed(lower[..., 1:], -1)

    @staticmethod
    def apply(bands, q):
        lower, diag, upper = bands
        pad = torch.zeros_like(q[..., :1])
        return diag*q + upper*torch.cat([q[..., 1:], pad], -1) + lower*torch.cat([pad, q[..., :-1]], -1)

    def burgers_load(self, u):
        """Galerkin load of -(u^2/2)_x with zero Dirichlet values; sum_i u_i N_i = 0 exactly."""
        pad = torch.zeros_like(u[..., :1])
        up, um = torch.cat([u[..., 1:], pad], -1), torch.cat([pad, u[..., :-1]], -1)
        return -(up*up + u*up - u*um - um*um)/6


class BurgersFEM(Problem):
    """u_t = (kappa(x) u_x)_x - a(x) u_x (skew form) - u u_x on [0,1], u = 0 at both ends, P1 FEM.

    Variable conductivity kappa = mu*(1 + 0.75 sin^2(3 pi x)) and advection a = 0.4 + 0.3 cos(2 pi x):
    a learned operator must discover spatially varying coefficients. Mass matrix M != I.
    """
    name, space, needs_coords, dim = 'burgers_fem', 'fem', True, 1
    elements, elements_ref = 128, 512
    h, frames, dt_ref = 0.01, 101, 2.5e-5
    mu_name, mu_train, mu_ref, mu_ood = 'kappa0', (4e-3, 1.2e-2), 7e-3, (2e-3, 2.4e-2)
    counts = dict(train=256, val=32, test=64, ood=32)

    def __init__(self, device='cpu'):
        super().__init__(device)
        self.grid = FEM1D(self.elements, device)
        self.fine = FEM1D(self.elements_ref, device)

    @staticmethod
    def kappa_shape(x):
        return 1 + 0.75*torch.sin(3*math.pi*x)**2

    @staticmethod
    def velocity(x):
        return 0.4 + 0.3*torch.cos(TWO_PI*x)

    def operator_parts(self, grid=None):
        """(diffusion bands at mu = 1, skew advection bands); L(mu) = mu*D + S."""
        grid = grid or self.grid
        k = self.kappa_shape(grid.xe)/grid.dx
        a = self.velocity(grid.x[:-1] + grid.dx/2)/2  # edge between interior nodes i and i+1
        zero_skew = torch.zeros_like(a)
        diffusion = grid.bands_from_elements(k, zero_skew)
        advection = grid.bands_from_elements(torch.zeros_like(k), -a)
        return diffusion, advection

    def bands(self, mu, grid=None):
        diffusion, advection = self.operator_parts(grid)
        mu = mu.double()[:, None]
        return tuple(mu*d[None] + s[None] for d, s in zip(diffusion, advection))

    def nonlinear(self, q, mu):
        return self.grid.burgers_load(q)

    def initial(self, count, generator):
        x = self.fine.x.cpu()
        u = torch.zeros(count, self.fine.n, dtype=torch.float64)
        for m in range(1, 6):
            u = u + torch.randn(count, 1, generator=generator, dtype=torch.float64)*torch.sin(m*math.pi*x)/m
        amp = 0.5 + 0.3*torch.rand(count, 1, generator=generator, dtype=torch.float64)
        return amp*u/u.abs().amax(-1, keepdim=True)

    def _generate(self, mu, generator):
        g, dt = self.fine, self.dt_ref
        u = self.initial(len(mu), generator).to(self.device)
        L = self.bands(mu, g)
        M = tuple(m[None].expand_as(l) for m, l in zip(g.mass_bands, L))
        lhs = tuple(m - dt/2*l for m, l in zip(M, L))
        per_frame = round(self.h/dt)
        out, previous = [u], None
        stride = self.elements_ref//self.elements
        for i in range(1, self.frames):
            for _ in range(per_frame):
                nl = g.burgers_load(u)
                explicit = nl if previous is None else 1.5*nl - 0.5*previous
                rhs = g.apply(M, u) + dt/2*g.apply(L, u) + dt*explicit
                previous, u = nl, tridiag_pcr(*lhs, rhs)
            out.append(u)
        frames = torch.stack(out, 1)[:, :, None, stride-1::stride]
        if frames.shape[-1] != self.grid.n or not torch.isfinite(frames).all():
            raise FloatingPointError('burgers_fem: bad reference trajectory')
        return frames


PROBLEMS = {p.name: p for p in (Burgers, KdV, KuramotoSivashinsky, NavierStokes2D, BurgersFEM)}


def get_problem(name, device='cpu'):
    return PROBLEMS[name](device)
