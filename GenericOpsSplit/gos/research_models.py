"""Physics-anchored implicit steps with a learned defect correction.

The nonlinear physics is evaluated at every Runge--Kutta stage. By default the
network is evaluated once per requested step and its load is held fixed through
all physics stages/substeps. The classical method is second order; a nonzero
frozen learned correction does not retain that formal order as dt changes.
"""
import math

import torch

from .models import StepModel
from .operators import FEMLinear, FourierLinear, make_linear
from .problems import FEM1D, etdrk4, etdrk4_coefficients

SCHEMES = ('ars222', 'midpoint', 'ros2_low')
GAMMA = 1 - 2**-0.5


class StableFourierLinear(FourierLinear):
    """Keep KS's unstable band explicit and preserve the periodic mean."""

    def _project(self, symbol):
        if self.problem.name == 'ks' and self.mode == 'known':
            symbol = torch.complex(symbol.real.clamp(max=0), symbol.imag)
        symbol = symbol.clone()
        if self.grid.dim == 1:
            symbol[..., 0] = 0
        else:
            symbol[..., 0, 0] = 0
        return symbol

    def components(self):
        return tuple(self._project(s) for s in super().components())

    def symbol(self, mu_t):
        return self._project(super().symbol(mu_t))


class ResearchModel(StepModel):
    """Exact physics plus zero-initialized FNO closure, with internal substeps.

    ``substeps`` counts physics steps per requested macro step, including at
    transferred dt. ``max_step=True`` instead caps each physics step at
    h/substeps, increasing physics work proportionally for larger requested dt.
    ``closure_each_stage=True`` restores the RK order conditions
    for the complete learned RHS at the cost of two network calls per substep.
    ``zero_equilibrium=True`` subtracts the network's zero-state output, doubling
    neural evaluations while making the learned correction vanish at q = 0.
    ``closure_in_range=True`` uses the learned defect only inside the inclusive
    training coefficient interval. Outside it the known classical step remains;
    this hard support rule is generally discontinuous at coefficient boundaries.
    """

    def __init__(self, problem, scheme, net_config, scale, device, substeps=1,
                 closure=True, linear_mode='known', closure_each_stage=False, max_step=False,
                 zero_equilibrium=False, closure_in_range=False):
        if scheme not in SCHEMES:
            raise ValueError(f'Unknown research scheme: {scheme}')
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError('substeps must be a positive integer')
        if linear_mode not in ('known', 'struct', 'free'):
            raise ValueError('linear_mode must be known, struct, or free')
        if closure_in_range and linear_mode != 'known':
            raise ValueError('Coefficient-gated closure requires the known physical operator')
        arm = 'implicit_' + linear_mode + ('_forced' if problem.name == 'ns2d' else '')
        super().__init__(problem, arm, net_config, scale, device)
        self.scheme, self.substeps = scheme, substeps
        self.closure_each_stage = closure_each_stage
        self.zero_equilibrium = zero_equilibrium
        self.closure_in_range = closure_in_range
        self.max_step = max_step
        self._solve_cache = None
        self.exact = not closure
        if not closure:
            del self.net
        if problem.space == 'fourier':
            self.linear = StableFourierLinear(problem, linear_mode, device)
            if linear_mode != 'known':
                self.linear.init_from(StableFourierLinear(problem, 'known', device))
        elif linear_mode != 'known':
            self.linear.init_from(make_linear(problem, 'known', device))
        if problem.name == 'ks':
            one = torch.ones(1, device=device)
            growth = problem.symbol(one*problem.mu_ref, self.linear.grid).real.clamp(min=0).float()
            self.register_buffer('growth_symbol', growth)

    def exact_load(self, q, mu_t):
        load = super().exact_load(q, mu_t)
        if self.problem.name == 'ks':
            grid = self.linear.grid
            load = load + grid.ifft(self.growth_symbol.unsqueeze(1)*grid.fft(q))
        return load

    def closure_load(self, q, mu_t):
        if self.exact:
            return torch.zeros_like(q)
        keep = None
        if self.closure_in_range:
            lo, hi = (value/self.problem.mu_ref for value in self.problem.mu_train)
            keep = (mu_t >= lo) & (mu_t <= hi)
            if not keep.any():
                return torch.zeros_like(q)
        increment = self.remainder_increment(q, mu_t)
        if self.zero_equilibrium:
            increment = increment - self.remainder_increment(torch.zeros_like(q), mu_t)
        if keep is not None:
            # Retain the batch layout so in-support neural predictions are unchanged.
            increment = torch.where(keep.view(-1, *([1]*(q.ndim-1))), increment, torch.zeros_like(increment))
        if self.problem.space == 'fourier':
            increment = increment - increment.mean(self.linear.grid.dims, keepdim=True)
        return self.linear.mass(increment/self.h)

    def load(self, q, mu_t):
        return self.exact_load(q, mu_t) + self.closure_load(q, mu_t)

    def step(self, q, mu_t, dt):
        if dt <= 0:
            raise ValueError('dt must be positive')
        count = max(1, math.ceil(self.substeps*dt/self.h - 1e-12)) if self.max_step else self.substeps
        lin, h = self.linear, dt/count
        shift = (0.5 if self.scheme == 'midpoint' else GAMMA)*h
        if isinstance(lin, FEMLinear):
            # All stages share the same matrix, including when its entries are learned.
            matrix = lin.mass_dense[None] - shift*FEM1D.dense(lin.bands(mu_t))
            if lin.mode == 'known' and not mu_t.requires_grad:
                cached = self._solve_cache.get(shift) if self._solve_cache is not None else None
                if cached is None:
                    # Known diffusion plus skew advection makes M-shift*L invertible.
                    factors, pivots, _ = torch.linalg.lu_factor_ex(matrix, check_errors=False)
                    if self._solve_cache is not None:
                        self._solve_cache[shift] = (factors, pivots)
                else:
                    factors, pivots = cached
            else:
                factors, pivots = torch.linalg.lu_factor(matrix)
            def solve(rhs):
                return torch.linalg.lu_solve(factors, pivots, rhs.squeeze(1).unsqueeze(-1)).squeeze(-1).unsqueeze(1)
        else:
            solve = lambda rhs: lin.shifted_solve(rhs, mu_t, shift)
        frozen = self.closure_load(q, mu_t) if not self.closure_each_stage else None
        def load(x):
            correction = self.closure_load(x, mu_t) if frozen is None else frozen
            return self.exact_load(x, mu_t) + correction
        for _ in range(count):
            f0 = load(q)
            if self.scheme == 'midpoint':
                mid = solve(lin.mass(q) + 0.5*h*f0)
                q = q + h*lin.mass_solve(lin.apply(mid, mu_t) + load(mid))
            elif self.scheme == 'ars222':
                stage = solve(lin.mass(q) + GAMMA*h*f0)
                delta = 1 - 1/(2*GAMMA)
                rhs = lin.mass(q) + h*((1-GAMMA)*lin.apply(stage, mu_t)
                                      + delta*f0 + (1-delta)*load(stage))
                q = solve(rhs)
            else:
                k1 = solve(lin.apply(q, mu_t) + f0)
                stage = q + h*k1
                k2 = solve(lin.apply(stage, mu_t) + load(stage) - 2*lin.mass(k1))
                q = q + h*(1.5*k1 + 0.5*k2)
        return q


    def forward(self, q, mu_t, steps, dt_mult=1, substeps=1):
        # Reuse known factors only inside this rollout; never retain an autograd graph.
        previous = self._solve_cache
        if isinstance(self.linear, FEMLinear) and self.linear.mode == 'known' and not mu_t.requires_grad:
            self._solve_cache = {}
        try:
            return super().forward(q, mu_t, steps, dt_mult, substeps)
        finally:
            self._solve_cache = previous


class ETDReference(StepModel):
    """Untrained Fourier ETDRK4 reference; exponential, not an implicit method.

    Coefficients are cached for identical coefficients and step size. State and
    nonlinear evaluation use float64 to match the fine-grid reference generator;
    the returned state retains the caller's dtype. This model has no parameters.
    """

    def __init__(self, problem, scale, device, substeps=1, max_step=False):
        if problem.space != 'fourier':
            raise ValueError('ETDReference requires a Fourier problem')
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError('substeps must be a positive integer')
        super().__init__(problem, 'imex_exact', {}, scale, device)
        self.substeps, self.max_step = substeps, max_step
        self.closure_each_stage = False
        self._coefficient_cache = {}

    @torch.no_grad()
    def step(self, q, mu_t, dt):
        if dt <= 0:
            raise ValueError('dt must be positive')
        count = max(1, math.ceil(self.substeps*dt/self.h - 1e-12)) if self.max_step else self.substeps
        h = dt/count
        cached = self._coefficient_cache.get(h)
        if cached is None or not torch.equal(cached[0], mu_t):
            symbol = self.problem.symbol(mu_t.double()*self.problem.mu_ref)
            coefficients = tuple(c.unsqueeze(1) for c in etdrk4_coefficients(symbol, h))
            self._coefficient_cache[h] = (mu_t.detach().clone(), coefficients)
        else:
            coefficients = cached[1]
        grid, dtype = self.problem.grid, q.dtype
        nonlinear = lambda v: grid.fft(self.exact_load(grid.ifft(v), mu_t))
        v = etdrk4(grid.fft(q.double()), nonlinear, coefficients, count)
        return grid.ifft(v).to(dtype)
