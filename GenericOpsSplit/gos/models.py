"""FNO backbone and the arms of the generic step family.

All arms share one backbone G_theta(q, log mu_t[, x]). With h the data interval and dt the model step:

  ar                q' = G(q)                                             (only dt = h)
  node              q' = q + dt G(q)/h
  explicit_known    q' = q + dt M^-1 (L q + M G(q)/h)                     (section 4.1.1, explicit Euler)
  imex_exact        (M - dt L) q' = M q + dt N_exact(q)                    (no learning: classical IMEX)
  implicit_known    (M - dt L) q' = M q + dt M G(q)/h                      (MAST-U analogue, known L)
  implicit_known_cn theta = 1/2 version of the above
  implicit_struct   L learned as J - D (dissipative by construction)       (section 4.1.2 target)
  implicit_free     L learned, same sparsity, no sign constraint           (structure ablation)
  *_tf              remainder with its tangent at q* = 0 removed, N = G(q) - DG(0)q (the native split's
                    definition of N), so the learned L is forced to carry the linear response
  ros2_*            (M - g dt L) k1 = L q + N(q);  (M - g dt L) k2 = L(q + dt k1) + N(q + dt k1) - 2 M k1;
                    q' = q + dt (3 k1 + k2)/2,  g = 1 + 1/sqrt(2)   (Verwer et al. 1999)
  node_rk2          explicit Heun with the learned rate
  oseen*            (I - dt [nu lap - u(q).grad]) q' = q + dt N(q): implicit linearised advection (ns2d)
"""
import torch
from torch import nn
import torch.nn.functional as F

from .operators import make_linear, OseenNS

ARMS = {
    'ar': dict(kind='ar'),
    'node': dict(kind='node'),
    'explicit_known': dict(kind='explicit', linear='known'),
    'imex_exact': dict(kind='implicit', linear='known', exact=True),
    'implicit_known': dict(kind='implicit', linear='known'),
    'implicit_known_cn': dict(kind='implicit', linear='known', theta=0.5),
    'implicit_struct': dict(kind='implicit', linear='struct'),
    'implicit_free': dict(kind='implicit', linear='free'),
    # tangent-free remainder N(q) = G(q) - DG(0) q: all linear response about q* = 0 must live in L
    'implicit_struct_tf': dict(kind='implicit', linear='struct', tangent_free=True),
    'implicit_free_tf': dict(kind='implicit', linear='free', tangent_free=True),
    # second-order: two-stage linearly implicit Rosenbrock-W (ROS2, order 2 for any W) and explicit Heun
    'ros2_known': dict(kind='ros2', linear='known'),
    'ros2_free': dict(kind='ros2', linear='free'),
    'node_rk2': dict(kind='node_rk2'),
    # 2D NS only: advection re-linearised every step and treated implicitly (GMRES)
    'oseen_exact': dict(kind='implicit', linear='oseen', exact=True),
    'oseen': dict(kind='implicit', linear='oseen'),
}
ROS2_GAMMA = 1 + 2**-0.5


class SpectralConv(nn.Module):
    def __init__(self, width, modes, dim):
        super().__init__()
        self.dim, self.modes = dim, modes
        self.weight = nn.Parameter(torch.randn(dim, width, width, *([modes]*dim), dtype=torch.cfloat)/(width*width))

    def forward(self, x):
        m = self.modes
        if self.dim == 1:
            X = torch.fft.rfft(x)
            out = torch.zeros_like(X)
            out[..., :m] = torch.einsum('bix,iox->box', X[..., :m], self.weight[0])
            return torch.fft.irfft(out, n=x.shape[-1])
        X = torch.fft.rfft2(x)
        out = torch.zeros_like(X)
        out[..., :m, :m] = torch.einsum('bixy,ioxy->boxy', X[..., :m, :m], self.weight[0])
        out[..., -m:, :m] = torch.einsum('bixy,ioxy->boxy', X[..., -m:, :m], self.weight[1])
        return torch.fft.irfft2(out, s=x.shape[-2:])


class FNO(nn.Module):
    def __init__(self, cin, cout, dim, modes, width, layers, pad=0):
        super().__init__()
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        self.dim, self.pad = dim, pad
        self.lift = conv(cin, width, 1)
        self.spectral = nn.ModuleList(SpectralConv(width, modes, dim) for _ in range(layers))
        self.pointwise = nn.ModuleList(conv(width, width, 1) for _ in range(layers))
        self.project = nn.Sequential(conv(width, 128, 1), nn.GELU(), conv(128, cout, 1))

    def forward(self, x):
        x = self.lift(x)
        if self.pad:
            x = F.pad(x, (0, self.pad)*self.dim)
        for i, (s, p) in enumerate(zip(self.spectral, self.pointwise)):
            x = s(x) + p(x)
            if i < len(self.spectral)-1:
                x = F.gelu(x)
        if self.pad:
            x = x[..., :-self.pad] if self.dim == 1 else x[..., :-self.pad, :-self.pad]
        return self.project(x)


class StepModel(nn.Module):
    def __init__(self, problem, arm, net_config, scale, device):
        super().__init__()
        spec = ARMS[arm]
        self.problem, self.arm, self.kind = problem, arm, spec['kind']
        self.theta, self.exact, self.h, self.scale = spec.get('theta', 1.), spec.get('exact', False), problem.h, scale
        self.tangent_free = spec.get('tangent_free', False)
        self.linear = make_linear(problem, spec['linear'], device) if 'linear' in spec else None
        if spec.get('linear') == 'oseen':
            self.linear.scale = scale
        grid = self.linear.grid if self.linear is not None else make_linear(problem, 'known', device).grid
        self.register_buffer('coords', grid.coords.float() if problem.needs_coords else torch.zeros(0, *grid.shape))
        if not self.exact:
            c = problem.channels
            cin = c + 1 + self.coords.shape[0]
            self.net = FNO(cin, c, problem.dim, **net_config)
            if self.kind != 'ar':
                nn.init.zeros_(self.net.project[-1].weight)
                nn.init.zeros_(self.net.project[-1].bias)

    def operator_parameters(self):
        return list(self.linear.parameters()) if self.linear is not None else []

    def network_parameters(self):
        return list(self.net.parameters()) if not self.exact else []

    def backbone(self, q, mu_t):
        b = q.shape[0]
        extra = [torch.log(mu_t).view(b, 1, *([1]*(q.ndim-2))).expand(b, 1, *q.shape[2:])]
        if self.coords.shape[0]:
            extra.append(self.coords[None].expand(b, *self.coords.shape))
        return self.net(torch.cat([q] + extra, 1))

    def exact_load(self, q, mu_t):
        """Exact remainder N(q) in load form, normalised units (for oseen: the forcing only)."""
        if self.kind == 'implicit' and isinstance(self.linear, OseenNS):
            g = self.problem.grid
            f = g.ifft(self.problem.forcing_hat(g)).to(q.dtype)/self.scale
            return f.expand_as(q)
        mu = mu_t.double()*self.problem.mu_ref
        return self.problem.nonlinear((q*self.scale).double(), mu).to(q.dtype)/self.scale

    def remainder_increment(self, q, mu_t):
        """Network remainder as an increment over h in nodal units (load = M g / h)."""
        g = self.backbone(q, mu_t)
        if self.tangent_free:
            _, tangent = torch.func.jvp(lambda x: self.backbone(x, mu_t), (torch.zeros_like(q),), (q,))
            g = g - tangent
        return g

    def exact_increment(self, q, mu_t):
        """Label for remainder_increment: h M^-1 N_exact(q)."""
        return self.h*self.linear.mass_solve(self.exact_load(q, mu_t))

    def load(self, q, mu_t):
        """Explicit forcing in M-weighted (load) form, normalised units."""
        if self.exact:
            return self.exact_load(q, mu_t)
        g = self.remainder_increment(q, mu_t)
        return self.linear.mass(g/self.h) if self.linear is not None else g/self.h

    def init_operator_from_known(self, device):
        self.linear.init_from(make_linear(self.problem, 'known', device))

    def step(self, q, mu_t, dt):
        if self.kind == 'ar':
            if abs(dt - self.h) > 1e-12:
                raise ValueError('AR is defined only at the data interval')
            return self.backbone(q, mu_t)
        if self.kind == 'node':
            return q + dt*self.backbone(q, mu_t)/self.h
        if self.kind == 'node_rk2':
            k1 = self.backbone(q, mu_t)/self.h
            return q + dt/2*(k1 + self.backbone(q + dt*k1, mu_t)/self.h)
        if self.kind == 'ros2':
            lin, shift = self.linear, ROS2_GAMMA*dt
            k1 = lin.shifted_solve(lin.apply(q, mu_t) + self.load(q, mu_t), mu_t, shift)
            q1 = q + dt*k1
            k2 = lin.shifted_solve(lin.apply(q1, mu_t) + self.load(q1, mu_t) - 2*lin.mass(k1), mu_t, shift)
            return q + dt*(1.5*k1 + 0.5*k2)
        load = self.load(q, mu_t)
        if self.kind == 'explicit':
            return q + dt*self.linear.mass_solve(self.linear.apply(q, mu_t) + load)
        return self.linear.solve(q, load, mu_t, dt, self.theta)

    def forward(self, q, mu_t, steps, dt_mult=1, substeps=1):
        dt = self.h*dt_mult/substeps
        out = []
        for _ in range(steps):
            for _ in range(substeps):
                q = self.step(q, mu_t, dt)
            out.append(q)
        return torch.stack(out, 1)
