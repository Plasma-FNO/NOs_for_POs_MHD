"""Unit tests: solve identities, the stability guarantee of the structured family, autograd/jvp paths."""
import math

import pytest
import torch

from gos.models import ARMS, StepModel
from gos.operators import make_linear
from gos.problems import FEM1D, get_problem, tridiag_pcr
from gos.train import CONFIG, step_jacobian_norm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)


def m_norm(lin, q):
    if lin.problem.space == 'fourier':
        return q.flatten(1).norm(dim=1)
    return (q.squeeze(1)*lin.mass(q).squeeze(1)).sum(-1).sqrt()


def randomise(lin, spread=3.):
    with torch.no_grad():
        for p in lin.parameters():
            p.copy_(spread*torch.randn_like(p))


@pytest.mark.parametrize('name', ['burgers', 'kdv', 'ns2d', 'burgers_fem'])
@pytest.mark.parametrize('theta', [0.5, 1.0])
def test_struct_is_non_expansive_for_any_step(name, theta):
    problem = get_problem(name, DEVICE)
    lin = make_linear(problem, 'struct', DEVICE).to(DEVICE)
    randomise(lin)
    q = torch.randn(4, 1, *lin.grid.shape, device=DEVICE)
    mu = torch.tensor([0.1, 1., 3., 10.], device=DEVICE)
    for dt in (1e-3, problem.h, 100*problem.h, 1e4*problem.h):
        out = lin.solve(q, torch.zeros_like(q), mu, dt, theta)
        assert torch.all(m_norm(lin, out) <= m_norm(lin, q)*(1+1e-4)), (name, dt)


def test_free_can_expand():
    problem = get_problem('burgers_fem', DEVICE)
    lin = make_linear(problem, 'free', DEVICE).to(DEVICE)
    randomise(lin)
    q = torch.randn(8, 1, *lin.grid.shape, device=DEVICE)
    out = lin.solve(q, torch.zeros_like(q), torch.ones(8, device=DEVICE), problem.h, 1.)
    assert torch.any(m_norm(lin, out) > m_norm(lin, q))


@pytest.mark.parametrize('name', ['burgers', 'kdv', 'ks', 'ns2d', 'burgers_fem'])
@pytest.mark.parametrize('mode', ['known', 'struct', 'free'])
def test_solve_satisfies_step_equation(name, mode):
    problem = get_problem(name, DEVICE)
    lin = make_linear(problem, mode, DEVICE).to(DEVICE)
    if mode != 'known':
        randomise(lin, 1.)
    q = torch.randn(3, 1, *lin.grid.shape, device=DEVICE, dtype=torch.float32)
    f = torch.randn_like(q)
    mu = torch.tensor([0.5, 1., 2.], device=DEVICE)
    theta, dt = 0.7, problem.h
    out = lin.solve(q, f, mu, dt, theta)
    lhs = lin.mass(out) - theta*dt*lin.apply(out, mu)
    rhs = lin.mass(q) + (1-theta)*dt*lin.apply(q, mu) + dt*f
    assert float((lhs-rhs).norm()/rhs.norm()) < 1e-3


def test_known_fem_matches_problem_bands():
    problem = get_problem('burgers_fem', DEVICE)
    lin = make_linear(problem, 'known', DEVICE).to(DEVICE)
    mu = torch.tensor([0.6, 1.7], device=DEVICE)
    exact = FEM1D.dense(problem.bands(mu.double()*problem.mu_ref)).float()
    assert torch.allclose(FEM1D.dense(lin.bands(mu)), exact, rtol=1e-5, atol=1e-6)
    S = exact + exact.transpose(-1, -2)
    assert torch.linalg.eigvalsh(S.double()).max() <= 1e-8  # true FEM operator is dissipative


def test_pcr_matches_dense():
    n = 37
    a, c = torch.randn(2, 5, n, dtype=torch.float64)
    b = 4 + torch.rand(5, n, dtype=torch.float64)
    a[:, 0] = 0
    c[:, -1] = 0
    d = torch.randn(5, n, dtype=torch.float64)
    x = tridiag_pcr(a, b, c, d)
    A = FEM1D.dense((a, b, c))
    assert torch.allclose(A @ x[..., None], d[..., None], atol=1e-10)


def test_fem_burgers_load_is_energy_neutral():
    g = FEM1D(64)
    u = torch.randn(4, g.n, dtype=torch.float64)
    assert torch.allclose((u*g.burgers_load(u)).sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-12)


@pytest.mark.parametrize('arm', sorted(ARMS))
@pytest.mark.parametrize('name', ['burgers', 'ns2d', 'burgers_fem'])
def test_arm_gradients_and_jacobian(arm, name):
    problem = get_problem(name, DEVICE)
    net = dict(CONFIG[name]['net'], modes=4, width=8, layers=2)
    model = StepModel(problem, arm, net, 0.3, DEVICE).to(DEVICE)
    grid = make_linear(problem, 'known', DEVICE).grid
    q = 0.1*torch.randn(2, 1, *grid.shape, device=DEVICE)
    mu = torch.tensor([0.8, 1.2], device=DEVICE)
    out = model(q, mu, 2)
    assert out.shape == (2, 2, 1, *grid.shape)
    params = list(model.parameters())
    if params:
        out.square().mean().backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    sigma = step_jacobian_norm(model, q, mu, iterations=3)
    assert sigma.shape == (2,)


def test_reference_converges_in_time():
    problem = get_problem('burgers', DEVICE)
    problem.frames = 5
    a, _ = problem.generate(2, torch.Generator().manual_seed(0), batch=2)
    problem.dt_ref = problem.dt_ref/2
    b, _ = problem.generate(2, torch.Generator().manual_seed(0), batch=2)
    assert float((a-b).norm()/b.norm()) < 1e-6


@pytest.mark.parametrize('name', ['burgers', 'burgers_fem'])
def test_tangent_free_remainder_has_no_linear_response(name):
    problem = get_problem(name, DEVICE)
    net = dict(CONFIG[name]['net'], modes=4, width=8, layers=2)
    model = StepModel(problem, 'implicit_free_tf', net, 0.3, DEVICE).to(DEVICE)
    torch.nn.init.normal_(model.net.project[-1].weight)
    grid = make_linear(problem, 'known', DEVICE).grid
    v = torch.randn(1, 1, *grid.shape, device=DEVICE)
    mu = torch.ones(1, device=DEVICE)
    base = model.load(torch.zeros_like(v), mu)
    r = [float((model.load(eps*v, mu) - base).norm()) for eps in (0.1, 0.2)]
    assert r[1]/r[0] > 3.0  # ~4 for a quadratic response, 2 for a linear one: the tangent at 0 is removed
