"""Matrix-free Oseen derivatives must match the solved linear system."""
import torch

from gos.operators import OseenNS
from gos.problems import get_problem

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def setup_operator():
    problem = get_problem('ns2d', DEVICE)
    problem.n = 16
    operator = OseenNS(problem, 'known', DEVICE, iterations=20)
    operator.scale = 0.6
    operator.implicit_gradient = True
    torch.manual_seed(17)
    q = 0.3*torch.randn(2, 1, 16, 16, device=DEVICE, dtype=torch.float64)
    load = 0.1*torch.randn_like(q)
    mu = torch.tensor([0.8, 1.2], device=DEVICE, dtype=torch.float64)
    return operator, q, load, mu


def test_oseen_transpose_identity():
    op, q, load, mu = setup_operator()
    A, _ = op.system(q, mu, 0.5)
    AT, _ = op.system(q, mu, 0.5, transpose=True)
    torch.testing.assert_close((A(q)*load).sum(), (q*AT(load)).sum(), rtol=1e-12, atol=1e-12)


def test_oseen_implicit_jvp_and_vjp_match_finite_difference():
    op, q, load, mu = setup_operator()
    step = lambda q, load, mu: op.solve(q, load, mu, 0.5, 1.)
    primals = (q, load, mu)
    tangents = tuple(torch.randn_like(x) for x in primals)
    out, jvp = torch.func.jvp(step, primals, tangents)
    eps = 1e-5
    plus = step(*(x + eps*v for x, v in zip(primals, tangents)))
    minus = step(*(x - eps*v for x, v in zip(primals, tangents)))
    torch.testing.assert_close(jvp, (plus-minus)/(2*eps), rtol=1e-5, atol=1e-7)
    _, pullback = torch.func.vjp(step, *primals)
    weight = torch.randn_like(out)
    grads = pullback(weight)
    torch.testing.assert_close((weight*jvp).sum(), sum((a*b).sum() for a, b in zip(grads, tangents)),
                               rtol=1e-7, atol=1e-9)
    assert all(torch.isfinite(g).all() for g in grads)


def test_gmres_handles_zero_rhs_and_exact_breakdown():
    from gos.operators import gmres
    b = torch.randn(2, 1, 16, 16, device=DEVICE, dtype=torch.float64)
    b[0].zero_()
    x, residual = gmres(lambda x: x, b, lambda x: x, iterations=20)
    torch.testing.assert_close(x, b, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(residual).all()


def test_oseen_zero_tangent_stays_zero():
    op, q, load, mu = setup_operator()
    step = lambda q: op.solve(q, load, mu, 0.5, 1.)
    _, tangent = torch.func.jvp(step, (q,), (torch.zeros_like(q),))
    assert torch.equal(tangent, torch.zeros_like(q))
