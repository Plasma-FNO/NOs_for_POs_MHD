"""Order, mass handling, conservation, split consistency, and closure cost."""
import pytest
import torch
from torch import nn

from gos.operators import make_linear
from gos.problems import get_problem
from gos.research_models import ResearchModel, SCHEMES

NET = dict(modes=4, width=8, layers=2)


class ScalarMassLinear(nn.Module):
    def mass(self, x):
        return 2*x

    def mass_solve(self, x):
        return x/2

    def apply(self, x, mu):
        return -8*x

    def shifted_solve(self, rhs, mu, shift):
        return rhs/(2 + 8*shift)


@pytest.mark.parametrize('scheme', SCHEMES)
def test_nonlinear_mass_problem_is_second_order(scheme):
    model = ResearchModel(get_problem('burgers'), scheme, NET, 1., 'cpu', closure=False)
    model.linear = ScalarMassLinear()
    model.exact_load = lambda x, mu: 0.5*x*x + 0.3
    q, mu = torch.ones(1, 1, 1, dtype=torch.float64), torch.ones(1)
    model.substeps = 512
    reference = model.step(q, mu, 0.5)
    errors = []
    for n in (4, 8, 16):
        model.substeps = n
        errors.append(float((model.step(q, mu, 0.5)-reference).abs().max()))
    assert min(errors[0]/errors[1], errors[1]/errors[2]) > 3.6, errors


def test_ks_split_reproduces_true_rhs_and_removes_implicit_growth():
    problem = get_problem('ks')
    model = ResearchModel(problem, 'ars222', NET, 0.7, 'cpu', closure=False)
    q, mu = torch.randn(2, 1, problem.n), torch.ones(2)
    true = make_linear(problem, 'known', 'cpu')
    expected = true.apply(q, mu) + problem.nonlinear(q*0.7, mu)/0.7
    actual = model.linear.apply(q, mu) + model.exact_load(q, mu)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert model.linear.symbol(mu).real.max() <= 0
    assert model.growth_symbol.max() > 0


def test_midpoint_preserves_energy_for_pure_skew_linear_physics():
    problem = get_problem('kdv')
    model = ResearchModel(problem, 'midpoint', NET, 1., 'cpu', closure=False, substeps=4)
    model.exact_load = lambda x, mu: torch.zeros_like(x)
    x = model.linear.grid.coords[0]
    q = (torch.sin(2*torch.pi*x) + 0.4*torch.cos(6*torch.pi*x))[None, None]
    pred = model.step(q, torch.ones(1), 0.08)
    torch.testing.assert_close(pred.norm(), q.norm(), atol=2e-5, rtol=2e-6)


@pytest.mark.parametrize('name', ['burgers', 'kdv', 'ks', 'ns2d', 'burgers_fem'])
def test_zero_closure_starts_at_classical_physics_and_has_gradients(name):
    problem = get_problem(name)
    model = ResearchModel(problem, 'ars222', NET, 0.7, 'cpu', substeps=2)
    reference = ResearchModel(problem, 'ars222', NET, 0.7, 'cpu', substeps=2, closure=False)
    q = 0.01*torch.randn(2, 1, *model.linear.grid.shape)
    mu = torch.ones(2)
    pred = model.step(q, mu, problem.h)
    torch.testing.assert_close(pred, reference.step(q, mu, problem.h))
    pred.square().mean().backward()
    gradients = [p.grad for p in model.network_parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().max() > 0 for g in gradients)
    if problem.space == 'fourier':
        torch.testing.assert_close(pred.mean(model.linear.grid.dims), q.mean(model.linear.grid.dims),
                                   atol=2e-7, rtol=1e-4)


@pytest.mark.parametrize('each_stage, calls', [(False, 1), (True, 6)])
def test_closure_evaluation_budget(each_stage, calls):
    model = ResearchModel(get_problem('burgers'), 'ars222', NET, 1., 'cpu', substeps=3,
                          closure_each_stage=each_stage)
    seen = []
    hook = model.net.register_forward_hook(lambda *args: seen.append(1))
    model.step(torch.zeros(1, 1, 256), torch.ones(1), 0.05)
    hook.remove()
    assert len(seen) == calls


@pytest.mark.parametrize('substeps', [0, -1, 1.5, True])
def test_invalid_substeps_are_rejected(substeps):
    with pytest.raises(ValueError, match='positive integer'):
        ResearchModel(get_problem('burgers'), 'ars222', NET, 1., 'cpu', substeps=substeps)


def test_fem_factor_reuse_preserves_operator_gradients(monkeypatch):
    problem = get_problem('burgers_fem')
    model = ResearchModel(problem, 'ars222', NET, 1., 'cpu', substeps=3, linear_mode='struct')
    original, calls = torch.linalg.lu_factor, []
    def count(matrix):
        calls.append(1)
        return original(matrix)
    monkeypatch.setattr(torch.linalg, 'lu_factor', count)
    q = 0.1*torch.sin(torch.pi*model.linear.grid.x)[None, None]
    model.step(q, torch.ones(1), problem.h).square().mean().backward()
    assert len(calls) == 1
    gradients = [p.grad for p in model.operator_parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().max() > 0 for g in gradients)


def test_etd_reference_is_fourth_order_and_cache_tracks_coefficients():
    from gos.research_models import ETDReference
    problem = get_problem('burgers')
    model = ETDReference(problem, 1., 'cpu', substeps=64)
    x = problem.grid.coords[0]
    q = (torch.sin(2*torch.pi*x) + 0.3*torch.cos(4*torch.pi*x))[None, None]
    mu = torch.ones(1)
    reference = model.step(q, mu, 0.3)
    errors = []
    for n in (1, 2, 4):
        model.substeps = n
        errors.append(float((model.step(q, mu, 0.3)-reference).norm()))
    assert min(errors[0]/errors[1], errors[1]/errors[2]) > 10, errors
    changed = model.step(q, mu*2, 0.3)
    fresh = ETDReference(problem, 1., 'cpu', substeps=4).step(q, mu*2, 0.3)
    torch.testing.assert_close(changed, fresh)
    assert not list(model.parameters())


@pytest.mark.parametrize('scheme', [*SCHEMES, 'etdrk4'])
def test_capped_physics_matches_repeated_original_steps(scheme):
    from gos.research_models import ETDReference
    problem = get_problem('burgers')
    if scheme == 'etdrk4':
        model = ETDReference(problem, 1., 'cpu', substeps=3, max_step=True)
    else:
        model = ResearchModel(problem, scheme, NET, 1., 'cpu', substeps=3, max_step=True, closure=False)
    x = problem.grid.coords[0].float()
    q = torch.sin(2*torch.pi*x)[None, None]
    mu = torch.ones(1)
    large = model.step(q, mu, 4*problem.h)
    repeated = q
    for _ in range(4):
        repeated = model.step(repeated, mu, problem.h)
    torch.testing.assert_close(large, repeated, atol=3e-6, rtol=3e-6)


def test_capped_large_step_still_evaluates_frozen_closure_once():
    model = ResearchModel(get_problem('burgers'), 'ars222', NET, 1., 'cpu', substeps=2, max_step=True)
    seen = []
    hook = model.net.register_forward_hook(lambda *args: seen.append(1))
    model.step(torch.zeros(1, 1, 256), torch.ones(1), 4*model.h)
    hook.remove()
    assert len(seen) == 1


def test_known_fem_rollout_cache_preserves_outputs_and_gradients(monkeypatch):
    from gos.models import StepModel
    problem = get_problem('burgers_fem')
    model = ResearchModel(problem, 'ars222', NET, 1., 'cpu', substeps=2)
    original, calls = torch.linalg.lu_factor_ex, []
    def count(matrix, **kwargs):
        calls.append(1)
        return original(matrix, **kwargs)
    monkeypatch.setattr(torch.linalg, 'lu_factor_ex', count)
    q = (0.1*torch.sin(torch.pi*model.linear.grid.x))[None, None].requires_grad_()
    mu = torch.ones(1)
    cached = model(q, mu, 4)
    gradients = torch.autograd.grad(cached.square().sum(), [q, *model.network_parameters()], allow_unused=True)
    assert len(calls) == 1
    assert model._solve_cache is None
    calls.clear()
    uncached = StepModel.forward(model, q, mu, 4)
    reference_gradients = torch.autograd.grad(uncached.square().sum(), [q, *model.network_parameters()], allow_unused=True)
    assert len(calls) == 4
    torch.testing.assert_close(cached, uncached)
    for actual, expected in zip(gradients, reference_gradients):
        if actual is not None:
            torch.testing.assert_close(actual, expected)


def test_fem_rollout_cache_clears_after_exception(monkeypatch):
    model = ResearchModel(get_problem('burgers_fem'), 'ars222', NET, 1., 'cpu')
    def fail(*args):
        raise RuntimeError('intentional test failure')
    monkeypatch.setattr(model, 'step', fail)
    with pytest.raises(RuntimeError, match='intentional'):
        model(torch.zeros(1, 1, 127), torch.ones(1), 2)
    assert model._solve_cache is None
