"""The coefficient-support rule preserves ID predictions and restores exact physics OOD."""
import pytest
import torch

from gos.problems import get_problem
from gos.research_models import ResearchModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NET = dict(modes=4, width=8, layers=2)


def models():
    problem = get_problem('burgers_fem', DEVICE)
    kwargs = dict(substeps=2, zero_equilibrium=True)
    original = ResearchModel(problem, 'ars222', NET, 0.7, DEVICE, **kwargs).to(DEVICE)
    with torch.no_grad():
        original.net.project[-1].weight.normal_(std=0.1)
    gated = ResearchModel(problem, 'ars222', NET, 0.7, DEVICE, closure_in_range=True, **kwargs).to(DEVICE)
    gated.load_state_dict(original.state_dict())
    classical = ResearchModel(problem, 'ars222', NET, 0.7, DEVICE, substeps=2, closure=False).to(DEVICE)
    return original, gated, classical


def state(model, count):
    return (0.02*torch.sin(torch.pi*model.linear.grid.x))[None, None].expand(count, -1, -1).clone()


def test_inclusive_support_and_mixed_batch_predictions():
    original, gated, classical = models()
    lo, hi = (v/gated.problem.mu_ref for v in gated.problem.mu_train)
    mu = torch.tensor([lo/2, lo, 1., hi, hi*2], device=DEVICE)
    q = state(gated, len(mu))
    assert float(original.closure_load(q, mu).abs().max()) > 0
    with torch.no_grad():
        predicted = gated.step(q, mu, gated.h)
        expected = original.step(q, mu, original.h)
        physical = classical.step(q, mu, classical.h)
    torch.testing.assert_close(predicted[1:4], expected[1:4], rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(predicted[[0, 4]], physical[[0, 4]], rtol=0., atol=0.)
    assert all(torch.equal(value, original.state_dict()[name]) for name, value in gated.state_dict().items())


def test_all_outside_skips_network_including_immediately_outside_bounds():
    _, gated, classical = models()
    lo, hi = torch.tensor([v/gated.problem.mu_ref for v in gated.problem.mu_train], device=DEVICE)
    mu = torch.stack([torch.nextafter(lo, lo-1), torch.nextafter(hi, hi+1)])
    q = state(gated, len(mu))
    calls = []
    hook = gated.net.register_forward_hook(lambda *args: calls.append(1))
    try:
        predicted = gated.step(q, mu, gated.h)
    finally:
        hook.remove()
    assert calls == []
    torch.testing.assert_close(predicted, classical.step(q, mu, classical.h), rtol=0., atol=0.)


def test_all_inside_matches_original_and_has_finite_jacobian():
    original, gated, _ = models()
    q = state(gated, 2)
    mu = torch.tensor([0.7, 1.3], device=DEVICE)
    torch.testing.assert_close(gated.step(q, mu, gated.h), original.step(q, mu, original.h), rtol=0., atol=0.)
    _, tangent = torch.func.jvp(lambda x: gated.step(x, mu, gated.h), (q,), (torch.ones_like(q),))
    assert torch.isfinite(tangent).all()


def test_gate_requires_known_operator_for_physics_fallback():
    with pytest.raises(ValueError, match='known physical operator'):
        ResearchModel(get_problem('burgers_fem', DEVICE), 'ars222', NET, 0.7, DEVICE,
                      closure_in_range=True, linear_mode='free')
