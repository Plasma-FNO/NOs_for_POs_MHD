"""Fixed forcing must be visible to learned NS2D maps and retained in the residual baseline."""
import pytest
import torch

from gos.models import StepModel
from gos.problems import get_problem

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NET = dict(modes=4, width=8, layers=2)


@pytest.mark.parametrize('arm', ['ar', 'node', 'implicit_known', 'implicit_struct', 'implicit_free'])
def test_forcing_feature_is_known_spatial_input(arm):
    problem = get_problem('ns2d', DEVICE)
    model = StepModel(problem, arm + '_forced', NET, 0.6, DEVICE).to(DEVICE)
    q = torch.zeros(2, 1, problem.n, problem.n, device=DEVICE)
    captured = []
    hook = model.net.lift.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    model.backbone(q, torch.ones(2, device=DEVICE))
    hook.remove()
    expected = problem.grid.ifft(problem.forcing_hat(problem.grid)).float()/0.6
    assert captured[0].shape[1] == 3
    assert torch.allclose(captured[0][:, -1], expected.expand(2, -1, -1))
    assert float(captured[0][:, -1].std()) > 0


def test_oseen_residual_starts_at_exact_physics():
    problem = get_problem('ns2d', DEVICE)
    model = StepModel(problem, 'oseen_residual', NET, 0.6, DEVICE).to(DEVICE)
    exact = StepModel(problem, 'oseen_exact', NET, 0.6, DEVICE).to(DEVICE)
    q = problem.initial(2, torch.Generator().manual_seed(0))[:, None, ::2, ::2].float().to(DEVICE)
    mu = torch.ones(2, device=DEVICE)
    with torch.no_grad():
        actual = model.step(q, mu, problem.h)
        expected = exact.step(q, mu, problem.h)
    torch.testing.assert_close(actual, expected)
    assert float(model.linear.last_residual.max()) < 1e-4
    assert torch.equal(model.remainder_increment(q, mu), torch.zeros_like(q))
    assert torch.equal(model.exact_increment(q, mu), torch.zeros_like(q))
