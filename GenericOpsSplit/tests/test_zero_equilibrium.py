"""The optional learned closure preserves the known FEM zero equilibrium."""
import pytest
import torch

from gos.problems import get_problem
from gos.research_models import ResearchModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NET = dict(modes=4, width=8, layers=2)


def test_fem_zero_equilibrium_survives_nonzero_network_and_has_gradients():
    problem = get_problem('burgers_fem', DEVICE)
    model = ResearchModel(problem, 'ars222', NET, 0.7, DEVICE, substeps=2,
                          zero_equilibrium=True).to(DEVICE)
    with torch.no_grad():
        for parameter in model.network_parameters():
            parameter.normal_(std=0.1)
    q = torch.zeros(2, 1, problem.grid.n, device=DEVICE)
    mu = torch.tensor([0.5, 2.], device=DEVICE)
    assert float(model.remainder_increment(q, mu).abs().max()) > 1e-5
    calls = []
    hook = model.net.register_forward_hook(lambda *args: calls.append(1))
    stationary = model.step(q, mu, problem.h)
    hook.remove()
    torch.testing.assert_close(stationary, q, rtol=0., atol=0.)
    assert len(calls) == 2
    q = (0.1*torch.sin(torch.pi*model.linear.grid.x)).repeat(2, 1, 1).requires_grad_()
    model.step(q, mu, problem.h).square().mean().backward()
    gradients = [p.grad for p in model.network_parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().max() > 0 for g in gradients)
    assert torch.isfinite(q.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='All training checks require CUDA')
def test_equilibrium_fem_complex_optimizer_resume_is_exact(tmp_path):
    from gos.train import train
    problem = get_problem('burgers_fem', 'cuda')
    x = problem.grid.x.float()
    q = torch.stack([(0.1 + 0.02*i)*torch.sin(torch.pi*x) for i in range(4)])[:, None]
    frames = torch.stack([q*0.97**i for i in range(5)], 1)
    mu = torch.tensor([0.7, 0.9, 1.1, 1.3], device='cuda')
    data = {'train': (frames, mu), 'val': (frames, mu)}
    config = dict(iters=4, window=1, batch=2, val_every=1, lr=1e-4, lr_min=1e-7)
    make = lambda: ResearchModel(problem, 'ars222', NET, 0.7, 'cuda', substeps=2,
                                 zero_equilibrium=True).cuda()
    torch.manual_seed(32)
    full = make()
    _, expected_history, _ = train(full, data, config, lambda _: None)
    expected_rng = torch.cuda.get_rng_state()
    torch.manual_seed(32)
    interrupted = make()
    checkpoint = tmp_path/'training_state.pt'
    _, _, partial = train(interrupted, data, config, lambda _: None,
                           checkpoint_path=checkpoint, max_cpu_seconds=0)
    assert partial['iterations'] == 1 and partial['status'] == 'checkpointed'
    resumed = make()
    _, actual_history, info = train(resumed, data, config, lambda _: None, checkpoint_path=checkpoint)
    assert info['iterations'] == 4 and info['resumes'] == 1
    assert any(torch.is_complex(p) for p in resumed.parameters())
    for name, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name
    assert actual_history == expected_history
    assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
