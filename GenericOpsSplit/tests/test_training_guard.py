"""A finite loss must not let nonfinite derivatives poison the optimizer."""
import pytest
import torch

from gos.train import train


@pytest.mark.skipif(not torch.cuda.is_available(), reason='All training checks require CUDA')
def test_nonfinite_gradient_does_not_update_parameters():
    class SingularModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((), device='cuda'))

        def network_parameters(self):
            return list(self.parameters())

        def operator_parameters(self):
            return []

        def forward(self, q, mu, steps, dt_mult=1):
            return self.weight.sqrt()*torch.ones(len(q), steps, *q.shape[1:], device=q.device)

    model = SingularModel()
    frames = torch.ones(2, 5, 1, 2, device='cuda')
    mu = torch.ones(2, device='cuda')
    data = {'train': (frames, mu), 'val': (frames, mu)}
    _, _, info = train(model, data, dict(iters=2, window=1, batch=2), lambda _: None)
    assert info['skipped_gradients'] == 2
    assert float(model.weight) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='All training checks require CUDA')
def test_resume_restores_optimizer_scheduler_and_rng_exactly(tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.9, device='cuda'))

        def network_parameters(self):
            return list(self.parameters())

        def operator_parameters(self):
            return []

        def forward(self, q, mu, steps, dt_mult=1):
            return torch.stack([q*self.weight**(i+1) for i in range(steps)], 1)

    frames = torch.arange(60, device='cuda').reshape(6, 5, 1, 2)/20+1
    mu = torch.ones(6, device='cuda')
    data = {'train': (frames, mu), 'val': (frames, mu)}
    config = dict(iters=8, window=1, batch=2, val_every=2)
    torch.manual_seed(23)
    full = TinyModel()
    _, full_history, _ = train(full, data, config, lambda _: None)
    expected_rng = torch.cuda.get_rng_state()
    torch.manual_seed(23)
    interrupted = TinyModel()
    path = tmp_path/'progress.pt'
    _, _, partial = train(interrupted, data, config, lambda _: None,
                           checkpoint_path=path, max_cpu_seconds=0)
    assert partial['status'] == 'checkpointed' and partial['iterations'] == 1
    resumed = TinyModel()
    _, resumed_history, finished = train(resumed, data, config, lambda _: None, checkpoint_path=path)
    assert finished['iterations'] == 8 and finished['resumes'] == 1
    assert torch.equal(full.weight, resumed.weight)
    assert full_history == resumed_history
    assert torch.equal(expected_rng, torch.cuda.get_rng_state())
