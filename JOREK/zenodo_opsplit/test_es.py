import json

import pytest
import torch

from model import SplitFNO
from test_workflow import grid, make_h5
from train_es import CachedWindows, range_statistics, relative_lp, sequence_loss


def test_ar_and_node_are_different_deployments_of_same_network():
    r, z = grid()
    ar = SplitFNO(r, z, width=4, modes=4, layers=1, monolithic=True, autoregressive=True, paper_architecture=True)
    node = SplitFNO(r, z, width=4, modes=4, layers=1, monolithic=True, paper_architecture=True)
    node.load_state_dict(ar.state_dict())
    x = torch.randn(2, 4, 12, 12)
    torch.testing.assert_close(node(x), x + ar(x))
    clone = SplitFNO(r, z, **ar.config)
    clone.load_state_dict(ar.state_dict())
    torch.testing.assert_close(clone(x), ar(x))


def test_range_fit_and_five_step_loss():
    values = torch.arange(2*7*4*3*3, dtype=torch.float32).reshape(2, 7, 4, 3, 3)
    mean, std = range_statistics(values)
    encoded = (values-mean[None, None, :, None, None])/std[None, None, :, None, None]
    torch.testing.assert_close(encoded.amin((0, 1, 3, 4)), -torch.ones(4))
    torch.testing.assert_close(encoded.amax((0, 1, 3, 4)), torch.ones(4))
    dataset = CachedWindows(encoded, 5)
    assert len(dataset) == 4 and dataset[0].shape[0] == 6
    frames = torch.arange(1, 7, dtype=torch.float32).reshape(1, 6, 1, 1, 1).expand(2, -1, -1, -1, -1)
    torch.testing.assert_close(sequence_loss(torch.nn.Identity(), frames),
                               sum(relative_lp(frames[:, 0], frames[:, t]) for t in range(1, 6)))
    assert relative_lp(torch.ones(2, 4), torch.zeros(2, 4)).isfinite()


def fixture_inputs(tmp_path, epochs=1):
    r, z = grid()
    entries = []
    for split in ('train', 'val', 'test'):
        row = make_h5(tmp_path/f'{split}.h5')
        row.update(split=split, r=r.tolist(), z=z.tolist())
        entries.append(row)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(entries=entries)))
    from pathlib import Path
    cfg = json.loads(Path(__file__).with_name('es_paper_protocol.json').read_text())
    cfg.update(epochs=epochs, train_trajectories=1, train_frames=5, batch_size=2,
               horizon=2, width=4, modes=4, layers=1, eval_horizon=4, scheduler_step=1)
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps(cfg))
    return manifest, protocol


@pytest.mark.parametrize('method', ['ar', 'node', 'opsplit'])
def test_es_train_checkpoint_and_evaluation(tmp_path, monkeypatch, method):
    import train_es
    manifest, protocol = fixture_inputs(tmp_path)
    output = tmp_path/method
    command = ['train_es.py', '--manifest', str(manifest), '--protocol', str(protocol),
               '--output', str(output), '--method', method, '--device', 'cpu', '--split-formulation', 'effective']
    monkeypatch.setattr('sys.argv', command)
    train_es.main()
    metrics = json.loads((output/'test_metrics.json').read_text())
    assert metrics['completed'] == 1 and metrics['best_epoch'] == 0
    saved = torch.load(output/'last.pt', weights_only=False)
    assert saved['scheduler']['last_epoch'] == 1
    assert saved['optimizer']['param_groups'][0]['lr'] == .0005
    monkeypatch.setattr('sys.argv', command+['--resume'])
    train_es.main()
    assert len((output/'history.jsonl').read_text().splitlines()) == 1


def test_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    import train_es
    manifest, protocol = fixture_inputs(tmp_path, epochs=2)
    base = ['train_es.py', '--manifest', str(manifest), '--protocol', str(protocol), '--method', 'node', '--device', 'cpu']
    monkeypatch.setattr('sys.argv', base+['--output', str(tmp_path/'continuous')])
    train_es.main()
    save = train_es.save_checkpoint
    def interrupt_after_epoch(path, payload):
        save(path, payload)
        if path.name == 'last.pt' and payload['epoch'] == 0:
            raise InterruptedError('simulated allocation loss')
    with monkeypatch.context() as patch:
        patch.setattr(train_es, 'save_checkpoint', interrupt_after_epoch)
        patch.setattr('sys.argv', base+['--output', str(tmp_path/'resumed')])
        with pytest.raises(InterruptedError):
            train_es.main()
    monkeypatch.setattr('sys.argv', base+['--output', str(tmp_path/'resumed'), '--resume'])
    train_es.main()
    a = torch.load(tmp_path/'continuous'/'last.pt', weights_only=False)
    b = torch.load(tmp_path/'resumed'/'last.pt', weights_only=False)
    for key, value in a['state_dict'].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, b['state_dict'][key], rtol=0, atol=0)


def test_sequential_jobs_and_snapshot(tmp_path, monkeypatch):
    import orchestrate_es
    manifest, protocol = fixture_inputs(tmp_path)
    calls = []
    def submit(command, text):
        calls.append(command)
        return str(100+len(calls))+'\n'
    monkeypatch.setattr(orchestrate_es.subprocess, 'check_output', submit)
    monkeypatch.setattr('sys.argv', ['orchestrate_es.py', '--manifest', str(manifest), '--protocol', str(protocol),
                                    '--output', str(tmp_path/'jobs'), '--dependency', '99', '--split-formulation', 'effective', '--submit'])
    orchestrate_es.main()
    assert [next(c[i+1] for i, v in enumerate(c) if v == '--dependency') for c in calls] == ['afterok:99', 'afterok:101', 'afterok:102']
    jobs = json.loads((tmp_path/'jobs'/'jobs.json').read_text())
    assert [j['method'] for j in jobs] == ['ar', 'node', 'opsplit']
    assert (tmp_path/'jobs'/'code'/'train_es.py').exists()
