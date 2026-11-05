import io
import json
from pathlib import Path
import tarfile

import h5py
import numpy as np
import pytest
import torch

torch.set_num_threads(1)
from data import family, inspect_file, layout, read_frames, split_for, statistics, Windows
from download import extract
from model import SplitFNO, laplacian
from train import evaluate, rollout_loss


def grid(n=12):
    return torch.linspace(9.501, 10.499, n), torch.linspace(-.499, .499, n)


def test_cylindrical_laplacian_and_nonperiodic_edges():
    r = torch.linspace(9.501, 10.499, 31, dtype=torch.float64)
    z = torch.linspace(-.499, .499, 31, dtype=torch.float64)
    zz, rr = torch.meshgrid(z, r, indexing='ij')
    x = (rr.double()**2 + zz.double()**2)[None, None]
    result = laplacian(x, r.double(), z.double())
    torch.testing.assert_close(result[..., 1:-1, 1:-1], torch.full_like(result[..., 1:-1, 1:-1], 6), rtol=1e-8, atol=1e-8)
    assert torch.count_nonzero(result[..., 0, :]) == 0


def test_transfer_exact_and_frozen_gradients():
    r, z = grid()
    source = SplitFNO(r, z, width=4, modes=4, layers=1)
    target = SplitFNO(r, z, channels=6, width=4, modes=4, layers=1)
    target.transfer(source, freeze=True)
    x = torch.randn(2, 6, 12, 12)
    before = {k: v.clone() for k, v in target.fluid.named_parameters()}
    torch.testing.assert_close(target(x, zero_shot=True)[:, :4], source(x[:, :4]))
    torch.testing.assert_close(target(x, zero_shot=True)[:, 4:], x[:, 4:])
    opt = torch.optim.Adam(p for p in target.parameters() if p.requires_grad)
    target(x).square().mean().backward()
    opt.step()
    for k, v in target.fluid.named_parameters():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert all(p.grad is None for p in target.fluid.parameters())
    assert any(p.grad is not None for p in target.magnetic.parameters())


def test_family_ignores_regime_and_blob_order():
    a = 'n_jropes = 2\nR_jropes(1)=10\nZ_jropes(1)=0\nw_jropes(1)=.1\nR_jropes(2)=10.2\nZ_jropes(2)=.2\nw_jropes(2)=.05'
    b = a.replace('(1)', '(TEMP)').replace('(2)', '(1)').replace('(TEMP)', '(2)') + '\nT_jropes(1)=.0002\n'
    assert family(a) == family(b)
    assert split_for(family(a)) == split_for(family(b))


def make_h5(path, case='es', offset=0):
    r, z = (v.numpy() for v in grid())
    rr, zz = np.meshgrid(r, z)
    fields = ['rho', 'T', 'Phi' if case == 'es' else 'u', 'omega']
    if case == 'em':
        fields += ['Psi', 'zj']
    with h5py.File(path, 'w') as f:
        f['R_mesh(nR,nZ)'], f['Z_mesh(nR,nZ)'], f['n_times'] = rr, zz, [5]
        for i, field in enumerate(fields):
            f[field] = np.stack([np.full((12, 12), offset+t+i, dtype=float) for t in range(5)])
    return dict(path=str(path), id=path.stem, case=case, frames=5, family=path.stem)


def test_reader_train_statistics_and_windows(tmp_path):
    train = make_h5(tmp_path/'train.h5')
    make_h5(tmp_path/'test.h5', offset=1000)
    mean, std = statistics([train])
    torch.testing.assert_close(mean[1:], torch.tensor([3., .004, 5.]))
    windows = Windows([train], 2)
    assert len(windows) == 3 and windows[0].shape == (3, 4, 12, 12)
    assert inspect_file(Path(train['path']), 'es')['shape'] == [12, 12]
    with h5py.File(train['path'], 'a') as f:
        f['T'][1, 0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        windows[0]


def test_safe_tar_rejects_traversal_and_links(tmp_path):
    for name, type_ in [('../escape', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)]:
        archive = tmp_path/'bad.tar'
        with tarfile.open(archive, 'w') as f:
            member = tarfile.TarInfo(name)
            member.type = type_
            f.addfile(member, io.BytesIO())
        with pytest.raises(ValueError, match='Unsafe'):
            extract(archive, tmp_path/'out')


def test_real_rollout_backward_and_failure_reporting(tmp_path):
    entry = make_h5(tmp_path/'sample.h5')
    r, z = grid()
    model = SplitFNO(r, z, width=4, modes=4, layers=1)
    frames = Windows([entry], 2)[0][None]
    mean, std = statistics([entry])
    loss = rollout_loss(model, frames, mean, std)
    loss.backward()
    assert torch.isfinite(loss) and model.log_diffusion.grad is not None
    report = evaluate(model, [entry], mean, std, 2)
    assert report['total'] == 1 and report['completed'] == 1
    with torch.no_grad():
        next(model.fluid.parameters()).fill_(float('nan'))
    report = evaluate(model, [entry], mean, std, 2)
    assert report['completed'] == 0 and report['trajectories'][0]['failure']


def test_all_training_modes_end_to_end(tmp_path, monkeypatch):
    import train as runner
    entries = []
    r, z = grid()
    for case in ('es', 'em'):
        for split in ('train', 'val', 'test'):
            entry = make_h5(tmp_path/f'{case}-{split}.h5', case)
            entry.update(split=split, r=r.tolist(), z=z.tolist())
            entries.append(entry)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'entries': entries}))
    for mode in ('source', 'zero-shot', 'frozen', 'finetune', 'scratch', 'monolithic'):
        arguments = ['train.py', '--manifest', str(manifest), '--output', str(tmp_path/mode),
                     '--mode', mode, '--epochs', '1', '--steps-per-epoch', '1', '--batch-size', '1',
                     '--horizon', '2', '--eval-horizon', '2', '--width', '4', '--modes', '4', '--layers', '1', '--device', 'cpu']
        if mode in ('zero-shot', 'frozen', 'finetune'):
            arguments += ['--source', str(tmp_path/'source'/'best.pt')]
        with monkeypatch.context() as patch:
            patch.setattr('sys.argv', arguments)
            if mode == 'zero-shot':
                def no_target_statistics(*args):
                    raise AssertionError('Zero-shot may not fit target statistics')
                patch.setattr(runner, 'statistics', no_target_statistics)
            runner.main()
        report = json.loads((tmp_path/mode/'test_metrics.json').read_text())
        assert report['completed'] == 1
        if mode == 'zero-shot':
            provenance = json.loads((tmp_path/mode/'provenance.json').read_text())
            assert provenance['trainable_parameters'] == 0 and not provenance['train_ids']
        assert len(report['fields']) == (4 if mode == 'source' else 6)


def test_summary_retains_incomplete_runs(tmp_path):
    from summarize import summarize
    run = tmp_path/'seed-0'/'scratch-8'
    run.mkdir(parents=True)
    (run/'provenance.json').write_text(json.dumps(dict(mode='scratch', seed=0, case='em', train_ids=['a'], parameters=10, trainable_parameters=10)))
    rows = summarize(tmp_path)
    assert rows[0]['completed_trajectories'] is None
    (run/'test_metrics.json').write_text(json.dumps(dict(completed=0, total=1, trajectories=[dict(relative_l2_by_frame=[], persistence_relative_l2_by_frame=[])])))
    rows = summarize(tmp_path)
    assert rows[0]['completed_trajectories'] == 0 and rows[0]['available_at_100'] == 0
    assert rows[0]['shared_relative_l2_100'] is None


def test_orchestration_dependencies_and_budget_guard(tmp_path, monkeypatch):
    import orchestrate
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'counts': {'em': {'train': 80}}}))
    calls = []
    def submit(command, text):
        calls.append(command)
        return str(100 + len(calls)) + '\n'
    monkeypatch.setattr(orchestrate.subprocess, 'check_output', submit)
    monkeypatch.setattr('sys.argv', ['orchestrate.py', '--manifest', str(manifest), '--output', str(tmp_path/'jobs'),
                                    '--account', 'test', '--dependency', '99', '--seeds', '0', '1', '--submit'])
    orchestrate.main()
    assert 'afterok:99' in calls[0] and 'afterok:99' in calls[1]
    assert 'afterany:101:102' in calls[2]
    jobs = json.loads((tmp_path/'jobs'/'jobs.json').read_text())
    assert jobs[-1]['stage'] == 'summary'
    script = (tmp_path/'jobs'/'seed-0.sbatch').read_text()
    assert script.index('--mode source') < script.index('--mode frozen')
    monkeypatch.setattr('sys.argv', ['orchestrate.py', '--manifest', str(manifest), '--output', str(tmp_path/'bad'),
                                    '--account', 'test', '--budgets', '81', '--submit'])
    with pytest.raises(SystemExit):
        orchestrate.main()
    assert len(calls) == 3
