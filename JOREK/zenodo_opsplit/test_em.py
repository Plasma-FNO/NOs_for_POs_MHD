import json
from pathlib import Path

import pytest
import torch

from test_workflow import grid, make_h5
from train_es import cached_trajectories


def inputs(tmp_path):
    r, z = grid()
    rows = []
    for case in ('es', 'em'):
        for split in ('train', 'val', 'test'):
            row = make_h5(tmp_path/f'{case}-{split}.h5', case=case)
            row.update(split=split, r=r.tolist(), z=z.tolist())
            rows.append(row)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(entries=rows)))
    cfg = json.loads(Path(__file__).with_name('em_paper_protocol.json').read_text())
    cfg.update(epochs=1, train_trajectories=1, train_frames=5, horizon=2, batch_size=2,
               width=4, modes=4, layers=1, eval_horizon=4)
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps(cfg))
    return manifest, protocol, rows


def test_em_fields_are_read_without_es_potential_key(tmp_path):
    _, _, rows = inputs(tmp_path)
    entry = next(e for e in rows if e['case'] == 'em')
    values = cached_trajectories([entry], 5)
    assert values.shape == (1, 5, 6, 12, 12)
    torch.testing.assert_close(values[0, 0, 4], torch.full((12, 12), 40.))


@pytest.mark.parametrize('method', ['ar', 'node'])
def test_em_training_and_checkpoint(tmp_path, monkeypatch, method):
    import train_es
    manifest, protocol, _ = inputs(tmp_path)
    output = tmp_path/method
    monkeypatch.setattr('sys.argv', ['train_es.py', '--manifest', str(manifest), '--protocol', str(protocol),
                                    '--output', str(output), '--case', 'em', '--method', method, '--device', 'cpu'])
    train_es.main()
    checkpoint = torch.load(output/'best.pt', weights_only=False)
    assert checkpoint['case'] == 'em' and checkpoint['model_config']['channels'] == 6
    assert checkpoint['model_config']['autoregressive'] == (method == 'ar')
    assert json.loads((output/'test_metrics.json').read_text())['completed'] == 1
    provenance = json.loads((output/'provenance.json').read_text())
    assert all(i.startswith('em-') for k in ('train_ids', 'val_ids', 'test_ids') for i in provenance[k])


@pytest.mark.parametrize('case', ['es', 'em'])
def test_native_split_redirects_legacy_entrypoint_before_output(tmp_path, monkeypatch, case):
    import train_es
    monkeypatch.setattr('sys.argv', ['train_es.py', '--manifest', str(tmp_path/'missing'), '--output', str(tmp_path/'out'),
                                    '--case', case, '--method', 'opsplit'])
    with pytest.raises(NotImplementedError, match='momentum inversion'):
        train_es.main()
    assert not (tmp_path/'out').exists()


def test_em_submission_points_to_dedicated_native_workflow(tmp_path, monkeypatch):
    import orchestrate_es
    manifest, protocol, _ = inputs(tmp_path)
    calls = []
    def submit(command, text):
        calls.append(command)
        return str(100+len(calls))+'\n'
    monkeypatch.setattr(orchestrate_es.subprocess, 'check_output', submit)
    monkeypatch.setattr('sys.argv', ['orchestrate_es.py', '--case', 'em', '--manifest', str(manifest),
                                    '--protocol', str(protocol), '--output', str(tmp_path/'jobs'), '--submit'])
    orchestrate_es.main()
    jobs = json.loads((tmp_path/'jobs'/'jobs.json').read_text())
    assert len(calls) == 2 and 'afterok:101' in calls[1]
    assert jobs[2]['method'] == 'opsplit' and jobs[2]['status'] == 'dedicated_native_workflow' and jobs[2]['job_id'] is None
    assert jobs[2]['primary_state'] == ['rho', 'p', 'u', 'psi']
    assert not (tmp_path/'jobs'/'opsplit.sbatch').exists()
    assert '--case em' in (tmp_path/'jobs'/'ar.sbatch').read_text()


def test_constraint_fit_does_not_assert_native_units():
    from audit_native import fit_constraint
    x = torch.arange(1., 10.)
    fit = fit_constraint(x, -3*x)
    assert fit == dict(scale=-3., relative_residual=0.)
    assert fit_constraint(x*0, x)['scale'] is None
