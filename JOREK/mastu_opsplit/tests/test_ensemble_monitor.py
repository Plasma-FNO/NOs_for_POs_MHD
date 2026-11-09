import importlib.util
import json
from pathlib import Path


def monitor():
    spec = importlib.util.spec_from_file_location('ensemble_monitor', Path(__file__).parents[1]/'scripts/monitor_ensemble.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_monitor_reports_scheduler_and_early_physical_failures(tmp_path):
    folder = tmp_path/'seed17_opsplit/opsplit'
    folder.mkdir(parents=True)
    (folder/'evaluation_000_substeps1.json').write_text(json.dumps(dict(failure=None, records=[dict(audit_passed=False)])))
    (tmp_path/'ensemble-123_1.log').write_text('noise\n'+json.dumps(dict(step=31, training_loss=.5))+'\n')
    status = monitor().inspect(tmp_path, '123', '123_0|OUT_OF_MEMORY|0:125\n123_1|RUNNING|0:0\n123_2|RUNNING|0:0')
    assert not status['finished']
    assert status['tasks'][1]['last_logged_update'] == 31
    assert len(status['alerts']) == 2
    assert 'OUT_OF_MEMORY' in status['alerts'][0]
    assert 'physical audit' in status['alerts'][1]


def test_monitor_requires_complete_artifacts_for_success(tmp_path):
    accounting = '\n'.join(f'123_{task}|COMPLETED|0:0' for task in range(9))
    status = monitor().inspect(tmp_path, '123', accounting)
    assert status['finished'] and len(status['alerts']) == 9
    for task in range(9):
        folder = tmp_path/f"seed{17+task//3}_{('ar', 'node', 'opsplit')[task%3]}"/('ar', 'node', 'opsplit')[task%3]
        folder.mkdir(parents=True)
        (folder/'metrics.json').write_text(json.dumps(dict(selected_step=2000,
            evaluation=dict(audits_passed=True, numerical_failures=0))))
    status = monitor().inspect(tmp_path, '123', accounting)
    assert status['finished'] and not status['alerts']


def test_watchdog_renewal_preserves_array_and_serial_limits(tmp_path, monkeypatch):
    module = monitor()
    calls = []
    monkeypatch.setenv('SLURM_JOB_ID', '456')
    monkeypatch.setattr(module.subprocess, 'check_output', lambda arguments, **kwargs: calls.append(arguments) or '789\n')
    assert module.renew(tmp_path, '123') == '789'
    assert '--dependency=afterany:456' in calls[0]
    assert '--partition=ptgr_all_serial' in calls[0]
    assert '--time=04:00:00' in calls[0]
    assert '--job-id 123' in calls[0][-1]
    assert json.loads((tmp_path/'monitor_job.json').read_text()) == {'job_id': '789', 'array_job_id': '123'}
