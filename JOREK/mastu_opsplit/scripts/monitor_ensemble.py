#!/usr/bin/env python3
"""Persist Slurm task failures and renew a lightweight watcher until the array ends."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def inspect(root, job_id, accounting):
    states = {}
    for line in accounting.splitlines():
        identity, state, exit_code = line.strip().split('|')[:3]
        if identity.startswith(job_id+'_') and identity[len(job_id)+1:].isdigit():
            states[int(identity[len(job_id)+1:])] = (state.split()[0], exit_code)
    tasks, alerts = [], []
    for task in range(9):
        family = ('ar', 'node', 'opsplit')[task % 3]
        seed = 17+task//3
        folder = root/f'seed{seed}_{family}'/family
        state, exit_code = states.get(task, ('ACCOUNTING_PENDING', None))
        row = dict(task=task, seed=seed, family=family, state=state, exit_code=exit_code)
        if state in TERMINAL and (state != 'COMPLETED' or exit_code != '0:0'):
            alerts.append(f'Task {task} ({seed}/{family}): {state}, exit {exit_code}')
        log = root/f'ensemble-{job_id}_{task}.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size-16384))
                tail = stream.read().decode(errors='replace')
            for line in reversed(tail.splitlines()):
                try:
                    progress = json.loads(line)
                except ValueError:
                    continue
                if 'step' in progress:
                    row['last_logged_update'] = progress['step']
                    row['last_training_loss'] = progress['training_loss']
                    break
            if 'Traceback (most recent call last)' in tail:
                alerts.append(f'Task {task}: Python traceback in {log.name}')
        for report_path in sorted(folder.glob('evaluation_*.json')):
            report = json.loads(report_path.read_text())
            if report['failure'] is not None or any(not record['audit_passed'] for record in report['records']):
                alerts.append(f'Task {task}: failed numerical/physical audit in {report_path.name}')
        if state == 'COMPLETED':
            metrics_path = folder/'metrics.json'
            if not metrics_path.exists():
                alerts.append(f'Task {task}: completed without metrics.json')
            else:
                metrics = json.loads(metrics_path.read_text())
                if metrics['selected_step'] != 2000 or not metrics['evaluation']['audits_passed'] or metrics['evaluation']['numerical_failures']:
                    alerts.append(f'Task {task}: incomplete budget or failed final audits')
        tasks.append(row)
    return dict(array_job_id=job_id, checked_at=datetime.now(timezone.utc).isoformat(), tasks=tasks,
                alerts=alerts, finished=all(task['state'] in TERMINAL for task in tasks))


def renew(root, job_id):
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--job-id', job_id])
    arguments = ['sbatch', '--parsable', '--account=fupa2_fauna', '--partition=ptgr_all_serial',
                 '--time=04:00:00', '--cpus-per-task=1', '--mem=1G', '--requeue',
                 '--job-name=mastu-ensemble-watch', f'--output={root}/monitor-%j.log']
    if os.environ.get('SLURM_JOB_ID'):
        arguments.append('--dependency=afterany:'+os.environ['SLURM_JOB_ID'])
    result = subprocess.check_output(arguments+['--wrap', 'exec '+command], text=True).strip()
    write(root/'monitor_job.json', dict(job_id=result, array_job_id=job_id))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not args.job_id.isdigit():
        raise ValueError('Require a numeric Slurm array job ID')
    root = args.root.resolve()
    deadline = time.monotonic()+13800
    alerts_path = root/'monitor_alerts.json'
    seen = json.loads(alerts_path.read_text()) if alerts_path.exists() else {}
    while True:
        try:
            accounting = subprocess.check_output(['sacct', '--array', '-X', '-n', '-P', '-j', args.job_id,
                                                  '--format=JobID%40,State%30,ExitCode'], text=True, timeout=60)
            status = inspect(root, args.job_id, accounting)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            status = dict(array_job_id=args.job_id, checked_at=datetime.now(timezone.utc).isoformat(),
                          alerts=[f'Monitor check failed: {error}'], finished=False)
        for alert in status['alerts']:
            if alert not in seen:
                seen[alert] = status['checked_at']
                print(json.dumps(dict(alert=alert, detected_at=status['checked_at'])), flush=True)
        write(alerts_path, seen)
        write(root/'monitor_status.json', status)
        if status['finished']:
            if not status['alerts'] and not (root/'comparison.json').exists():
                subprocess.run([sys.executable, str(root/'study/scripts/summarize_ensemble.py'),
                                '--root', str(root), '--output', str(root/'comparison.json')], check=True)
            print(json.dumps(dict(event='array_finished', **status)), flush=True)
            return
        if args.once:
            return
        if time.monotonic() >= deadline:
            print(json.dumps(dict(event='monitor_renewed', job_id=renew(root, args.job_id))), flush=True)
            return
        time.sleep(300)


if __name__ == '__main__':
    main()
