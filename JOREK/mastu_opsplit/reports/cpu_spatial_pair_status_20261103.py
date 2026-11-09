"""Refresh paired-run status and account each of its allocations once."""
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    reports = Path(__file__).resolve().parent
    path = reports / 'cpu_spatial_pair_20261103.json'
    record = json.loads(path.read_text())
    entries = [*record['runs'].values(), *record.get('previous_analysis_attempts', [])]
    if 'analysis' in record:
        entries.append(record['analysis'])
    jobs = {entry['job_id']: entry for entry in entries}
    query = subprocess.run(['sacct', '-n', '-P', '-X', '-j', ','.join(jobs),
                            '--format=JobIDRaw,State,ElapsedRaw,AllocCPUS,NNodes,ExitCode'],
                           capture_output=True, text=True, check=True)
    cpu, nodes, accounted = 0., 0., 0
    for line in query.stdout.splitlines():
        row = line.split('|')
        if row[0] not in jobs:
            continue
        accounting = dict(state=row[1], elapsed_seconds=int(row[2]), allocated_cpus=int(row[3]),
                          allocated_nodes=int(row[4]), exit_code=row[5])
        jobs[row[0]].update(accounting=accounting, scheduler_state=row[1])
        cpu += accounting['elapsed_seconds'] * accounting['allocated_cpus'] / 3600
        nodes += accounting['elapsed_seconds'] * accounting['allocated_nodes'] / 3600
        accounted += 1
    summary = {}
    for label, entry in record['runs'].items():
        work = Path(entry['bulk_output'])
        result = work / 'result.json'
        if result.exists():
            entry['result'] = json.loads(result.read_text())
            entry['status'] = entry['result']['status']
        log = work / 'stdout.log'
        steps = re.findall(r'After step\s+(\d+)', log.read_text()) if log.exists() else []
        entry['steps_logged'] = int(steps[-1]) if steps else 0
        summary[label] = {key: entry.get(key) for key in ['job_id', 'scheduler_state', 'status', 'steps_logged']}
    comparison = reports / 'cpu_spatial_convergence_20261103.json'
    if comparison.exists():
        record['comparison_report'] = str(comparison)
        record['comparison_status'] = json.loads(comparison.read_text())['status']
    record['status'] = 'completed' if comparison.exists() and all(
        entry.get('status') == 'completed' for entry in record['runs'].values()) else 'active'
    record['as_of_utc'] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(record, indent=2) + '\n')
    ledger_path = reports / 'resource_ledger.json'
    ledger = json.loads(ledger_path.read_text())
    base = record['ledger_before_pair']
    ledger.update(as_of_utc=record['as_of_utc'],
                  actual_hpc_cpu_core_hours=base['actual_hpc_cpu_core_hours'] + cpu,
                  actual_hpc_node_hours=base['actual_hpc_node_hours'] + nodes,
                  simulation_jobs_submitted=base['simulation_jobs_submitted'] + len(record['runs']),
                  scientific_jobs_submitted=base['scientific_jobs_submitted'] + len(record['runs']),
                  development_jobs_with_accounting=base['development_jobs_with_accounting'] + accounted)
    retries = len(record.get('previous_analysis_attempts', []))
    ledger['spatial_pair'] = {'record': path.name, 'job_ids': list(jobs), 'requested_cpu_core_hours_max': 17 + retries,
                              'requested_node_hours_max': 5 + retries, 'scientific_gates_passed': [],
                              'usage_note': 'Unique allocations include two simulations and dependent analysis; running elapsed usage is provisional.'}
    ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps({'runs': summary, 'analysis': record.get('analysis', {}).get('scheduler_state'),
                      'comparison': record.get('comparison_status'), 'allocated_cpu_hours': cpu}, indent=2))


if __name__ == '__main__':
    main()
