"""Refresh MUMPS recovery status and account for its single allocation exactly once."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

R = Path(__file__).resolve().parent
p = R / 'cpu_mumps_workspace_recovery_20261103.json'
record = json.loads(p.read_text())
ledger_path = R / 'resource_ledger.json'
ledger = json.loads(ledger_path.read_text())
if 'ledger_before_recovery' not in record:
    record['ledger_before_recovery'] = {key: ledger[key] for key in ('actual_hpc_cpu_core_hours', 'actual_hpc_node_hours',
        'simulation_jobs_submitted', 'scientific_jobs_submitted', 'development_jobs_with_accounting')}
base = record['ledger_before_recovery']
job = record['job_id']
query = subprocess.run(['sacct', '-n', '-P', '-X', '-j', job,
    '--format=JobIDRaw,State,ElapsedRaw,AllocCPUS,NNodes,ExitCode,AllocTRES,Partition'], capture_output=True, text=True, check=True)
rows = [line.split('|') for line in query.stdout.splitlines() if line.split('|')[0] == job]
cpu, node = 0., 0.
if rows:
    t = rows[0]
    a = {'state': t[1], 'elapsed_seconds': int(t[2]), 'allocated_cpus': int(t[3]), 'allocated_nodes': int(t[4]),
         'exit_code': t[5], 'alloc_tres': t[6], 'partition': t[7]}
    record.update(accounting=a, scheduler_state=t[1])
    cpu = a['elapsed_seconds'] * a['allocated_cpus'] / 3600
    node = a['elapsed_seconds'] * a['allocated_nodes'] / 3600
output = Path(record['bulk_output'])
if (output / 'binary.sha256').exists():
    record.update(build_status='compiled_and_linked', binary_sha256=(output / 'binary.sha256').read_text().split()[0])
if (output / 'result.json').exists():
    result = json.loads((output / 'result.json').read_text())
    record.update(result=result, status=result['status'])
record['as_of_utc'] = datetime.now(timezone.utc).isoformat()
p.write_text(json.dumps(record, indent=2) + '\n')
ledger.update(as_of_utc=record['as_of_utc'], actual_hpc_cpu_core_hours=base['actual_hpc_cpu_core_hours'] + cpu,
    actual_hpc_node_hours=base['actual_hpc_node_hours'] + node,
    simulation_jobs_submitted=base['simulation_jobs_submitted'] + 1,
    scientific_jobs_submitted=base['scientific_jobs_submitted'] + 1,
    development_jobs_with_accounting=base['development_jobs_with_accounting'] + bool(rows))
ledger['mumps_workspace_recovery'] = {'record': 'reports/' + p.name, 'budget_record': record['budget_record'],
    'job_id': job, 'requested_cpu_core_hours': 8, 'requested_node_hours': 2,
    'usage_note': 'One composite build/regression/probe allocation; counted once. Running elapsed usage is provisional.',
    'scientific_training_enabled': False, 'production_ensemble_enabled': False}
ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
print(json.dumps({'job_id': job, 'scheduler_state': record.get('scheduler_state'), 'stage_status': record['status'],
                  'total_allocated_cpu_hours': ledger['actual_hpc_cpu_core_hours']}, indent=2))
