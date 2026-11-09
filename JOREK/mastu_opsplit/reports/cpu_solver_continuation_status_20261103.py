"""Refresh solver evidence and count each continuation allocation once."""
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

R = Path(__file__).resolve().parent
L = R / 'resource_ledger.json'
U = R / 'cpu_solver_continuation_usage_20261103.json'
ledger = json.loads(L.read_text())
usage = json.loads(U.read_text()) if U.exists() else {'ledger_before': {k: ledger[k] for k in [
    'actual_hpc_cpu_core_hours', 'actual_hpc_node_hours', 'development_jobs_with_accounting',
    'simulation_jobs_submitted', 'scientific_jobs_submitted']}}
jobs = {'2771312': {'kind': 'analysis', 'record': 'cpu_newton_equilibrium_comparison_20261103.json'}}
for path in R.glob('*.json'):
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        continue
    job = str(data.get('job_id', ''))
    if job.isdigit() and int(job) > 2770975 and 'bulk_output' in data:
        jobs[job] = {'kind': data.get('kind', 'simulation'), 'record': path.name}
        work = Path(data['bulk_output'])
        results = {} if data.get('native_attempt_started') is False else {p.parent.name: json.loads(p.read_text()) for p in work.glob('*/result.json')}
        if results:
            data['stage_results'] = results
        if job == '2771301':
            data.update(status='diagnosis_completed_expected_native_grid_rejection',
                        conclusion='Diagnostic-only build reproduces the four native folds. Trace identifies reversed exact FE flux between neighboring core levels.')
        for label in results:
            log = work / label / 'stdout.log'
            if log.exists():
                text = log.read_text()
                iterations = re.findall(r'Iteration\s+(\d+): diff=\s*(\S+)', text)
                if iterations:
                    results[label]['equilibrium_summary'] = {'iterations': int(iterations[-1][0]),
                        'last_raw_update': float(iterations[-1][1]),
                        'converged': 'Fixed boundary equilibrium converged:' in text}
        path.write_text(json.dumps(data, indent=2) + '\n')
rows = subprocess.check_output(['sacct', '-n', '-P', '-X', '-j', ','.join(jobs),
    '--format=JobIDRaw,State,ElapsedRaw,AllocCPUS,NNodes,ExitCode'], text=True)
cpu = node = 0.
for line in rows.splitlines():
    job, state, elapsed, cpus, nodes, code, *_ = line.split('|')
    if job not in jobs:
        continue
    jobs[job]['accounting'] = dict(state=state, elapsed_seconds=int(elapsed), allocated_cpus=int(cpus),
                                  allocated_nodes=int(nodes), exit_code=code)
    cpu += int(elapsed) * int(cpus) / 3600
    node += int(elapsed) * int(nodes) / 3600
usage.update(as_of_utc=datetime.now(timezone.utc).isoformat(), jobs=jobs,
             allocated_cpu_hours=cpu, allocated_node_hours=node,
             scope='Additional top-level recorded solver allocations after 2770975; unique allocations, running elapsed usage provisional. Scientific acceptance remains separate.')
U.write_text(json.dumps(usage, indent=2) + '\n')
before = usage['ledger_before']
ledger.update(as_of_utc=usage['as_of_utc'], actual_hpc_cpu_core_hours=before['actual_hpc_cpu_core_hours']+cpu,
              actual_hpc_node_hours=before['actual_hpc_node_hours']+node,
              development_jobs_with_accounting=before['development_jobs_with_accounting']+len(jobs),
              simulation_jobs_submitted=before['simulation_jobs_submitted']+sum(j['kind']=='simulation' for j in jobs.values()),
              scientific_jobs_submitted=before['scientific_jobs_submitted']+sum(j['kind']=='simulation' for j in jobs.values()),
              solver_continuation={'record': U.name, 'job_ids': list(jobs), 'scientific_gates_passed': []})
L.write_text(json.dumps(ledger, indent=2)+'\n')
print(json.dumps({'jobs': {k: v.get('accounting', {}) for k, v in jobs.items()}, 'additional_cpu_hours': cpu}, indent=2))
