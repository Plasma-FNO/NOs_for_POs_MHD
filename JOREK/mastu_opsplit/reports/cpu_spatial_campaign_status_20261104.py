"""Read current scheduler/native progress without modifying frozen attempt records."""
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

R = Path(__file__).resolve().parent
records = {}
paths = set(R.glob('cpu_combined_*_20261104.json')) | set(R.glob('cpu_poloidal880_dt*_20261104.json')) | set(R.glob('cpu_mumps_scaling8_20261104.json')) | set(R.glob('cpu_scaling8_*_20261104.json')) | set(R.glob('cpu_corner_handle*_20261104.json')) | set(R.glob('cpu_xpoint_handle*_20261104.json')) | set(R.glob('cpu_reversed_crossing*_20261104.json'))
for path in sorted(paths):
    record = json.loads(path.read_text())
    if record.get('job_id'):
        records[str(record['job_id'])] = (path, record)
output = subprocess.check_output(['sacct', '-n', '-P', '-X', '-j', ','.join(records),
                                  '--format=JobID,State,Elapsed,ExitCode'], text=True)
states = {row[0]: row[1:] for line in output.splitlines() if (row := line.split('|')) and len(row) >= 4}
jobs = {}
for job, (path, record) in records.items():
    entry = {'record': path.name, 'scheduler': states.get(job, []), 'native': {}}
    base = Path(record.get('bulk_output', str(R)))
    stages = [record['stage']] if 'stage' in record else record.get('stages', [])
    if not stages and record.get('budget', {}).get('mpi_ranks', 0) > 1 and (base / 'manifest.json').exists():
        stages = json.loads((base / 'manifest.json').read_text()).get('stages', [])
    for stage in stages:
        log = base / stage / 'stdout.log'
        if not log.exists():
            continue
        text = log.read_text()
        iterations = re.findall(r'Iteration\s+(\d+): diff=\s*(\S+)', text)
        steps = re.findall(r'time step :\s+(\d+)\s+(\d+)\s+(\d+)', text)
        info = {'last_newton': iterations[-1] if iterations else None,
                'last_started_step': int(steps[-1][-1]) if steps else None}
        for variable, number in [('rho', 5), ('T', 6)]:
            samples = re.findall(rf'min/max\s+{number}\s+{variable}\s+(\S+)\s+(\S+)', text)
            if samples:
                info['minimum_' + variable] = min(float(x[0]) for x in samples)
        errors = re.findall(r'MUMPS refinement steps and backward errors:\s*\d+\s+(\S+)\s+(\S+)', text)
        if errors:
            info['maximum_backward_error'] = max(float(a) + float(b) for a, b in errors)
        result = base / stage / 'result.json'
        if result.exists():
            proof = json.loads(result.read_text())
            info.update(status=proof['status'], error=proof.get('error'))
        entry['native'][stage] = info
    jobs[job] = entry
    state = entry['scheduler'][0] if entry['scheduler'] else 'UNKNOWN'
    print(job, state, path.stem.replace('cpu_combined_', '').replace('_20261104', ''),
          json.dumps(entry['native'], separators=(',', ':')))
report = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'jobs': jobs,
          'scope': 'Live progress only. Completed native steps do not establish full spatial convergence.',
          'scientific_gates_passed': []}
(R / 'cpu_spatial_campaign_status_20261104.json').write_text(json.dumps(report, indent=2) + '\n')
