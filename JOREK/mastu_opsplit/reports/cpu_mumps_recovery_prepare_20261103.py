"""Build the MUMPS fix and rerun guarded parity plus the refined three-step probe."""
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import sha256_file


def main():
    reports = ROOT / 'reports'
    now = datetime.now(timezone.utc).isoformat()
    previous = json.loads((reports / 'cpu_cached_rz_recovery_20261103.json').read_text())
    source = Path(previous['source_directory'])
    if sha256_file(source / 'jorek_model303') != previous['binary_sha256']:
        raise ValueError('parent executable identity differs')
    patch = ROOT / 'patches/0011-mumps-workspace-retry-and-errors.patch'
    credit = subprocess.run(['saldo', '-b', '--dcgp', '-n'], text=True, capture_output=True, check=True)
    row = next(line for line in credit.stdout.splitlines() if line.split()[0] == 'FUPA2_FAUNA')
    parts = row.split()
    remaining, monthly = int(parts[3]) - int(parts[4]), int(parts[7]) - int(parts[8])
    if min(remaining, monthly) < 512:
        raise ValueError('insufficient current project credit for one two-hour CPU allocation')
    budget = {'created_utc': now, 'approval_source': 'Explicit user: Can you diagnose and fix the MUMPS crash, then rerun the refined three-step probe?',
              'scope': 'One MUMPS recovery allocation: build patch 0011, ordinary/cached eight-step parity, then guarded refined three-step probe',
              'slurm_account': 'fupa2_fauna', 'partition': 'dcgp_fua_prod', 'budget_account': 'FUPA2_FAUNA',
              'command': ['saldo', '-b', '--dcgp', '-n'], 'selected_raw_row': row,
              'remaining_local_resource_hours': remaining, 'monthly_remaining_local_resource_hours': monthly,
              'billing_note': 'Shared project resource-hours include memory; requested rank CPU-hours differ from final site billing.',
              'job_submissions_max': 1, 'nodes': 1, 'mpi_ranks': 4, 'omp_threads': 1, 'memory_GiB': 128,
              'wall_seconds': 7200, 'attempts_per_stage': 1, 'requested_cpu_core_hours_max': 8,
              'requested_node_hours_max': 2, 'conservative_local_resource_hours_max': 512,
              'compiler_processes_max': 2, 'scientific_training_enabled': False, 'production_ensemble_enabled': False,
              'previous_phase_preserved': 'reports/cpu_refinement_continuation_20261102.json',
              'previous_recovery_preserved': 'reports/cpu_cached_rz_recovery_20261103.json'}
    budgetpath = reports / 'cpu_mumps_recovery_authorization_20261103.json'
    with budgetpath.open('x') as stream:
        json.dump(budget, stream, indent=2)
        stream.write('\n')
    bulk = source.parents[1]
    base = bulk / 'mumps_recovery_20261103'
    base.mkdir()
    target = base / 'source'
    shutil.copytree(source, target)
    apply = subprocess.run(['patch', '--batch', '-p1', '-i', str(patch)], cwd=target, text=True, capture_output=True, check=True)
    ordinary = bulk / 'cpu_refine_stable_parity8_20261102/attempt_1/work'
    refined = bulk / 'fixed_rz_nested_probe3_20261102'
    for stage in ['ordinary8', 'cached8', 'refined3']:
        work = base / stage
        work.mkdir()
        old = refined if stage == 'refined3' else ordinary
        shutil.copyfile(old / 'eqdsk.dat', work / 'eqdsk.dat')
        text = (old / 'input').read_text()
        if stage == 'cached8':
            text = namelist_overrides(text, {'regrid_from_rz': True})
        (work / 'input').write_text(text)
        if stage != 'ordinary8':
            shutil.copyfile(ordinary / 'jorek_equil_rz.h5', work / 'jorek_equil_rz.h5')
    reference = bulk / 'cpu_io_baseline48_20261102/attempt_1/work/jorek000008.h5'
    shutil.copyfile(reference, base / 'reference8.h5')
    shutil.copyfile(reports / 'cpu_native_parity_protocol_20261102.json', base / 'parity_protocol.json')
    worker = reports / 'cpu_cached_rz_recovery_worker_20261103.py'
    shutil.copyfile(worker, base / 'worker.py')
    shutil.copyfile(ROOT / 'runs/prepared_equilibrium_dcgp_authorized_20261102/jorek-env.sh', base / 'jorek-env.sh')
    inputs = [p for stage in ['ordinary8', 'cached8', 'refined3'] for p in (base / stage).iterdir()]
    inputs += [base / name for name in ['worker.py', 'reference8.h5', 'parity_protocol.json', 'jorek-env.sh']]
    inputs += [target / 'solvers/mod_mumps.f90']
    record = {'created_utc': now, 'status': 'prepared', 'scope': budget['scope'], 'budget_record': str(budgetpath),
              'budget_record_sha256': sha256_file(budgetpath), 'parent_source': str(source),
              'parent_binary_sha256': previous['binary_sha256'], 'patch': str(patch), 'patch_sha256': sha256_file(patch),
              'bulk_output': str(base), 'source_directory': str(target), 'worker_sha256': sha256_file(base / 'worker.py'),
              'preparation_source_sha256': sha256_file(__file__), 'input_hashes': {str(p.relative_to(base)): sha256_file(p) for p in inputs},
              'requested_cpu_core_hours': 8, 'requested_node_hours': 2, 'scientific_gates_passed': [],
              'stage_limits_s': {'build': 300, 'ordinary8': 600, 'cached8': 600, 'refined3': 5400},
              'ordinary_parity': 'Exact fields, geometry, indices, times and energies',
              'cached_parity': 'Frozen q8 field/gradient protocol; exact geometry, time and steps',
              'patch_application_stdout': apply.stdout,
              'focused_tests': {'passed': 18, 'report': 'cpu_mumps_recovery_tests_20261103.xml',
                                'sha256': sha256_file(reports / 'cpu_mumps_recovery_tests_20261103.xml')},
              'ledger_before_recovery': {k: json.loads((reports / 'resource_ledger.json').read_text())[k]
                  for k in ('actual_hpc_cpu_core_hours', 'actual_hpc_node_hours', 'simulation_jobs_submitted',
                            'scientific_jobs_submitted', 'development_jobs_with_accounting')}}
    (base / 'authorization.json').write_text(json.dumps(record, indent=2) + '\n')
    (base / 'startup.sha256').write_text(''.join(f'{value}  {name}\n' for name, value in record['input_hashes'].items()))
    script = reports / 'cpu_mumps_recovery_20261103.sbatch'
    python = '/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-mumps-recovery
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
cd {base}
sha256sum --check startup.sha256 > startup_verification.log
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd {target}
timeout --signal=TERM --kill-after=10s 300s make -j2 jorek_model303 > {base}/build.out 2> {base}/build.err
sha256sum jorek_model303 > {base}/binary.sha256
{python} {base}/worker.py {base} {ROOT}
''')
    subprocess.run(['bash', '-n', str(script)], check=True)
    record.update(script=str(script), script_sha256=sha256_file(script))
    command = ['sbatch', '--parsable', '--chdir', str(base), '--output', str(base / 'batch-%j.out'),
               '--error', str(base / 'batch-%j.err'), str(script)]
    check = subprocess.run(['sbatch', '--test-only', *command[1:]], text=True, capture_output=True)
    record['scheduler_preflight'] = {'returncode': check.returncode, 'stdout': check.stdout, 'stderr': check.stderr}
    path = reports / 'cpu_mumps_recovery_20261103.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    if check.returncode:
        raise ValueError(check.stderr)
    out = subprocess.run(command, text=True, capture_output=True)
    record['submission'] = {'command': command, 'returncode': out.returncode, 'stdout': out.stdout, 'stderr': out.stderr}
    if out.returncode:
        path.write_text(json.dumps(record, indent=2) + '\n')
        raise ValueError(out.stderr)
    record.update(job_id=out.stdout.strip(), status='submitted', submitted_utc=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'job_id': record['job_id'], 'output': str(base), 'remaining_cpu_project_resource_hours': remaining,
                      'monthly_remaining_resource_hours': monthly, 'requested_cpu_core_hours': 8, 'requested_node_hours': 2}, indent=2))


if __name__ == '__main__':
    main()
