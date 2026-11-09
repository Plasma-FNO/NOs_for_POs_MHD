"""Freeze the authorized refined pair and submit its two independent runs."""
import json
import re
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
    path = reports / 'cpu_spatial_pair_20261103.json'
    if path.exists():
        raise ValueError('paired-run record already exists; do not duplicate submissions')
    previous = json.loads((reports / 'cpu_restart_capacity_recovery_20261103.json').read_text())
    if previous['status'] != 'completed':
        raise ValueError('validated refined probe required')
    binary = Path(previous['source_directory']) / 'jorek_model303'
    if sha256_file(binary) != previous['binary_sha256']:
        raise ValueError('validated binary identity changed')
    credit = subprocess.run(['saldo', '-b', '--dcgp', '-n'], capture_output=True, text=True, check=True)
    row = next(line for line in credit.stdout.splitlines() if line.split()[0] == 'FUPA2_FAUNA')
    parts = row.split()
    remaining, monthly = int(parts[3]) - int(parts[4]), int(parts[7]) - int(parts[8])
    if min(remaining, monthly) < 1280:
        raise ValueError('insufficient verified CPU-project credit for the bounded pair and analysis')
    now = datetime.now(timezone.utc).isoformat()
    budget = {'created_utc': now,
              'approval_source': 'Explicit user: Can you now run the refined baseline and perturbed cases to the same final time as the existing runs and check spatial convergence?',
              'scope': 'Two refined 48-step cases and one dependent spatial-analysis allocation; no production or training.',
              'slurm_account': 'fupa2_fauna', 'partition': 'dcgp_fua_prod', 'selected_raw_credit_row': row,
              'remaining_local_resource_hours': remaining, 'monthly_remaining_local_resource_hours': monthly,
              'simulation_allocations_max': 2, 'analysis_allocations_max': 1, 'concurrent_simulation_allocations_max': 2,
              'simulation': {'nodes': 1, 'mpi_ranks': 4, 'omp_threads': 1, 'memory_GiB': 128,
                             'wall_seconds': 7200, 'process_timeout_seconds': 5400, 'attempts': 1},
              'analysis': {'nodes': 1, 'cpus': 1, 'threads': 1, 'memory_GiB': 64, 'wall_seconds': 3600},
              'requested_cpu_core_hours_max': 17, 'requested_node_hours_max': 5,
              'conservative_local_resource_hours_max': 1280,
              'billing_note': 'Shared-project resource-hours include memory and differ from rank CPU-hours.',
              'scientific_training_enabled': False, 'production_ensemble_enabled': False}
    budget_path = reports / 'cpu_spatial_pair_authorization_20261103.json'
    with budget_path.open('x') as stream:
        json.dump(budget, stream, indent=2); stream.write('\n')
    probe = Path(previous['bulk_output']) / 'refined3'
    base = probe.parents[1] / 'spatial_pair_20261103'
    base.mkdir()
    shutil.copyfile(binary, base / 'jorek.exe')
    (base / 'jorek.exe').chmod(0o555)
    shutil.copyfile(probe / 'jorek000000.h5', base / 'refined_reference_initial.h5')
    shutil.copyfile(probe.parent / 'jorek-env.sh', base / 'jorek-env.sh')
    shutil.copyfile(reports / 'cpu_spatial_pair_worker_20261103.py', base / 'worker.py')
    prior_parameters = json.loads((reports / 'cpu_enabled_initializer_validation_20261102.json').read_text())['parameters']
    perturbation = {f'opsplit_{name}({i + 1})': value
                    for name, values in [('center', prior_parameters['center_m']), ('width', prior_parameters['width_m']),
                                         ('psin_taper', prior_parameters['psi_n_taper'])]
                    for i, value in enumerate(values)}
    runs = {}
    for label, amplitude in [('baseline48', 0.), ('perturbed48', .05)]:
        work = base / label
        work.mkdir()
        overrides = {'nstep_n': 48, 'tstep_n': .0625, 'nout': 8, **perturbation,
                     'opsplit_amplitude(1)': amplitude, 'opsplit_amplitude(2)': amplitude}
        (work / 'input').write_text(namelist_overrides((probe / 'input').read_text(), overrides))
        for name in ['eqdsk.dat', 'jorek_equil_rz.h5']:
            shutil.copyfile(probe / name, work / name)
        runs[label] = {'amplitudes': [amplitude, amplitude], 'steps': 48, 'tstep_code_units': .0625,
                       'output_cadence_steps': 8, 'bulk_output': str(work)}
    # The pair differs only in its two explicitly enabled amplitudes.
    normalize = lambda text: re.sub(r'opsplit_amplitude\([12]\) = [^\n]+', '', text)
    assert normalize((base / 'baseline48/input').read_text()) == normalize((base / 'perturbed48/input').read_text())
    initial_text = (probe / 'stdout.log').read_text().split(' initial energies :')[0]
    axis = float(re.findall(r'Psi_axis\s*=\s*([\-+0-9.Ee]+)', initial_text)[-1])
    boundary = float(re.findall(r'Psi_bnd\s*=\s*([\-+0-9.Ee]+)', initial_text)[-1])
    clean = json.loads((reports / 'cpu_energy_complete_continuation_20261102.json').read_text())
    references = {}
    for run in clean['runs']:
        if run['label'] not in ['energy_complete_baseline48', 'energy_complete_perturbed48']:
            continue
        work = Path(run['bulk_output']) / 'attempt_1/work'
        final = work / 'jorek_restart.h5'
        expected = run['result']['attempts'][0]['health']['restart_sha256']
        if sha256_file(final) != expected or run['status'] != 'completed':
            raise ValueError('clean existing reference identity/status changed')
        references[run['label'].removeprefix('energy_complete_')] = {
            'work': str(work), 'initial_sha256': sha256_file(work / 'jorek000000.h5'), 'final_sha256': expected,
            'job_id': run['job_id']}
    files = [p for label in runs for p in (base / label).iterdir()]
    files += [base / name for name in ['jorek.exe', 'refined_reference_initial.h5', 'jorek-env.sh', 'worker.py']]
    record = {'created_utc': now, 'status': 'prepared', 'bulk_output': str(base),
              'budget_record': str(budget_path), 'budget_sha256': sha256_file(budget_path),
              'validated_probe_record': str(reports / 'cpu_restart_capacity_recovery_20261103.json'),
              'binary_sha256': sha256_file(base / 'jorek.exe'), 'source_directory': previous['source_directory'],
              'preparation_source_sha256': sha256_file(__file__),
              'input_hashes': {str(p.relative_to(base)): sha256_file(p) for p in files},
              'runs': runs, 'references': references, 'reference_psi_axis': axis, 'reference_psi_boundary': boundary,
              'final_time_s': 1.4059131112280784e-6, 'quadrature_order_comparison': 4,
              'field_relative_target': .01, 'integral_relative_target': .02,
              'perturbation_parameters': prior_parameters, 'scientific_gates_passed': [],
              'ledger_before_pair': {key: json.loads((reports / 'resource_ledger.json').read_text())[key]
                  for key in ['actual_hpc_cpu_core_hours', 'actual_hpc_node_hours', 'simulation_jobs_submitted',
                              'scientific_jobs_submitted', 'development_jobs_with_accounting']}}
    (base / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    path.write_text(json.dumps(record, indent=2) + '\n')
    python = '/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
    for label in runs:
        script = reports / f'cpu_spatial_{label}_20261103.sbatch'
        script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-spatial-{label}
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
{python} {base}/worker.py {base} {ROOT} {label}
''')
        subprocess.run(['bash', '-n', str(script)], check=True)
        command = ['sbatch', '--parsable', '--chdir', str(base), '--output', str(base / f'{label}-%j.out'),
                   '--error', str(base / f'{label}-%j.err'), str(script)]
        check = subprocess.run(['sbatch', '--test-only', *command[1:]], capture_output=True, text=True, check=True)
        submitted = subprocess.run(command, capture_output=True, text=True, check=True)
        runs[label].update(job_id=submitted.stdout.strip(), status='submitted', script=str(script),
                           script_sha256=sha256_file(script), submission_command=command,
                           scheduler_preflight=check.stderr, submitted_utc=datetime.now(timezone.utc).isoformat())
        record['status'] = 'submitted'
        path.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'label': label, 'job_id': runs[label]['job_id'], 'final_time_s': record['final_time_s']}), flush=True)


if __name__ == '__main__':
    main()
