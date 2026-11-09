"""Test the existing native Newton GS solver with unchanged physical equations."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import sha256_file


def main():
    reference = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/spatial_radial_pair_20261103')
    base = reference.parent / 'newton_damped_wall_20261103'
    reports = ROOT / 'reports'
    record_path = reports / 'cpu_newton_damped_wall_20261103.json'
    if base.exists() or record_path.exists():
        raise ValueError('preserve existing relaxation controls')
    row = next(line for line in subprocess.check_output(['saldo', '-b', '--dcgp', '-n'], text=True).splitlines() if line.split()[0] == 'FUPA2_FAUNA')
    cols = row.split()
    if min(int(cols[3])-int(cols[4]), int(cols[7])-int(cols[8])) < 768:
        raise ValueError('insufficient current CPU project credit')
    original = json.loads((reference / 'manifest.json').read_text())
    if sha256_file(reference / 'source/jorek_model303') != original['validated_unchanged_binary_sha256']:
        raise ValueError('validated executable changed')
    (base / 'source').mkdir(parents=True)
    shutil.copyfile(reference / 'source/jorek_model303', base / 'source/jorek_model303')
    (base / 'source/jorek_model303').chmod(0o555)
    for name in ['worker.py', 'jorek-env.sh']:
        shutil.copyfile(reference / name, base / name)
    anchor_path = reports / 'cpu_newton_flux_anchor_20261103.json'
    anchor = json.loads(anchor_path.read_text())
    if sha256_file(anchor['source']) != anchor['source_sha256']:
        raise ValueError('reference flux anchor changed')
    stages = {}
    for label in ['probe3']:
        work = base / label
        work.mkdir()
        overrides = {'amix': 0.5, 'newton_GS_fixbnd': True, 'delta_psi_GS': anchor['delta_psi_GS'], 'nstep_n': 3}
        if label == 'probe3':
            overrides.update({f'n_ext_block({i})': 114 if i == 3 else 60 for i in range(1, 12)})
        (work / 'input').write_text(namelist_overrides((reference / 'probe3/input').read_text(), overrides))
        for name in ['eqdsk.dat', 'jorek_equil_rz.h5']:
            shutil.copyfile(reference / 'probe3' / name, work / name)
        stages[label] = {'steps': 3, 'amplitudes': [0., 0.], 'timeout_seconds': 4800}
    files = [anchor_path, Path(anchor['source']), base / 'source/jorek_model303', base / 'worker.py', base / 'jorek-env.sh'] + [p for label in stages for p in (base / label).iterdir()]
    manifest = {key: original[key] for key in ['final_time_s', 'tstep_code_units', 'field_relative_target', 'integral_relative_target', 'validated_unchanged_binary_sha256', 'equil_accuracy']}
    manifest.update(stages=stages, input_hashes={str(p): sha256_file(p) for p in files},
                    numerical_change='Enable existing newton_GS_fixbnd with half Newton updates (amix=0.5); pressure derivative normalized to the independently measured converged reference span; same raw GS stopping tolerance 1e-10 and native 200-iteration limit.',
                    control_policy='Original-wall native Newton control already passed strict equilibrium and three steps; this run damps only the failed wall1.5 Newton updates. Passing probes alone cannot establish final-time wall convergence.')
    (base / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    budget = {'authorization': 'User request to get the numerical solver to converge and finish the science plan.',
              'scope': 'One three-step wall1.5 proof using damped native Newton to suppress the upper/lower X-point cycle.',
              'allocations_max': 1, 'attempts_each_max': 1, 'mpi_ranks': 4, 'threads': 1, 'memory_GiB': 256,
              'wall_hours_max': 3, 'rank_CPU_hours_max': 12, 'node_hours_max': 3, 'conservative_resource_hours_max': 768,
              'saldo_row': row, 'equilibrium_tolerance_changed': False, 'physical_equivalence_requires_validation': True, 'delta_psi_GS': anchor['delta_psi_GS']}
    script = reports / 'cpu_newton_damped_wall_20261103.sbatch'
    python = '/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-newton-damped-wall
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=03:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
{python} {base}/worker.py {base} {ROOT} probe3
''')
    subprocess.run(['bash', '-n', str(script)], check=True)
    command = ['sbatch', '--parsable', '--chdir', str(base), '--output', str(base / 'batch-%j.out'), '--error', str(base / 'batch-%j.err'), str(script)]
    preflight = subprocess.run(['sbatch', '--test-only', *command[1:]], capture_output=True, text=True, check=True)
    record = {'status': 'prepared', 'bulk_output': str(base), 'budget': budget,
              'manifest_sha256': sha256_file(base / 'manifest.json'), 'script': str(script), 'script_sha256': sha256_file(script),
              'scheduler_preflight': preflight.stderr, 'scientific_gates_passed': []}
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    job = subprocess.check_output(command, text=True).strip()
    record.update(status='submitted', job_id=job)
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'job_id': job, 'amix': 0.5, 'newton_GS_fixbnd': True, 'equil_accuracy': 1e-10}))


if __name__ == '__main__':
    main()
