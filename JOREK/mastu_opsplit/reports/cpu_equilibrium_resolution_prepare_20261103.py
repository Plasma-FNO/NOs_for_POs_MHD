"""Test the radial mesh against a freshly refined preliminary R/Z equilibrium."""
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
    reports = ROOT / 'reports'
    reference = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/spatial_radial_pair_20261103')
    base = reference.parent / 'equilibrium_resolution_probe_20261103'
    record_path = reports / 'cpu_equilibrium_resolution_probe_20261103.json'
    if base.exists() or record_path.exists():
        raise ValueError('preserve existing R/Z resolution probe')
    row = next(line for line in subprocess.check_output(['saldo', '-b', '--dcgp', '-n'], text=True).splitlines() if line.split()[0] == 'FUPA2_FAUNA')
    cols = row.split()
    if min(int(cols[3])-int(cols[4]), int(cols[7])-int(cols[8])) < 1024:
        raise ValueError('insufficient CPU project credit')
    original = json.loads((reference / 'manifest.json').read_text())
    if sha256_file(reference / 'source/jorek_model303') != original['validated_unchanged_binary_sha256']:
        raise ValueError('validated solver identity changed')
    (base / 'source').mkdir(parents=True)
    shutil.copyfile(reference / 'source/jorek_model303', base / 'source/jorek_model303')
    (base / 'source/jorek_model303').chmod(0o555)
    for name in ['worker.py', 'jorek-env.sh']:
        shutil.copyfile(reference / name, base / name)
    work = base / 'probe3'
    work.mkdir()
    overrides = {'regrid_from_rz': False, 'n_R': 180, 'n_Z': 520, 'n_flux': 160, 'nstep_n': 3}
    (work / 'input').write_text(namelist_overrides((reference / 'probe3/input').read_text(), overrides))
    shutil.copyfile(reference / 'probe3/eqdsk.dat', work / 'eqdsk.dat')
    files = [base / 'source/jorek_model303', base / 'worker.py', base / 'jorek-env.sh', work / 'input', work / 'eqdsk.dat']
    manifest = {key: original[key] for key in ['final_time_s', 'tstep_code_units', 'field_relative_target', 'integral_relative_target', 'validated_unchanged_binary_sha256', 'equil_accuracy']}
    manifest.update(stages={'probe3': {'steps': 3, 'amplitudes': [0., 0.], 'timeout_seconds': 13800}},
                    input_hashes={str(p): sha256_file(p) for p in files},
                    hypothesis='Four radial160 folds already occur before extensions. Test sensitivity to the cached preliminary R/Z equilibrium, without editing geometry or loosening equilibrium tolerance.')
    (base / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    budget = {'authorization': 'User request to resolve Gates A/B/C blockers for validated training data.',
              'scope': 'One unchanged-solver native three-step proof: fresh doubled R/Z counts, radial160 and original wall/poloidal counts.',
              'allocations_max': 1, 'attempts_max': 1, 'mpi_ranks': 4, 'threads': 1, 'memory_GiB': 256,
              'wall_hours_max': 4, 'rank_CPU_hours_max': 16, 'node_hours_max': 4, 'conservative_resource_hours_max': 1024,
              'saldo_row': row, 'spatial_targets_changed': False, 'physics_changed': False}
    script = reports / 'cpu_equilibrium_resolution_probe_20261103.sbatch'
    python = '/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-fresh-rz-radial160
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=04:00:00
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
    print(json.dumps({'job_id': job, 'overrides': overrides}))


if __name__ == '__main__':
    main()
