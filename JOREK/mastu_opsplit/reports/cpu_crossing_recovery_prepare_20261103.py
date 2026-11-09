"""Build the bounded contour crossing candidate and validate two native probes."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import sha256_file


def main():
    bulk = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
    reference = bulk / 'spatial_radial_pair_20261103'
    base = bulk / 'crossing_recovery_20261103'
    reports = ROOT / 'reports'
    record_path = reports / 'cpu_crossing_recovery_20261103.json'
    if base.exists() or record_path.exists():
        raise ValueError('preserve existing crossing experiment')
    row = next(line for line in subprocess.check_output(['saldo', '-b', '--dcgp', '-n'], text=True).splitlines() if line.split()[0] == 'FUPA2_FAUNA')
    cols = row.split()
    if min(int(cols[3])-int(cols[4]), int(cols[7])-int(cols[8])) < 768:
        raise ValueError('insufficient current project credit')
    patch = ROOT / 'patches/0019-bound-native-contour-crossings.patch'
    parent = bulk / 'fast_mesh_20261103/source160'
    base.mkdir()
    shutil.copytree(parent, base / 'source')
    subprocess.run(['patch', '--batch', '-p1', '-i', str(patch)], cwd=base / 'source', capture_output=True, check=True)
    for name in ['worker.py', 'jorek-env.sh']:
        shutil.copyfile(reference / name, base / name)
    stages = {}
    for label, flux in [('radial160', 160), ('control80', 80)]:
        work = base / label
        work.mkdir()
        (work / 'input').write_text(namelist_overrides((reference / 'probe3/input').read_text(), {'n_flux': flux, 'amix': .6, 'nstep_n': 3}))
        for name in ['eqdsk.dat', 'jorek_equil_rz.h5']:
            shutil.copyfile(reference / 'probe3' / name, work / name)
        stages[label] = {'steps': 3, 'amplitudes': [0., 0.], 'timeout_seconds': 4800}
    original = json.loads((reference / 'manifest.json').read_text())
    budget = {'authorization': 'User requested numerical convergence and completion of the science plan.',
              'scope': 'One isolated crossing build and two three-step probes; radial160 must pass before control80 runs. No production acceptance.',
              'allocations_max': 1, 'attempts_each_max': 1, 'mpi_ranks': 4, 'threads': 1, 'memory_GiB': 256,
              'wall_hours_max': 3, 'rank_CPU_hours_max': 12, 'node_hours_max': 3, 'conservative_resource_hours_max': 768,
              'saldo_row': row, 'equilibrium_tolerance': 1e-10, 'physics_changed': False, 'spatial_targets_changed': False}
    files = [base / 'worker.py', base / 'jorek-env.sh', patch, reports / 'cpu_crossing_tests_20261103.xml',
             base / 'source/grids/grid_utils/find_crossing_on_surface_piece.f90']
    files += [p for label in stages for p in (base / label).iterdir()]
    manifest = {key: original[key] for key in ['final_time_s', 'tstep_code_units', 'field_relative_target', 'integral_relative_target', 'equil_accuracy']}
    manifest.update(stages=stages, budget=budget, input_hashes={str(p): sha256_file(p) for p in files},
                    limitation='Crossing changes may change geometry; repeat matched references before spatial acceptance.',
                    source_parent=str(parent), patch_sha256=sha256_file(patch))
    (base / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    script = reports / 'cpu_crossing_recovery_20261103.sbatch'
    python = '/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-bounded-crossing
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
cd {base}/source
timeout --signal=TERM --kill-after=10s 600s make -j2 jorek_model303 > {base}/build.out 2> {base}/build.err
{python} {base}/worker.py {base} {ROOT} radial160
{python} {base}/worker.py {base} {ROOT} control80
''')
    subprocess.run(['bash', '-n', str(script)], check=True)
    command = ['sbatch', '--parsable', '--chdir', str(base), '--output', str(base / 'batch-%j.out'), '--error', str(base / 'batch-%j.err'), str(script)]
    preflight = subprocess.run(['sbatch', '--test-only', *command[1:]], capture_output=True, text=True, check=True)
    record = {'status': 'prepared', 'bulk_output': str(base), 'budget': budget, 'manifest_sha256': sha256_file(base / 'manifest.json'),
              'script': str(script), 'script_sha256': sha256_file(script), 'scheduler_preflight': preflight.stderr, 'scientific_gates_passed': []}
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    record.update(status='submitted', job_id=subprocess.check_output(command, text=True).strip())
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'job_id': record['job_id'], 'bulk_output': str(base)}))


if __name__ == '__main__':
    main()
