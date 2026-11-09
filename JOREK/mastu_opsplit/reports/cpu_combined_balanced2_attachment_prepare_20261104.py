"""Probe balanced refinement with the original verified topological wall attachment."""
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
    base = bulk / 'combined_balanced2_attachment_20261104'
    reports = ROOT / 'reports'
    record_path = reports / 'cpu_combined_balanced2_attachment_20261104.json'
    if base.exists() or record_path.exists():
        raise ValueError('preserve existing crossing experiment')
    row = next(line for line in subprocess.check_output(['saldo', '-b', '--dcgp', '-n'], text=True).splitlines() if line.split()[0] == 'FUPA2_FAUNA')
    cols = row.split()
    if min(int(cols[3])-int(cols[4]), int(cols[7])-int(cols[8])) < 3072:
        raise ValueError('insufficient current project credit')
    patch = ROOT / 'patches/0014-cover-next-spatial-grid-capacity.patch'
    parent = bulk / 'combined_refined_solver_20261103/source'
    reference = parent.parent
    qualification = json.loads((reports / 'cpu_combined_refined_qualification_20261103.json').read_text())
    if qualification['status'] != 'combined_numerical_probes_verified' or sha256_file(parent / 'jorek_model303') != qualification['binary_sha256']:
        raise ValueError('qualified combined parent required')
    for path, digest in json.loads((reference / 'manifest.json').read_text())['input_hashes'].items():
        if sha256_file(path) != digest:
            raise ValueError('parent source/input changed')
    mpi = json.loads((reports / 'cpu_combined_mpi16_parity_20261104.json').read_text())
    if not mpi['passed'] or mpi['binary_sha256'] != qualification['binary_sha256']:
        raise ValueError('qualified same-source MPI16 proof required')
    capacity = bulk / 'combined_balanced2_20261104'
    import h5py
    import numpy as np
    capacity_proof = bulk / 'combined_balanced2_mpi16_20261104/control3/result.json'
    proof = json.loads(capacity_proof.read_text())
    if proof['status'] != 'completed' or not proof['parity']['passed'] or not proof['initial_parity']['passed'] or proof['binary_sha256'] != sha256_file(capacity / 'source/jorek_model303'):
        raise ValueError('capacity/MPI16 physical parity required')
    grid_reference = bulk / 'combined_balanced2_mpi16_20261104/probe3/grid_no_patch.h5'
    with h5py.File(reference / 'control3/grid_no_patch.h5') as old, h5py.File(grid_reference) as fine:
        original_corner = old['x'][:, 0, 0, 22766]
        refined_corner = fine['x'][:, 0, 0, 90966]
        if not np.array_equal(original_corner, refined_corner) or old['boundary'][22766] != 3 or fine['boundary'][90966] != 3:
            raise ValueError('original topological corner must be unchanged')
    base.mkdir()
    (base / 'source').mkdir()
    shutil.copyfile(capacity / 'source/jorek_model303', base / 'source/jorek_model303')
    (base / 'source/jorek_model303').chmod(0o555)
    for name in ['worker.py', 'jorek-env.sh']:
        shutil.copyfile(reference / name, base / name)
    worker = base / 'worker.py'
    text = worker.read_text().replace('from cpu_cached_rz_recovery_worker_20261103 import parity', 'from cpu_native_snapshot_parity_20261104 import parity')
    text = text.replace("'-n', '4', str(executable)", "'-n', str(config.get('mpi_ranks',4)), str(executable)")
    text = text.replace("if stage == 'regression8':", "if 'parity_reference' in config:")
    anchor = "            result['parity'] = parity(work/'jorek_restart.h5', reference,"
    text = text.replace(anchor, "            result['initial_parity'] = parity(work/'jorek000000.h5', reference.parent/'jorek000000.h5', manifest['parity_protocol'])\n" + anchor)
    anchor = "        result['initial'] = audit_restart(work/'jorek000000.h5', 0, 0.)"
    text = text.replace(anchor, "        import h5py\n        import numpy as np\n        with h5py.File(work/'grid_no_patch.h5') as actual, h5py.File(manifest['attachment_grid_reference']) as expected:\n            for key in ['x','vertex','size','boundary']:\n                if not np.array_equal(actual[key][()], expected[key][()]):\n                    raise ValueError('pre-extension geometry changed: ' + key)\n" + anchor)
    worker.write_text(text)
    counts = {'n_flux':160, 'n_tht':440, 'n_open':8, 'n_outer':64, 'n_inner':16,
              'n_leg':24, 'n_leg_out':160, 'n_private':16, 'n_up_leg':24, 'n_up_leg_out':160, 'n_up_priv':16,
              **{f'n_ext_block({i})':152 if i==3 else 80 for i in range(1,12)}}
    stages = {}
    counts.update({'R_block_points_right(11,1)': float(original_corner[0]), 'Z_block_points_right(11,1)': float(original_corner[1])})
    for label in ['probe3']:
        work = base / label
        work.mkdir()
        (work / 'input').write_text(namelist_overrides((reference / 'control3/input').read_text(), counts if label == 'probe3' else {}))
        for name in ['eqdsk.dat', 'jorek_equil_rz.h5']:
            shutil.copyfile(reference / 'control3' / name, work / name)
        stages[label] = {'steps':3, 'amplitudes':[0.,0.], 'timeout_seconds':18000 if label == 'probe3' else 1800, 'mpi_ranks':16}
        if label == 'control3':
            stages[label]['parity_reference'] = str(reference / 'control3/jorek_restart.h5')
    original = json.loads((reference / 'manifest.json').read_text())
    budget = {'authorization': 'User requested numerical convergence and completion of the science plan.',
              'scope': 'Qualified capacity/MPI16 binary; balanced full-grid three-step probe with patch11 interior guide snapped to the exact original coarse topological corner. Physical wall endpoints and physics unchanged.',
              'allocations_max': 1, 'attempts_each_max': 1, 'mpi_ranks': 16, 'threads': 1, 'memory_GiB': 512,
              'wall_hours_max': 6, 'rank_CPU_hours_max': 96, 'node_hours_max': 6, 'conservative_resource_hours_max': 3072,
              'saldo_row': row, 'equilibrium_tolerance': 1e-10, 'physics_changed': False, 'spatial_targets_changed': False}
    files = [capacity_proof, grid_reference, reference / 'control3/grid_no_patch.h5', base / 'worker.py', base / 'jorek-env.sh', base / 'source/jorek_model303',
             reports / 'cpu_native_snapshot_parity_20261104.py', reports / 'cpu_native_parity_protocol_20261102.json',
             reports / 'cpu_combined_mpi16_parity_20261104.json', reference / 'control3/result.json',
             reference / 'control3/jorek000000.h5', reference / 'control3/jorek_restart.h5',
             capacity / 'source/models/mod_settings.f90']
    files += [p for label in stages for p in (base / label).iterdir()]
    manifest = {key: original[key] for key in ['final_time_s', 'tstep_code_units', 'field_relative_target', 'integral_relative_target', 'equil_accuracy']}
    manifest.update(stages=stages, budget=budget, input_hashes={str(p): sha256_file(p) for p in files},
                    parity_protocol=json.loads((reports / 'cpu_native_parity_protocol_20261102.json').read_text()),
                    mesh_overrides=counts, attachment_grid_reference=str(grid_reference), attachment_reason='Rounded guide snaps past the same original corner after refinement; use its exact coordinates.',
                    limitation='Balanced-grid health and capacity parity only. Full matched spatial trajectories and complete physical coverage remain required; failed anisotropic meshes are retained.',
                    source_parent=str(parent), patch_sha256=sha256_file(patch))
    (base / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    script = reports / 'cpu_combined_balanced2_attachment_20261104.sbatch'
    python = '/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-balanced-attachment
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=06:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
{python} {base}/worker.py {base} {ROOT} probe3
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
