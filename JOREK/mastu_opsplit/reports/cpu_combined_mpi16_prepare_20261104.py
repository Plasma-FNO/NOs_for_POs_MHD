"""Test corrected-solver MPI parity before using more ranks for larger spatial grids."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'reports'
sys.path.insert(0, str(ROOT / 'src'))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import sha256_file

B = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
parent = B / 'combined_refined_solver_20261103'
base = B / 'combined_mpi16_20261104'
if base.exists():
    raise ValueError('preserve existing MPI attempts')
qualification = json.loads((R / 'cpu_combined_refined_qualification_20261103.json').read_text())
if qualification['status'] != 'combined_numerical_probes_verified':
    raise ValueError('qualified corrected solver required')
if sha256_file(parent / 'source/jorek_model303') != qualification['binary_sha256']:
    raise ValueError('qualified binary changed')
original = json.loads((parent / 'manifest.json').read_text())
for path, digest in original['input_hashes'].items():
    if sha256_file(path) != digest:
        raise ValueError('qualified source changed')
row = next(x for x in subprocess.check_output(['saldo', '-b', '--dcgp', '-n'], text=True).splitlines()
           if x.split()[0] == 'FUPA2_FAUNA')
cols = row.split()
if min(int(cols[3])-int(cols[4]), int(cols[7])-int(cols[8])) < 896:
    raise ValueError('insufficient credit for capped parity checks')
(base / 'source').mkdir(parents=True)
shutil.copyfile(parent / 'source/jorek_model303', base / 'source/jorek_model303')
(base / 'source/jorek_model303').chmod(0o555)
shutil.copyfile(parent / 'jorek-env.sh', base / 'jorek-env.sh')
text = (parent / 'worker.py').read_text()
old = "'mpirun', '--bind-to', 'none', '-n', '4', str(executable)"
if text.count(old) != 1:
    raise ValueError('unexpected native launch command')
(base / 'worker.py').write_text(text.replace(old,
    "'mpirun', '--bind-to', 'none', '-n', str(config['mpi_ranks']), str(executable)"))
files = [base / 'source/jorek_model303', base / 'jorek-env.sh', base / 'worker.py']
manifest = {key: original[key] for key in ['final_time_s', 'tstep_code_units', 'field_relative_target',
                                         'integral_relative_target', 'equil_accuracy']}
manifest['stages'] = {}
for label, ranks, amplitude in [('baseline16', 16, 0.), ('perturbed16', 16, .05), ('perturbed4', 4, .05)]:
    work = base / label
    work.mkdir()
    (work / 'input').write_text(namelist_overrides((parent / 'control3/input').read_text(),
        {'opsplit_amplitude(1)': amplitude, 'opsplit_amplitude(2)': amplitude}))
    for name in ['eqdsk.dat', 'jorek_equil_rz.h5']:
        shutil.copyfile(parent / 'control3' / name, work / name)
    files.extend(work.iterdir())
    manifest['stages'][label] = {'steps': 3, 'amplitudes': [amplitude, amplitude], 'timeout_seconds': 3300, 'mpi_ranks': ranks}
manifest.update(input_hashes={str(p): sha256_file(p) for p in files},
                binary_sha256=qualification['binary_sha256'],
                limitation='Diagnostic only until both baseline and perturbed initial/final q8 field and gradient parity pass at unchanged tolerances. Legacy 16-rank failure remains preserved.')
(base / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
python = '/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
jobs = []
for name, ranks, memory, hours, stages in [('mpi16', 16, 256, 2, ['baseline16', 'perturbed16']),
                                          ('mpi4', 4, 256, 1, ['perturbed4'])]:
    script = base / f'{name}.sbatch'
    body = '\n'.join(f'{python} {base}/worker.py {base} {ROOT} {stage}' for stage in stages)
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-combined-{name}-parity
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks={ranks}
#SBATCH --cpus-per-task=1
#SBATCH --mem={memory}G
#SBATCH --time={hours:02d}:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
{body}
''')
    command = ['sbatch', '--parsable', '--chdir', str(base), '--output', str(base / f'{name}-%j.out'),
               '--error', str(base / f'{name}-%j.err'), str(script)]
    subprocess.run(['sbatch', '--test-only', *command[1:]], check=True, capture_output=True)
    record = {'status': 'prepared', 'bulk_output': str(base), 'stages': stages,
              'script_sha256': sha256_file(script), 'manifest_sha256': sha256_file(base / 'manifest.json'),
              'budget': {'mpi_ranks': ranks, 'memory_GiB': memory, 'wall_hours_max': hours,
                         'rank_CPU_hours_max': ranks * hours, 'saldo_row': row}, 'scientific_gates_passed': []}
    path = R / f'cpu_combined_{name}_parity_job_20261104.json'
    with path.open('x') as stream:
        json.dump(record, stream, indent=2)
    job = subprocess.check_output(command, text=True).strip()
    record.update(status='submitted', job_id=job)
    path.write_text(json.dumps(record, indent=2) + '\n')
    jobs.append(job)
print(json.dumps(jobs), flush=True)
