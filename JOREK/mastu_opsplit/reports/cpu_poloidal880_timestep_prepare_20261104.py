"""Diagnose the first poloidal880 step from its preserved initial state at three timesteps."""
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
parent = B / 'combined_core_poloidal880_20261104'
base = B / 'poloidal880_timestep_20261104'
if base.exists():
    raise ValueError('preserve existing timestep diagnosis')
geometry = json.loads((R / 'cpu_core_poloidal880_geometry_diagnosis_20261104.json').read_text())
initial = parent / 'probe3/jorek000000.h5'
if sha256_file(initial) != geometry['sha256'] or any(x['folds'] for x in geometry['quadrature'].values()):
    raise ValueError('preserved positive q8/q16 initial geometry required')
mpi = json.loads((R / 'cpu_combined_mpi16_parity_20261104.json').read_text())
if not mpi['passed'] or sha256_file(parent / 'source/jorek_model303') != mpi['binary_sha256']:
    raise ValueError('same-binary MPI16 control qualification required')
row = next(x for x in subprocess.check_output(['saldo', '-b', '--dcgp', '-n'], text=True).splitlines()
           if x.split()[0] == 'FUPA2_FAUNA')
c = row.split()
if min(int(c[3])-int(c[4]), int(c[7])-int(c[8])) < 1536:
    raise ValueError('insufficient current credit for capped diagnosis')
(base / 'source').mkdir(parents=True)
shutil.copyfile(parent / 'source/jorek_model303', base / 'source/jorek_model303')
(base / 'source/jorek_model303').chmod(0o555)
shutil.copyfile(parent / 'jorek-env.sh', base / 'jorek-env.sh')
text = (parent / 'worker.py').read_text().replace("'-n', '4', str(executable)", "'-n', '16', str(executable)")
text = text.replace('import json\n', 'import json\nimport shutil\n', 1)
text = text.replace('    start = time.monotonic()', "    shutil.copyfile(work/'initial_reference.h5', work/'jorek_restart.h5')\n    shutil.copyfile(work/'initial_reference.h5', work/'jorek000000.h5')\n    start = time.monotonic()")
text = text.replace("manifest['final_time_s']*config['steps']/48", "config['expected_time_s']")
(base / 'worker.py').write_text(text)
original = json.loads((parent / 'manifest.json').read_text())
manifest = {key: original[key] for key in ['final_time_s', 'tstep_code_units', 'field_relative_target',
                                         'integral_relative_target', 'equil_accuracy']}
files = [base / 'source/jorek_model303', base / 'worker.py', base / 'jorek-env.sh',
         R / 'cpu_core_poloidal880_geometry_diagnosis_20261104.json', R / 'cpu_combined_mpi16_parity_20261104.json']
manifest['stages'] = {}
for label, dt in [('dt1', .0625), ('dt_half', .03125), ('dt_quarter', .015625)]:
    work = base / label
    work.mkdir()
    shutil.copyfile(initial, work / 'initial_reference.h5')
    for name in ['eqdsk.dat', 'jorek_equil_rz.h5']:
        shutil.copyfile(parent / 'probe3' / name, work / name)
    (work / 'input').write_text(namelist_overrides((parent / 'probe3/input').read_text(),
        {'restart': True, 'nstep_n': 1, 'nout': 1, 'tstep_n': dt}))
    files.extend(work.iterdir())
    manifest['stages'][label] = {'steps': 1, 'amplitudes': [0., 0.], 'timeout_seconds': 3300,
                               'tstep_code_units': dt, 'expected_time_s': dt * 4.686377037426928e-7}
manifest.update(input_hashes={str(p): sha256_file(p) for p in files},
                scope='Diagnostic one-step restarts only. No full spatial or timestep acceptance; no production trajectory.',
                initial_restart_sha256=sha256_file(initial), binary_sha256=mpi['binary_sha256'])
(base / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
python = '/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
for label in manifest['stages']:
    script = base / f'{label}.sbatch'
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-poloidal880-{label}
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=01:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
{python} {base}/worker.py {base} {ROOT} {label}
''')
    command = ['sbatch', '--parsable', '--chdir', str(base), '--output', str(base / f'{label}-%j.out'),
               '--error', str(base / f'{label}-%j.err'), str(script)]
    subprocess.run(['sbatch', '--test-only', *command[1:]], check=True, capture_output=True)
    record = {'status': 'prepared', 'bulk_output': str(base), 'stage': label,
              'script_sha256': sha256_file(script), 'manifest_sha256': sha256_file(base / 'manifest.json'),
              'budget': {'mpi_ranks': 16, 'memory_GiB': 512, 'wall_hours_max': 1, 'rank_CPU_hours_max': 16, 'saldo_row': row},
              'scientific_gates_passed': []}
    output = R / f'cpu_poloidal880_{label}_20261104.json'
    with output.open('x') as stream:
        json.dump(record, stream, indent=2)
    record.update(status='submitted', job_id=subprocess.check_output(command, text=True).strip())
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(label, record['job_id'], flush=True)
