"""Isolated scaling/conditioning diagnosis; retain equations, initial states and tolerances."""
import difflib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'reports'
sys.path.insert(0, str(ROOT / 'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.case import namelist_overrides

B = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
parent = B / 'combined_refined_solver_20261103'
base = B / 'mumps_scaling8_20261104'
record_path = R / 'cpu_mumps_scaling8_20261104.json'
if base.exists() or record_path.exists():
    raise ValueError('preserve existing scaling diagnosis')
qualification = json.loads((R / 'cpu_combined_refined_qualification_20261103.json').read_text())
if qualification['status'] != 'combined_numerical_probes_verified' or sha256_file(parent / 'source/jorek_model303') != qualification['binary_sha256']:
    raise ValueError('qualified parent required')
for path, digest in json.loads((parent / 'manifest.json').read_text())['input_hashes'].items():
    if sha256_file(path) != digest:
        raise ValueError('parent changed: ' + path)
row = next(line for line in subprocess.check_output(['saldo', '-b', '--dcgp', '-n'], text=True).splitlines() if line.split()[0] == 'FUPA2_FAUNA')
c = row.split()
if min(int(c[3])-int(c[4]), int(c[7])-int(c[8])) < 1024:
    raise ValueError('insufficient current credit')
base.mkdir()
shutil.copytree(parent / 'source', base / 'source', ignore=shutil.ignore_patterns('.git', 'docs', 'reg_tests'))
source = base / 'source/solvers/mod_mumps.f90'
old = source.read_text()
new = old.replace('mmss%mumps_par%icntl(11) = 2', 'mmss%mumps_par%icntl(11) = 1')
new = new.replace('mmss%mumps_par%icntl(8)  = 7                            ! row and column scaling  7: automatic scaling', 'mmss%mumps_par%icntl(8)  = 8                            ! rigorous iterative row/column scaling (MUMPS 5.7.3)')
anchor = '      mmss%mumps_par%infog(15),mmss%mumps_par%rinfog(7:8)'
new = new.replace(anchor, anchor + "\n    if (my_id == 0) write(*,*) 'MUMPS norms, residual, backward/forward errors, conditions:', &\n      mmss%mumps_par%rinfog(4:11)")
if new == old or new.count('rigorous iterative') != 1:
    raise ValueError('unexpected source')
source.write_text(new)
patch = ROOT / 'patches/0030-diagnose-rigorous-mumps-scaling.patch'
with patch.open('x') as stream:
    stream.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='a/solvers/mod_mumps.f90', tofile='b/solvers/mod_mumps.f90'))
shutil.copyfile(parent / 'jorek-env.sh', base / 'jorek-env.sh')
worker = (B / 'combined_balanced2_mpi16_20261104/worker.py').read_text()
worker = worker.replace('import json\n', 'import json\nimport shutil\n', 1)
worker = worker.replace('    start = time.monotonic()', "    if config.get('restart_diagnostic'):\n        shutil.copyfile(work/'initial_reference.h5', work/'jorek_restart.h5')\n        shutil.copyfile(work/'initial_reference.h5', work/'jorek000000.h5')\n    start = time.monotonic()")
(base / 'worker.py').write_text(worker)
original = json.loads((parent / 'manifest.json').read_text())
manifest = {key: original[key] for key in ['final_time_s', 'tstep_code_units', 'field_relative_target', 'integral_relative_target', 'equil_accuracy']}
stages = {}
refs = {'control3': B / 'combined_mpi16_20261104/baseline16', 'perturbed3': B / 'combined_mpi16_20261104/perturbed16', 'poloidal880_step1': B / 'combined_core_poloidal880_20261104/probe3'}
files = [base / 'worker.py', base / 'jorek-env.sh', source, patch, R / 'cpu_native_snapshot_parity_20261104.py', R / 'cpu_native_parity_protocol_20261102.json']
for label, ref in refs.items():
    work = base / label
    work.mkdir()
    for name in ['eqdsk.dat', 'jorek_equil_rz.h5']:
        shutil.copyfile(ref / name, work / name)
    config = {'steps': 3, 'mpi_ranks': 16, 'amplitudes': [0., 0.], 'timeout_seconds': 1800}
    overrides = {}
    if label == 'poloidal880_step1':
        overrides = {'restart': True, 'nstep_n': 1, 'nout': 1}
        config.update(steps=1, restart_diagnostic=True, timeout_seconds=3300)
        shutil.copyfile(ref / 'jorek000000.h5', work / 'initial_reference.h5')
    else:
        if json.loads((ref / 'result.json').read_text())['status'] != 'completed':
            raise ValueError('healthy parity reference required')
        config['parity_reference'] = str(ref / 'jorek_restart.h5')
        files.extend([ref / 'jorek000000.h5', ref / 'jorek_restart.h5', ref / 'result.json'])
    (work / 'input').write_text(namelist_overrides((ref / 'input').read_text(), overrides))
    stages[label] = config
    files.extend(work.iterdir())
manifest.update(stages=stages, input_hashes={str(p):sha256_file(p) for p in files}, parity_protocol=json.loads((R / 'cpu_native_parity_protocol_20261102.json').read_text()),
                scope='Numerical scaling experiment only. All stages run independently for diagnosis; neither health nor spatial convergence is assumed. No physics changes.', source_parent=str(parent / 'source'))
(base / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
python = '/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
script = base / 'run.sbatch'
script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-scaling8-diagnosis
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=02:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd {base}/source
timeout --signal=TERM --kill-after=10s 1200s make -j2 jorek_model303 > {base}/build.out 2> {base}/build.err
status=0
{python} {base}/worker.py {base} {ROOT} control3 || status=1
{python} {base}/worker.py {base} {ROOT} perturbed3 || status=1
{python} {base}/worker.py {base} {ROOT} poloidal880_step1 || status=1
exit "$status"
''')
subprocess.run(['bash', '-n', str(script)], check=True)
command = ['sbatch', '--parsable', '--chdir', str(base), '--output', str(base / 'batch-%j.out'), '--error', str(base / 'batch-%j.err'), str(script)]
preflight = subprocess.run(['sbatch', '--test-only', *command[1:]], check=True, capture_output=True, text=True)
record = {'status': 'prepared', 'bulk_output': str(base), 'manifest_sha256': sha256_file(base / 'manifest.json'), 'script_sha256': sha256_file(script),
          'budget': {'mpi_ranks':16, 'memory_GiB':512, 'wall_hours_max':2, 'rank_CPU_hours_max':32, 'conservative_resource_hours_max':1024, 'saldo_row':row},
          'scheduler_preflight':preflight.stderr, 'scientific_gates_passed':[]}
record_path.write_text(json.dumps(record, indent=2)+'\n')
record.update(status='submitted', job_id=subprocess.check_output(command, text=True).strip())
record_path.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({'job_id':record['job_id'], 'bulk_output':str(base)}))
