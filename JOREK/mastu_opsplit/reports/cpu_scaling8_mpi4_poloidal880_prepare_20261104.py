"""Independently replay the rescued poloidal880 step at four MPI ranks with strict physical parity."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102');parent=B/'mumps_scaling8_20261104';base=B/'scaling8_mpi4_poloidal880_20261104';ref=parent/'poloidal880_step1'
if base.exists():raise ValueError('preserve prior attempt')
qualification=R/'cpu_mumps_scaling8_qualification_20261104.json';proof=json.loads(qualification.read_text());digest=sha256_file(parent/'source/jorek_model303')
assert proof['status']=='scaling8_control_parity_and_restart_rescue_verified' and proof['binary_sha256']==digest
r=json.loads((ref/'result.json').read_text());assert r['status']=='completed' and r['binary_sha256']==digest
row=next(x for x in subprocess.check_output(['saldo','-b','--dcgp','-n'],text=True).splitlines() if x.split()[0]=='FUPA2_FAUNA');c=row.split()
if min(int(c[3])-int(c[4]),int(c[7])-int(c[8]))<1024:raise ValueError('insufficient current credit')
(base/'source').mkdir(parents=True);work=base/'replay1';work.mkdir()
for name in ['worker.py','jorek-env.sh']:shutil.copyfile(parent/name,base/name)
shutil.copyfile(parent/'source/jorek_model303',base/'source/jorek_model303');(base/'source/jorek_model303').chmod(0o555)
for name in ['input','eqdsk.dat','jorek_equil_rz.h5','initial_reference.h5']:shutil.copyfile(ref/name,work/name)
original=json.loads((parent/'manifest.json').read_text());m={k:original[k] for k in ['final_time_s','tstep_code_units','field_relative_target','integral_relative_target','equil_accuracy','parity_protocol']}
files=[base/'worker.py',base/'jorek-env.sh',base/'source/jorek_model303',qualification,ref/'result.json',ref/'jorek_restart.h5',ref/'jorek000000.h5',*work.iterdir()]
m.update(stages={'replay1':{'steps':1,'mpi_ranks':4,'amplitudes':[0.,0.],'timeout_seconds':6900,'restart_diagnostic':True,'parity_reference':str(ref/'jorek_restart.h5')}},input_hashes={str(p):sha256_file(p) for p in files},scope='Independent one-step MPI4/16 physical parity only, not spatial convergence.')
(base/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
script=base/'run.sbatch';script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-scaling8-p880-mpi4
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=02:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {base}/worker.py {base} {ROOT} replay1
''')
cmd=['sbatch','--parsable','--chdir',str(base),'--output',str(base/'batch-%j.out'),'--error',str(base/'batch-%j.err'),str(script)];subprocess.run(['sbatch','--test-only',*cmd[1:]],check=True,capture_output=True)
d={'status':'submitted','bulk_output':str(base),'binary_sha256':digest,'manifest_sha256':sha256_file(base/'manifest.json'),'script_sha256':sha256_file(script),'budget':{'mpi_ranks':4,'memory_GiB':512,'wall_hours_max':2,'rank_CPU_hours_max':8,'saldo_row':row},'job_id':subprocess.check_output(cmd,text=True).strip(),'scientific_gates_passed':[]}
(R/'cpu_scaling8_mpi4_poloidal880_20261104.json').write_text(json.dumps(d,indent=2)+'\n');print(d['job_id'])
