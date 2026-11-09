"""Capacity-only build, strict control parity, then fresh poloidal1760 probe."""
import difflib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.case import namelist_overrides
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102');parent=B/'mumps_scaling8_20261104';base=B/'scaling8_capacity400_20261104'
record_path=R/'cpu_scaling8_capacity400_20261104.json'
if base.exists() or record_path.exists():raise ValueError('preserve prior attempt')
proof=json.loads((R/'cpu_mumps_scaling8_qualification_20261104.json').read_text())
assert proof['binary_sha256']==sha256_file(parent/'source/jorek_model303')
for p,h in json.loads((parent/'manifest.json').read_text())['input_hashes'].items():
 if sha256_file(p)!=h:raise ValueError('parent changed: '+p)
row=next(line for line in subprocess.check_output(['saldo','-b','--dcgp','-n'],text=True).splitlines() if line.split()[0]=='FUPA2_FAUNA');c=row.split()
if min(int(c[3])-int(c[4]),int(c[7])-int(c[8]))<6144:raise ValueError('insufficient credit')
base.mkdir();shutil.copytree(parent/'source',base/'source',ignore=shutil.ignore_patterns('.git','docs','reg_tests'))
p=base/'source/models/mod_settings.f90';old=p.read_text();new=old.replace('= 160001','= 400001').replace('n_boundary_max    = 4001','n_boundary_max    = 8001')
assert old.count('= 160001')==2 and new.count('n_boundary_max    = 8001')==1
p.write_text(new);patch=ROOT/'patches/0032-capacity-for-poloidal1760.patch'
with patch.open('x') as f:f.writelines(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/models/mod_settings.f90',tofile='b/models/mod_settings.f90'))
for name in ['worker.py','jorek-env.sh']:shutil.copyfile(parent/name,base/name)
original=json.loads((parent/'manifest.json').read_text());manifest={k:original[k] for k in ['final_time_s','tstep_code_units','field_relative_target','integral_relative_target','equil_accuracy','parity_protocol']}
files=[p,patch,base/'worker.py',base/'jorek-env.sh',R/'cpu_native_snapshot_parity_20261104.py'];stages={}
for label,ref in [('control3',parent/'control3'),('perturbed3',parent/'perturbed3'),('poloidal880',B/'scaling8_poloidal880_20261104/probe3'),('probe3',B/'scaling8_poloidal880_20261104/probe3')]:
 assert json.loads((ref/'result.json').read_text())['status']=='completed'
 work=base/label;work.mkdir();overrides={'n_tht':1760} if label=='probe3' else {}
 (work/'input').write_text(namelist_overrides((ref/'input').read_text(),overrides))
 for name in ['eqdsk.dat','jorek_equil_rz.h5']:shutil.copyfile(ref/name,work/name)
 stages[label]=dict(steps=3,mpi_ranks=16,amplitudes=[0.,0.],timeout_seconds=10800 if label=='probe3' else 4500)
 if label!='probe3':stages[label]['parity_reference']=str(ref/'jorek_restart.h5')
 files.extend([*work.iterdir(),ref/'result.json',ref/'jorek000000.h5',ref/'jorek_restart.h5'])
manifest.update(stages=stages,input_hashes={str(p):sha256_file(p) for p in files},source_parent=str(parent/'source'),scope='Storage capacity only: 400001 nodes/elements and 8001 boundary elements. Original baseline/perturbed and poloidal880 strict parity required before poloidal1760; no equations, physics, tolerances or geometry algorithms changed.')
(base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');script=base/'run.sbatch'
script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-scaling8-poloidal1760-probe
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=720G
#SBATCH --time=06:00:00
set -euo pipefail
source {base}/jorek-env.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd {base}/source
timeout --signal=TERM --kill-after=10s 1500s make -j2 jorek_model303 > {base}/build.out 2> {base}/build.err
for stage in control3 perturbed3 poloidal880 probe3; do
 /pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {base}/worker.py {base} {ROOT} "$stage"
done
''')
cmd=['sbatch','--parsable','--chdir',str(base),'--output',str(base/'batch-%j.out'),'--error',str(base/'batch-%j.err'),str(script)]
subprocess.run(['sbatch','--test-only',*cmd[1:]],check=True,capture_output=True)
record=dict(status='prepared',bulk_output=str(base),manifest_sha256=sha256_file(base/'manifest.json'),script_sha256=sha256_file(script),budget=dict(authorization='Continued full spatial convergence requested by user.',mpi_ranks=16,memory_GiB=720,wall_hours_max=6,rank_CPU_hours_max=96,saldo_row=row),scientific_gates_passed=[])
record_path.write_text(json.dumps(record,indent=2)+'\n');record.update(status='submitted',job_id=subprocess.check_output(cmd,text=True).strip());record_path.write_text(json.dumps(record,indent=2)+'\n');print(record['job_id'])
