"""Build an isolated interior-handle repair and require unchanged control parity."""
import difflib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'reports'
sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
parent=B/'mumps_scaling8_20261104'
base=B/'xpoint_handle_20261104'
record_path=R/'cpu_xpoint_handle_20261104.json'
resume=sys.argv[1:]==['--resume-preparation']
if record_path.exists() or (base.exists() and not resume):raise ValueError('preserve previous attempt')
if resume and (not base.exists() or sorted(p.name for p in base.iterdir())!=['source']):
    raise ValueError('resume only the source-copy preparation before any native attempt')
qualification=json.loads((R/'cpu_mumps_scaling8_qualification_20261104.json').read_text())
if sha256_file(parent/'source/jorek_model303')!=qualification['binary_sha256']:raise ValueError('qualified parent changed')
for name,digest in json.loads((parent/'manifest.json').read_text())['input_hashes'].items():
    if sha256_file(name)!=digest:raise ValueError('frozen parent changed: '+name)
row=next(line for line in subprocess.check_output(['saldo','-b','--dcgp','-n'],text=True).splitlines() if line.split()[0]=='FUPA2_FAUNA')
c=row.split()
if min(int(c[3])-int(c[4]),int(c[7])-int(c[8]))<3072:raise ValueError('insufficient credit')
if not resume:
    base.mkdir()
    shutil.copytree(parent/'source',base/'source',ignore=shutil.ignore_patterns('.git','docs','reg_tests'))
source=base/'source/grids/grid_utils/finish_grid.f90'
old=source.read_text()
if old!=(parent/'source/grids/grid_utils/finish_grid.f90').read_text():raise ValueError('copied source changed')
new=old.replace('use constants\n','use constants\nuse phys_module, only: grid_to_wall\n',1)
anchor='if (n_order .ge. 5) then\n  call set_high_order_sizes(newelement_list)'
assert new.count(anchor)==1
new=new.replace(anchor,'if (grid_to_wall .and. n_order == 3) call opsplit_limit_xpoint_handle(newnode_list,newelement_list)\n\n'+anchor,1)
helper=ROOT/'native/opsplit_limit_xpoint_handle.f90'
new+='\n'+helper.read_text()
source.write_text(new)
patch=ROOT/'patches/0033-limit-folded-xpoint-adjacent-handle.patch'
with patch.open('x') as stream:stream.writelines(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/grids/grid_utils/finish_grid.f90',tofile='b/grids/grid_utils/finish_grid.f90'))
for name in ['worker.py','jorek-env.sh']:shutil.copyfile(parent/name,base/name)
worker_path=base/'worker.py'
worker=worker_path.read_text()
refined_worker=(B/'corner_handle_wall120_20261104/worker.py').read_text()
a=refined_worker.index('        import h5py\n')
z=refined_worker.index("        if stage == 'probe3' and 'planned_geometry_reference' in manifest:",a)
audit=refined_worker[a:z]
worker=worker.replace("        if stage == 'probe3' and 'planned_geometry_reference' in manifest:",audit+"        if stage == 'probe3' and 'planned_geometry_reference' in manifest:")
worker_path.write_text(worker)
original=json.loads((parent/'manifest.json').read_text())
manifest={key:original[key] for key in ['final_time_s','tstep_code_units','field_relative_target','integral_relative_target','equil_accuracy','parity_protocol']}
stages={}
files=[helper,patch,source,base/'worker.py',base/'jorek-env.sh',R/'cpu_native_snapshot_parity_20261104.py']
for label,ref in [('control3',parent/'control3'),('perturbed3',parent/'perturbed3'),('wall80',B/'scaling8_wall80_20261104/probe3'),('probe3',B/'combined_divertor160_20261104/probe3')]:
    result=json.loads((ref/'result.json').read_text())
    if label!='probe3' and result['status']!='completed':raise ValueError('healthy reference required')
    work=base/label;work.mkdir()
    for name in ['input','eqdsk.dat','jorek_equil_rz.h5']:shutil.copyfile(ref/name,work/name)
    stages[label]=dict(steps=3,mpi_ranks=16,amplitudes=[0.,0.],timeout_seconds=7200 if label=='probe3' else 3600)
    if label!='probe3':stages[label]['parity_reference']=str(ref/'jorek_restart.h5')
    files.extend([*work.iterdir(),ref/'result.json',ref/'jorek000000.h5',ref/'jorek_restart.h5'])
manifest.update(stages=stages,input_hashes={str(p):sha256_file(p) for p in files},scope='X-point-adjacent interior handle repair only. Three healthy controls must retain strict physical parity before the failed divertor160 mesh is retried. Equations, boundary curves and tolerances are unchanged.',source_parent=str(parent/'source'))
(base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
script=base/'run.sbatch'
script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-xpoint-handle-control
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
cd {base}/source
timeout --signal=TERM --kill-after=10s 1200s make -j2 jorek_model303 > {base}/build.out 2> {base}/build.err
for stage in control3 perturbed3 wall80 probe3; do
 /pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {base}/worker.py {base} {ROOT} "$stage"
done
''')
command=['sbatch','--parsable','--chdir',str(base),'--output',str(base/'batch-%j.out'),'--error',str(base/'batch-%j.err'),str(script)]
subprocess.run(['sbatch','--test-only',*command[1:]],check=True,capture_output=True)
record=dict(status='prepared',bulk_output=str(base),manifest_sha256=sha256_file(base/'manifest.json'),script_sha256=sha256_file(script),budget=dict(mpi_ranks=16,memory_GiB=512,wall_hours_max=6,rank_CPU_hours_max=96,saldo_row=row),scientific_gates_passed=[])
record_path.write_text(json.dumps(record,indent=2)+'\n')
record.update(status='submitted',job_id=subprocess.check_output(command,text=True).strip())
record_path.write_text(json.dumps(record,indent=2)+'\n')
print(record['job_id'])
