"""Require unchanged native controls before retrying radial640 with segment bounds."""
import difflib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'native'))
from mastu_opsplit.common import sha256_file
from boundary_incidence import replace_boundary_incidence
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102');parent=B/'scaling8_capacity400_20261104';base=B/'bounded_core_20261104';record_path=R/'cpu_bounded_core_20261104.json'
if base.exists() or record_path.exists():raise ValueError('preserve previous attempt')
qualification=json.loads((R/'cpu_scaling8_capacity400_qualification_20261104.json').read_text())
assert sha256_file(parent/'source/jorek_model303')==qualification['binary_sha256']
for p,h in qualification['input_hashes'].items():
 if sha256_file(p)!=h:raise ValueError('qualification changed: '+p)
row=next(line for line in subprocess.check_output(['saldo','-b','--dcgp','-n'],text=True).splitlines() if line.split()[0]=='FUPA2_FAUNA');c=row.split()
if min(int(c[3])-int(c[4]),int(c[7])-int(c[8]))<4320:raise ValueError('insufficient credit')
helper=ROOT/'native/opsplit_refine_bounded_core_crossing.f90';name='grids/grid_utils/find_crossing.f90';old=(parent/'source'/name).read_text()
anchor='call opsplit_refine_core_crossing('
assert old.count(anchor)==1
new=old.replace(anchor,'call opsplit_refine_bounded_core_crossing(')+'\n'+helper.read_text()
base.mkdir();shutil.copytree(parent/'source',base/'source',ignore=shutil.ignore_patterns('.git','docs','reg_tests'))
source=base/'source'/name;source.write_text(new);patch=ROOT/'patches/0038-reject-core-roots-outside-polar-segment.patch'
with patch.open('x') as f:f.writelines(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
fast_source=base/'source/grids/grid_utils/update_boundary_types.f90';fast_source.write_text(replace_boundary_incidence(fast_source.read_text()))
shutil.copyfile(parent/'jorek-env.sh',base/'jorek-env.sh')
shutil.copyfile(B/'xpoint_handle_20261104/worker.py',base/'worker.py')
original=json.loads((parent/'manifest.json').read_text());manifest={k:original[k] for k in ['final_time_s','tstep_code_units','field_relative_target','integral_relative_target','equil_accuracy','parity_protocol']}
files=[helper,patch,source,fast_source,ROOT/'native/boundary_incidence.py',ROOT/'patches/0036-cache-distinct-boundary-node-incidence.patch',base/'worker.py',base/'jorek-env.sh',R/'cpu_native_snapshot_parity_20261104.py'];stages={}
refs=[('control3',parent/'control3'),('perturbed3',parent/'perturbed3'),('radial320',B/'scaling8_capacity400_radial320_20261104/probe3'),('poloidal880',parent/'poloidal880'),('probe3',B/'scaling8_capacity400_radial640_20261104/probe3')]
for label,ref in refs:
 work=base/label;work.mkdir()
 for name in ['input','eqdsk.dat','jorek_equil_rz.h5']:shutil.copyfile(ref/name,work/name)
 stages[label]=dict(steps=3,mpi_ranks=16,amplitudes=[0.,0.],timeout_seconds=10800 if label=='probe3' else 4500)
 if label!='probe3':
  proof=json.loads((ref/'result.json').read_text());assert proof['status']=='completed'
  stages[label]['parity_reference']=str(ref/'jorek_restart.h5');files.extend([ref/'result.json',ref/'jorek000000.h5',ref/'jorek_restart.h5'])
 files.extend(work.iterdir())
manifest.update(stages=stages,input_hashes={str(p):sha256_file(p) for p in files},scope='Only reject exact closed-core roots outside their original polar segment, letting native contour search continue to other pieces. Same-capacity original baseline/perturbed, radial320 and poloidal880 strict field/gradient parity required. Cached boundary incidence preserves classifications; independent q16 geometry audit required. No physics or acceptance change.')
(base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');script=base/'run.sbatch'
script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-bounded-core-radial640
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
for stage in control3 perturbed3 radial320 poloidal880 probe3; do
 /pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {base}/worker.py {base} {ROOT} "$stage"
done
''')
cmd=['sbatch','--parsable','--chdir',str(base),'--output',str(base/'batch-%j.out'),'--error',str(base/'batch-%j.err'),str(script)]
pre=subprocess.run(['sbatch','--test-only',*cmd[1:]],check=True,text=True,capture_output=True)
record=dict(status='prepared',bulk_output=str(base),manifest_sha256=sha256_file(base/'manifest.json'),script_sha256=sha256_file(script),scheduler_preflight=pre.stderr,budget=dict(authorization='Continue full spatial convergence as requested.',mpi_ranks=16,memory_GiB=720,wall_hours_max=6,rank_CPU_hours_max=96,saldo_row=row),scientific_gates_passed=[])
record_path.write_text(json.dumps(record,indent=2)+'\n');record.update(status='submitted',job_id=subprocess.check_output(cmd,text=True).strip());record_path.write_text(json.dumps(record,indent=2)+'\n');print(record['job_id'])
