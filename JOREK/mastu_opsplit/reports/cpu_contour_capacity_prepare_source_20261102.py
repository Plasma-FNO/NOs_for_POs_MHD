import json,shutil,subprocess,difflib,sys
from pathlib import Path
from datetime import datetime,timezone
root=Path('/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit');sys.path.insert(0,str(root/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.case import namelist_overrides
r=root/'reports';phasepath=r/'cpu_refinement_continuation_20261102.json';phase=json.loads(phasepath.read_text())
budget=json.loads(Path(phase['budget_record']).read_text());assert sha256_file(phase['budget_record'])==phase['budget_record_sha256']
assert len(phase['runs'])<budget['phase']['simulation_submissions_max']
cpu=sum(x['requested_cpu_core_hours'] for x in phase['builds']+phase['runs'])+8
node=sum(x['requested_node_hours'] for x in phase['builds']+phase['runs'])+2
assert cpu<=128 and node<=8
parent=json.loads((r/'cpu_refinement_capacity160k_build_20261102.json').read_text()); assert sha256_file(parent['binary_path'])==parent['binary_sha256']
bulk=Path(parent['source_directory']).parents[1];diagnostic=bulk/'contour_bounds_debug_20261102'
error=(diagnostic/'stderr.log').read_text();assert "Index '10001'" in error and "'r_outer_contour'" in error
base=bulk/'contour_capacity_fix_20261102';base.mkdir();src=base/'source';shutil.copytree(parent['source_directory'],src)
f=src/'grids/grid_xpoint_data.f90';old=f.read_text();assert old.count('n_contour_max = 10000')==1;new=old.replace('n_contour_max = 10000','n_contour_max = 40000');f.write_text(new)
patch=root/'patches/0009-refined-flux-contour-capacity.patch';patch.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/grids/grid_xpoint_data.f90',tofile='b/grids/grid_xpoint_data.f90')))
(src/'.obj/reorder_and_clean_flux_surfaces.o').unlink()
for name,template in [('parity8','cpu_refine_stable_parity8_20261102'),('quad_probe3','cpu_refine_quad_probe3_20261102')]:
 work=base/name;work.mkdir();oldwork=bulk/template/'attempt_1/work'
 shutil.copyfile(oldwork/'eqdsk.dat',work/'eqdsk.dat')
 text=(oldwork/'input').read_text()
 if name=='quad_probe3':
  text=namelist_overrides(text,{'regrid_from_rz':True});shutil.copyfile(oldwork/'jorek_equil_rz.h5',work/'jorek_equil_rz.h5')
 (work/'input').write_text(text)
shutil.copyfile(bulk/'cpu_io_baseline48_20261102/attempt_1/work/jorek000008.h5',base/'reference8.h5')
worker=r/'cpu_contour_fix_worker_source_20261102.py'
worker.write_text('''"""Single-allocation capacity regression followed by guarded refined-mesh probe."""
import json,subprocess,sys,time
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.runner import restart_health,classify
base=Path(sys.argv[1]);binary=base/'source/jorek_model303'
for name,steps,horizon in [('parity8',8,2.343188518713464e-7),('quad_probe3',3,8.78695694517549e-8)]:
 work=base/name;t=time.monotonic()
 with (work/'stdout.log').open('w') as out,(work/'stderr.log').open('w') as err,(work/'input').open() as inp:
  proc=subprocess.run(['timeout','--signal=TERM','--kill-after=10s','3000s','mpirun','--bind-to','none','-n','4',str(binary)],cwd=work,stdin=inp,stdout=out,stderr=err)
 text=(work/'stdout.log').read_text()+(work/'stderr.log').read_text()
 result={'label':name,'returncode':proc.returncode,'status':classify(proc.returncode,proc.returncode==124,text),'wall_seconds':time.monotonic()-t,'binary_sha256':sha256_file(binary),'scientific_gates_passed':[]}
 try:
  assert result['status']=='completed',result['status']
  result['health']=restart_health(work/'jorek_restart.h5',expected_steps=steps,final_time_s=horizon)
  if name=='parity8':
   with h5py.File(work/'jorek_restart.h5') as a,h5py.File(base/'reference8.h5') as b:
    keys=['x','values','vertex','size','boundary','constrained','n_sons','xtime','energies']
    result['exact_reference_arrays']={key:bool(np.array_equal(a[key][()],b[key][()])) for key in keys}
   assert all(result['exact_reference_arrays'].values()),'capacity regression changed native arrays'
 except Exception as e:
  result.update(status='failed_validation',error=str(e))
 (work/'result.json').write_text(json.dumps(result,indent=2)+'\\n')
 print(json.dumps(result),flush=True)
 if result['status']!='completed':sys.exit(1)
''')
script=r/'cpu_contour_capacity_fix_20261102.sbatch';env=root/'runs/prepared_equilibrium_dcgp_authorized_20261102/jorek-env.sh';python='/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-contour-fix
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
source {env}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd {src}
make -j2 .obj/reorder_and_clean_flux_surfaces.o F90FLAGS='-g -fcheck=bounds -fbacktrace' > {base}/build.out 2> {base}/build.err
make -j2 jorek_model303 >> {base}/build.out 2>> {base}/build.err
sha256sum jorek_model303 > {base}/binary.sha256
{python} {worker} {base}
''')
record={'label':'contour_capacity_fix','status':'prepared','scope':'Capacity-only source build, exact 8-step existing-mesh regression, then cached-RZ nested 3-step probe; sequential and guarded in one allocation','source_directory':str(src),'bulk_output':str(base),'script':str(script),'script_sha256':sha256_file(script),'worker':str(worker),'worker_sha256':sha256_file(worker),'patch':str(patch),'patch_sha256':sha256_file(patch),'requested_cpu_core_hours':8,'requested_node_hours':2,'mpi_ranks':4,'omp_threads':1,'memory_GiB':128,'wall_seconds':7200,'maximum_simultaneous_compilers':2,'n_contour_max':40000,'equations_modified':False,'startup_hashes':{str(p.relative_to(base)):sha256_file(p) for name in ['parity8','quad_probe3'] for p in (base/name).iterdir()},'diagnosis_log_sha256':sha256_file(diagnostic/'stderr.log'),'scientific_gates_passed':[]}
cmd=['sbatch','--parsable','--chdir',str(base),'--output',str(base/'batch-%j.out'),'--error',str(base/'batch-%j.err'),str(script)]
check=subprocess.run(['sbatch','--test-only',*cmd[1:]],capture_output=True,text=True);assert check.returncode==0,check.stderr
out=subprocess.run(cmd,capture_output=True,text=True);assert out.returncode==0,out.stderr
record.update(job_id=out.stdout.strip(),status='submitted',submitted_utc=datetime.now(timezone.utc).isoformat())
phase['runs'].append(record);phase.update(reserved_cpu_core_hours=cpu,reserved_node_hours=node)
phasepath.write_text(json.dumps(phase,indent=2)+'\n');(r/'cpu_contour_capacity_fix_20261102.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'job_id':record['job_id'],'reserved_cpu_hours':cpu,'reserved_node_hours':node}))
