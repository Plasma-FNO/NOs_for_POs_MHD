import json,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
root=Path('/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit');sys.path.insert(0,str(root/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.case import namelist_overrides
r=root/'reports';phasepath=r/'cpu_refinement_continuation_20261102.json';phase=json.loads(phasepath.read_text());budget=json.loads(Path(phase['budget_record']).read_text())
assert sha256_file(phase['budget_record'])==phase['budget_record_sha256'];assert len(phase['runs'])<budget['phase']['simulation_submissions_max']
parent=next(x for x in phase['runs'] if x['label']=='contour_capacity_fix');bulk=Path(parent['bulk_output']).parent;base=bulk/'fixed_rz_nested_probe3_20261102';base.mkdir()
parity=json.loads((Path(parent['bulk_output'])/'parity8/result.json').read_text());assert parity['status']=='completed' and all(parity['exact_reference_arrays'].values())
previous=json.loads((Path(parent['bulk_output'])/'quad_probe3/result.json').read_text());assert previous['status']=='failed_validation'
assert 'supposed to find 4 strikes' in (Path(parent['bulk_output'])/'quad_probe3/stdout.log').read_text()
binary=Path(parent['source_directory'])/'jorek_model303';assert sha256_file(binary)==parity['binary_sha256']
query=subprocess.run(['squeue','-h','-j',parent['job_id'],'-o','%T'],text=True,capture_output=True);assert not query.stdout.strip(),query.stdout
cpu=sum(x['requested_cpu_core_hours'] for x in phase['builds']+phase['runs'])+8;node=sum(x['requested_node_hours'] for x in phase['builds']+phase['runs'])+2
assert cpu<=128 and node<=8
old=Path(parent['bulk_output'])/'quad_probe3';fine=bulk/'cpu_refine_stable_parity8_20261102/attempt_1/work'
for name,source in [('eqdsk.dat',old/'eqdsk.dat'),('jorek_equil_rz.h5',fine/'jorek_equil_rz.h5')]:shutil.copyfile(source,base/name)
(base/'input').write_text(namelist_overrides((old/'input').read_text(),{'n_R':90,'n_Z':260,'regrid_from_rz':True}))
worker=r/'cpu_fixed_rz_probe_worker_source_20261102.py'
worker.write_text('''"""One bounded probe refining only the final mesh on the validated R/Z equilibrium."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.runner import classify,restart_health
from mastu_opsplit.common import sha256_file
base=Path(sys.argv[1]);binary=Path(sys.argv[2]);start=time.monotonic()
with (base/'input').open() as inp,(base/'stdout.log').open('w') as out,(base/'stderr.log').open('w') as err:
 p=subprocess.run(['timeout','--signal=TERM','--kill-after=10s','6900s','mpirun','--bind-to','none','-n','4',str(binary)],cwd=base,stdin=inp,stdout=out,stderr=err)
text=(base/'stdout.log').read_text()+(base/'stderr.log').read_text()
r={'returncode':p.returncode,'status':classify(p.returncode,p.returncode==124,text),'wall_seconds':time.monotonic()-start,'binary_sha256':sha256_file(binary),'scientific_gates_passed':[]}
try:
 assert r['status']=='completed',r['status']
 r['health']=restart_health(base/'jorek_restart.h5',expected_steps=3,final_time_s=8.78695694517549e-8)
except Exception as e:r.update(status='failed_validation',error=str(e))
(base/'result.json').write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r),flush=True);sys.exit(0 if r['status']=='completed' else 1)
''')
script=r/'cpu_fixed_rz_nested_probe3_20261102.sbatch';env=root/'runs/prepared_equilibrium_dcgp_authorized_20261102/jorek-env.sh';python='/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-fixedrz-nested
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
{python} {worker} {base} {binary}
''')
record={'label':'fixed_rz_nested_probe3','status':'prepared','scope':'Refine final flux-aligned mesh by 2x while holding validated intermediate R/Z equilibrium and all physical inputs fixed; three-step cost/health probe only','bulk_output':str(base),'binary':str(binary),'binary_sha256':sha256_file(binary),'script':str(script),'script_sha256':sha256_file(script),'worker':str(worker),'worker_sha256':sha256_file(worker),'startup_hashes':{p.name:sha256_file(p) for p in base.iterdir()},'prior_capacity_parity':parity,'intermediate_equilibrium_source':str(fine/'jorek_equil_rz.h5'),'requested_cpu_core_hours':8,'requested_node_hours':2,'mpi_ranks':4,'omp_threads':1,'memory_GiB':128,'wall_seconds':7200,'scientific_gates_passed':[]}
cmd=['sbatch','--parsable','--chdir',str(base),'--output',str(base/'batch-%j.out'),'--error',str(base/'batch-%j.err'),str(script)]
check=subprocess.run(['sbatch','--test-only',*cmd[1:]],text=True,capture_output=True);assert check.returncode==0,check.stderr
out=subprocess.run(cmd,text=True,capture_output=True);assert out.returncode==0,out.stderr
record.update(job_id=out.stdout.strip(),status='submitted',submitted_utc=datetime.now(timezone.utc).isoformat());parent.update(status='nested_probe_failed',quad_probe3_result=previous,stage='completed',failure_site='Native strategic-point guard found 6 strikes, expected 4, on doubled intermediate R/Z equilibrium')
phase['runs'].append(record);phase.update(reserved_cpu_core_hours=cpu,reserved_node_hours=node);phasepath.write_text(json.dumps(phase,indent=2)+'\n');(r/'cpu_fixed_rz_nested_probe3_20261102.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'job_id':record['job_id'],'reserved_cpu_hours':cpu,'reserved_node_hours':node}))
