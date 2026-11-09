"""Recover the observed Picard limit cycle with native damping on both meshes."""
import json,shutil,subprocess,sys,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import sha256_file


def main():
 r=ROOT/'reports';record_path=r/'cpu_spatial_next_20261103.json';record=json.loads(record_path.read_text())
 out=r/'cpu_equilibrium_damping_20261103.json'
 if out.exists():raise ValueError('damping recovery already prepared')
 old=Path(record['bulk_output']);bulk=old.parent;base=bulk/'spatial_damped_pair_20261103'
 if base.exists():raise ValueError('preserve existing recovery')
 text=(old/'probe3/stdout.log').read_text();updates=[(int(i),float(x)) for i,x in re.findall(r'Iteration\s+(\d+): diff=\s*(\S+)',text)]
 if len(updates)<80 or min(x for _,x in updates[-26:])<1e-10:raise ValueError('stagnation evidence not established')
 row=next(x for x in subprocess.check_output(['saldo','-b','--dcgp','-n'],text=True).splitlines() if x.split()[0]=='FUPA2_FAUNA');c=row.split()
 if min(int(c[3])-int(c[4]),int(c[7])-int(c[8]))<1536:raise ValueError('insufficient verified credit')
 evidence={'probe_job':record['jobs']['probe']['job_id'],'last_updates':updates[-39:],'stdout_sha256':sha256_file(old/'probe3/stdout.log'),
           'reason':'Repeated Picard limit cycle above unchanged1e-10 raw-update stopping criterion. Native amix0 full updates replaced by amix.5 on both meshes; no physical coefficients changed.'}
 (r/'cpu_equilibrium_limit_cycle_20261103.json').write_text(json.dumps(evidence,indent=2)+'\n')
 subprocess.run(['scancel',record['jobs']['probe']['job_id']],check=True)
 shutil.copyfile(record_path,r/'cpu_spatial_before_damping_20261103.json')
 base.mkdir();(base/'source').mkdir();shutil.copyfile(old/'source/jorek_model303',base/'source/jorek_model303');(base/'source/jorek_model303').chmod(0o555)
 for name in ['worker.py','jorek-env.sh']:shutil.copyfile(old/name,base/name)
 manifest=json.loads((old/'manifest.json').read_text());manifest.pop('input_hashes')
 refs={label:bulk/f'equilibrium_damped_{label}_20261103' for label in ['baseline48','perturbed48']}
 stages={}
 for label,config in manifest['stages'].items():
  work=base/label;work.mkdir();(work/'input').write_text(namelist_overrides((old/label/'input').read_text(),{'amix':.5}))
  for name in ['eqdsk.dat','jorek_equil_rz.h5']:shutil.copyfile(old/label/name,work/name)
  stages[label]={**config,'bulk_output':str(work)}
 manifest.update(stages=stages,analysis_reference_works={k:str(v) for k,v in refs.items()},equilibrium_relaxation_amix=.5)
 files=[base/'worker.py',base/'jorek-env.sh',base/'source/jorek_model303',Path(manifest['planned_geometry_reference'])]+[p for label in stages for p in (base/label).iterdir()]
 manifest['input_hashes']={str(p):sha256_file(p) for p in files};(base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 for label,work in refs.items():
  work.mkdir();reference=Path(json.loads((old/'manifest.json').read_text())['analysis_reference_works'][label])
  (work/'input').write_text(namelist_overrides((reference/'input').read_text(),{'amix':.5}))
  for name in ['eqdsk.dat','jorek_equil_rz.h5']:shutil.copyfile(reference/name,work/name)
  shutil.copyfile(r/'cpu_equilibrium_tolerance_worker_20261103.py',work/'worker.py')
  m={'binary':str(base/'source/jorek_model303'),'final_time_s':manifest['final_time_s'],'reference_work':str(reference)}
  if label=='perturbed48':m['initializer_reference']=str(refs['baseline48'])
  m['input_hashes']={str(p):sha256_file(p) for p in [work/'input',work/'eqdsk.dat',work/'jorek_equil_rz.h5',work/'worker.py',base/'source/jorek_model303']}
  (work/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 budget={'authorization':'Explicit current diagnosis and next refinement request','scope':'One damped three-step finer probe and two matched48-step strict reference cases; conditional full-pair reservation reused.',
         'allocations_max':3,'attempts_each_max':1,'mpi_ranks_each':4,'threads':1,'wall_seconds_each':7200,'rank_CPU_hours_max':24,'node_hours_max':6,
         'memory_GiB_probe':256,'memory_GiB_each_reference':128,'conservative_resource_hours_max':1536,'saldo_row':row,'equil_accuracy':1e-10,'amix':.5}
 recovery={'budget':budget,'bulk_output':str(base),'reference_works':{k:str(v) for k,v in refs.items()},'manifest_sha256':sha256_file(base/'manifest.json'),'jobs':{}}
 out.write_text(json.dumps(recovery,indent=2)+'\n')
 record['jobs']['undamped_probe_stagnated']={**record['jobs'].pop('probe'),'active':False}
 record.update(bulk_output=str(base),manifest_sha256=sha256_file(base/'manifest.json'),stages=stages,analysis_reference_works={k:str(v) for k,v in refs.items()},status='damped_equilibrium_recovery_submitted',damping_budget_record=str(out))
 for label,work in [('probe',base/'probe3'),*refs.items()]:
  memory=256 if label=='probe' else 128;worker=base/'worker.py' if label=='probe' else work/'worker.py'
  args=f'{base} {ROOT} probe3' if label=='probe' else f'{work} {ROOT}'
  script=r/f'cpu_equilibrium_damped_{label}_20261103.sbatch'
  script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-damped-{label}
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem={memory}G
#SBATCH --time=02:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {worker} {args}
''')
  command=['sbatch','--parsable','--chdir',str(work),'--output',str(work/'batch-%j.out'),'--error',str(work/'batch-%j.err'),str(script)]
  subprocess.run(['bash','-n',str(script)],check=True);subprocess.run(['sbatch','--test-only',*command[1:]],check=True)
  job=subprocess.check_output(command,text=True).strip();entry={'job_id':job,'kind':'simulation','script':str(script),'script_sha256':sha256_file(script),'budget_record':str(out)}
  recovery['jobs'][label]=entry;out.write_text(json.dumps(recovery,indent=2)+'\n')
  record['jobs']['probe' if label=='probe' else 'damped_reference_'+label]=entry;record_path.write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(recovery['jobs']))


if __name__=='__main__':main()
