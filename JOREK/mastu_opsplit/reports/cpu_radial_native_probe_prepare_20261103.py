"""Select a radial refinement with direct, unchanged-solver three-step probes."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import sha256_file


def main():
 r=ROOT/'reports';path=r/'cpu_spatial_next_20261103.json';record=json.loads(path.read_text());out=r/'cpu_radial_native_probe_20261103.json'
 if out.exists() or 'probe' in record['jobs']:raise ValueError('preserve existing native probes')
 old=Path(record['bulk_output']);base=old.parent/'radial_native_probe_20261103';base.mkdir()
 m=json.loads((old/'manifest.json').read_text());binary=old/'source/jorek_model303'
 if sha256_file(binary)!=m['validated_unchanged_binary_sha256']:raise ValueError('validated executable identity changed')
 row=next(x for x in subprocess.check_output(['saldo','-b','--dcgp','-n'],text=True).splitlines() if x.split()[0]=='FUPA2_FAUNA');c=row.split()
 if min(int(c[3])-int(c[4]),int(c[7])-int(c[8]))<768:raise ValueError('insufficient verified credit')
 (base/'source').mkdir();shutil.copyfile(binary,base/'source/jorek_model303');(base/'source/jorek_model303').chmod(0o555)
 for n in ['worker.py','jorek-env.sh']:shutil.copyfile(old/n,base/n)
 stages={}
 labels=['radial80wall1eq','radial80wall1','radial60wall1']
 for label,flux,equidistant in zip(labels,[80,80,60],[True,False,False]):
  work=base/label;work.mkdir();overrides={'n_flux':flux,'nstep_n':3,'nout':8,'amix':.5,'equil_accuracy':1e-10,
    **{f'n_ext_block({i})':76 if i==3 else 40 for i in range(1,12)},**{f'n_ext_equidistant({i})':equidistant for i in range(1,12)}}
  (work/'input').write_text(namelist_overrides((old/'probe3/input').read_text(),overrides))
  for n in ['eqdsk.dat','jorek_equil_rz.h5']:shutil.copyfile(old/'probe3'/n,work/n)
  stages[label]={'steps':3,'amplitudes':[0.,0.],'timeout_seconds':3600,'bulk_output':str(work)}
 budget={'authorization':'Current explicit diagnosis and next-refinement request','scope':'One allocation containing at most three frozen direct native3-step probes; first healthy geometry, strict equilibrium and time evolution selected before full48-step errors.',
  'allocations_max':1,'candidates_max':3,'attempts_each_max':1,'mpi_ranks':4,'threads':1,'memory_GiB':256,'wall_seconds':10800,'rank_CPU_hours_max':12,'node_hours_max':3,'conservative_resource_hours_max':768,'saldo_row':row,
  'frozen_candidates':labels,'equil_accuracy':1e-10,'amix':.5,'spatial_target':.01,'physics_changed':False,'limitation':'Directional core radial test; original wall counts retained after doubled-wall geometry or strict-equilibrium failures.'}
 for key in ['input_hashes','planned_geometry_reference','planned_geometry_sha256','mesh_selection','planning_manifest_sha256']:m.pop(key,None)
 m.update(stages=stages,budget=budget,selection_policy='First direct native probe passing positive finite q8 geometry, strict1e-10 equilibrium and3 steps; no48-step convergence errors used for selection.')
 files=[base/'worker.py',base/'jorek-env.sh',base/'source/jorek_model303']+[p for label in stages for p in (base/label).iterdir()];m['input_hashes']={str(p):sha256_file(p) for p in files};(base/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 controller=base/'probe_selection.py';controller.write_text('''import json,subprocess,sys,h5py\nfrom pathlib import Path\nb=Path(sys.argv[1]);root=sys.argv[2];attempts=[]\nfor label in ['radial80wall1eq','radial80wall1','radial60wall1']:\n run=subprocess.run([sys.executable,str(b/'worker.py'),str(b),root,label],check=False)\n result=json.loads((b/label/'result.json').read_text());attempts.append({'label':label,'returncode':run.returncode,'status':result['status']})\n (b/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\\n')\n if run.returncode:continue\n with h5py.File(b/label/'jorek000000.h5') as f:nodes=int(f['n_nodes'][()].item());elements=int(f['n_elements'][()].item())\n if max(nodes,elements)<=155000:\n  (b/'selection.json').write_text(json.dumps({'selected':label,'nodes':nodes,'elements':elements,'policy':'First healthy direct native3-step probe under frozen1e-10 equilibrium criterion.'},indent=2)+'\\n');break\nelse:raise ValueError('All direct native probes failed frozen validation')\n''')
 script=r/'cpu_radial_native_probe_20261103.sbatch';script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-radial-native-probe
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=03:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {controller} {base} {ROOT}
''')
 rec={'budget':budget,'bulk_output':str(base),'manifest_sha256':sha256_file(base/'manifest.json'),'controller_sha256':sha256_file(controller),'script_sha256':sha256_file(script)};out.write_text(json.dumps(rec,indent=2)+'\n')
 command=['sbatch','--parsable','--chdir',str(base),'--output',str(base/'batch-%j.out'),'--error',str(base/'batch-%j.err'),str(script)]
 subprocess.run(['bash','-n',str(script)],check=True);subprocess.run(['sbatch','--test-only',*command[1:]],check=True);job=subprocess.check_output(command,text=True).strip();rec['job_id']=job;out.write_text(json.dumps(rec,indent=2)+'\n')
 shutil.copyfile(path,r/'cpu_spatial_before_direct_native_probe_20261103.json')
 record['jobs']['native_wall_geometry_folded']={**record['jobs'].pop('directional_plan'),'active':False}
 record['jobs']['probe']={'job_id':job,'kind':'simulation','budget_record':str(out),'script':str(script),'script_sha256':sha256_file(script)}
 record.update(bulk_output=str(base),manifest_sha256=rec['manifest_sha256'],stages=stages,status='direct_radial_probe_selection_submitted');path.write_text(json.dumps(record,indent=2)+'\n');print(job)


if __name__=='__main__':main()
