"""Freeze the matched full pair from an already completed native radial probe."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import sha256_file


def main():
 r=ROOT/'reports';path=r/'cpu_spatial_next_20261103.json';record=json.loads(path.read_text());old=Path(record['bulk_output'])
 if record.get('radial_pair_prepared'):raise ValueError('preserve prepared full pair')
 selection=json.loads((old/'selection.json').read_text());selected=old/selection['selected'];proof=json.loads((selected/'result.json').read_text())
 state=subprocess.check_output(['sacct','-n','-P','-X','-j',record['jobs']['probe']['job_id'],'--format=State,ExitCode'],text=True).strip()
 if state!='COMPLETED|0:0' or proof['status']!='completed':raise ValueError('completed healthy native probe allocation required')
 m=json.loads((old/'manifest.json').read_text());binary=old/'source/jorek_model303'
 if sha256_file(binary)!=m['validated_unchanged_binary_sha256'] or proof['binary_sha256']!=sha256_file(binary):raise ValueError('validated executable identity changed')
 for moment,name in [('initial','jorek000000.h5'),('final','jorek_restart.h5')]:
  if sha256_file(selected/name)!=proof[moment]['restart_sha256']:raise ValueError('native probe output changed')
 if proof['initial']['metadata']['minimum_signed_jacobian_m2']<=0 or proof['final']['health']['steps']!=3:raise ValueError('positive geometry and3-step native proof required')
 base=old.parent/'spatial_radial_pair_20261103'
 if base.exists():raise ValueError('preserve existing radial pair')
 shutil.copyfile(path,r/'cpu_spatial_before_radial_pair_20261103.json');base.mkdir();(base/'source').mkdir();shutil.copyfile(binary,base/'source/jorek_model303');(base/'source/jorek_model303').chmod(0o555)
 for n in ['worker.py','jorek-env.sh']:shutil.copyfile(old/n,base/n)
 (base/'probe3').mkdir()
 for n in ['input','eqdsk.dat','jorek_equil_rz.h5','stdout.log','stderr.log','result.json','jorek000000.h5','jorek_restart.h5']:shutil.copyfile(selected/n,base/'probe3'/n)
 stages={'probe3':{'steps':3,'amplitudes':[0.,0.],'timeout_seconds':3600,'bulk_output':str(base/'probe3')}}
 for label,amp in [('baseline48',0.),('perturbed48',.05)]:
  work=base/label;work.mkdir();(work/'input').write_text(namelist_overrides((selected/'input').read_text(),{'nstep_n':48,'opsplit_amplitude(1)':amp,'opsplit_amplitude(2)':amp}))
  for n in ['eqdsk.dat','jorek_equil_rz.h5']:shutil.copyfile(selected/n,work/n)
  stages[label]={'steps':48,'amplitudes':[amp,amp],'timeout_seconds':20400,'bulk_output':str(work)}
 m.pop('budget',None)
 m['full_pair_budget_record']=record['budget_record']
 m.update(stages=stages,mesh_selection=selection,native_probe_original_work=str(selected),native_probe_manifest_sha256=sha256_file(old/'manifest.json'),
          spatial_refinement_scope='Core radial count refinement with original wall and poloidal counts; wall placement follows the selected frozen candidate. Poloidal and wall-count convergence is not established by this pair.')
 files=[base/'worker.py',base/'jorek-env.sh',base/'source/jorek_model303']+[p for label in stages for p in (base/label).iterdir()];m['input_hashes']={str(p):sha256_file(p) for p in files};(base/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 record.update(bulk_output=str(base),manifest_sha256=sha256_file(base/'manifest.json'),stages=stages,status='native_radial_probe_validated',radial_pair_prepared=True)
 record['native_probe_selection']=selection;record['native_probe_original_work']=str(selected);path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(selection))


if __name__=='__main__':main()
