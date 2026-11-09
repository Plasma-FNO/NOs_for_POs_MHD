"""Repeat the residual-verified refined solves against their own frozen one-step references."""
from pathlib import Path
import json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102');old=B/'mumps_refinement_20261103';base=B/'mumps_refinement_repeat_20261103';assert not base.exists()
row=next(x for x in subprocess.check_output(['saldo','-b','--dcgp','-n'],text=True).splitlines() if x.split()[0]=='FUPA2_FAUNA');c=row.split();assert min(int(c[3])-int(c[4]),int(c[7])-int(c[8]))>=128
base.mkdir();(base/'source').mkdir();shutil.copyfile(old/'source/jorek_model303',base/'source/jorek_model303');(base/'source/jorek_model303').chmod(0o555)
for name in ['worker.py','jorek-env.sh']:shutil.copyfile(old/name,base/name)
p=base/'worker.py';s=p.read_text();target="        shutil.copyfile(work / 'initial_reference.h5', work / 'jorek_restart.h5')"
guard='''        reference = Path(manifest['old_bridge']) / label
        proof = json.loads((reference / 'result.json').read_text())
        if proof.get('native_linearized_volume_balance', {}).get('status') != 'interior_linearized_balance_verified':
            raise ValueError('refined one-step reference must pass the frozen balance test')
        if proof['binary_sha256'] != sha256_file(executable):
            raise ValueError('repeatability requires identical refined solver')
        if proof['final']['restart_sha256'] != sha256_file(reference / 'jorek000009.h5'):
            raise ValueError('refined reference changed')
'''
assert s.count(target)==1;p.write_text(s.replace(target,guard+target))
m=json.loads((old/'manifest.json').read_text());files=[base/'worker.py',base/'jorek-env.sh',base/'source/jorek_model303']
for label in m['stages']:
 w=base/label;w.mkdir()
 for name in ['input','initial_reference.h5','eqdsk.dat','jorek_equil_rz.h5']:shutil.copyfile(old/label/name,w/name)
 files.extend(w.iterdir())
budget={'authorization':'User requested solver convergence and science-plan completion.','scope':'Two independent single-step repeats from exactly the same saved inputs and binary; references must first pass native linearized balance.','mpi_ranks':4,'memory_GiB':128,'wall_hours_max':1,'rank_CPU_hours_max':4,'saldo_row':row}
m.update(old_bridge=str(old),budget=budget,input_hashes={str(p):sha256_file(p) for p in files});(base/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
p=R/'cpu_mumps_repeat_20261103.sbatch';p.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-mumps-repeat
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {base}/worker.py {base} {ROOT}
''')
command=['sbatch','--parsable','--dependency=afterany:2771526','--chdir',str(base),'--output',str(base/'batch-%j.out'),'--error',str(base/'batch-%j.err'),str(p)];subprocess.run(['sbatch','--test-only',*command[1:]],capture_output=True,check=True)
record={'status':'prepared','bulk_output':str(base),'budget':budget,'dependencies':['2771526'],'script_sha256':sha256_file(p),'manifest_sha256':sha256_file(base/'manifest.json'),'scientific_gates_passed':[]};p=R/'cpu_mumps_repeat_20261103.json';p.write_text(json.dumps(record,indent=2)+'\n');record.update(status='submitted',job_id=subprocess.check_output(command,text=True).strip());p.write_text(json.dumps(record,indent=2)+'\n');print(record['job_id'])
