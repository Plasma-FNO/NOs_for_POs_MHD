"""Fresh core-poloidal probes after rigorous-scaling control and failed-step rescue."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'reports'
sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.case import namelist_overrides

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('resolution',type=int,choices=[440,880])
args=parser.parse_args()
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
parent=B/'mumps_scaling8_20261104'
base=B/f'scaling8_poloidal{args.resolution}_20261104'
record_path=R/f'cpu_scaling8_poloidal{args.resolution}_20261104.json'
if base.exists() or record_path.exists():raise ValueError('preserve prior attempt')
binary=parent/'source/jorek_model303';digest=sha256_file(binary)
original=json.loads((parent/'manifest.json').read_text())
for name,expected in original['input_hashes'].items():
 if sha256_file(name)!=expected:raise ValueError('frozen scaling experiment changed: '+name)
for label in ['control3','perturbed3','poloidal880_step1']:
 result=json.loads((parent/label/'result.json').read_text())
 if result['status']!='completed' or result['binary_sha256']!=digest:raise ValueError('scaling control and rescue must pass')
 if label!='poloidal880_step1' and not all(result[k]['passed'] for k in ['initial_parity','parity']):raise ValueError('strict field/gradient parity required')
row=next(line for line in subprocess.check_output(['saldo','-b','--dcgp','-n'],text=True).splitlines() if line.split()[0]=='FUPA2_FAUNA')
c=row.split()
if min(int(c[3])-int(c[4]),int(c[7])-int(c[8]))<2048:raise ValueError('insufficient current credit')
(base/'source').mkdir(parents=True)
shutil.copyfile(binary,base/'source/jorek_model303');(base/'source/jorek_model303').chmod(0o555)
shutil.copyfile(parent/'jorek-env.sh',base/'jorek-env.sh')
text=(B/'combined_balanced2_mpi16_20261104/worker.py').read_text()
anchor="        result['final'] = audit_restart(work/'jorek_restart.h5', config['steps'],"
text=text.replace(anchor,"        if 'initial_parity_reference' in config:\n            result['initial_parity'] = parity(work/'jorek000000.h5', Path(config['initial_parity_reference']), manifest['parity_protocol'])\n"+anchor)
(base/'worker.py').write_text(text)
ref=B/'combined_refined_solver_20261103/core_poloidal440' if args.resolution==440 else B/'combined_core_poloidal880_20261104/probe3'
work=base/'probe3';work.mkdir()
(work/'input').write_text(namelist_overrides((ref/'input').read_text(),{'nstep_n':3,'nout':1}))
for name in ['eqdsk.dat','jorek_equil_rz.h5']:shutil.copyfile(ref/name,work/name)
config={'steps':3,'mpi_ranks':16,'amplitudes':[0.,0.],'timeout_seconds':13500,'initial_parity_reference':str(ref/'jorek000000.h5')}
if args.resolution==440:config['parity_reference']=str(ref/'jorek_restart.h5')
files=[base/'source/jorek_model303',base/'worker.py',base/'jorek-env.sh',R/'cpu_native_snapshot_parity_20261104.py',ref/'jorek000000.h5',*(parent/label/'result.json' for label in ['control3','perturbed3','poloidal880_step1']),*work.iterdir()]
if args.resolution==440:files.extend([ref/'result.json',ref/'jorek_restart.h5'])
manifest={key:original[key] for key in ['final_time_s','tstep_code_units','field_relative_target','integral_relative_target','equil_accuracy','parity_protocol']}
manifest.update(stages={'probe3':config},input_hashes={str(p):sha256_file(p) for p in files},binary_sha256=digest,
 scope='Fresh equilibrium and three-step health qualification only; full spatial comparison still required. Stronger MUMPS scaling, unchanged equations and thresholds.')
(base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
script=base/'probe.sbatch'
script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-scaling8-poloidal{args.resolution}
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=04:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {base}/worker.py {base} {ROOT} probe3
''')
command=['sbatch','--parsable','--chdir',str(base),'--output',str(base/'batch-%j.out'),'--error',str(base/'batch-%j.err'),str(script)]
preflight=subprocess.run(['sbatch','--test-only',*command[1:]],check=True,capture_output=True,text=True)
record={'status':'prepared','bulk_output':str(base),'binary_sha256':digest,'manifest_sha256':sha256_file(base/'manifest.json'),'script_sha256':sha256_file(script),'scheduler_preflight':preflight.stderr,
 'budget':{'mpi_ranks':16,'memory_GiB':512,'wall_hours_max':4,'rank_CPU_hours_max':64,'saldo_row':row},'scientific_gates_passed':[]}
record_path.write_text(json.dumps(record,indent=2)+'\n')
record.update(status='submitted',job_id=subprocess.check_output(command,text=True).strip())
record_path.write_text(json.dumps(record,indent=2)+'\n')
print(args.resolution,record['job_id'])
