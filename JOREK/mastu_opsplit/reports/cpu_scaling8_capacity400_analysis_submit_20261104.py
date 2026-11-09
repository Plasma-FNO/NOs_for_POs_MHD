"""Submit unchanged spatial comparisons after their immutable native pairs finish."""
import argparse,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'src'))
from mastu_opsplit.common import sha256_file
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('direction',choices=['radial_next','poloidal_next']);a=p.parse_args()
family='scaling8_capacity400'
coarse,fine=('radial320','radial640') if a.direction=='radial_next' else ('poloidal880','poloidal1760')
records=[json.loads((R/f'cpu_{family}_pair_{mesh}_{stage}_20261104.json').read_text()) for mesh in [coarse,fine] for stage in ['baseline48','perturbed48']]
all_ids=[str(r['job_id']) for r in records]
active=[]
for job in all_ids:
 state=subprocess.check_output(['sacct','-n','-X','-j',job,'-o','State'],text=True).strip()
 if state=='COMPLETED':continue
 if state not in ['RUNNING','PENDING','CONFIGURING']:raise ValueError('native dependency is not healthy: '+job+' '+state)
 active.append(job)
base=Path(records[-1]['bulk_output']);out=R/f'cpu_{family}_{a.direction}_analysis_job_20261104.json'
if out.exists():raise ValueError('preserve existing analysis')
script=R/f'cpu_{family}_{a.direction}_analysis_20261104.sbatch'
script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-{family}-{a.direction}-analysis
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {R}/cpu_scaling8_capacity400_spatial_analysis_20261104.py {a.direction}
''')
cmd=['sbatch','--parsable',*(['--dependency=afterok:'+':'.join(active)] if active else []),'--chdir',str(base),'--output',str(base/'analysis-%j.out'),'--error',str(base/'analysis-%j.err'),str(script)]
pre=subprocess.run(['sbatch','--test-only',*cmd[1:]],check=True,text=True,capture_output=True)
record=dict(status='prepared',kind='analysis',native_attempt_started=False,bulk_output=str(base),dependencies=all_ids,active_dependencies=active,script_sha256=sha256_file(script),analysis_source_sha256=sha256_file(R/'cpu_scaling8_capacity400_spatial_analysis_20261104.py'),budget=dict(mpi_ranks=1,memory_GiB=128,wall_hours_max=2),scheduler_preflight=pre.stderr,scientific_gates_passed=[])
with out.open('x') as f:json.dump(record,f,indent=2)
record.update(status='submitted',job_id=subprocess.check_output(cmd,text=True).strip())
out.write_text(json.dumps(record,indent=2)+'\n');print(record['job_id'])
