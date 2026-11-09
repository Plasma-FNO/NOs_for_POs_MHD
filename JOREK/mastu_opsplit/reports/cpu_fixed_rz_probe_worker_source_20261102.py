"""One bounded probe refining only the final mesh on the validated R/Z equilibrium."""
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
(base/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(0 if r['status']=='completed' else 1)
