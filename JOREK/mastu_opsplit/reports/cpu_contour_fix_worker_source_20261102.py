"""Single-allocation capacity regression followed by guarded refined-mesh probe."""
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
 (work/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
 if result['status']!='completed':sys.exit(1)
