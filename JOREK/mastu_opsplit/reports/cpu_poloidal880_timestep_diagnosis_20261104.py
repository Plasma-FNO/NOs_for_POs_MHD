"""Summarize immutable one-step timestep diagnoses and locate the failed wall response."""
import json
from pathlib import Path
import re
import sys
import h5py
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'src'))
from mastu_opsplit.common import sha256_file
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/poloidal880_timestep_20261104')
manifest=json.loads((B/'manifest.json').read_text())
output={'scope':'One-step restart diagnosis only; no spatial or restart-continuation qualification.', 'manifest_sha256':sha256_file(B/'manifest.json'), 'cases':{}, 'scientific_gates_passed':[]}
for label,config in manifest['stages'].items():
 work=B/label
 result=json.loads((work/'result.json').read_text())
 log=(work/'stdout.log').read_text()
 errors=re.findall(r'MUMPS refinement steps and backward errors:\s*(\d+)\s+(\S+)\s+(\S+)',log)
 with h5py.File(work/'jorek_restart.h5') as f, h5py.File(work/'initial_reference.h5') as initial:
  x=f['x'][:,:,0,:]; v=f['values'][:,:,0,:]; vi=initial['values'][:,:,0,:];b=f['boundary'][()]
  worst=np.argsort(np.abs(v[6,0]))[-8:][::-1]
  nodes=[{'node':int(i+1),'RZ':x[:,0,i].tolist(),'boundary':int(b[i]),'Vpar':float(v[6,0,i]),'initial_psi_t':float(vi[0,2,i])} for i in worst]
 output['cases'][label]={'dt':config['tstep_code_units'],'status':result['status'],'error':result.get('error'),
  'binary_sha256':result['binary_sha256'],'initial_sha256':sha256_file(work/'initial_reference.h5'),'final_sha256':sha256_file(work/'jorek_restart.h5'),
  'linear_errors':[{'refinements':int(n),'omega1':float(a),'omega2':float(b)} for n,a,b in errors],
  'printed_minima':{name:min(float(a) for a,b in re.findall(rf'min/max\s+{n}\s+{name}\s+(\S+)\s+(\S+)',log)) for name,n in [('rho',5),('T',6)]},
  'largest_nodal_parallel_velocities':nodes}
 record=R/f'cpu_poloidal880_{label}_20261104.json'
 row=json.loads(record.read_text());row.update(status=result['status'],failure_evidence={'error':result.get('error'),'result':str(work/'result.json')})
 record.write_text(json.dumps(row,indent=2)+'\n')
output['conclusion']='All three timesteps lose density/temperature positivity on step one. Reducing dt is not a remedy in these diagnostics. Largest Vpar is near the straight outer wall R=1.5551043, Z=-0.12; initial tangential flux derivative there is not near zero. Stronger linear scaling is being tested separately.'
(R/'cpu_poloidal880_timestep_diagnosis_20261104.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({k:{'status':v['status'],'minima':v['printed_minima']} for k,v in output['cases'].items()}))
