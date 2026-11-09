"""Locate central-grid folds independently of wall extensions using native bicubic maps."""
import h5py
import numpy as np
import sys
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.restart import cubic_basis
b=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/exact_poloidal_crossing_20261103/poloidal440')
p,_=np.polynomial.legendre.leggauss(8);p=(p+1)/2;s,t=np.meshgrid(p,p,indexing='ij');_,hs,ht=cubic_basis(s.ravel(),t.ravel())
rows={}
for name in ['grid_no_patch.h5', *[f'grid_patch{i}.h5' for i in range(1,12)], 'jorek_stopped.h5']:
 with h5py.File(b/name) as f:
  v=f['vertex'][()].T-1;size=f['size'][()].transpose(2,1,0);xy=f['x'][()][:,:,0,:].transpose(2,1,0)
  bad=[]
  for start in range(0,len(v),1024):
   vv=v[start:start+1024];ss=size[start:start+1024];loc=xy[vv].reshape(len(vv),16,2)
   rs,rt=[np.einsum('eqd,edc->eqc',np.einsum('qvd,evd->eqvd',h,ss).reshape(len(vv),-1,16),loc) for h in [hs,ht]]
   j=rs[...,0]*rt[...,1]-rs[...,1]*rt[...,0];ids=np.where((j.min(1)<0)&(j.max(1)>0))[0]
   bad.extend([{'element':int(start+i+1),'center':xy[vv[i],0].mean(0).tolist(),'corners':xy[vv[i],0].tolist(),'min_jacobian':float(j[i].min()),'max_jacobian':float(j[i].max())} for i in ids])
  rows[name]={'elements':len(v),'folded_elements':bad}
print(json.dumps(rows,indent=2));(ROOT/'reports/cpu_exact_poloidal_geometry_diagnosis_20261103.json').write_text(json.dumps(rows,indent=2)+'\n')
