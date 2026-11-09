"""Offline interior-vertex handle trials; no native state or acceptance is changed."""
import argparse,json,sys
from pathlib import Path
import h5py,numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'src'))
from mastu_opsplit.restart import cubic_basis
from mastu_opsplit.common import sha256_file
p=argparse.ArgumentParser(description=__doc__);p.add_argument('mesh',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
if args.output.exists():raise ValueError('preserve previous diagnosis')
with h5py.File(args.mesh) as f:
 xy=f['x'][:,:,0,:].transpose(2,1,0);v=f['vertex'][()].T-1;sizes=f['size'][()].transpose(2,1,0);flags=f['boundary'][()]
def jacobian_min(ids,values,order):
 q=np.polynomial.legendre.leggauss(order)[0];a,b=np.meshgrid((q+1)/2,(q+1)/2,indexing='ij');_,hs,ht=cubic_basis(a.ravel(),b.ravel());out=np.empty(len(ids))
 for start in range(0,len(ids),512):
  ix=ids[start:start+512];jets=xy[v[ix]]*values[start:start+512,...,None]
  rs,rt=[np.einsum('qvd,evdc->eqc',h,jets) for h in [hs,ht]]
  jac=rs[...,0]*rt[...,1]-rs[...,1]*rt[...,0]
  if not np.isfinite(jac).all():raise ValueError('nonfinite geometry')
  out[start:start+512]=jac.min(axis=1)
 return out
ids=np.arange(len(v));original=jacobian_min(ids,sizes,8);bad=ids[original<=0]
counts=np.bincount(v.ravel(),minlength=len(xy));active=np.zeros(len(xy),bool);active[np.unique(v[bad])]=True;active[:8]=False;active &= (counts==4)&(flags==0)
# Include one ring of interior nodes to avoid abrupt changes at the failed component's edge.
affected=np.any(active[v],axis=1);active[np.unique(v[affected])]=True;active[:8]=False;active &= (counts==4)&(flags==0)
affected=np.flatnonzero(np.any(active[v],axis=1));node_mask=active[v[affected]]
report=dict(scope=__doc__,mesh=str(args.mesh),mesh_sha256=sha256_file(args.mesh),source_sha256=sha256_file(__file__),original_q8_fold_count=len(bad),original_q8_fold_elements=(bad+1).tolist(),eligible_interior_nodes=int(active.sum()),affected_cells=len(affected),trials=[],scientific_gates_passed=[])
for attempt in range(11):
 factor=.8**attempt;trial=sizes[affected].copy();trial[:,:,1:3]*=np.where(node_mask,factor,1.)[...,None]
 trial[:,:,3]=np.where(node_mask,trial[:,:,1]*trial[:,:,2],trial[:,:,3]);mins=jacobian_min(affected,trial,16)
 row=dict(factor=factor,minimum_affected_q16_jacobian_m2=float(mins.min()) if len(mins) else None,remaining_affected_q16_folds=int((mins<=0).sum()))
 if len(mins) and (mins>0).all():
  strict=jacobian_min(affected,trial,32);row.update(minimum_affected_q32_jacobian_m2=float(strict.min()),remaining_affected_q32_folds=int((strict<=0).sum()))
 report['trials'].append(row);print(json.dumps(row),flush=True)
 if row.get('remaining_affected_q32_folds')==0:break
with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
