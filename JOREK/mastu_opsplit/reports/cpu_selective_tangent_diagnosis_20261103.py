"""Offline test of centering only tangents that reverse between adjacent chords."""
from pathlib import Path
import sys,json,re
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.restart import cubic_basis
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
q,_=np.polynomial.legendre.leggauss(8);s,t=np.meshgrid((q+1)/2,(q+1)/2,indexing='ij');_,hs,ht=cubic_basis(s.ravel(),t.ravel())
def measure(x,v,size):
 jets=x[v]*size[...,None];rs,rt=[np.einsum('qvd,evdc->eqc',h,jets) for h in (hs,ht)]
 j=rs[...,0]*rt[...,1]-rs[...,1]*rt[...,0]
 return {'minimum':float(j.min()),'invalid_elements':(np.where((j<=0).any(1))[0]+1).tolist()}
def unit(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)
results={}
for case in ['wall80_newton_20261103/wall80','exact_poloidal_crossing_20261103/poloidal440','normalized_newton_20261103/control1']:
 b=B/case;rows={}
 for patch in range(1,12):
  with h5py.File(b/f'grid_patch{patch}.h5') as f:
   v=f['vertex'][()].T-1;size=f['size'][()].transpose(2,1,0);xy=f['x'][()][:,:,0,:].transpose(2,1,0)
  edges=[abs(int(v[0,j]-v[0,0])) for j in [1,3]]
  assert min(edges)==1
  npol=max(edges);nstart=int(v.min());nrad=(len(xy)-nstart)//npol
  assert len(xy)-nstart==nrad*npol and len(v)==(nrad-1)*(npol-1)
  x=xy.copy();nodes=x[nstart:].reshape(nrad,npol,4,2);coords=nodes[:,:,0].copy()
  stride=edges[0]
  changes=0
  for axis,dof in [(0,1 if stride==npol else 2),(1,2 if stride==npol else 1)]:
   incoming=coords[1:-1,1:-1]-(coords[:-2,1:-1] if axis==0 else coords[1:-1,:-2])
   outgoing=(coords[2:,1:-1] if axis==0 else coords[1:-1,2:])-coords[1:-1,1:-1]
   old=nodes[1:-1,1:-1,dof]
   reversed_chart=np.sum(old*incoming,axis=-1)*np.sum(old*outgoing,axis=-1)<0
   average=unit(incoming)+unit(outgoing)
   valid=np.linalg.norm(average,axis=-1)>1e-8
   mask=reversed_chart&valid
   average=unit(average)
   average*=np.where(np.sum(old*average,axis=-1)>=0,1,-1)[...,None]
   old[mask]=average[mask];changes+=int(mask.sum())
  before=measure(xy,v,size)
  corners=x[v,0]
  sedge=np.stack([corners[:,1]-corners[:,0],corners[:,1]-corners[:,0],corners[:,2]-corners[:,3],corners[:,2]-corners[:,3]],axis=1)
  tedge=np.stack([corners[:,3]-corners[:,0],corners[:,2]-corners[:,1],corners[:,2]-corners[:,1],corners[:,3]-corners[:,0]],axis=1)
  for dof,edge,ends in [(1,sedge,[1,-1,-1,1]),(2,tedge,[1,1,-1,-1])]:
   size[:,:,dof]=abs(size[:,:,dof])*np.sign(np.sum(edge*x[v,dof],axis=-1))*np.array(ends)
  size[:,:,3]=size[:,:,1]*size[:,:,2]
  rows[str(patch)]={'before':before,'after':measure(x,v,size),'changed_tangents':changes}
 results[case]=rows
out=ROOT/'reports/cpu_selective_tangent_diagnosis_20261103.json';out.write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({case:{p:{'before':len(x['before']['invalid_elements']),'after':len(x['after']['invalid_elements']),'changed':x['changed_tangents']} for p,x in rows.items()} for case,rows in results.items()},indent=2))
