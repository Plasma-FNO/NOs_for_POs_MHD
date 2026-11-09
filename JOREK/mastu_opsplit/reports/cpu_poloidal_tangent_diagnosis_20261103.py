from pathlib import Path
import sys,json
import h5py,numpy as np
r=Path('/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit');sys.path.insert(0,str(r/'src'))
from mastu_opsplit.restart import cubic_basis
b=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/exact_poloidal_crossing_20261103/poloidal440')
with h5py.File(b/'grid_patch1.h5') as f:
 v=f['vertex'][()].T-1;size=f['size'][()].transpose(2,1,0);xy=f['x'][()][:,:,0,:].transpose(2,1,0)
q,_=np.polynomial.legendre.leggauss(8);s,t=np.meshgrid((q+1)/2,(q+1)/2,indexing='ij');_,hs,ht=cubic_basis(s.ravel(),t.ravel())
def measure(x):
 jets=x[v]*size[...,None];rs,rt=[np.einsum('qvd,evdc->eqc',h,jets) for h in (hs,ht)]
 j=rs[...,0]*rt[...,1]-rs[...,1]*rt[...,0]
 return {'minimum':float(j.min()),'invalid_elements':(np.where((j<=0).any(1))[0]+1).tolist()}
nrad=40;npol=len(v)//(nrad-1)+1;nstart=len(xy)-nrad*npol
x=xy.copy();rows=x[nstart:].reshape(nrad,npol,4,2)
def unit(a):return a/np.linalg.norm(a,axis=-1,keepdims=True)
coords=rows[:,:,0,:].copy()
# Only interior nodes: shared patch boundaries and the wall are untouched.
rows[1:-1,1:-1,1]=unit(unit(coords[1:-1,1:-1]-coords[:-2,1:-1])+unit(coords[2:,1:-1]-coords[1:-1,1:-1]))
rows[1:-1,1:-1,2]=unit(unit(coords[1:-1,1:-1]-coords[1:-1,:-2])+unit(coords[1:-1,2:]-coords[1:-1,1:-1]))
result={'before':measure(xy),'centered_before_recomputing_chart_signs':measure(x),'dimensions':[nrad,npol,nstart]}
corners=x[v,0,:]
sedge=np.stack([corners[:,1]-corners[:,0],corners[:,1]-corners[:,0],corners[:,2]-corners[:,3],corners[:,2]-corners[:,3]],axis=1)
tedge=np.stack([corners[:,3]-corners[:,0],corners[:,2]-corners[:,1],corners[:,2]-corners[:,1],corners[:,3]-corners[:,0]],axis=1)
for dof,edge,ends in [(1,sedge,[1,-1,-1,1]),(2,tedge,[1,1,-1,-1])]:
 size[:,:,dof]=np.abs(size[:,:,dof])*np.sign(np.sum(edge*x[v,dof],axis=-1))*np.array(ends)
size[:,:,3]=size[:,:,1]*size[:,:,2]
result['centered_with_recomputed_chart_signs']=measure(x)
result['status']='offline_geometry_diagnosis_only'
result['limitation']='No native source or restart changed. Shared patch boundary nodes retained. The separate divertor fold remains unresolved.'
print(json.dumps(result,indent=2))
(r/'reports/cpu_poloidal_tangent_diagnosis_20261103.json').write_text(json.dumps(result,indent=2)+'\n')
