from pathlib import Path
import sys,json
import h5py,numpy as np
from scipy.optimize import root
from numpy.polynomial.polynomial import polyder,polyval2d
b=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/wall_equilibrium_relaxation_20261103/probe3')
with h5py.File(b/'jorek000000.h5') as f:
 v=f['vertex'][()].T-1;ss=f['size'][()].transpose(2,1,0);xy=f['x'][()][:,:,0,:].transpose(2,1,0);psi=f['values'][()][0,:,0,:].T
basis=np.array([[1,0,-3,2],[0,3,-6,3],[0,0,3,-2],[0,0,3,-3]],float)
def coefficients(jets):
 c=np.zeros((4,4))
 for k,(i,j) in enumerate([(0,0),(1,0),(1,1),(0,1)]):
  for d,(di,dj) in enumerate([(0,0),(1,0),(0,1),(1,1)]):c+=jets[k,d]*np.outer(basis[2*i+di],basis[2*j+dj])
 return c
rows=[]
for sign in [-1,1]:
 target=np.array([.4837,sign*1.2445]);corners=xy[v,0];lower=corners.min(1);upper=corners.max(1)
 distance=np.linalg.norm(np.maximum(np.maximum(lower-target,target-upper),0),axis=1)
 for e in np.flatnonzero(distance<.012):
  c=coefficients(psi[v[e]]*ss[e]);cs=polyder(c,axis=0);ct=polyder(c,axis=1);css=polyder(cs,axis=0);cst=polyder(cs,axis=1);ctt=polyder(ct,axis=1)
  def gradient(st):return np.array([polyval2d(*st,cs),polyval2d(*st,ct)])
  def hessian(st):return np.array([[polyval2d(*st,css),polyval2d(*st,cst)],[polyval2d(*st,cst),polyval2d(*st,ctt)]])
  roots=[]
  for x in [.1,.5,.9]:
   for y in [.1,.5,.9]:
    sol=root(gradient,[x,y],jac=hessian,tol=1e-11);st=sol.x
    if np.linalg.norm(gradient(st))>1e-11 or (st<0).any() or (st>1).any() or any(np.linalg.norm(st-p)<1e-6 for p in roots):continue
    roots.append(st)
    rz=[float(polyval2d(*st,coefficients(xy[v[e],:,d]*ss[e]))) for d in range(2)]
    rows.append({'element':int(e+1),'st':st.tolist(),'rz':rz,'psi':float(polyval2d(*st,c)),'hessian_eigenvalues_st':np.linalg.eigvalsh(hessian(st)).tolist(),'hessian_determinant_st':float(np.linalg.det(hessian(st)))})
print(json.dumps({'source':str(b/'jorek000000.h5'),'stationary_points':rows},indent=2))
