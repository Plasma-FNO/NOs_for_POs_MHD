"""Read-only local cached-flux evaluation at concave cells; never mesh acceptance."""
import json,sys
from pathlib import Path
import h5py,numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'src'))
from mastu_opsplit.restart import cubic_basis
from mastu_opsplit.common import sha256_file
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/combined_balanced2_attachment_20261104/probe3')
with h5py.File(B/'jorek_equil_rz.h5') as f:
 xy=f['x'][:,:,0,:].transpose(2,1,0);v=f['vertex'][()].T-1;size=f['size'][()].transpose(2,1,0);psi=f['values'][0,:,0,:].T
jets=xy[v]*size[...,None];pv=psi[v]*size;tree=cKDTree(jets[:,:,0].mean(axis=1))
with h5py.File(B/'jorek_grid.h5') as f:
 points=f['x'][:,0,0,:].T;gv=f['vertex'][()].T-1
folds=[x for x in json.loads((R/'cpu_balanced_fold_geometry_20261104.json').read_text())['folds'] if not x['polygon_convex_ccw']]
node_ids=np.unique([n for x in folds for n in x['vertices']]);mapped={}
for n in node_ids:
 pt=points[n-1];found=None;best=None
 for e in np.atleast_1d(tree.query(pt,k=128)[1]):
  def fun(st):return np.einsum('vd,vdc->c',cubic_basis(*st)[0],jets[e])-pt
  def jac(st):
   _,a,b=cubic_basis(*st);return np.stack([np.einsum('vd,vdc->c',x,jets[e]) for x in [a,b]],axis=1)
  fit=least_squares(fun,[.5,.5],jac=jac,bounds=([0,0],[1,1]),xtol=1e-14,ftol=None,gtol=1e-14,max_nfev=100)
  err=float(np.linalg.norm(fit.fun));det=float(np.linalg.det(jac(fit.x)))
  if best is None or err<best['inverse_error_m']:best=dict(rz_m=pt.tolist(),inverse_error_m=err,local_jacobian_m2=det,cache_element=int(e+1),resolved=False)
  if err<1e-10 and abs(det)>1e-14:
   h=cubic_basis(*fit.x)[0];found=dict(rz_m=pt.tolist(),psi=float(np.sum(h*pv[e])),cache_element=int(e+1),inverse_error_m=err,local_jacobian_m2=det);break
 if found is None:found=best
 print(int(n),json.dumps(found),flush=True)
 mapped[int(n)]=found
result=dict(scope=__doc__,input_hashes={str(B/n):sha256_file(B/n) for n in ['jorek_grid.h5','jorek_equil_rz.h5']},nodes=mapped,cells=[],scientific_gates_passed=[])
for cell in folds:
 ids=cell['vertices']
 if any('psi' not in mapped[n] for n in ids):
  result['cells'].append(dict(element=cell['element'],status='unresolved_local_cache_mapping'));continue
 p=[mapped[n]['psi'] for n in ids];row=dict(element=cell['element'],vertices=ids,psi=p,opposite_edge_flux_changes=[p[1]-p[0],p[2]-p[3],p[3]-p[0],p[2]-p[1]])
 result['cells'].append(row);print(json.dumps(row),flush=True)
with (R/'cpu_balanced_flux_diagnosis_20261104.json').open('x') as f:json.dump(result,f,indent=2)
