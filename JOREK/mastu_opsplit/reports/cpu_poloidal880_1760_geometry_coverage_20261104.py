"""Frozen q4 physical-domain coverage for the next poloidal refinement."""
import json,sys
from pathlib import Path
import h5py,numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.restart import read_restart,cubic_basis
from mastu_opsplit.queries import sample_restart
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
base=B/'scaling8_capacity400_20261104';coarse=base/'poloidal880/jorek000000.h5';fine=base/'probe3/jorek000000.h5'
for label in ['control3','perturbed3','poloidal880','probe3']:
 proof=json.loads((base/label/'result.json').read_text());assert proof['status']=='completed'
 if label!='probe3':assert proof['initial_parity']['passed'] and proof['parity']['passed']
 assert sha256_file(base/label/'jorek000000.h5')==proof['initial']['restart_sha256']
hashes={str(p):sha256_file(p) for p in [coarse,fine,base/'manifest.json']}
with h5py.File(fine) as f:v=f['vertex'][()].T-1;s=f['size'][()].transpose(2,1,0);x=f['x'][:,:,0,:].transpose(2,1,0)
q,_=np.polynomial.legendre.leggauss(16);a,b=np.meshgrid((q+1)/2,(q+1)/2,indexing='ij');_,hs,ht=cubic_basis(a.ravel(),b.ravel());minimum=float('inf')
for start in range(0,len(v),512):
 jets=x[v[start:start+512]]*s[start:start+512,...,None];rs,rt=[np.einsum('qvd,evdc->eqc',h,jets) for h in [hs,ht]];jac=rs[...,0]*rt[...,1]-rs[...,1]*rt[...,0]
 assert np.isfinite(jac).all() and (jac>0).all(),'fine q16 geometry fold'
 minimum=min(minimum,float(jac.min()))
g,_,meta=read_restart(fine,quadrature_order=4);points=g.quadrature_rz.reshape(-1,2);weights=g.volume_weights.ravel();sample=sample_restart(coarse,points);mask=sample['mask']
result=dict(scope='Geometry-only q4 physical-query coverage, independent q16 fine geometry. No extrapolation or tolerance waiver; field convergence remains separate.',input_hashes=hashes,analysis_source_sha256=sha256_file(__file__),queries_source_sha256=sha256_file(R.parent/'src/mastu_opsplit/queries.py'),fine_metadata=meta,minimum_q16_jacobian_m2=minimum,total_queries=len(mask),covered_queries=int(mask.sum()),full_domain_coverage=bool(mask.all()),uncovered_volume_m3=float(weights[~mask].sum()),uncovered_rz_m=points[~mask].tolist(),scientific_gates_passed=[])
for p,h in hashes.items():assert sha256_file(p)==h
with (R/'cpu_poloidal880_1760_geometry_coverage_20261104.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:result[k] for k in ['total_queries','covered_queries','full_domain_coverage']}))
