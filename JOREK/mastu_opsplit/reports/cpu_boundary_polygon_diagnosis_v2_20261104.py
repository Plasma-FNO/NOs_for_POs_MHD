from pathlib import Path
import re,json,sys,hashlib
import h5py,numpy as np
sys.path.insert(0,str(Path('JOREK/mastu_opsplit/src').resolve()))
from mastu_opsplit.restart import cubic_basis
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102');R=Path('JOREK/mastu_opsplit/reports')
refs={'poloidal220':B/'mumps_scaling8_20261104/control3','poloidal440':B/'scaling8_poloidal440_20261104/probe3','poloidal880':B/'scaling8_poloidal880_20261104/probe3'}
text=(refs['poloidal220']/'input').read_text();rr=re.findall(r'R_limiter\(\s*\d+\)\s*=\s*([+\-\d.eE]+)',text);zz=re.findall(r'Z_limiter\(\s*\d+\)\s*=\s*([+\-\d.eE]+)',text);poly=np.array([rr,zz],float).T
start=poly;end=np.roll(poly,-1,axis=0);vec=end-start;length2=(vec**2).sum(axis=1);keep=length2>0;start,vec,length2=start[keep],vec[keep],length2[keep]
results={};q,w=np.polynomial.legendre.leggauss(16);q=(q+1)/2;w=w/2
for label,base in refs.items():
 p=base/'jorek000000.h5'
 with h5py.File(p) as f:x=f['x'][:,:,0,:].transpose(2,1,0);v=f['vertex'][()].T-1;s=f['size'][()].transpose(2,1,0);boundary_flags=f['boundary'][()]
 edges=np.stack([v,np.roll(v,-1,axis=1)],axis=-1).reshape(-1,2);_,idx,cnt=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True);boundary=idx[cnt==1];boundary=boundary[(boundary_flags[edges[boundary]]>0).all(axis=1)];samples=[];weights=[]
 for side in range(4):
  es=boundary[boundary%4==side]//4
  ss,tt=[(q,np.zeros_like(q)),(np.ones_like(q),q),(q,np.ones_like(q)),(np.zeros_like(q),q)][side]
  h,hs,ht=cubic_basis(ss,tt);jets=x[v[es]]*s[es,...,None];pos=np.einsum('qvd,evdc->eqc',h,jets);tangent=np.einsum('qvd,evdc->eqc',hs if side%2==0 else ht,jets)
  samples.append(pos.reshape(-1,2));weights.append((np.linalg.norm(tangent,axis=-1)*w).ravel())
 points=np.concatenate(samples);weights=np.concatenate(weights);dmin=np.full(len(points),np.inf)
 for a,b,c in zip(start,vec,length2):
  d=points-a;t=np.clip(d@b/c,0,1);dist=np.linalg.norm(d-t[:,None]*b,axis=1);dmin=np.minimum(dmin,dist)
 i=int(dmin.argmax());results[label]=dict(restart_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),boundary_edges=len(boundary),maximum_distance_to_input_limiter_polygon_m=float(dmin[i]),arclength_rms_distance_m=float(np.sqrt(np.sum(weights*dmin*dmin)/weights.sum())),distance_quantiles_m=np.quantile(dmin,[.5,.9,.99,.999]).tolist(),maximum_point_rz_m=points[i].tolist(),fraction_boundary_length_over_1mm=float(weights[dmin>.001].sum()/weights.sum()))
print(json.dumps(results,indent=2))
with (R/'cpu_boundary_polygon_diagnosis_v2_20261104.json').open('x') as f:json.dump(dict(scope='Native positive-boundary-flag-filtered geometry-only boundary approximation diagnosis against unchanged input limiter polygon; no acceptance threshold or physical boundary change.',results=results,scientific_gates_passed=[]),f,indent=2)
