import ctypes,json,subprocess,sys,tempfile
from pathlib import Path
import h5py,numpy as np
root=Path('/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit');sys.path.insert(0,str(root/'src'))
from mastu_opsplit.restart import read_restart
from mastu_opsplit.numerical import evaluate_native
from mastu_opsplit.common import sha256_file
patch=root/'patches/0001-model303-axisymmetric-perturbation.patch'
add='\n'.join(s[1:] for s in patch.read_text().splitlines() if s.startswith('+') and not s.startswith('+++'))
helper=add[add.index('subroutine opsplit_perturb_node('):add.index('end subroutine opsplit_perturb_node')+len('end subroutine opsplit_perturb_node')]
report={'scope':'Preliminary shape screening using rounded native flux values; not enabled native runtime validation or physical acceptance','initializer_patch_sha256':sha256_file(patch),'script_sha256':sha256_file(__file__),'flux_axis_screening':-.10331,'flux_boundary_screening':-.01228,'normalization_limitation':'Native stdout values rounded to 5 decimals; use conservative support margin and verify actual enabled runtime','candidates':[]}
with tempfile.TemporaryDirectory(prefix='mastu-screen-') as temp:
 src=Path(temp)/'helper.f90';src.write_text(helper);lib=Path(temp)/'helper.so';subprocess.run(['gfortran','-shared','-fPIC','-fcheck=all',str(src),'-o',str(lib)],check=True)
 fn=ctypes.CDLL(str(lib)).opsplit_perturb_node_;pointer=ctypes.POINTER(ctypes.c_double);fn.argtypes=[pointer]*11
 for label in ['axis_order_equilibrium','mesh2_axis_order_equilibrium']:
  path=root/f'runs/cpu_{label}_20261102/attempt_1/work/jorek_restart.h5';g,state,meta=read_restart(path,quadrature_order=8)
  with h5py.File(path) as f:
   xy=f['x'][()][:,:,0,:].transpose(2,1,0).copy();orig=f['values'][()][:,:,0,:].transpose(2,1,0).copy();boundary=f['boundary'][()];indices=f['index'][()].T;vertex=f['vertex'][()].T-1;sizes=f['size'][()].transpose(2,1,0)
  base=evaluate_native(state,g);psin=(base[...,0]+.10331)/(.10331-.01228)
  for width in [[.04,.08],[.08,.16],[.12,.24]]:
   changed=orig.copy();center=np.array([.95,0.]);width=np.array(width);taper=np.array([.55,.75]);amplitude=np.array([.05,.05])
   for i in range(len(changed)):
    if boundary[i] or abs(xy[i,0,1])>=1.25:continue
    args=[xy[i,:,0].copy(),xy[i,:,1].copy(),orig[i,:,0].copy(),np.array([-.10331]),np.array([-.01228]),center,width,taper,amplitude,orig[i,:,4].copy(),orig[i,:,5].copy()]
    fn(*(a.ctypes.data_as(pointer) for a in args));changed[i,:,4]=args[-2];changed[i,:,5]=args[-1]
   expanded=changed.reshape(-1,7);flatids=indices.reshape(-1)-1;canonical=state.copy();canonical[flatids]=expanded
   assert np.allclose(expanded,canonical[flatids],atol=1e-13,rtol=1e-13)
   interp=evaluate_native(canonical,g);delta=interp[...,4:6]-base[...,4:6]
   def cutoff(x,a,b):
    v=np.clip((x-a)/(b-a),0,1);return (1-v)**3*(1+3*v+6*v*v)
   radius2=np.sum(((g.quadrature_rz-center)/width)**2,axis=-1)
   shape=np.exp(-.5*radius2)*cutoff(radius2,9,16)*cutoff(psin,*taper)
   target=base[...,4:6]*amplitude*shape[...,None];weights=g.volume_weights[...,None]
   err=np.sqrt(np.sum(weights*(delta-target)**2,axis=(0,1))/np.sum(weights*target**2,axis=(0,1)))
   affected=np.any(np.abs(changed[vertex,:,4:6]-orig[vertex,:,4:6])>1e-14,axis=(1,2,3));jets=orig[vertex,:,0]*sizes
   control=np.stack([jets[:,:,0],jets[:,:,0]+jets[:,:,1],jets[:,:,0]+jets[:,:,2],jets.sum(axis=2)],axis=2)
   maxflux=float(((control[affected]+.10331)/(.10331-.01228)).max())
   result={'mesh':label,'restart_sha256':sha256_file(path),'center_m':center.tolist(),'width_m':width.tolist(),'amplitudes':amplitude.tolist(),'psin_taper':taper.tolist(),'relative_perturbation_interpolation_l2':err.tolist(),'affected_elements':int(affected.sum()),'affected_elements_psi_n_control_hull_max':maxflux,'affected_boundary_vertices':int(np.count_nonzero(boundary[vertex[affected]])),'rho_min_sampled':float(interp[...,4].min()),'temperature_min_sampled':float(interp[...,5].min())}
   report['candidates'].append(result);print(label,width.tolist(),err.tolist(),maxflux,result['affected_boundary_vertices'],flush=True)
(root/'reports/cpu_initializer_shape_screen_20261102.json').write_text(json.dumps(report,indent=2)+'\n')
