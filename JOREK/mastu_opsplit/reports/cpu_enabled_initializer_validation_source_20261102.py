import json,sys,re
from pathlib import Path
from datetime import datetime,timezone
import h5py,numpy as np
root=Path('/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit');sys.path.insert(0,str(root/'src'))
from mastu_opsplit.restart import read_restart,diagnose_restart
from mastu_opsplit.numerical import evaluate_native
from mastu_opsplit.common import sha256_file
baseline=root/'runs/cpu_mesh2_axis_order_equilibrium_20261102/attempt_1/work/jorek_restart.h5'
paths=[root/f'runs/cpu_fine_initializer_mpi{n}_20261102/attempt_1/work/jorek_restart.h5' for n in [1,4]]
for path in paths:
 result=json.loads((path.parents[2]/'result.json').read_text());assert result['status']=='completed', result['status']
with h5py.File(baseline) as f:base=f['values'][()];boundary=f['boundary'][()];vertex=f['vertex'][()].T-1;sizes=f['size'][()].transpose(2,1,0);psi=f['values'][()][0,:,0,:].T
with h5py.File(paths[0]) as f:actual=f['values'][()]
with h5py.File(paths[1]) as f:parallel=f['values'][()]
assert np.array_equal(actual,parallel),'MPI equivalence failed'
assert np.array_equal(base[[0,1,2,3,6]],actual[[0,1,2,3,6]]),'Unselected fields changed'
assert np.array_equal(base[:,:,:,boundary!=0],actual[:,:,:,boundary!=0]),'Boundary coefficients changed'
geom,state,metadata=read_restart(paths[0],quadrature_order=8);_,basestate,_=read_restart(baseline,quadrature_order=8)
vals=evaluate_native(state,geom);initial=evaluate_native(basestate,geom);weights=geom.volume_weights
changed=np.any(np.abs((actual-base)[4:6,:,0,:].transpose(2,1,0)[vertex])>1e-14,axis=(1,2,3));jets=psi[vertex]*sizes;control=np.stack([jets[:,:,0],jets[:,:,0]+jets[:,:,1],jets[:,:,0]+jets[:,:,2],jets.sum(axis=2)],axis=2)
# Conservative interval accounts for native stdout rounding of both flux extrema.
axis_interval=[-.103315,-.103305];boundary_interval=[-.012285,-.012275]
maxpsin=max(float(((control[changed]-a)/(b-a)).max()) for a in axis_interval for b in boundary_interval)
assert maxpsin<.85,'Affected elements approach separatrix'
assert not np.any(boundary[vertex[changed]]),'Affected elements touch boundary'
assert vals[...,4:6].min()>0,'Sampled nonpositive density/temperature'
relative=actual[4:6,0,0,:]/base[4:6,0,0,:]-1
assert relative.min()>-1e-12 and relative.max()<=.05+1e-12
report={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Enabled native zero-step initialization validation only; physical pilot and Gates A/B/C remain incomplete','script_sha256':sha256_file(__file__),'baseline_sha256':sha256_file(baseline),'enabled_restart_sha256':sha256_file(paths[0]),'enabled_mpi4_restart_sha256':sha256_file(paths[1]),'parameters':{'amplitudes':[.05,.05],'center_m':[.95,0.],'width_m':[.08,.16],'psi_n_taper':[.55,.75]},'mpi1_mpi4_fields_exact':True,'unselected_fields_exact':True,'boundary_coefficients_exact':True,'affected_elements':int(changed.sum()),'affected_elements_touch_boundary':False,'affected_elements_psi_n_control_hull_max_with_rounding_interval':maxpsin,'flux_normalization_limitation':'Conservative +/-5e-6 intervals for final printed native axis/boundary flux; runtime initializer itself uses full-precision ES values','max_nodal_relative_perturbation':relative.max(axis=1).tolist(),'metadata':metadata,'diagnostics':diagnose_restart(paths[0],baseline,quadrature_order=8),'gate_A_passed':False,'balance_checked':False,'native_constraints_independently_checked':False}
out=root/'reports/cpu_enabled_initializer_validation_20261102.json'
with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({k:report[k] for k in ['mpi1_mpi4_fields_exact','unselected_fields_exact','boundary_coefficients_exact','affected_elements','affected_elements_psi_n_control_hull_max_with_rounding_interval','max_nodal_relative_perturbation']},indent=2))
