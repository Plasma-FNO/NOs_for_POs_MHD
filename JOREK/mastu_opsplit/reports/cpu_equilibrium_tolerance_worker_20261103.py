"""Isolate equilibrium stopping error at fixed geometry, physics and timestep."""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(sys.argv[2]);sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'reports'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.numerical import evaluate_native, evaluate_native_grad
from mastu_opsplit.restart import read_restart
from cpu_spatial_pair_worker_20261103 import audit_restart
from cpu_spatial_pair_analysis_20261103 import pressure_fields, field_metrics


def main():
    work=Path(sys.argv[1]);manifest=json.loads((work/'manifest.json').read_text())
    for name,value in manifest['input_hashes'].items():
        if sha256_file(name)!=value:
            raise ValueError('frozen input changed: '+name)
    start=time.monotonic()
    with (work/'input').open() as inp,(work/'stdout.log').open('x') as out,(work/'stderr.log').open('x') as err:
        proc=subprocess.run(['timeout','--signal=TERM','--kill-after=10s','6600s','mpirun','--bind-to','none','-n','4',manifest['binary']],
                            cwd=work,stdin=inp,stdout=out,stderr=err)
    result={'returncode':proc.returncode,'process_wall_seconds':time.monotonic()-start,
            'binary_sha256':sha256_file(manifest['binary']),'scientific_gates_passed':[]}
    try:
        if proc.returncode or (work/'stderr.log').stat().st_size:
            raise ValueError('native run failed or stderr nonempty')
        text=(work/'stdout.log').read_text()
        if 'equilibrium not fully converged' in text or 'MUMPS failure during' in text:
            raise ValueError('equilibrium or solver convergence failure')
        result['initial']=audit_restart(work/'jorek000000.h5',0,0.)
        result['final']=audit_restart(work/'jorek_restart.h5',48,manifest['final_time_s'])
        if 'initializer_reference' in manifest:
            import re
            from cpu_spatial_pair_worker_20261103 import check_initializer
            reference=Path(manifest['initializer_reference'])
            initial_text=(reference/'stdout.log').read_text().split(' initial energies :')[0]
            axis=float(re.findall(r'Psi_axis\s*=\s*([\-+0-9.Ee]+)',initial_text)[-1])
            boundary=float(re.findall(r'Psi_bnd\s*=\s*([\-+0-9.Ee]+)',initial_text)[-1])
            result['initializer']=check_initializer(work/'jorek000000.h5',reference/'jorek000000.h5',
                                                     [.05,.05],axis,boundary)
        geometry,state,_=read_restart(work/'jorek_restart.h5',quadrature_order=8)
        original_geometry,original,_=read_restart(Path(manifest['reference_work'])/'jorek_restart.h5',quadrature_order=8)
        for name in ['dof_map','dof_ids','basis','basis_grad','quadrature_rz','poloidal_weights','boundary_labels']:
            if not np.array_equal(getattr(geometry,name),getattr(original_geometry,name)):
                raise ValueError('equilibrium tolerance changed physical mesh')
        a=pressure_fields(evaluate_native(original,geometry).reshape(-1,7))
        b=pressure_fields(evaluate_native(state,geometry).reshape(-1,7))
        result['same_mesh_final_field_changes_q8']=field_metrics(a,b,geometry.volume_weights.ravel(),.01)
        grad=evaluate_native_grad(state,geometry)[...,1,:]
        change=evaluate_native_grad(original-state,geometry)[...,1,:]
        weights=geometry.volume_weights*geometry.quadrature_rz[...,0]**2
        result['poloidal_velocity_relative_l2_change_q8']=float(np.sqrt(np.sum(weights[...,None]*change**2)/np.sum(weights[...,None]*grad**2)))
        result['status']='completed'
    except Exception as error:
        result.update(status='failed_validation',error=str(error))
    (work/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':
    sys.exit(main())
