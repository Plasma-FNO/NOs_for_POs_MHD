"""Check native current/vorticity weak consistency and quadrature sensitivity."""
import gc
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.restart import read_restart
from mastu_opsplit.numerical import model303_constraint_blocks


def main():
    base=Path(sys.argv[1])
    manifest=json.loads((base/'manifest.json').read_text())
    out=ROOT/'reports/cpu_spatial_constraint_diagnosis_20261103.json'
    if out.exists():
        raise ValueError('preserve completed diagnostic')
    report={'source_sha256':sha256_file(__file__),'native_gauss_order':4,
            'scope':'Interior weak current/vorticity blocks only; native boundary/axis rows excluded; no Gate B approval.',
            'normalization':'Euclidean native weak-vector norm divided by mass-vector norm; diagnostic within each mesh, not a mesh-independent field norm.',
            'comparisons':{}}
    for mesh in ['existing','refined']:
        old=Path(manifest['references']['baseline48']['work']) if mesh=='existing' else base/'baseline48'
        pert=Path(manifest['references']['perturbed48']['work']) if mesh=='existing' else base/'perturbed48'
        for order in [4,8]:
            geometry,state,metadata=read_restart(old/'jorek_restart.h5',quadrature_order=order)
            with h5py.File(old/'jorek_restart.h5') as stream:
                indices=stream['index'][()].T.ravel()
                _,first,inverse=np.unique(indices,return_index=True,return_inverse=True)
                xy=np.repeat(stream['x'][:,0,0,:].T,4,axis=0)[first]
                multiplicity=np.bincount(inverse)
            # Remove native axis aliases and a conservative circle around the magnetic axis.
            interior=(geometry.boundary_labels==0)&(multiplicity==1)&(np.linalg.norm(xy-np.array([.9,0.]),axis=1)>.1)
            with h5py.File(pert/'jorek_restart.h5') as stream:
                for name in ['x','vertex','size','index','boundary','constrained','n_sons']:
                    with h5py.File(old/'jorek_restart.h5') as reference:
                        if not np.array_equal(stream[name][()],reference[name][()]):
                            raise ValueError('paired mesh changed')
                pert_state=stream['values'][()][:,:,0].transpose(2,1,0).reshape(-1,7)[first]
            for label,values in [('baseline',state),('perturbed',pert_state),('paired_response',pert_state-state)]:
                mass,gradient=model303_constraint_blocks(values,geometry)
                error=np.linalg.norm((mass+gradient)[interior],axis=0)
                norm=np.linalg.norm(mass[interior],axis=0)
                metrics={name:{'relative_weak_residual':float(error[i]/norm[i]) if norm[i] else None,
                               'mass_vector_norm':float(norm[i]),'residual_vector_norm':float(error[i])}
                         for i,name in enumerate(['zj','w'])}
                report['comparisons'][f'{mesh}_q{order}_{label}']={'interior_test_dofs':int(interior.sum()),'fields':metrics}
                print(json.dumps({f'{mesh}_q{order}_{label}':metrics}),flush=True)
                del mass,gradient
            del geometry,state,pert_state,values
            gc.collect()
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
