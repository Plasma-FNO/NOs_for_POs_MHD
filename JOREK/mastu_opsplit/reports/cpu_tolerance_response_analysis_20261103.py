"""Measure equilibrium-tolerance sensitivity of paired responses at fixed mesh."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from cpu_spatial_pair_analysis_20261103 import (
    ROOT, GEOMETRY_KEYS, evaluate_native, field_metrics, pressure_fields, read_restart,
)
from mastu_opsplit.common import sha256_file


def main():
    bulk=Path(sys.argv[1])
    out=ROOT/'reports/cpu_tolerance_response_20261103.json'
    if out.exists():
        raise ValueError('preserve existing numerical-sensitivity evidence')
    folders={'default_baseline':bulk/'spatial_pair_20261103/baseline48',
             'default_perturbed':bulk/'spatial_pair_20261103/perturbed48',
             'strict_baseline':bulk/'equilibrium_tolerance_20261103',
             'strict_perturbed':bulk/'equilibrium_tolerance_perturbed_20261103'}
    reference=folders['strict_baseline']/'jorek_restart.h5'
    geometry,_,metadata=read_restart(reference,quadrature_order=8)
    with h5py.File(reference) as stream:
        expected={key:stream[key][()] for key in GEOMETRY_KEYS}
    _,first=np.unique(expected['index'].T.ravel(),return_index=True)
    fields={};hashes={}
    for name,folder in folders.items():
        result=json.loads((folder/'result.json').read_text())
        path=folder/'jorek_restart.h5'
        if result['status']!='completed' or sha256_file(path)!=result['final']['restart_sha256']:
            raise ValueError('unvalidated or changed native output: '+name)
        if result['final']['health']['steps']!=48 or abs(result['final']['health']['final_time_s']-metadata['time_seconds'])>1e-15:
            raise ValueError('physical times/steps differ')
        with h5py.File(path) as stream:
            for key,value in expected.items():
                if not np.array_equal(stream[key][()],value):
                    raise ValueError('physical mesh changed: '+key)
            nodal=stream['values'][()][:,:,0].transpose(2,1,0).reshape(-1,7)
        fields[name]=pressure_fields(evaluate_native(nodal[first],geometry).reshape(-1,7))
        hashes[str(path)]=sha256_file(path)
    original=fields['default_perturbed']-fields['default_baseline']
    strict=fields['strict_perturbed']-fields['strict_baseline']
    report={'scope':'Numerical equilibrium stopping sensitivity, not spatial convergence; same complete physical mesh/time.',
            'equil_accuracy_before':1e-6,'equil_accuracy_after':1e-10,'quadrature_order':8,
            'source_sha256':sha256_file(__file__),'restart_sha256':hashes,
            'final_time_s':metadata['time_seconds'],'full_volume_m3':float(geometry.volume_weights.sum()),
            'paired_response_changes':field_metrics(original,strict,geometry.volume_weights.ravel(),.01),
            'scientific_gates_passed':[]}
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['paired_response_changes']),flush=True)


if __name__=='__main__':
    main()
