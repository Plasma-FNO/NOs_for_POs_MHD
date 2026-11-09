"""Qualify the exact integer search, then audit a balanced directional geometry."""
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(sys.argv[2]);sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.common import sha256_file


def main():
    base=Path(sys.argv[1])
    authorization=json.loads((base/'authorization.json').read_text())
    for name,value in authorization['input_hashes'].items():
        if sha256_file(name)!=value:
            raise ValueError('frozen build input changed: '+name)
    scientific_objects={name:sha256_file(base/'source160/.obj'/name) for name in authorization['scientific_objects_before']}
    if scientific_objects!=authorization['scientific_objects_before']:
        raise ValueError('scientific matrix/solver objects changed during integer-only rebuild')
    stages=authorization['stages']
    stages['regression8']['binary']=str(base/'solver160.exe')
    for label in ['balanced200','balanced160']:
        stages[label]['binary']=str(base/'planner240.exe')
    manifest={**authorization,'stages':stages,'final_time_s':1.4059131112280784e-6,
              'input_hashes':{**authorization['input_hashes'],str(base/'solver160.exe'):sha256_file(base/'solver160.exe'),
                              str(base/'planner240.exe'):sha256_file(base/'planner240.exe')}}
    (base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    command=lambda label:[sys.executable,str(base/'worker.py'),str(base),str(ROOT),label]
    subprocess.run(command('regression8'),check=False)
    regression=json.loads((base/'regression8/result.json').read_text())
    if regression.get('returncode')!=0 or 'initial' not in regression:
        raise ValueError('native integration run failed before geometry qualification')
    with h5py.File(base/'regression8/jorek000000.h5') as actual,h5py.File(base/'reference0.h5') as expected:
        geometry={key:bool(np.array_equal(actual[key][()],expected[key][()]))
                  for key in ['x','vertex','size','index','boundary','constrained','n_sons','neighbours']}
    if not all(geometry.values()):
        raise ValueError('optimized native geometry/neighbours changed: '+str(geometry))
    qualification={'solver160_accepted':regression['status']=='completed','geometry_and_neighbours_exact':geometry,
                   'scientific_objects_identical':scientific_objects,'solver160_sha256':sha256_file(base/'solver160.exe'),
                   'planner240_sha256':sha256_file(base/'planner240.exe'),'regression':regression,
                   'field_gradient_tolerances_unchanged':authorization['parity_protocol']}
    (base/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
    for label in ['balanced200','balanced160']:
        subprocess.run(command(label),check=True)
        with h5py.File(base/label/'jorek_restart.h5') as stream:
            nodes=int(stream['n_nodes'][()].item());elements=int(stream['n_elements'][()].item())
        if max(nodes,elements)<=155000:
            selection={'selected':label,'nodes':nodes,'elements':elements,
                       'policy':'Balanced1.5x exterior/core-poloidal counts,2 candidates in descending core-radial count; first positive q8 geometry <=155000.'}
            (base/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
            print(json.dumps({'qualification':qualification['solver160_accepted'],'selection':selection}),flush=True)
            return
    raise ValueError('Neither balanced geometry fits the frozen validated-capacity margin')


if __name__=='__main__':
    main()
