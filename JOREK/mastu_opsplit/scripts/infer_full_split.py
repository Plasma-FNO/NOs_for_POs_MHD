"""Called once per native step, on rank zero, using only the solver's own state."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import h5py
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mastu_opsplit.full_rollout import FullOpsSplitOperators, native_jets, normalize_current_fields, write_prediction
from mastu_opsplit.restart import read_restart


def infer(checkpoint, restart, linear, output):
    checkpoint=Path(checkpoint)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload.get('schema')!='mastu_full_opsplit_v1': raise ValueError('Not a full OpsSplit checkpoint')
    if hashlib.sha256(Path(linear).read_bytes()).hexdigest()!=payload['linear_sha256']:
        raise ValueError('Checkpoint frozen operator mismatch')
    linear_manifest=json.loads((Path(linear).parent/'manifest.json').read_text()) if (Path(linear).parent/'manifest.json').is_file() else json.loads(Path('opsplit_manifest.json').read_text())
    with h5py.File(restart) as f,h5py.File(linear_manifest['baseline']) as b:
        for key in ('x','size','index','vertex','boundary','n_sons','constrained'):
            if not np.array_equal(f[key][()],b[key][()]):raise ValueError('Inference geometry mismatch: '+key)
    geometry,coeff,metadata=read_restart(restart)
    model=FullOpsSplitOperators(geometry.bounds,payload['model_config'])
    model.load_state_dict(payload['model_state']);model.eval()
    mean=torch.as_tensor(payload['input_mean'],dtype=torch.float64)
    scale=torch.as_tensor(payload['input_scale'],dtype=torch.float64)
    out_mean=torch.as_tensor(payload['output_mean'],dtype=torch.float64)
    out_scale=torch.as_tensor(payload['output_scale'],dtype=torch.float64)
    if mean.shape!=(7,) or scale.shape!=(7,) or out_mean.shape!=(5,3) or out_scale.shape!=(5,3):
        raise ValueError('Invalid normalization layout')
    if not all(torch.isfinite(a).all() for a in (mean,scale,out_mean,out_scale)) or not (scale>0).all() or not (out_scale>0).all():
        raise ValueError('Invalid normalization statistics')
    with torch.no_grad():
        values,_=native_jets(torch.as_tensor(coeff).unsqueeze(0),geometry)
        fields=normalize_current_fields(values.flatten(1,2),mean,scale)
        coords=torch.as_tensor(geometry.quadrature_rz.reshape(-1,2),dtype=torch.float32)
        result=model(fields,coords,torch.tensor([[payload['s_perp']]],dtype=torch.float32))
        result=(result.double()*out_scale+out_mean).numpy().reshape(*geometry.basis.shape[:2],5,3)
    with h5py.File(restart) as f: vertices=f['vertex'][()].T
    write_prediction(output,int(metadata['steps']),result,vertices,linear)
    audit=dict(step=int(metadata['steps']),state_sha256=hashlib.sha256(Path(restart).read_bytes()).hexdigest(),
               forcing_sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest(),
               checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),trained=payload.get('trained',False))
    with Path(f"opsplit_inference_{metadata['steps']:06d}.json").open('x') as f:json.dump(audit,f,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--restart',default='opsplit_state.h5');p.add_argument('--linear',default='opsplit_linear.bin')
    p.add_argument('--output',default='opsplit_prediction.bin');a=p.parse_args()
    infer(a.checkpoint,a.restart,a.linear,a.output)
