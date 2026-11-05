"""Cache native equation-term supervision and conversion evidence from training prefixes."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from data import layout, select
from native import (decode_release, common_coefficients, read_input, wall_trace, NativeReference)
from native_terms import features, labels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=['es','em'],required=True)
    p.add_argument('--trajectories',type=int,default=0)
    p.add_argument('--frames',type=int,default=50)
    p.add_argument('--stride',type=int,default=5)
    args=p.parse_args()
    if args.frames<2 or args.stride<1: p.error('Positive frame count and stride required')
    torch.set_num_threads(1)
    m=json.loads(args.manifest.read_text())
    limit=args.trajectories or (100 if args.case=='es' else 76)
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(case=args.case,phase='term_pretraining',frames=args.frames,stride=args.stride,crop=4,
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  target_kind='coarse-grid FD approximation to source equations; not exported JOREK term residuals',
                  native_rollout_ready=False, splits={},
                  source_adapter='https://github.com/spamela/neural_parareal/blob/5e4203a3edff9aa8e78cca954a8c2b39fcc8ec85/coarse_solver/coarse_solver.py#L112',
                  code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    for split in ('train','val','test'):
        entries=select(m,args.case,split,limit if split=='train' else 0,907)
        if not entries or (split=='train' and len(entries)!=limit):raise ValueError('Insufficient trajectory splits')
        xs,ys,ids=[],[],[]
        rejected=[];zero_pressure=[]
        for entry in entries:
            c=common_coefficients(read_input(entry['input']),args.case)
            # Labels use only A, fluid momentum and coefficient-free scalar D.
            # EM viscosity selection does not affect these pretraining targets.
            with h5py.File(entry['path']) as f:
                keys,r,z=layout(f,args.case)
                if entry['frames']<args.frames:raise ValueError('Insufficient saved frames')
                raw=torch.tensor(np.stack([f[key][:args.frames:args.stride] for key in keys],axis=1))
            q=decode_release(raw)
            valid=torch.isfinite(q).flatten(1).all(1)&(q[:,0]>0).flatten(1).all(1)
            for i in torch.where(~valid)[0].tolist():
                rejected.append(dict(id=entry['id'],frame=i*args.stride,reason='nonfinite state or nonpositive density'))
            for i in torch.where((q[:,1]<=0).flatten(1).any(1))[0].tolist():
                zero_pressure.append(dict(id=entry['id'],frame=i*args.stride))
            initial=q[:1]
            q=q[valid]
            if not len(q):continue
            r,z=torch.tensor(r),torch.tensor(z)
            reference=NativeReference(r,z,wall_trace(initial,r,z,c),c)
            with torch.no_grad():
                x,y=features(q).float(),labels(reference,q).float()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite native labels')
            xs.append(x);ys.append(y);ids.extend([entry['id']]*len(q))
        x,y=torch.cat(xs),torch.cat(ys)
        torch.save(dict(x=x,y=y,ids=ids),args.output/f'{split}.pt')
        manifest['splits'][split]=dict(trajectories=len(entries),samples=len(x),ids=[e['id'] for e in entries],rejected_states=rejected,nonpositive_pressure_states=zero_pressure)
        if split=='train':
            lo,hi=x.amin((0,2,3)),x.amax((0,2,3))
            norms=dict(input_mean=(lo+hi)/2,input_scale=((hi-lo)/2).clamp_min(1e-20),
                       target_scale=y[:,:,4:-4,4:-4].double().square().mean((0,2,3)).sqrt().float().clamp_min(1e-20))
            torch.save(norms,args.output/'normalization.pt')
        print(json.dumps(dict(split=split,**{k:v for k,v in manifest['splits'][split].items() if k!='ids'})),flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
