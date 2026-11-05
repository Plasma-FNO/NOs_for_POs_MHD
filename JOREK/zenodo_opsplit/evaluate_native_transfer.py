"""Evaluate frozen native operators with analytic EM coupling, without EM training."""
import argparse,json,time,hashlib
import h5py,numpy as np
from pathlib import Path
import torch
from data import select,layout
from native import read_input,common_coefficients,wall_trace,NativeReference,encode_release
from native_terms import NativeOperators,NAMES
from native_rollout import NativeRollout
from train_native_rollout import load_trajectory
from validate_native import relative


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',choices=['es','em'],default='em')
    p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--trajectories',type=int,default=2)
    p.add_argument('--horizon',type=int,default=5);p.add_argument('--substeps',type=int,default=8)
    p.add_argument('--seed',type=int,default=907)
    p.add_argument('--modules',nargs='+',choices=NAMES,default=list(NAMES));p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(1)
    if min(args.horizon,args.substeps)<1 or args.trajectories<0:p.error('Invalid evaluation sizes')
    model=None;source_case=None
    if args.checkpoint:
        ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        model=NativeOperators(**ckpt['normalization'],**ckpt['model_config']).to(args.device)
        model.load_state_dict(ckpt['state_dict']);model.eval();model.requires_grad_(False);source_case=ckpt['case']
    manifest=json.loads(args.manifest.read_text())
    report=dict(case=args.case,source_case=source_case,checkpoint=str(args.checkpoint),split=args.split,modules=args.modules,
                horizon=args.horizon,substeps=args.substeps,training_performed=False,trajectories=[],
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
                manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                primary_metrics='unweighted relative L2, four-cell crop',
                observation_metrics='R-weighted relative L2, full sampled grid',
                code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    for entry in select(manifest,args.case,args.split,args.trajectories,args.seed):
        started=time.monotonic();row=dict(id=entry['id'],frames=[],status='ok')
        try:
            q,r,z=load_trajectory(entry,args.horizon+1)
            with h5py.File(entry['path']) as handle:
                keys,_,_=layout(handle,args.case)
                observed=torch.tensor(np.stack([handle[k][:args.horizon+1] for k in keys],axis=1))
            if (len(q)!=args.horizon+1 or not torch.isfinite(q).all() or
                    not torch.isfinite(observed).all() or torch.any(q[:,:2]<=0)):
                row['status']='invalid_release_state'
                row['error']='Invalid or insufficient release states; excluded before model evaluation'
                continue
            c=common_coefficients(read_input(entry['input']),args.case);c['viscosity']='old'
            ref=NativeReference(r,z,wall_trace(q[:1],r,z,c),c)
            solver=NativeRollout(ref,q[:1],model,modules=args.modules,device=args.device)
            ref=solver.ref;q=q.to(args.device);r=r.to(args.device);observed=observed.to(args.device);x=q[:1]
            with torch.no_grad():
                for step in range(1,len(q)):
                    x=solver.step(x,c['nominal_frame_dt'],args.substeps)
                    item=dict(frame=step,primary_relative_l2=relative(x,q[step:step+1])[0])
                    if args.case=='em':item['delta_psi_relative_l2']=relative(x[:,3]-q[:1,3],q[step:step+1,3]-q[:1,3])[0]
                    omega,current=ref.constraints(x)
                    decoded=encode_release(x,omega[:,1:-1,1:-1],current[:,1:-1,1:-1] if args.case=='em' else None)
                    truth=observed[step:step+1]
                    item['observed_relative_l2']=(((decoded-truth).square()*r).sum((-2,-1))/(truth.square()*r).sum((-2,-1)).clamp_min(1e-30)).sqrt()[0].tolist()
                    item['persistence_observed_relative_l2']=(((observed[:1]-truth).square()*r).sum((-2,-1))/(truth.square()*r).sum((-2,-1)).clamp_min(1e-30)).sqrt()[0].tolist()
                    row['frames'].append(item)
            row['limiter']=dict(solver.limiter_stats)
        except (ValueError,RuntimeError,FloatingPointError) as error:
            row.update(status='failed',error=str(error))
        finally:
            row['seconds']=time.monotonic()-started;report['trajectories'].append(row)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps(row),flush=True)
    if (any(row['status']=='failed' for row in report['trajectories']) or
            not any(row['status']=='ok' for row in report['trajectories'])):raise SystemExit(1)


if __name__=='__main__':main()
