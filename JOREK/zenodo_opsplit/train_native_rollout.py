"""Five-step native JOREK rollout tuning with term anchoring and implicit adjoints.

GPU sparse direct solves and FNOs, batch20 via gradient accumulation.
The CPU elliptic backend remains available for validation. Resume commits
every optimizer step; a timed exit code 75 requests Slurm continuation.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from data import layout,select
from native import decode_release,read_input,common_coefficients,wall_trace,NativeReference
from native_terms import NativeOperators,features,labels,NAMES
from native_rollout import NativeRollout
from train_es import save_checkpoint,relative_lp,range_statistics
from train_native_terms import term_loss


def load_trajectory(entry,frames):
    with h5py.File(entry['path']) as f:
        keys,r,z=layout(f,entry['case'])
        raw=torch.tensor(np.stack([f[k][:frames] for k in keys],axis=1))
    q=decode_release(raw)
    return q,torch.tensor(r),torch.tensor(z)


def admissible_windows(values,horizon):
    valid=torch.isfinite(values).flatten(2).all(2)&(values[:,:,:2]>0).flatten(2).all(2)
    windows=[];excluded=[]
    for trajectory in range(len(values)):
        for start in range(values.shape[1]-horizon):
            (windows if valid[trajectory,start:start+horizon+1].all() else excluded).append((trajectory,start))
    return windows,excluded


def sequence_objective(solver,frames,mean,scale,dt,substeps):
    x=frames[:1];loss=x.new_zeros(())
    for truth in frames[1:]:
        x=solver.step(x,dt,substeps)
        loss=loss+relative_lp((x-mean[None,:,None,None])/scale[None,:,None,None],
                              (truth[None]-mean[None,:,None,None])/scale[None,:,None,None])
    return loss


def require_preflight(path,provenance):
    report=json.loads(path.read_text())
    if not report.get('passed') or report.get('samples')!=0:
        raise ValueError('Full training requires a passed exhaustive window audit')
    for key,value in dict(case=provenance['case'],manifest_sha256=provenance['manifest_sha256'],
        checkpoint_sha256=provenance['source_sha256'],code_sha256=provenance['code_sha256'],
        train_ids=provenance['train_ids'],val_ids=provenance['val_ids'],
        substeps=provenance['protocol']['substeps']).items():
        if report.get(key)!=value:raise ValueError('Window audit does not match training: '+key)
    if provenance['protocol']['blend']!=1.:raise ValueError('Window audit uses blend=1')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--pretrained',type=Path,required=True)
    p.add_argument('--validation-report',type=Path,help='Required exhaustive window audit for full training')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',choices=['es','em'],required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--substeps',type=int,default=8)
    p.add_argument('--elliptic-device',choices=['cpu','cuda'],help='Default: same device as FNOs')
    p.add_argument('--max-seconds',type=float,default=0,help='Checkpoint and exit75 at this wall-time budget; 0 disables')
    p.add_argument('--epochs',type=int,default=250);p.add_argument('--batch-size',type=int,default=20)
    p.add_argument('--train-trajectories',type=int,default=0,help='Optional target-training budget; 0 uses ES100/EM76')
    p.add_argument('--seed',type=int,default=907)
    p.add_argument('--checkpoint-steps',action='store_true',help='Recompute implicit steps during backward to bound GPU memory')
    p.add_argument('--term-weight',type=float,default=1.);p.add_argument('--blend',type=float,default=1.)
    p.add_argument('--freeze',nargs='*',choices=NAMES,default=[])
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if not args.smoke and not args.validation_report:p.error('Full training requires --validation-report')
    if min(args.substeps,args.epochs,args.batch_size)<1 or args.term_weight<0:p.error('Invalid training settings')
    if len(args.freeze)==len(NAMES):p.error('Fully frozen transfer is inference; use evaluate_native_transfer.py')
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    args.elliptic_device=args.elliptic_device or ('cuda' if args.device.startswith('cuda') else 'cpu')
    if args.max_seconds<0:p.error('Nonnegative wall-time budget required')
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    m=json.loads(args.manifest.read_text());count=args.train_trajectories or (100 if args.case=='es' else 76)
    if count<1:p.error('Positive target trajectory count required')
    train=select(m,args.case,'train',2 if args.smoke else count,args.seed)
    val=select(m,args.case,'val',1 if args.smoke else 0,args.seed)
    if len(train)!=(2 if args.smoke else count) or not val:raise ValueError('Insufficient family-separated data')
    cfg=dict(horizon=5,frames=50,epochs=1 if args.smoke else args.epochs,batch_size=1 if args.smoke else args.batch_size,
             substeps=args.substeps,cfl=.4,max_substeps_per_frame=4096,term_weight=args.term_weight,blend=args.blend,freeze=args.freeze,
             elliptic_device=args.elliptic_device,max_seconds=args.max_seconds,optimizer='Adam',lr=.001,weight_decay=1e-4,clip=1.,scheduler='StepLR50 gamma0.5',seed=args.seed,
             checkpoint_steps=args.checkpoint_steps)
    checkpoint=torch.load(args.pretrained,map_location='cpu',weights_only=False)
    if checkpoint.get('phase') not in ('term_pretraining','residual_pretraining','native_rollout'):raise ValueError('Native checkpoint required')
    model=NativeOperators(**checkpoint['normalization'],**checkpoint['model_config']).to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    for name in args.freeze:model.blocks[name].requires_grad_(False)
    optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=.001,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,50,.5)
    values=torch.stack([load_trajectory(e,cfg['frames'])[0] for e in train])
    windows,excluded=admissible_windows(values,cfg['horizon'])
    if not windows:raise ValueError('No admissible native training windows')
    valid_state=torch.isfinite(values).flatten(2).all(2)&(values[:,:,:2]>0).flatten(2).all(2)
    mean,scale=range_statistics(values[valid_state][None])
    mean,scale=mean.to(args.elliptic_device),scale.to(args.elliptic_device)
    provenance=dict(case=args.case,phase='native_rollout',protocol=cfg,smoke=args.smoke,
        source_case=checkpoint['case'],source_checkpoint=str(args.pretrained.resolve()),
        transfer=checkpoint.get('transfer'),
        source_sha256=hashlib.sha256(args.pretrained.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        train_ids=[e['id'] for e in train],val_ids=[e['id'] for e in val],
        excluded_windows=[dict(id=train[t]['id'],start=s) for t,s in excluded],
        viscosity='old; release profile inferred from training residuals',
        time='nominal common-native dt=5; exact original timestamps unavailable',
        normalization='state: train-prefix primary-variable minmax; operator interface: source checkpoint',
        loss=('batch-summed five-step normalized relative L2 + weighted normalized correction L2 at each window start'
              if model.is_residual else 'batch-summed five-step normalized relative L2 + weighted independent term MSE at each window start'),
        integrator='first-order IMEX, CFL<=0.4 for fluid/poloidal Alfven waves, analytic boundary strip; sparse LU with implicit adjoint',
        elliptic_backend='cuDSS 0.7.1 GPU FP64' if args.elliptic_device=='cuda' else 'SciPy CPU FP64',
        positivity='implicit upwind/diffusion positive scalar fallback, global convex correction with 1% reserve; no state clipping',
        code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    if args.elliptic_device=='cuda':
        import os
        from gpu_sparse import DEFAULT_LIBRARY,library
        library()
        runtime=Path(os.environ.get('JOREK_CUDSS_LIBRARY',DEFAULT_LIBRARY))
        provenance['cudss_runtime']=dict(path=str(runtime),sha256=hashlib.sha256(runtime.read_bytes()).hexdigest())
    if not args.smoke:
        require_preflight(args.validation_report,provenance)
        provenance['validation_report']=str(args.validation_report.resolve())
        provenance['validation_sha256']=hashlib.sha256(args.validation_report.read_bytes()).hexdigest()
    epoch0,offset0,best=0,0,float('inf')
    if args.resume:
        if json.loads((args.output/'provenance.json').read_text())!=provenance:raise ValueError('Resume requires unchanged code/data/protocol')
        saved=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        epoch0,offset0,best=saved['next_epoch'],saved['next_offset'],saved['best']
        torch.set_rng_state(saved['torch_rng'])
        if args.device.startswith('cuda'):torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

    @lru_cache(maxsize=8)
    def solver_for(split,index):
        entry=(train if split=='train' else val)[index]
        initial,r,z=load_trajectory(entry,1)
        c=common_coefficients(read_input(entry['input']),args.case);c['viscosity']='old'
        ref=NativeReference(r,z,wall_trace(initial,r,z,c),c)
        return NativeRollout(ref,initial,model,args.blend,device=args.elliptic_device,checkpoint_steps=args.checkpoint_steps)

    def payload(next_epoch,next_offset):
        return dict(state_dict=model.state_dict(),normalization=checkpoint['normalization'],model_config=model.config,
                    case=args.case,phase='native_rollout',smoke=args.smoke,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    next_epoch=next_epoch,next_offset=next_offset,best=best,mean=mean,scale=scale,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
                    protocol=cfg,manifest_sha256=provenance['manifest_sha256'])

    started=time.monotonic()
    for epoch in range(epoch0,cfg['epochs']):
        order=torch.randperm(len(windows),generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        if args.smoke:order=[windows.index((0,0))]
        model.train()
        for offset in range(offset0 if epoch==epoch0 else 0,len(order),cfg['batch_size']):
            if args.max_seconds and time.monotonic()-started>=args.max_seconds:
                save_checkpoint(args.output/'last.pt',payload(epoch,offset))
                raise SystemExit(75)
            optimizer.zero_grad(set_to_none=True);loss_sum=0.;limiter_totals={}
            for wi in order[offset:offset+cfg['batch_size']]:
                trajectory,start=windows[wi];solver=solver_for('train',trajectory);solver.reset_limiter_stats()
                frames=values[trajectory,start:start+cfg['horizon']+1].to(args.elliptic_device)
                try:
                    loss=sequence_objective(solver,frames,mean,scale,solver.ref.c['nominal_frame_dt'],args.substeps)
                except (FloatingPointError,RuntimeError) as error:
                    raise RuntimeError(f"Native window failed: case={args.case}, epoch={epoch}, offset={offset}, id={train[trajectory]['id']}, start={start}") from error
                predicted=model(features(frames[:1]).to(device=args.device,dtype=torch.float32))
                with torch.no_grad():
                    target=(torch.zeros_like(predicted) if model.is_residual else
                            labels(solver.ref,frames[:1]).to(device=args.device,dtype=torch.float32)/model.target_scale[None,:,None,None])
                anchor=term_loss(predicted,target)
                objective=loss+args.term_weight*anchor.to(loss.device)
                if not torch.isfinite(objective):raise FloatingPointError('Nonfinite native rollout objective')
                objective.backward();loss_sum+=float(objective.detach())
                for key,value in solver.limiter_stats.items():
                    limiter_totals[key]=min(limiter_totals.get(key,1.),value) if key.startswith('minimum') else limiter_totals.get(key,0)+value
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            save_checkpoint(args.output/'last.pt',payload(epoch,min(offset+cfg['batch_size'],len(order))))
            row=dict(epoch=epoch,offset=offset,objective=loss_sum,seconds=time.monotonic()-started,limiter=limiter_totals)
            print(json.dumps(row),flush=True)
            with (args.output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        model.eval();validation=[]
        with torch.no_grad():
            for i,entry in enumerate(val):
                if args.max_seconds and time.monotonic()-started>=args.max_seconds:
                    save_checkpoint(args.output/'last.pt',payload(epoch,len(order)))
                    raise SystemExit(75)
                frames,_,_=load_trajectory(entry,6 if args.smoke else 50)
                frames=frames.to(args.elliptic_device)
                if not torch.isfinite(frames).all() or torch.any(frames[:,:2]<=0):
                    validation.append(dict(id=entry['id'],status='invalid_release_state'));continue
                solver=solver_for('val',i);solver.reset_limiter_stats()
                score=sequence_objective(solver,frames,mean,scale,solver.ref.c['nominal_frame_dt'],args.substeps)/(len(frames)-1)
                validation.append(dict(id=entry['id'],status='ok',relative_lp=float(score),limiter=dict(solver.limiter_stats)))
        scored=[v['relative_lp'] for v in validation if v['status']=='ok']
        if not scored:raise ValueError('No admissible validation trajectories')
        score=sum(scored)/len(scored);scheduler.step()
        if score<best:
            best=score;save_checkpoint(args.output/'best.pt',payload(epoch+1,0))
        save_checkpoint(args.output/'last.pt',payload(epoch+1,0))
        (args.output/'validation.json').write_text(json.dumps(dict(epoch=epoch,trajectories=validation),indent=2)+'\n')
        offset0=0


if __name__=='__main__':main()
