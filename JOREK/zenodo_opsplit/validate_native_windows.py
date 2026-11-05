"""Gate native training on every admissible five-frame window and validation rollout.

Independent worker processes share the allocated GPU. Each training trajectory
also backpropagates its worst-loss window, including all four term anchors.
This checks the initial checkpoint, not future optimizer iterates or accuracy.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import torch
from data import select
from native import NativeReference,common_coefficients,read_input,wall_trace
from native_rollout import NativeRollout
from native_terms import NativeOperators,features,labels
from train_es import range_statistics
from train_native_rollout import admissible_windows,load_trajectory,sequence_objective
from train_native_terms import term_loss


def initialize(checkpoint,mean,scale,device,substeps):
    global MODEL,MEAN,SCALE,DEVICE,SUBSTEPS
    torch.set_num_threads(1)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    MODEL=NativeOperators(**saved['normalization'],**saved['model_config']).to(device)
    MODEL.load_state_dict(saved['state_dict']);MODEL.eval()
    MEAN,SCALE=mean.to(device),scale.to(device)
    DEVICE,SUBSTEPS=device,substeps


def check_trajectory(task):
    entry,split=task;started=time.monotonic()
    q,r,z=load_trajectory(entry,50)
    c=common_coefficients(read_input(entry['input']),entry['case']);c['viscosity']='old'
    ref=NativeReference(r,z,wall_trace(q[:1],r,z,c),c)
    solver=NativeRollout(ref,q[:1],MODEL,device=DEVICE,checkpoint_steps=True)
    row=dict(id=entry['id'],split=split,windows=0,excluded=[],failures=[],gradient_checked=False)
    if split=='train':
        windows,excluded=admissible_windows(q[None],5)
        starts=[s for _,s in windows];row['excluded']=[s for _,s in excluded]
        horizon=5
    else:
        valid=torch.isfinite(q).all() and torch.all(q[:,:2]>0)
        starts=[0] if valid else [];horizon=49
        if not valid:row['excluded']=[0]
    q=q.to(DEVICE);worst=(-1.,None)
    for start in starts:
        try:
            with torch.no_grad():
                score=sequence_objective(solver,q[start:start+horizon+1],MEAN,SCALE,c['nominal_frame_dt'],SUBSTEPS)
            if not torch.isfinite(score):raise FloatingPointError('Nonfinite rollout loss')
            row['windows']+=1
            if row['windows']%10==0:
                print(json.dumps(dict(id=entry['id'],split=split,checked=row['windows'],seconds=time.monotonic()-started)),flush=True)
            if float(score)>worst[0]:worst=(float(score),start)
        except (RuntimeError,ValueError,FloatingPointError) as error:
            row['failures'].append(dict(start=start,error=str(error)))
    if split=='train' and worst[1] is not None:
        try:
            MODEL.zero_grad(set_to_none=True)
            frames=q[worst[1]:worst[1]+6]
            score=sequence_objective(solver,frames,MEAN,SCALE,c['nominal_frame_dt'],SUBSTEPS)
            predicted=MODEL(features(frames[:1]).float())
            with torch.no_grad():
                target=(torch.zeros_like(predicted) if MODEL.is_residual else
                        labels(solver.ref,frames[:1]).float()/MODEL.target_scale[None,:,None,None])
            loss=score+term_loss(predicted,target)
            loss.backward()
            for name,block in MODEL.blocks.items():
                gradients=[p.grad for p in block.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError('Missing/nonfinite gradient in '+name)
            row['gradient_checked']=True
            row['gradient_window']=worst[1]
        except (RuntimeError,ValueError,FloatingPointError) as error:
            row['failures'].append(dict(start=worst[1],gradient=True,error=str(error)))
    MODEL.zero_grad(set_to_none=True)
    row.update(worst_loss=worst[0],limiter=dict(solver.limiter_stats),seconds=time.monotonic()-started)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--pretrained',type=Path,required=True)
    p.add_argument('--case',choices=['es','em'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--substeps',type=int,default=8)
    p.add_argument('--train-trajectories',type=int,default=0)
    p.add_argument('--seed',type=int,default=907)
    p.add_argument('--samples',type=int,default=0,help='Development subset only; never passes the full-training gate')
    args=p.parse_args();torch.set_num_threads(1)
    if args.workers<1 or args.substeps<1 or args.samples<0:p.error('Invalid validation limits')
    m=json.loads(args.manifest.read_text());count=args.train_trajectories or (100 if args.case=='es' else 76)
    if count<1:p.error('Positive target trajectory count required')
    train=select(m,args.case,'train',count,args.seed);val=select(m,args.case,'val',0,args.seed)
    if len(train)!=count or not val:raise ValueError('Insufficient family-separated data')
    values=torch.stack([load_trajectory(e,50)[0] for e in train])
    valid=torch.isfinite(values).flatten(2).all(2)&(values[:,:,:2]>0).flatten(2).all(2)
    mean,scale=range_statistics(values[valid][None]);windows,excluded=admissible_windows(values,5)
    del values
    tasks=[(e,'train') for e in train]+[(e,'val') for e in val]
    if args.samples:tasks=[(e,'train') for e in train[:args.samples]]+[(e,'val') for e in val[:args.samples]]
    report=dict(case=args.case,passed=False,samples=args.samples,expected_training_windows=len(windows),
                excluded_training_windows=len(excluded),substeps=args.substeps,
                manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256(args.pretrained.read_bytes()).hexdigest(),
                code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},
                train_ids=[e['id'] for e in train],val_ids=[e['id'] for e in val],trajectories=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save();started=time.monotonic()
    with mp.get_context('spawn').Pool(args.workers,initializer=initialize,
            initargs=(str(args.pretrained),mean,scale,args.device,args.substeps)) as pool:
        for row in pool.imap_unordered(check_trajectory,tasks):
            report['trajectories'].append(row);report['seconds']=time.monotonic()-started;save()
            print(json.dumps(row),flush=True)
    report['completed']=len(report['trajectories'])==len(tasks)
    report['passed']=bool(not args.samples and report['completed'] and
        all(not r['failures'] and (r['gradient_checked'] if r['split']=='train' else bool(r['windows'] or r['excluded'])) for r in report['trajectories']) and
        any(r['windows'] for r in report['trajectories'] if r['split']=='val') and
        sum(r['windows'] for r in report['trajectories'] if r['split']=='train')==len(windows))
    save()
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
