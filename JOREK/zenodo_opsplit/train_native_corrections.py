"""Small ES rollout pretraining for bounded additive operator corrections.

The old checkpoint supplies architecture and train-only scales, never weights.
Trajectory loss learns discretization/model discrepancy. Zero-centered term
regularization limits corrections; it does not identify individual true terms.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from comparison import digest, write_json
from data import select
from native_terms import NativeOperators, features
from pilot_native_transfer import verify_source, load_cohort, make_solver, evaluate, score
from train_native_rollout import sequence_objective
from train_native_terms import term_loss
from train_es import range_statistics, save_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','source','source-data','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fraction',type=float,default=.1)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--trajectories',type=int,default=8)
    p.add_argument('--seed',type=int,default=907)
    p.add_argument('--substeps',type=int,default=8)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if min(args.epochs,args.trajectories,args.substeps)<1 or not 0<args.fraction<=1:
        p.error('Positive sizes and fraction in (0,1] required')
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    base=torch.load(args.source,map_location='cpu',weights_only=False)
    verify_source(base,args.source_data,args.manifest)
    manifest=json.loads(args.manifest.read_text())
    entries={s:select(manifest,'es',s,args.trajectories if s=='train' else 2,args.seed) for s in ('train','val')}
    if len(entries['train'])!=args.trajectories or len(entries['val'])!=2:
        raise ValueError('Insufficient source trajectories')
    cohorts={s:load_cohort(es,7 if s=='train' else 6) for s,es in entries.items()}
    config=dict(base['model_config'],correction_fraction=args.fraction)
    model=NativeOperators(**base['normalization'],**config).to(args.device)
    mean,scale=range_statistics(torch.stack([q for _,_,q in cohorts['train']]))
    mean,scale=mean.to(args.device),scale.to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    solvers=[make_solver(e,q,model,args.device) for e,_,q in cohorts['train']]
    windows=[(i,s) for i in range(args.trajectories) for s in (0,4)]
    args.output.mkdir(parents=True,exist_ok=False)
    provenance=dict(case='es',phase='residual_pretraining',source_weights_used=False,
        scale_checkpoint_sha256=digest(args.source),manifest_sha256=digest(args.manifest),
        train_ids=[e['id'] for e in entries['train']],val_ids=[e['id'] for e in entries['val']],
        windows=windows,epochs=args.epochs,horizon=2,seed=args.seed,substeps=args.substeps,
        fraction=args.fraction,model_config=config,lr=.001,weight_decay=1e-4,regularization_weight=1.,
        loss='Two-step primary-state relative L2 plus equal-operator normalized correction L2',
        validation='Five-step observed shared-field error; initial analytic model eligible; no EM or test selection',
        code_sha256={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')})
    write_json(args.output/'provenance.json',provenance)

    def payload(epoch):
        return dict(case='es',phase='residual_pretraining',state_dict=model.state_dict(),
            normalization=base['normalization'],model_config=config,epoch=epoch,
            dataset_sha256=base['dataset_sha256'],
            correction_training_ids=provenance['train_ids'],correction_validation_ids=provenance['val_ids'],
            provenance_sha256=digest(args.output/'provenance.json'))

    initial=evaluate(cohorts['val'],model,args.device,5,args.substeps)
    write_json(args.output/'validation-initial.json',initial)
    metric=score(initial,5)
    if metric is None:raise RuntimeError('Analytic ES validation failed')
    best=metric['shared_relative_l2'];best_epoch=-1
    save_checkpoint(args.output/'best.pt',payload(-1))
    started=time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        order=torch.randperm(len(windows),generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        for index in order:
            i,start=windows[index];solver=solvers[i];solver.reset_limiter_stats()
            q=cohorts['train'][i][2][start:start+3].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss=sequence_objective(solver,q,mean,scale,solver.ref.c['nominal_frame_dt'],args.substeps)
            correction=model(features(q[:1]).float())
            penalty=term_loss(correction,torch.zeros_like(correction))
            objective=loss+penalty
            if not torch.isfinite(objective):raise FloatingPointError('Nonfinite source correction objective')
            objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            row=dict(epoch=epoch,id=cohorts['train'][i][0]['id'],start=start,rollout_loss=float(loss.detach()),
                     correction_penalty=float(penalty.detach()),limiter=dict(solver.limiter_stats))
            with (args.output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        rows=evaluate(cohorts['val'],model,args.device,5,args.substeps)
        write_json(args.output/f'validation-{epoch}.json',rows)
        metric=score(rows,5)
        if metric is not None and metric['shared_relative_l2']<best:
            best,best_epoch=metric['shared_relative_l2'],epoch
            save_checkpoint(args.output/'best.pt',payload(epoch))
        save_checkpoint(args.output/'last.pt',payload(epoch))
        summary=dict(epoch=epoch,best_epoch=best_epoch,best_validation=best,
            initial_validation=score(initial,5)['shared_relative_l2'],seconds=time.monotonic()-started)
        write_json(args.output/'summary.json',summary)
        print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
