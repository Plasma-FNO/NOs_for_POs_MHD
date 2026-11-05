"""Train separate native physical operators against equation terms before rollout tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from native_terms import NativeOperators, NAMES, SLICES
from train_es import save_checkpoint


def term_loss(pred, target):
    # Equal weight per operator; no cross-operator cancellation in the loss.
    diff=(pred-target)[:,:,4:-4,4:-4]
    return torch.stack([diff[:,s].square().mean() for s in SLICES]).mean()


@torch.no_grad()
def evaluate(model, loader, scale, device):
    model.eval();total=torch.zeros(4,device=device);count=0
    for x,y in loader:
        pred=model(x.to(device)); truth=y.to(device)/scale[None,:,None,None]
        err=(pred-truth)[:,:,4:-4,4:-4]
        total+=torch.stack([err[:,s].square().mean() for s in SLICES])*len(x);count+=len(x)
    return (total/count).cpu().tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=250);p.add_argument('--batch-size',type=int,default=20)
    p.add_argument('--width',type=int,default=64);p.add_argument('--modes',type=int,default=32)
    p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.width,args.modes)<1:p.error('Positive training settings required')
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA allocation required')
    torch.set_num_threads(1);torch.manual_seed(907)
    meta=json.loads((args.data/'manifest.json').read_text())
    norms=torch.load(args.data/'normalization.pt',weights_only=True)
    model=NativeOperators(**norms,width=args.width,modes=args.modes).to(args.device)
    loaders={}
    generator=torch.Generator().manual_seed(907)
    for split in ('train','val','test'):
        data=torch.load(args.data/f'{split}.pt',weights_only=False)
        if args.smoke:data={k:v[:40] for k,v in data.items()}
        loaders[split]=DataLoader(TensorDataset(data['x'],data['y']),batch_size=args.batch_size,shuffle=split=='train',
                                 generator=generator if split=='train' else None)
    provenance=dict(case=meta['case'],phase='term_pretraining',native_rollout_ready=False,data=str(args.data.resolve()),
                    dataset_sha256=hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),
                    parameters=sum(p.numel() for p in model.parameters()),model_config=model.config,
                    optimizer='Adam lr0.001 weight_decay0.0001 betas0.9/0.999',scheduler='StepLR50 gamma0.5',
                    loss='equal mean operator-normalized MSE; separate term targets; not five-step rollout loss',
                    smoke=args.smoke,epochs=1 if args.smoke else args.epochs,
                    code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,50,.5)
    scale=model.target_scale;best=float('inf');start_epoch=0;elapsed=0.
    if args.resume:
        if json.loads((args.output/'provenance.json').read_text())!=provenance:raise ValueError('Resume requires unchanged code/data/protocol')
        saved=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        generator.set_state(saved['sampler_rng']);torch.set_rng_state(saved['torch_rng'])
        if args.device.startswith('cuda'):torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_epoch,best,elapsed=saved['epoch']+1,saved['best'],saved['elapsed']
        history=args.output/'history.jsonl'
        history.write_text(''.join(json.dumps(row)+'\n' for row in map(json.loads,history.read_text().splitlines()) if row['epoch']<start_epoch))
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    started=time.monotonic()
    for epoch in range(start_epoch,provenance['epochs']):
        model.train();total,count=0.,0
        for i,(x,y) in enumerate(loaders['train']):
            if args.smoke and i==2:break
            optimizer.zero_grad(set_to_none=True)
            loss=term_loss(model(x.to(args.device)),y.to(args.device)/scale[None,:,None,None])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite term loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total+=loss.item()*len(x);count+=len(x)
        metrics=evaluate(model,loaders['val'],scale,args.device)
        if not all(torch.isfinite(torch.tensor(metrics))):raise FloatingPointError('Nonfinite validation')
        score=sum(metrics)/4
        row=dict(epoch=epoch,train_loss=total/count,val_terms=dict(zip(NAMES,metrics)),lr=optimizer.param_groups[0]['lr'],seconds=elapsed+time.monotonic()-started)
        with (args.output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True);scheduler.step()
        payload=dict(state_dict=model.state_dict(),normalization=norms,model_config=model.config,epoch=epoch,
                     case=meta['case'],phase='term_pretraining',native_rollout_ready=False,dataset_sha256=provenance['dataset_sha256'])
        if score<best:best=score;save_checkpoint(args.output/'best.pt',payload)
        payload.update(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),best=best,elapsed=row['seconds'],
                       sampler_rng=generator.get_state(),torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [])
        save_checkpoint(args.output/'last.pt',payload)
    saved=torch.load(args.output/'best.pt',map_location=args.device,weights_only=False)
    model.load_state_dict(saved['state_dict'])
    metrics=dict(zip(NAMES,evaluate(model,loaders['test'],scale,args.device)))
    (args.output/'test_metrics.json').write_text(json.dumps(dict(phase='term_pretraining',normalized_term_mse=metrics),indent=2)+'\n')


if __name__=='__main__':main()
