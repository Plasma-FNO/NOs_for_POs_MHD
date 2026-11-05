"""Replay a failed immutable campaign and retain field/rate traces for every validation prefix."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--skip-replay',action='store_true',help='Inspect the last saved weights without completing the epoch')
    p.add_argument('--first-failure',action='store_true')
    a=p.parse_args()
    sys.path.insert(0,str(a.campaign/'code'))
    import torch
    from train_comparison import Experiment
    import parallel
    torch.set_num_threads(a.threads)
    setup=json.loads((a.campaign/'setup.json').read_text())
    ex=Experiment(Path(setup['data']),'composite',a.device)
    saved=torch.load(a.campaign/'train/last.pt',map_location='cpu',weights_only=False)
    ex.model.load_state_dict(saved['state_dict'])
    opt=torch.optim.Adam(ex.model.parameters(),lr=ex.cfg['lr'],weight_decay=ex.cfg['weight_decay'])
    opt.load_state_dict(saved['optimizer'])
    parallel.restore_rng(saved['rank_rng'][0],ex.device)
    epoch,offset=saved['next_epoch'],saved['next_offset']
    order=torch.randperm(len(ex.meta['windows']),generator=torch.Generator().manual_seed(ex.cfg['seed']+epoch)).tolist()
    ex.model.train()
    for start in ([] if a.skip_replay else range(offset,len(order),ex.cfg['batch_size'])):
        batch=[ex.meta['windows'][i] for i in order[start:start+ex.cfg['batch_size']]]
        opt.zero_grad(set_to_none=True);ex.batch_loss(batch)
        torch.nn.utils.clip_grad_norm_(ex.model.parameters(),ex.cfg['grad_clip'],error_if_nonfinite=True);opt.step()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.save(dict(state_dict=ex.model.state_dict(),epoch=epoch),a.output/'failing.pt')
    fields=['rho','T','potential','omega']+(['psi','j'] if ex.model.channels==6 else [])
    def ranges(x):
        return {name:dict(min=float(v.min()),max=float(v.max()),nonfinite=int((~v.isfinite()).sum()))
                for name,v in zip(fields,x[0].detach().double())}
    rates={}
    def hook(name):
        def record(module,args,output):rates[name]=ranges(output)
        return record
    ex.model.fluid.register_forward_hook(hook('fluid'))
    if ex.model.channels==6:ex.model.magnetic.register_forward_hook(hook('magnetic'))
    def diffusion_hook(module,args,output):
        rates['particle_before']=ranges(args[0][:,None])
        rates['particle_after']=ranges(output[:,None])
    ex.model.particle.register_forward_hook(diffusion_hook)
    ex.model.eval();rows=[]
    with torch.no_grad():
        for index in ex.meta['val_indices']:
            exported=ex.values['val']['exported'][index,:50].to(ex.device).double()
            x=(exported[:1]-ex.mean[None,:,None,None])/ex.scale[None,:,None,None]
            row=dict(id=ex.meta['entries']['val'][index]['id'],index=index,steps=[],failure=None)
            for step in range(1,len(exported)):
                record=dict(step=step,input=ranges(x*ex.scale[None,:,None,None]+ex.mean[None,:,None,None]))
                try:
                    x=ex.model(x)
                    record['output']=ranges(x*ex.scale[None,:,None,None]+ex.mean[None,:,None,None])
                except (FloatingPointError,RuntimeError) as error:row['failure']=str(error)
                record['rates']=rates.copy();row['steps'].append(record)
                if row['failure']:break
            rows.append(row)
            (a.output/'diagnosis.json').write_text(json.dumps(dict(case=setup['case'],epoch=epoch,replayed_from_offset=None if a.skip_replay else offset,device=a.device,trajectories=rows),indent=2))
            print(json.dumps(dict(id=row['id'],steps=len(row['steps']),failure=row['failure'])),flush=True)
            if row['failure'] and a.first_failure:break


if __name__=='__main__':main()
