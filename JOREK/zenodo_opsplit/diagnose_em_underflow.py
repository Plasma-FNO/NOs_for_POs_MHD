"""Read-only replay of the physical-state EM campaign, including softplus underflow."""
import argparse
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--steps',type=int,default=49)
    p.add_argument('--checkpoint',choices=['best.pt','last.pt','failed-validation.pt'])
    p.add_argument('--trajectory')
    a=p.parse_args()
    sys.path.insert(0,str(a.campaign/'code'))
    import torch
    import composite
    from comparison import digest, normalize, write_json
    from train_comparison import Experiment
    from train_es import relative_lp
    # Match the campaign's single-threaded eigenbasis construction before loading weights.
    torch.set_num_threads(1)
    a.output.mkdir(parents=True,exist_ok=False)
    setup=json.loads((a.campaign/'setup.json').read_text())
    ex=Experiment(Path(setup['data']),'composite',a.device)
    torch.set_num_threads(a.threads)
    assert ex.meta['case']=='em' and ex.model.config['state_representation']=='exported-float64'
    names=['rho','T','potential','omega','psi','j']
    def scalar(value):
        number=float(value)
        return number if math.isfinite(number) else str(number)
    def ranges(value):
        return {name:dict(min=scalar(field.min()),max=scalar(field.max())) for name,field in zip(names,value[0])}
    original=composite.positive_increment
    detail={};calls=[]
    def traced(value,rate,reference):
        result=original(value,rate,reference)
        field=('rho','T')[len(calls)%2]
        coordinate=value/reference+torch.log(-torch.expm1(-value/reference))
        after=coordinate+rate
        location=tuple(int(i) for i in torch.unravel_index(after.argmin(),after.shape))
        before=float(value[location]);r=float(rate[location]);ref=float(reference);z=float(after[location])
        record=dict(field=field,input_min=scalar(value.min()),output_min=scalar(result.min()),
                    zeros=int((result==0).sum()),rate_min=scalar(rate.min()),rate_max=scalar(rate.max()),
                    worst_location_batch_Z_R=location,input_at_worst=before,rate_at_worst=r,
                    reference=ref,coordinate_before=scalar(coordinate[location]),coordinate_after=z)
        if z < -30:
            # At this magnitude log(softplus(z)) differs from z by less than 5e-14.
            record['log10_positive_native_value']=(math.log(ref)+z)/math.log(10)
        if record['zeros']:
            cpu=original(torch.tensor(before,dtype=torch.float64),torch.tensor(r,dtype=torch.float64),torch.tensor(ref,dtype=torch.float64))
            record['cpu_float64_result']=scalar(cpu)
            with localcontext() as context:
                context.prec=80
                v,rate_d,ref_d=map(Decimal,map(str,(before,r,ref)))
                scaled=v/ref_d
                # expm1 would cancel at extremely small scaled values; log(expm1(x))~log(x) there.
                coord=scaled.ln() if scaled < Decimal('1e-60') else (scaled.exp()-1).ln()
                out_coord=coord+rate_d
                result_d=ref_d*out_coord.exp() if out_coord < -150 else ref_d*(1+out_coord.exp()).ln()
                record['decimal80_positive_native_value']=str(result_d)
                record['below_smallest_float64']=result_d<Decimal(2)**-1074
        calls.append(record)
        return result
    original_check=ex.model.check_state
    def check(value,stage):
        detail[stage]=ranges(value)
        original_check(value,stage)
    rates={}
    def capture(name):
        def hook(module,args,output):rates[name]=output.detach()
        return hook
    handles=[]
    summary=dict(campaign=str(a.campaign),device=a.device,requested_steps=a.steps,smallest_positive_float64=float.fromhex('0x0.0000000000001p-1022'),checkpoints={})
    for filename in ([a.checkpoint] if a.checkpoint else ('best.pt','last.pt','failed-validation.pt')):
        checkpoint=a.campaign/'train'/filename
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        # Fresh initialization may differ across CPU hosts; every saved tensor is restored below.
        expected=dict(ex.identity,device=saved['identity']['device'],initialization_sha256=saved['identity']['initialization_sha256'])
        if saved['identity']!=expected:
            write_json(a.output/'identity-mismatch.json',dict(saved=saved['identity'],reconstructed=expected))
            raise ValueError('Use the exact frozen campaign code/data for replay')
        assert saved['model_config']==ex.model.config
        ex.model.load_state_dict(saved['state_dict']);ex.model.eval()
        for name,value in ex.model.state_dict().items():
            if isinstance(value,torch.Tensor):assert torch.equal(value.cpu(),saved['state_dict'][name].cpu()), name
        tracing=filename=='failed-validation.pt'
        if tracing:
            composite.positive_increment=traced;ex.model.check_state=check
            handles=[ex.model.fluid.register_forward_hook(capture('fluid')),ex.model.magnetic.register_forward_hook(capture('magnetic'))]
        rows=[]
        with torch.no_grad():
            for index in ex.meta['val_indices']:
                if a.trajectory and ex.meta['entries']['val'][index]['id']!=a.trajectory:continue
                exported=ex.values['val']['exported'][index,:ex.cfg['train_frames']].to(ex.device).double()
                state=exported[:1];loss=0.
                row=dict(id=ex.meta['entries']['val'][index]['id'],completed_steps=0,failure=None)
                trace=[]
                for step in range(1,a.steps+1):
                    detail.clear();calls.clear();record=dict(step=step)
                    if tracing:
                        record['input_exported']=ranges(state)
                        record['normalized_input_absmax']={n:scalar(v.abs().max()) for n,v in zip(names,normalize(state,ex.mean,ex.scale)[0])}
                        if step<len(exported):record['source_exported']=ranges(exported[step:step+1])
                    try:
                        state=ex.model(state)
                        if step<len(exported):loss+=float(relative_lp(normalize(state,ex.mean,ex.scale),normalize(exported[step:step+1],ex.mean,ex.scale)))
                        row['completed_steps']=step
                    except (FloatingPointError,RuntimeError) as error:row['failure']=str(error)
                    if tracing:
                        record['stages']=detail.copy();record['positive_updates']=calls.copy()
                        if calls and calls[-1]['field']=='T':
                            location=tuple(calls[-1]['worst_location_batch_Z_R']);b,z,r=location
                            record['T_rate_components_at_worst']={n:scalar(v[b,1,z,r]) for n,v in rates.items()}
                        trace.append(record)
                    if row['failure']:break
                if row['completed_steps']>=min(a.steps,len(exported)-1):row['relative_l2']=loss/min(a.steps,len(exported)-1)
                if tracing:
                    write_json(a.output/(row['id']+'.json'),dict(**row,steps=trace))
                rows.append(row);print(json.dumps(dict(checkpoint=filename,**row)),flush=True)
        if tracing:
            composite.positive_increment=original;ex.model.check_state=original_check
            for handle in handles:handle.remove()
        summary['checkpoints'][filename]=dict(sha256=digest(checkpoint),next_epoch=saved['next_epoch'],next_offset=saved['next_offset'],
                    trajectories=rows,failed=sum(r['failure'] is not None for r in rows),
                    mean_relative_l2=None if any(r['failure'] for r in rows) else sum(r['relative_l2'] for r in rows)/len(rows))
        write_json(a.output/'summary.json',summary)


if __name__=='__main__':main()
