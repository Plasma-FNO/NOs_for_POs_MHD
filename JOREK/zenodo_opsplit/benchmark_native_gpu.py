"""Compare complete five-frame training-window values/gradients and wall time."""
import argparse,json,time
from pathlib import Path
import torch
from data import select
from native import read_input,common_coefficients,NativeReference,wall_trace
from native_rollout import NativeRollout
from native_terms import NativeOperators,features,labels
from train_native_rollout import load_trajectory,sequence_objective
from train_es import range_statistics
from train_native_terms import term_loss
from gpu_sparse import clear_cache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(1)
    manifest=json.loads(args.manifest.read_text());report=dict(device=torch.cuda.get_device_name(),cases={})
    for case in ('es','em'):
        entry=select(manifest,case,'train',1,907)[0]
        q,r,z=load_trajectory(entry,50)
        mean,scale=range_statistics(q[None])
        c=common_coefficients(read_input(entry['input']),case);c['viscosity']='old'
        reference=NativeReference(r,z,wall_trace(q[:1],r,z,c),c)
        saved=torch.load(args.runs/f'{case}-term-training/best.pt',map_location='cpu',weights_only=False)
        model=NativeOperators(**saved['normalization'],**saved['model_config']).cuda()
        model.load_state_dict(saved['state_dict']);model.train()
        rows=[];baseline_gradient=None
        for device in ('cpu','cuda','cuda'):
            solver=NativeRollout(reference,q[:1],model,device=device)
            frames=q[:6].to(device);mm,ss=mean.to(device),scale.to(device)
            model.zero_grad(set_to_none=True);torch.cuda.synchronize();started=time.monotonic()
            value=sequence_objective(solver,frames,mm,ss,c['nominal_frame_dt'],8)
            with torch.no_grad():target=labels(solver.ref,frames[:1]).cuda().float()/model.target_scale[None,:,None,None]
            anchor=term_loss(model(features(frames[:1]).cuda().float()),target)
            loss=value+anchor.to(device);loss.backward();torch.cuda.synchronize()
            elapsed=time.monotonic()-started
            gradient=torch.cat([p.grad.flatten().cpu() for p in model.parameters() if p.grad is not None])
            if baseline_gradient is None:baseline_gradient=gradient
            error=float((gradient-baseline_gradient).norm()/baseline_gradient.norm())
            row=dict(elliptic_device=device,seconds=elapsed,objective=float(loss.detach()),relative_gradient_error=error,limiter=dict(solver.limiter_stats))
            if device=='cuda' and (error>2e-3 or abs(row['objective']-rows[0]['objective'])>1e-5):raise RuntimeError(f'CPU/GPU training mismatch: {row}')
            rows.append(row);print(case,json.dumps(row),flush=True)
            del loss,value,anchor,solver
        report['cases'][case]=dict(id=entry['id'],measurements=rows,speedup=rows[0]['seconds']/rows[-1]['seconds'])
        args.output.write_text(json.dumps(report,indent=2)+'\n');clear_cache()


if __name__=='__main__':main()
