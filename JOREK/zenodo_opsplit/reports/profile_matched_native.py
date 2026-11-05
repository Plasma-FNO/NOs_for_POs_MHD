"""Profile one unchanged matched window; measurements identify contention explicitly."""
from pathlib import Path
import sys,argparse,json,time,cProfile,pstats,io,gc
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from train_comparison import Experiment
from gpu_sparse import clear_cache

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',choices=['es','em'],default='em');p.add_argument('--arm',default='opsplit-pretrained');p.add_argument('--run');p.add_argument('--start',type=int,default=0);p.add_argument('--horizon',type=int,default=5);p.add_argument('--output',type=Path,required=True);p.add_argument('--operator-profile',action='store_true');p.add_argument('--no-step-checkpoint',action='store_true');p.add_argument('--compile-terms',action='store_true');a=p.parse_args()
 torch.set_num_threads(1);root=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-matched-20261104')/a.case
 ex=Experiment(root/'data',a.arm,'cuda',root/'terms/best.pt' if a.arm.endswith('pretrained') else None)
 i=next(i for i,e in enumerate(ex.meta['entries']['train']) if e['id']==a.run) if a.run else 0
 solver=ex.solver('train',i);solver.checkpoint_steps=not a.no_step_checkpoint;q=ex.values['train']['q'][i,a.start:a.start+1].cuda()
 if a.compile_terms:solver.ref._terms=torch.compile(solver.ref._terms,fullgraph=True)
 warm=ex.objective('train',i,a.start,a.horizon);warm.backward();del warm;ex.model.zero_grad(set_to_none=True)
 solver.reset_limiter_stats();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();now=time.monotonic();prof=cProfile.Profile();prof.enable()
 loss=ex.objective('train',i,a.start,a.horizon);torch.cuda.synchronize();forward=time.monotonic()-now;forward_memory=torch.cuda.memory_allocated();loss.backward();torch.cuda.synchronize();elapsed=time.monotonic()-now;prof.disable()
 text=io.StringIO();pstats.Stats(prof,stream=text).sort_stats('cumtime').print_stats(45);a.output.with_suffix('.profile.txt').write_text(text.getvalue())
 row=dict(case=a.case,arm=a.arm,id=ex.meta['entries']['train'][i]['id'],start=a.start,horizon=a.horizon,shape=list(q.shape),seconds=elapsed,forward_seconds=forward,forward_memory_bytes=forward_memory,peak_memory_bytes=torch.cuda.max_memory_allocated(),loss=float(loss.detach()),limiter=solver.limiter_stats,identity=ex.identity)
 a.output.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k!='identity'}),flush=True)
 del loss;ex.model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
 if a.operator_profile:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],profile_memory=True) as trace:
   result=solver._trial_step(q,.01);result.square().sum().backward();torch.cuda.synchronize()
  a.output.with_suffix('.operators.txt').write_text(trace.key_averages().table(sort_by='self_cuda_time_total',row_limit=45)+'\n'+trace.key_averages().table(sort_by='self_cuda_memory_usage',row_limit=25))
 clear_cache()
if __name__=='__main__':main()
