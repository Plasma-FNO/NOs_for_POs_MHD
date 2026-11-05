"""Four independent GPU checks: full matched failures and first 49-step prefix."""
from pathlib import Path
import sys,json,time,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from train_comparison import Experiment

def main():
 torch.set_num_threads(1);rank=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(rank)
 case=('es','es','em','em')[rank];arm=('opsplit-scratch','opsplit-pretrained')[rank%2]
 root=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-matched-20261104')/case
 ex=Experiment(root/'data',arm,'cuda',root/'terms/best.pt' if arm.endswith('pretrained') else None)
 runs=[('jorek_run0476',36),('jorek_run0432',44),('jorek_run0472',39)] if case=='es' else [('jorek_run063',21)]
 tasks=[('train',next(i for i,e in enumerate(ex.meta['entries']['train']) if e['id']==run),start,5,True) for run,start in runs]
 tasks.append(('val',ex.meta['val_indices'][0],0,49,False));rows=[]
 for split,i,start,horizon,backward in tasks:
  ex.model.zero_grad(set_to_none=True);now=time.monotonic();torch.cuda.reset_peak_memory_stats()
  row=dict(case=case,arm=arm,split=split,id=ex.meta['entries'][split][i]['id'],start=start,horizon=horizon)
  try:
   with torch.set_grad_enabled(backward):loss=ex.objective(split,i,start,horizon)
   if backward:
    loss.backward();torch.nn.utils.clip_grad_norm_(ex.model.parameters(),1.,error_if_nonfinite=True)
    row['module_gradient_norms']={name:sum(float(p.grad.abs().square().sum()) for p in block.parameters() if p.grad is not None)**.5 for name,block in ex.model.blocks.items()}
    assert all(v>0 for v in row['module_gradient_norms'].values())
   row.update(passed=True,loss=float(loss.detach()));del loss
  except (FloatingPointError,RuntimeError) as error:row.update(passed=False,error=str(error))
  row.update(seconds=time.monotonic()-now,peak_memory_bytes=torch.cuda.max_memory_allocated(),limiter=dict(ex.solver(split,i).limiter_stats));rows.append(row)
  Path(__file__).with_name(f'optimized_initializations_rank{rank}.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
if __name__=='__main__':main()
