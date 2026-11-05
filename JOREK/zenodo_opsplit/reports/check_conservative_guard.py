"""Verify the continuity-defect correction on the EM matched initializations."""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from train_comparison import Experiment
from gpu_sparse import clear_cache

def main():
 torch.set_num_threads(1);root=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-matched-20261104/em');rows=[]
 for arm in ('opsplit-scratch','opsplit-pretrained'):
  ex=Experiment(root/'data',arm,'cuda',root/'terms/best.pt' if arm.endswith('pretrained') else None)
  index=next(i for i,e in enumerate(ex.meta['entries']['train']) if e['id']=='jorek_run063')
  for split,i,start,horizon,gradient in [('train',index,21,5,True),('val',ex.meta['val_indices'][0],0,49,False)]:
   now=time.monotonic();row=dict(arm=arm,split=split,id=ex.meta['entries'][split][i]['id'],start=start,horizon=horizon)
   try:
    with torch.set_grad_enabled(gradient):loss=ex.objective(split,i,start,horizon)
    if gradient:
     loss.backward();torch.nn.utils.clip_grad_norm_(ex.model.parameters(),1.,error_if_nonfinite=True)
     row['module_gradients']={name:sum(float(p.grad.abs().square().sum()) for p in block.parameters() if p.grad is not None)**.5 for name,block in ex.model.blocks.items()}
     assert all(v>0 for v in row['module_gradients'].values())
    row.update(passed=True,loss=float(loss.detach()))
   except (FloatingPointError,RuntimeError) as error:row.update(passed=False,error=str(error))
   row.update(seconds=time.monotonic()-now,limiter=dict(ex.solver(split,i).limiter_stats));rows.append(row)
   Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
  ex.solver.cache_clear();del ex;clear_cache();torch.cuda.empty_cache()
if __name__=='__main__':main()
