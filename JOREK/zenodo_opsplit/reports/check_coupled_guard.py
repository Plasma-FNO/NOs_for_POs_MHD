"""Targeted GPU gradients and full validation prefixes; not an exhaustive audit."""
from pathlib import Path
import sys,json,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from train_comparison import Experiment
from gpu_sparse import clear_cache
ROOT=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-matched-20261104')

def check(task):
 case,arm=task;torch.set_num_threads(1);rows=[]
 ex=Experiment(ROOT/case/'data',arm,'cuda',ROOT/case/'terms/best.pt' if arm.endswith('pretrained') else None)
 run,start=('jorek_run0472',39) if case=='es' else ('jorek_run063',21)
 index=next(i for i,e in enumerate(ex.meta['entries']['train']) if e['id']==run)
 for split,i,s,h,gradient in [('train',index,start,5,True),('val',ex.meta['val_indices'][0],0,49,False)]:
  began=time.monotonic();solver=ex.solver(split,i);original=solver.stable_timestep
  def bounded(q,maximum):
   h=original(q,maximum)
   if h<1e-6 or solver.limiter_stats['substeps']>2048:raise FloatingPointError('Diagnostic work limit; no production acceptance')
   return h
  solver.stable_timestep=bounded
  row=dict(case=case,arm=arm,split=split,id=ex.meta['entries'][split][i]['id'],start=s,horizon=h)
  try:
   with torch.set_grad_enabled(gradient):loss=ex.objective(split,i,s,h)
   if gradient:
    loss.backward();torch.nn.utils.clip_grad_norm_(ex.model.parameters(),1.,error_if_nonfinite=True)
    row['module_gradient_norms']={name:sum(float(p.grad.abs().square().sum()) for p in block.parameters() if p.grad is not None)**.5 for name,block in ex.model.blocks.items()}
    if not all(v>0 for v in row['module_gradient_norms'].values()):raise RuntimeError('Missing learned-module gradient')
   row.update(passed=True,loss=float(loss.detach()))
  except (FloatingPointError,RuntimeError) as e:row.update(passed=False,error=str(e))
  row.update(seconds=time.monotonic()-began,limiter=dict(solver.limiter_stats));rows.append(row)
  path=Path(__file__).with_name(f'coupled_guard_{case}_{arm}.json');path.write_text(json.dumps(rows,indent=2)+'\n')
  print(json.dumps(row),flush=True);solver.stable_timestep=original
 ex.solver.cache_clear();clear_cache()
 return rows
if __name__=='__main__':
 tasks=[(c,a) for c in ('es','em') for a in ('opsplit-scratch','opsplit-pretrained')]
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  rows=[]
  for future in as_completed([pool.submit(check,t) for t in tasks]):rows.extend(future.result())
 Path(__file__).with_name('coupled_guard_targeted.json').write_text(json.dumps(rows,indent=2)+'\n')
 raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
