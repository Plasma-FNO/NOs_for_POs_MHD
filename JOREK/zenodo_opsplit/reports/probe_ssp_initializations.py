import sys,json,time
from pathlib import Path
sys.path.insert(0,'/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/zenodo_opsplit')
import torch
from train_comparison import Experiment
from gpu_sparse import clear_cache
torch.set_num_threads(1)
root=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-matched-20261104');rows=[]
for case,run,start in [('es','jorek_run0472',39),('em','jorek_run063',21)]:
 for arm in ['opsplit-scratch','opsplit-pretrained']:
  ex=Experiment(root/case/'data',arm,'cuda',root/case/'terms/best.pt' if arm.endswith('pretrained') else None)
  i=next(i for i,e in enumerate(ex.meta['entries']['train']) if e['id']==run);now=time.monotonic()
  solver=ex.solver('train',i);first_order=solver._trial_step
  def ssp(q,dt,substeps=1,correction_scale=1.):
   a=first_order(q,dt,substeps,correction_scale)
   b=.75*q+.25*first_order(a,dt,substeps,correction_scale)
   return q/3+2/3*first_order(b,dt,substeps,correction_scale)
  solver._trial_step=ssp
  try:
   loss=ex.objective('train',i,start,5);loss.backward()
   grad=torch.nn.utils.clip_grad_norm_(ex.model.parameters(),1.,error_if_nonfinite=True)
   row=dict(case=case,arm=arm,id=run,start=start,passed=True,loss=float(loss.detach()),grad_norm=float(grad),seconds=time.monotonic()-now)
  except (FloatingPointError,RuntimeError) as error:
   row=dict(case=case,arm=arm,id=run,start=start,passed=False,error=str(error),seconds=time.monotonic()-now)
  row['limiter']=dict(ex.solver('train',i).limiter_stats);rows.append(row)
  Path('/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/zenodo_opsplit/reports/matched_ssp_regressions.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(json.dumps(row),flush=True)
  ex.solver.cache_clear();del ex;clear_cache();torch.cuda.empty_cache()
