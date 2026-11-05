import sys,json,time
from pathlib import Path
sys.path.insert(0,'/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/zenodo_opsplit')
import torch
from train_comparison import Experiment
from gpu_sparse import clear_cache
from native_rollout import solve,mv
torch.set_num_threads(1)
root=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-matched-20261104');rows=[]
for case,run,start in [('es','jorek_run0472',39),('em','jorek_run063',21)]:
 for arm in ['opsplit-scratch','opsplit-pretrained']:
  ex=Experiment(root/case/'data',arm,'cuda',root/case/'terms/best.pt' if arm.endswith('pretrained') else None)
  i=next(i for i,e in enumerate(ex.meta['entries']['train']) if e['id']==run);now=time.monotonic()
  solver=ex.solver('train',i);original=solver._trial_step;limits=[0]
  def energy_limited(q,dt,substeps=1,correction_scale=1.):
   result=original(q,dt,substeps,correction_scale)
   c=solver.ref.c;old=q[0,2].flatten();new=result[0,2].flatten()
   ka=solver.stiffness_tensor((q[0,0].flatten()*solver.R**3).reshape(solver.shape))
   kn=solver.stiffness_tensor((result[0,0].flatten()*solver.R**3).reshape(solver.shape))
   terms=solver.ref.terms(q);extended=solver.ref.extend(q)
   pressure=(-2*solver.ref.r*solver.ref.op.grad(extended[:,1])[1])[:,1:-1,1:-1]
   force=pressure+terms['lorentz']+terms['conservative']
   forced=solve(ka+dt*c['mu']*solver.H+dt*c['mu_hyper']*solver.Hu,mv(ka,old)-dt*solver.V*force.flatten())
   ceiling=torch.maximum(.5*(old*mv(ka,old)).sum(),.5*(forced*mv(kn,forced)).sum())
   actual=.5*(new*mv(kn,new)).sum()
   if actual>ceiling:
    factor=(ceiling.clamp_min(0)/actual).sqrt() if ceiling>0 else actual*0
    result=torch.cat((result[:,:2],result[:,2:3]*factor,result[:,3:]),dim=1);limits[0]+=1
   return result
  solver._trial_step=energy_limited
  try:
   loss=ex.objective('train',i,start,5);loss.backward()
   grad=torch.nn.utils.clip_grad_norm_(ex.model.parameters(),1.,error_if_nonfinite=True)
   row=dict(case=case,arm=arm,id=run,start=start,passed=True,loss=float(loss.detach()),grad_norm=float(grad),seconds=time.monotonic()-now)
  except (FloatingPointError,RuntimeError) as error:
   row=dict(case=case,arm=arm,id=run,start=start,passed=False,error=str(error),seconds=time.monotonic()-now)
  row['limiter']=dict(ex.solver('train',i).limiter_stats);row['energy_limited']=limits[0];rows.append(row)
  Path('/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/zenodo_opsplit/reports/matched_energy_regressions.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(json.dumps(row),flush=True)
  ex.solver.cache_clear();del ex;clear_cache();torch.cuda.empty_cache()
