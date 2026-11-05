from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from train_comparison import Experiment
from native_rollout import mv
from gpu_sparse import clear_cache
torch.set_num_threads(1);root=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-matched-20261104/es');rows=[]
ex=Experiment(root/'data','opsplit-pretrained','cuda',root/'terms/best.pt')
for run,start in [('jorek_run0476',36),('jorek_run0432',44)]:
 i=next(i for i,e in enumerate(ex.meta['entries']['train']) if e['id']==run);solver=ex.solver('train',i);original=solver.stable_timestep;trace=[];began=time.monotonic()
 def monitor(q,maximum):
  h=original(q,maximum);n=solver.limiter_stats['substeps']
  if n%50==0:
   x=solver.ref.extend(q);div=-2*solver.ref.op.grad(x[:,2])[1][0,1:-1,1:-1]
   raw=mv(solver.upwind_transport(x),torch.ones(solver.n,device='cuda',dtype=torch.float64))/(solver.V*solver.R)
   row=dict(id=run,substeps=n,rho_min=float(q[:,0].min()),rho_max=float(q[:,0].max()),p_max=float(q[:,1].max()),p_min=float(q[:,1].min()),T_max=float((q[:,1]/q[:,0]).max()),u_max=float(q[:,2].abs().max()),physical_div_max=float(div.max()),upwind_div_max=float(raw.max()),h=h)
   trace.append(row);print(json.dumps(row),flush=True)
  if n>512 or h<1e-5:raise FloatingPointError('Diagnostic work limit; not a full-window certificate')
  return h
 solver.stable_timestep=monitor;row=dict(id=run,start=start)
 try:
  with torch.no_grad():loss=ex.objective('train',i,start,5)
  row.update(passed=True,loss=float(loss))
 except (FloatingPointError,RuntimeError) as error:row.update(passed=False,error=str(error))
 row.update(seconds=time.monotonic()-began,trace=trace,limiter=dict(solver.limiter_stats));rows.append(row)
 Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k!='trace'}),flush=True)
 solver.stable_timestep=original
ex.solver.cache_clear();clear_cache()
