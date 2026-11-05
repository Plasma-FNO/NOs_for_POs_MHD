"""Bounded diagnostic; abort accelerated states instead of waiting for CFL collapse."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from train_comparison import Experiment
from gpu_sparse import clear_cache
from native_terms import NAMES

def main():
 torch.set_num_threads(1)
 root=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-matched-20261104');rows=[]
 for case,run,start in [('es','jorek_run0472',39),('em','jorek_run063',21)]:
  ex=Experiment(root/case/'data','opsplit-pretrained','cuda',root/case/'terms/best.pt')
  i=next(i for i,e in enumerate(ex.meta['entries']['train']) if e['id']==run)
  for modules in [(),('scalar_advection',),('fluid_momentum',),('particle_diffusion','thermal_diffusion'),('scalar_advection','fluid_momentum'),NAMES]:
   ex.solver.cache_clear();clear_cache();solver=ex.solver('train',i);solver.modules=set(modules)
   original=solver.stable_timestep;initial=ex.values['train']['q'][i,start:start+1].cuda();initial_h=original(initial,1.);trace=[]
   def controlled(q,maximum):
    h=original(q,maximum)
    if len(trace)%10==0:print(json.dumps(dict(case=case,modules=modules,substep=len(trace),rho=float(q[:,0].min()),p=float(q[:,1].max()),u=float(q[:,2].abs().max()),h=h)),flush=True)
    trace.append(dict(rho_min=float(q[:,0].min()),rho_max=float(q[:,0].max()),p_max=float(q[:,1].max()),u_max=float(q[:,2].abs().max()),h=h))
    if original(q,1.)<initial_h/100 or len(trace)>512:raise FloatingPointError('Diagnostic acceleration limit (not a production acceptance test)')
    return h
   solver.stable_timestep=controlled;now=time.monotonic()
   try:
    with torch.no_grad():loss=float(ex.objective('train',i,start,5))
    row=dict(case=case,modules=modules,passed=True,loss=loss)
   except (FloatingPointError,RuntimeError) as e:row=dict(case=case,modules=modules,passed=False,error=str(e))
   row.update(seconds=time.monotonic()-now,trace=trace);rows.append(row)
   Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n')
   print(json.dumps({k:v for k,v in row.items() if k!='trace'}),flush=True)
  ex.solver.cache_clear();del ex;clear_cache();torch.cuda.empty_cache()
if __name__=='__main__':main()
