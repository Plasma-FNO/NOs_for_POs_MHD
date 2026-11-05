import sys,time,json
from pathlib import Path
sys.path.insert(0,'/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/zenodo_opsplit')
import torch
from comparison import reference_for,decoded_fields,normalize,load_fields
from train_es import relative_lp
from native_rollout import NativeRollout
from native_terms import NativeOperators
from gpu_sparse import clear_cache
torch.set_num_threads(1)
root=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs');report=[]
for case,run,start in [('es','jorek_run0472',39),('em','jorek_run063',21)]:
 meta=json.loads((root/f'jorek-matched-20261104/{case}/data/comparison.json').read_text())
 e=next(e for e in meta['entries']['train'] if e['id']==run)
 raw,q=load_fields(e,50)
 data=torch.load(root/f'jorek-matched-20261104/{case}/data/data.pt',weights_only=False)
 mean,scale=data['mean'].cuda(),data['scale'].cuda();del data
 saved=torch.load(root/f'jorek-native-20261104/{case}-term-training/best.pt',weights_only=False,map_location='cpu')
 model=NativeOperators(**saved['normalization'],**saved['model_config']).cuda();model.load_state_dict(saved['state_dict']);model.train()
 solver=NativeRollout(reference_for(e,q[:1]),q[:1],model,modules=('scalar_advection','fluid_momentum'),device='cuda')
 now=time.monotonic();x=q[start:start+1].cuda();loss=x.new_zeros(())
 try:
  for frame in range(5):
   x=solver.step(x,solver.ref.c['nominal_frame_dt'],8)
   loss=loss+relative_lp(normalize(decoded_fields(solver.ref,x),mean,scale),normalize(raw[start+frame+1:start+frame+2].cuda(),mean,scale))
   print(json.dumps(dict(case=case,id=run,frame=frame+1,limiter=solver.limiter_stats)),flush=True)
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
  row=dict(case=case,id=run,start=start,passed=True,loss=float(loss.detach()),grad_norm=float(grad),seconds=time.monotonic()-now,limiter=dict(solver.limiter_stats))
 except (FloatingPointError,RuntimeError) as error:
  row=dict(case=case,id=run,start=start,passed=False,error=str(error),seconds=time.monotonic()-now,limiter=dict(solver.limiter_stats))
 report.append(row);print(json.dumps(row),flush=True)
 Path('/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/zenodo_opsplit/reports/matched_analytic_diffusion_gpu.json').write_text(json.dumps(report,indent=2)+'\n')
 del x,loss,solver,model,saved;clear_cache();torch.cuda.empty_cache()
