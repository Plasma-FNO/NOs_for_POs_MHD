"""Forward/backward and optimizer updates on the two originally failing windows."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from data import select
from native import *
from native_terms import NativeOperators,features,labels
from native_rollout import NativeRollout
from train_native_rollout import load_trajectory,sequence_objective
from train_es import range_statistics
from train_native_terms import term_loss

torch.set_num_threads(1);root=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-native-20261104');rows=[]
for case,ident,start in [('es','jorek_run0497',42),('em','jorek_run017',35),('em','jorek_run016',28)]:
 m=json.loads((root/f'{case}-gpu-full/manifest.json').read_text())
 e=next(e for e in select(m,case,'train',0,907) if e['id']==ident)
 q,r,z=load_trajectory(e,50);mean,scale=range_statistics(q[None]);mean,scale=mean.cuda(),scale.cuda()
 c=common_coefficients(read_input(e['input']),case);c['viscosity']='old'
 ref=NativeReference(r,z,wall_trace(q[:1],r,z,c),c)
 path=root/f'{case}-term-training/best.pt'
 if ident=='jorek_run017':path=root/'em-gpu-full/rollout/last.pt'
 saved=torch.load(path,map_location='cpu',weights_only=False)
 model=NativeOperators(**saved['normalization'],**saved['model_config']).cuda();model.load_state_dict(saved['state_dict']);model.train()
 optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
 solver=NativeRollout(ref,q[:1],model,device='cuda');frames=q[start:start+6].cuda()
 for update in range(2):
  began=time.monotonic();model.zero_grad(set_to_none=True);solver.reset_limiter_stats()
  loss=sequence_objective(solver,frames,mean,scale,5.,8)
  with torch.no_grad():target=labels(solver.ref,frames[:1]).float()/model.target_scale[None,:,None,None]
  loss=loss+term_loss(model(features(frames[:1]).float()),target);loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
  for block in model.blocks.values():
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
  optimizer.step()
  row=dict(case=case,id=ident,start=start,update=update,loss=float(loss.detach()),seconds=time.monotonic()-began,limiter=dict(solver.limiter_stats))
  rows.append(row);print(json.dumps(row),flush=True)
 del loss,optimizer,solver,model
Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n')
