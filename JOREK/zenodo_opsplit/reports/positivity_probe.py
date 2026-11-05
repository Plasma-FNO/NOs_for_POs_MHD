import sys,json,time
sys.path.insert(0,'/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/zenodo_opsplit')
from pathlib import Path
import torch
from data import select
from native import *
from native_rollout import NativeRollout
from native_terms import NativeOperators
from train_native_rollout import load_trajectory

torch.set_num_threads(1);root=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-native-20261104')
case,ident,start,device=sys.argv[1:];start=int(start)
m=json.loads((root/f'{case}-gpu-full/manifest.json').read_text())
e=next(e for e in select(m,case,'train',0,907) if e['id']==ident)
q,r,z=load_trajectory(e,50);c=common_coefficients(read_input(e['input']),case);c['viscosity']='old'
ref=NativeReference(r,z,wall_trace(q[:1],r,z,c),c)
saved=torch.load(root/f'{case}-term-training/best.pt',map_location='cpu',weights_only=False)
model=NativeOperators(**saved['normalization'],**saved['model_config']).to(device);model.load_state_dict(saved['state_dict']);model.eval()
for operators in [None,model]:
 solver=NativeRollout(ref,q[:1],operators,device=device);x=q[start:start+1].to(device);t=time.monotonic()
 with torch.no_grad():
  for frame in range(5):
   x=solver.step(x,5.,8)
   print(case,ident,start,'learned',operators is not None,'frame',frame+1,'min rho p',x[:,:2].flatten(2).min(2).values.tolist(),solver.limiter_stats,flush=True)
 print('PASS seconds',time.monotonic()-t,flush=True)
