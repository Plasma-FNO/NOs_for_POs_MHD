"""Reproduce the saved EM optimizer-state failure; retest the same weights/window."""
import sys,json,importlib.util,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from data import select
from native import *
from native_rollout import NativeRollout
from native_terms import NativeOperators
from train_native_rollout import load_trajectory
root=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-native-20261104')
torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('original_rollout',root/'em-gpu-full/code/native_rollout.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
m=json.loads((root/'em-gpu-full/manifest.json').read_text());train=select(m,'em','train',76,907)
values=torch.stack([load_trajectory(e,50)[0] for e in train])
valid=torch.isfinite(values).flatten(2).all(2)&(values[:,:,0]>0).flatten(2).all(2)
windows=[(t,s) for t in range(len(values)) for s in range(45) if valid[t,s:s+6].all()]
order=torch.randperm(len(windows),generator=torch.Generator().manual_seed(907)).tolist()
saved=torch.load(root/'em-gpu-full/rollout/last.pt',map_location='cpu',weights_only=False)
model=NativeOperators(**saved['normalization'],**saved['model_config']).cuda();model.load_state_dict(saved['state_dict']);model.eval()
for offset in range(saved['next_offset'],saved['next_offset']+20):
 ti,start=windows[order[offset]];entry=train[ti];q,r,z=load_trajectory(entry,1)
 c=common_coefficients(read_input(entry['input']),'em');c['viscosity']='old'
 ref=NativeReference(r,z,wall_trace(q,r,z,c),c)
 solver=old.NativeRollout(ref,q,model,device='cuda');x=values[ti,start:start+1].cuda()
 with torch.no_grad():
  try:
   for frame in range(5):x=solver.step(x,5.,8)
  except (FloatingPointError,RuntimeError) as error:
   row=dict(id=entry['id'],start=start,offset=offset,original_error=str(error),original_failed_frame=frame+1)
   print(json.dumps(row),flush=True)
   solver=NativeRollout(ref,q,model,device='cuda');x=values[ti,start:start+1].cuda();began=time.monotonic()
   for frame in range(5):x=solver.step(x,5.,8)
   row.update(fixed_completed=True,limiter=dict(solver.limiter_stats),seconds=time.monotonic()-began)
   Path(__file__).with_suffix('.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True);break
else:raise RuntimeError('Did not reproduce the original EM failure')
