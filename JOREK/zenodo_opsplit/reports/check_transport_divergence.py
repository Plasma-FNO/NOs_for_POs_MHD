from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from comparison import reference_for
from native_rollout import NativeRollout,mv
torch.set_num_threads(1)
root=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-matched-20261104/es/data')
meta=json.loads((root/'comparison.json').read_text());data=torch.load(root/'data.pt',weights_only=False);rows=[]
for name,start in [('jorek_run0472',39),('jorek_run0432',36),('jorek_run0494',25),('jorek_run0476',31),('jorek_run0234',37)]:
 i=next(i for i,e in enumerate(meta['entries']['train']) if e['id']==name);entry=meta['entries']['train'][i]
 q0=data['train']['q'][i,:1];q=data['train']['q'][i,start:start+1]
 solver=NativeRollout(reference_for(entry,q0),q0);extended=solver.ref.extend(q)
 analytic=-2*solver.ref.op.grad(extended[:,2])[1][0,1:-1,1:-1]
 actual=(mv(solver.upwind_transport(extended),torch.ones(solver.n,dtype=torch.float64))/(solver.V*solver.R)).reshape(solver.shape)
 error=actual-analytic
 row=dict(id=name,start=start,analytic_min=float(analytic.min()),analytic_max=float(analytic.max()),discrete_min=float(actual.min()),discrete_max=float(actual.max()),max_error=float(error.abs().max()),interior_error=float(error[4:-4,4:-4].abs().max()),h=solver.stable_timestep(q,.625))
 rows.append(row);print(json.dumps(row),flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n')
