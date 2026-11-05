import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from comparison import write_json
from train_comparison import Experiment

torch.set_num_threads(1)
root=Path('/pitagora_scratch/userexternal/vgopakum/jorek-composite-transfer-20261105/em')
ex=Experiment(root/'data','composite','cpu')
torch.set_num_threads(16)
assert not any(p.requires_grad for p in ex.model.fluid.parameters())
assert all(p.requires_grad for p in ex.model.magnetic.parameters())
assert torch.count_nonzero(ex.model.magnetic.projection.fcs[-1].weight)==0
assert torch.count_nonzero(ex.model.magnetic.projection.fcs[-1].bias)==0
started=time.monotonic()
report=dict(device='cpu',transfer=ex.transfer,identity=ex.identity,
            horizon=ex.cfg['train_frames']-1,trajectories=len(ex.meta['val_indices']),
            trainable_parameters=sum(p.numel() for p in ex.model.parameters() if p.requires_grad),
            frozen_parameters=sum(p.numel() for p in ex.model.parameters() if not p.requires_grad))
try:
    score,rows=ex.validation()
    report.update(passed=True,mean_relative_l2=score,rows=rows)
except (RuntimeError,FloatingPointError) as error:
    report.update(passed=False,error=str(error))
report['seconds']=time.monotonic()-started
write_json(root/'initial-validation-cpu.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ['identity','transfer','rows']},indent=2))
raise SystemExit(0 if report['passed'] else 2)
