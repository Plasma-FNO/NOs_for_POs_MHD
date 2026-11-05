"""Trace the saved EM failure across the positive update and affine state encoding."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.campaign/'code'))
    import torch
    from train_comparison import Experiment
    torch.set_num_threads(1)
    setup=json.loads((a.campaign/'setup.json').read_text())
    ex=Experiment(Path(setup['data']),'composite','cuda')
    saved=torch.load(a.campaign/'train/failed-validation.pt',map_location='cpu',weights_only=False)
    ex.model.load_state_dict(saved['state_dict']);ex.model.eval()
    index=next(i for i in ex.meta['val_indices'] if ex.meta['entries']['val'][i]['id']=='jorek_run083')
    initial=ex.values['val']['exported'][index,:1].to(ex.device).double()
    state=(initial-ex.mean[None,:,None,None])/ex.scale[None,:,None,None]
    stages={};check=ex.model.check_state
    def traced(value,stage):
        t=value[:,1]
        stages[stage]=dict(min_native_T=float(t.min()),native_T_at_49_23=float(t[0,49,23]))
        check(value,stage)
    ex.model.check_state=traced
    report=dict(epoch=saved['next_epoch'],trajectory='jorek_run083',device='cuda',
                T_mean=float(ex.mean[1]),T_scale=float(ex.scale[1]),T_native_factor=float(ex.model.native_factor[1]),steps=[])
    with torch.no_grad():
        for step in range(1,50):
            stages.clear();row=dict(step=step)
            try:
                state=ex.model(state)
                decoded=state*ex.scale[None,:,None,None]+ex.mean[None,:,None,None]
                row['decoded_min_exported_T']=float(decoded[:,1].min())
                row['decoded_T_at_49_23']=float(decoded[0,1,49,23])
                row['normalized_T_at_49_23']=float(state[0,1,49,23])
            except (FloatingPointError,RuntimeError) as error:row['failure']=str(error)
            row['stages']=stages.copy();report['steps'].append(row)
            if 'failure' in row:break
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['steps'][-2:],indent=2),flush=True)


if __name__=='__main__':main()
