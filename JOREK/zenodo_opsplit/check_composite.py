"""Check the corrected representation against every validation prefix at failed weights."""
import argparse
from pathlib import Path
import torch
from comparison import digest, write_json
from train_comparison import Experiment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--threads',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(a.threads)
    if a.output.exists():raise FileExistsError(a.output)
    ex=Experiment(a.data,'composite',a.device)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ex.model.load_state_dict(saved['state_dict']);ex.model.eval()
    report=dict(identity=ex.identity,checkpoint_sha256=digest(a.checkpoint),case=ex.meta['case'],
                positive_update=ex.model.config['positive_update'],horizon=ex.cfg['train_frames']-1,
                completed=False,passed=False,trajectories=[])
    with torch.no_grad():
        for index in ex.meta['val_indices']:
            row=dict(id=ex.meta['entries']['val'][index]['id'],failure=None)
            try:row['relative_l2']=float(ex.objective('val',index,0,report['horizon']))/report['horizon']
            except (FloatingPointError,RuntimeError) as error:row['failure']=str(error)
            report['trajectories'].append(row);write_json(a.output,report)
            print(row,flush=True)
    report['completed']=True
    report['passed']=len(report['trajectories'])==len(ex.meta['val_indices']) and not any(r['failure'] for r in report['trajectories'])
    if report['passed']:report['mean_relative_l2']=sum(r['relative_l2'] for r in report['trajectories'])/len(report['trajectories'])
    write_json(a.output,report)
    if not report['passed']:raise SystemExit(2)


if __name__=='__main__':main()
