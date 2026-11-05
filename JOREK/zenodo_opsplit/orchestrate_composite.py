"""Snapshot and launch composite ES/EM training against existing matched data.

Independent optimizer smoke and exhaustive initialization audit gate each full
run. Existing native/AR/NODE campaigns and data are never rewritten.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from comparison import write_json, digest
from orchestrate import PYTHON


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--python',default=PYTHON)
    p.add_argument('--submit',action='store_true')
    p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args();data=a.data.resolve()
    meta=json.loads((data/'comparison.json').read_text());case=meta['case']
    if digest(data/'data.pt')!=meta['data_sha256']:raise ValueError('Changed shared data cache')
    stability_epochs=meta['protocol'].get('composite_stability_epochs',26 if case=='em' else 12)
    if not 1<=stability_epochs<=meta['protocol']['epochs']:raise ValueError('Invalid composite stability epoch budget')
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    code=root/'code';code.mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,code/path.name)
    write_json(root/'setup.json',dict(case=case,data=str(data),comparison_sha256=digest(data/'comparison.json'),
               protocol=meta['protocol'],formulation='composite-imex-auxiliary-defect',
               positive_update='Softplus rho/T coordinates: logarithmic near zero, additive for large fields; shared train-only scales, no clipping.',
               state_representation='Exported physical fields in float64; center only network inputs and loss, never the carried state.',
               stability_epochs=stability_epochs,normalization=meta['normalization'],transfer=meta['protocol'].get('composite_transfer'),
               linear='Nominal dt5 backward Euler for particle diffusion and EM resistivity; separable batched interior solve.',
               boundary='Outer four sampled cells evolve through neural rates; no physical wall clamp.',
               auxiliaries='Fourth-order interior constraints plus learned full-grid defects; defects are not physical PDE terms.',
               training=meta['protocol'].get('notes',{}).get('transfer','Trajectory loss only; no term pretraining or anchors. Shared ES FNO6 layers; EM adds a two-layer correction.'),
               limitations=['Implicit linear update, softplus rho/T increments and auxiliary defects adapt the NS Euler protocol.',
                            'Stable linear branch does not certify nonlinear stability or positivity.',
                            'Existing six-layer EM baselines do not match this larger EM capacity.',
                            'Initial scratch training only; no claim of ES-to-EM transfer.']))
    base=[a.python,str(code/'train_comparison.py'),'--data',str(data),'--arm','composite']
    stages=[('smoke',base+['--output',str(root/'smoke'),'--smoke'],None,root/'smoke/last.pt','01:00:00')]
    if not a.smoke_only:
        stages += [('audit',base+['--output',str(root/'audit.json'),'--audit','--max-seconds','79200'],'smoke',root/'audit.json','24:00:00'),
                   ('probe',base+['--output',str(root/'train'),'--audit-report',str(root/'audit.json'),'--stop-after-epoch',str(stability_epochs),'--train-only','--max-seconds','79200'],'audit',root/'train/last.pt','24:00:00'),
                   ('train',base+['--output',str(root/'train'),'--audit-report',str(root/'audit.json'),'--max-seconds','79200'],'probe',root/'train/last.pt','24:00:00')]
    jobs=[]
    for name,command,dependency,checkpoint,walltime in stages:
        script=root/(name+'.sbatch')
        gate=('test -f '+shlex.quote(str(root/'train/stability.json'))+'\n') if name=='train' else ''
        script.write_text('#!/bin/bash\nset -euo pipefail\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'+gate+'set +e\n'
                          +'if [ -f '+shlex.quote(str(checkpoint))+' ]; then\n'+shlex.join(command+['--resume'])+'\nelse\n'+shlex.join(command)+'\nfi\n'
                          +'status=$?\nset -e\nif [ "$status" -eq 75 ]; then\n  scontrol requeue "$SLURM_JOB_ID"\n  exit 0\nfi\nexit "$status"\n')
        request=['sbatch','--parsable','--account=fupb2_ukaea_ml','--partition=boost_fua_prod','--nodes=1','--ntasks=1',
                 '--cpus-per-task=4','--mem=32G','--gres=gpu:1','--time='+walltime,'--requeue','--open-mode=append',
                 '--job-name=jorek-composite-'+case+'-'+name,'--output='+str(root/(name+'-%j.log'))]
        jobs.append(dict(stage=name,dependency=dependency,job_id=None,request=request,script=str(script)))
    write_json(root/'jobs.json',jobs)
    if a.submit:
        ids={}
        for job in jobs:
            request=job['request'].copy()
            if job['dependency']:request+=['--dependency=afterok:'+ids[job['dependency']],'--kill-on-invalid-dep=yes']
            request.append(job['script']);job['submission']=request
            job['job_id']=subprocess.check_output(request,text=True).strip().split(';')[0]
            ids[job['stage']]=job['job_id'];write_json(root/'jobs.json',jobs)
    print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
