"""Create and optionally submit isolated three-field ES campaigns with dependent checks."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import xml.etree.ElementTree as ET

PYTHON='/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python'
ARMS=('ar','node','opsplit','opsplit-vorticity')


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare(source,data,root,weight,validation,python=PYTHON,arms=ARMS,opsplit_update='additive'):
    if weight<=0:raise ValueError('Weighted run requires a positive omega weight')
    if not arms or len(set(arms))!=len(arms) or any(a not in ARMS for a in arms):raise ValueError('Invalid campaign arms')
    if opsplit_update not in ('additive','softplus'):raise ValueError('Invalid OpsSplit update map')
    root.mkdir(parents=True,exist_ok=False);code=root/'code';code.mkdir()
    for pattern in ('*.py','*_paper_protocol.json'):
        for path in source.glob(pattern):shutil.copy2(path,code/path.name)
    meta=json.loads((data/'comparison.json').read_text())
    if meta['case']!='es':raise ValueError('Only ES supported')
    numeric=('epochs','train_trajectories','train_frames','batch_size','horizon','width','modes','layers','lr','weight_decay',
             'scheduler_step','scheduler_gamma','grad_clip','eval_horizon','seed')
    cfg={k:meta['protocol'][k] for k in numeric}
    cfg.update(fields=['rho','T','potential'],scalar_divisors=[1e20,1.,1e3],normalization='training-only fieldwise minmax',
               native_state_conversion=False,weighted_vorticity_loss=weight,omega_factor=-6.460061919207894e-5,
               opsplit_update=opsplit_update,
               omega_crop=4,density_diffusion_per_saved_frame=5e-6,composite_stability_epochs=min(12,cfg['epochs']),
               checkpoint_selection='49-step validation primary-field relative L2; no auxiliary term in checkpoint selection')
    write(root/'protocol.json',cfg)
    write(root/'setup.json',dict(data=str(data),arms=list(arms),protocol=cfg,validation=validation,
          code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')},
          training_windows=len(meta['windows']),validation_trajectories=len(meta['val_indices']),
          test_cohorts={str(h):sum(v>=h for v in meta['test_prefixes'].values()) for h in (10,49,100,200)},
          notes=['All FNO inputs and outputs have three fields; omega is only an interior loss target/diagnostic.',
                 f'AR/NODE use direct/residual normalized predictions; OpsSplit uses {opsplit_update} updates then saved-frame density diffusion.',
                 'Weighted and unweighted OpsSplit share initialization, architecture and data order.',
                 'Weight100 is an exploratory pilot-informed setting, not an optimum.',
                 'Normalization comes from all100 training prefixes, with unchanged common source-valid windows.',
                 'No source dataset, previous checkpoint or previous campaign is modified.']))
    stages=[]
    for arm in arms:
        base=[python,str(code/'es_threefield.py'),'--data',str(data),'--arm',arm,'--omega-weight',str(weight),
              '--opsplit-update',opsplit_update]
        out=root/arm;smoke=arm+'-smoke'
        stages.append(dict(stage=smoke,command=base+['--output',str(root/smoke),'--smoke'],dependencies=[],checkpoint=root/smoke/'last.pt',walltime='00:30:00'))
        dependencies=[smoke]
        if arm.startswith('opsplit'):
            name=arm+'-audit';audit=root/(name+'.json')
            stages.append(dict(stage=name,command=base+['--output',str(audit),'--audit','--max-seconds','79200'],dependencies=dependencies,checkpoint=audit,walltime='24:00:00'))
            base+=['--audit-report',str(audit)]
            name_probe=arm+'-probe'
            stages.append(dict(stage=name_probe,command=base+['--output',str(out),'--stop-after-epoch',str(cfg['composite_stability_epochs']),'--max-seconds','79200'],dependencies=[name],checkpoint=out/'last.pt',walltime='24:00:00'))
            dependencies=[name_probe]
        stages.append(dict(stage=arm,command=base+['--output',str(out),'--max-seconds','79200'],dependencies=dependencies,checkpoint=out/'last.pt',walltime='24:00:00'))
        stages.append(dict(stage=arm+'-evaluate',command=base+['--output',str(out),'--evaluate-only','--max-seconds','79200'],dependencies=[arm],checkpoint=None,walltime='24:00:00'))
    jobs=[]
    for stage in stages:
        name,command=stage['stage'],stage['command'];checkpoint=stage['checkpoint'];script=root/(name+'.sbatch')
        body='#!/bin/bash\nset -euo pipefail\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\nset +e\n'
        if checkpoint:
            body+='if [ -f '+shlex.quote(str(checkpoint))+' ]; then\n'+shlex.join(command+['--resume'])+'\nelse\n'+shlex.join(command)+'\nfi\n'
        else:body+=shlex.join(command)+'\n'
        body+='status=$?\nset -e\nif [ "$status" -eq 75 ]; then\n  scontrol requeue "$SLURM_JOB_ID"\n  exit 0\nfi\nexit "$status"\n'
        script.write_text(body)
        subprocess.run(['bash','-n',str(script)],check=True)
        request=['sbatch','--parsable','--account=fupb2_ukaea_ml','--partition=boost_fua_prod','--nodes=1','--ntasks=1',
                 '--cpus-per-task=4','--mem=32G','--gres=gpu:1','--time='+stage['walltime'],'--requeue','--open-mode=append',
                 '--job-name=es3-'+name,'--output='+str(root/(name+'-%j.log'))]
        jobs.append(dict(stage=name,dependencies=stage['dependencies'],job_id=None,request=request,script=str(script)))
    assert len(jobs)==sum(5 if arm.startswith('opsplit') else 3 for arm in arms)
    write(root/'jobs.json',jobs)
    return jobs


def submit(root,jobs):
    ids={}
    for job in jobs:
        request=job['request'].copy()
        if job['dependencies']:request+=['--dependency=afterok:'+':'.join(ids[d] for d in job['dependencies']),'--kill-on-invalid-dep=yes']
        request.append(job['script']);job['submission']=request
        job['job_id']=subprocess.check_output(request,text=True).strip().split(';')[0]
        ids[job['stage']]=job['job_id'];write(root/'jobs.json',jobs)
    return ids


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--omega-weight',type=float,default=100.)
    p.add_argument('--arms',choices=ARMS,nargs='+',default=list(ARMS))
    p.add_argument('--opsplit-update',choices=('additive','softplus'),default='additive')
    p.add_argument('--tests',type=Path,required=True);p.add_argument('--submit',action='store_true')
    a=p.parse_args()
    suites=ET.parse(a.tests).getroot();tests=sum(int(s.attrib.get('tests',0)) for s in suites.iter('testsuite'))
    failures=sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites.iter('testsuite'))
    if tests<15 or failures:raise ValueError('Require the passing three-field regression suite before launch')
    jobs=prepare(a.source.resolve(),a.data.resolve(),a.output.resolve(),a.omega_weight,
                 dict(tests=tests,failures=failures,report=str(a.tests.resolve())),arms=a.arms,opsplit_update=a.opsplit_update)
    if a.submit:print(json.dumps(submit(a.output.resolve(),jobs),indent=2))
    else:print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
