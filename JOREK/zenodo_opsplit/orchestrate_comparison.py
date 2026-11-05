"""Prepare immutable four-arm ES/EM campaigns, with smoke and native audit dependencies."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from comparison import ARMS, write_json
from orchestrate import PYTHON


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-from',type=Path,help='Relaunch only native arms using an existing matched case campaign and term checkpoint')
    p.add_argument('--case',choices=['es','em'],required=True);p.add_argument('--python',default=PYTHON)
    p.add_argument('--gpus',type=int,choices=[1,2,4],default=1,help='Native arm GPUs; global batch remains fixed')
    p.add_argument('--protocol',type=Path);p.add_argument('--submit',action='store_true')
    p.add_argument('--smoke-only',action='store_true',help='Prepare data and term pretraining, then validate bounded optimizer runs only')
    a=p.parse_args()
    if (a.manifest is None)==(a.native_from is None):p.error('Provide exactly one of --manifest and --native-from')
    source=a.native_from.resolve() if a.native_from else None
    if source:
        shared=json.loads((source/'data/comparison.json').read_text())
        if shared['case']!=a.case:p.error('Reused data case differs')
        if not (source/'terms/best.pt').is_file():p.error('Existing matched term checkpoint required')
        if a.protocol:p.error('Reused matched data fixes the protocol')
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    code=root/'code';code.mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,code/path.name)
    shutil.copy2(source/'manifest.json' if source else a.manifest,root/'manifest.json')
    shutil.copy2(source/'protocol.json' if source else a.protocol or Path(__file__).with_name(f'{a.case}_paper_protocol.json'),root/'protocol.json')
    data=source/'data' if source else root/'data'
    terms=source/'terms' if source else root/'terms'
    if source:write_json(root/'reused_data.json',dict(campaign=str(source),data=str(data),terms=str(terms/'best.pt')))
    stages=[('data',[a.python,str(code/'comparison.py'),'--manifest',str(root/'manifest.json'),
                     '--case',a.case,'--protocol',str(root/'protocol.json'),'--output',str(root/'data')],[],False,False),
            ('terms',[a.python,str(code/'train_native_terms.py'),'--data',str(root/'data/terms'),
                      '--output',str(root/'terms')],['data'],True,True)]
    if source:stages=[]
    for arm in ARMS:
        if source and not arm.startswith('opsplit'):continue
        base=[a.python,str(code/'train_comparison.py'),'--data',str(data),'--arm',arm]
        deps=[] if source else ['terms'] if arm=='opsplit-pretrained' else ['data']
        if arm=='opsplit-pretrained':base+=['--pretrained',str(terms/'best.pt')]
        smoke=arm+'-smoke'
        stages.append((smoke,base+['--output',str(root/smoke),'--smoke'],deps,True,True))
        if a.smoke_only:continue
        deps=[smoke]
        if arm.startswith('opsplit'):
            audit=arm+'-audit'
            stages.append((audit,base+['--output',str(root/(audit+'.json')),'--audit','--max-seconds','79200'],deps,True,True))
            deps=[audit];base+=['--audit-report',str(root/(audit+'.json'))]
        stages.append((arm,base+['--output',str(root/arm),'--max-seconds','79200'],deps,True,True))
        if a.gpus>1 and arm.startswith('opsplit'):
            stages.append((arm+'-evaluate',base+['--output',str(root/arm),'--evaluate-only','--max-seconds','79200'],[arm],True,False))
    jobs=[];ids={}
    for name,command,deps,gpu,resumable in stages:
        count=a.gpus if name.startswith('opsplit') and not name.endswith('-evaluate') else 1
        if count>1:
            if name.endswith('-audit'):command+=['--audit-workers',str(count),'--audit-gpus',str(count)]
            else:command=[a.python,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(count)]+command[1:]+['--train-only']
        script=root/(name+'.sbatch')
        body='#!/bin/bash\nset -euo pipefail\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=1\n'
        marker=root/name/'requeue.json'
        if count>1 and not name.endswith('-audit'):body+='rm -f '+shlex.quote(str(marker))+'\n'
        body+='set +e\n'
        if resumable:
            checkpoint=root/(name+'.json') if name.endswith('-audit') else root/name/'last.pt'
            body+='if [ -f '+shlex.quote(str(checkpoint))+' ]; then\n'+shlex.join(command+['--resume'])+'\nelse\n'+shlex.join(command)+'\nfi\n'
        else:body+=shlex.join(command)+'\n'
        body+='status=$?\nset -e\n'
        condition='[ "$status" -eq 75 ]'
        if count>1 and not name.endswith('-audit'):
            condition+=' || { [ "$status" -eq 0 ] && [ -f '+shlex.quote(str(marker))+' ]; }'
        body+='if '+condition+'; then\n  scontrol requeue "$SLURM_JOB_ID"\n  exit 0\nfi\nexit "$status"\n'
        script.write_text(body)
        wall_time='00:30:00' if name=='terms' or name.endswith('-smoke') else '24:00:00' if gpu else '02:00:00'
        request=['sbatch','--parsable','--account=fupb2_ukaea_ml','--partition='+('boost_fua_prod' if gpu else 'ptgr_all_serial'),
                 '--nodes=1','--ntasks=1',f'--cpus-per-task={4*count}','--mem='+('96G' if count>1 else '32G' if gpu else '11000M'),'--time='+wall_time,
                 '--requeue','--open-mode=append',f'--job-name=jorek-compare-{a.case}-{name}',f'--output={root}/{name}-%j.log']
        if gpu:request+=[f'--gres=gpu:{count}']
        if deps:request+=['--dependency=afterok:'+':'.join(ids[d] for d in deps),'--kill-on-invalid-dep=yes']
        request+=[str(script)]
        ids[name]='<'+name+'-job-id>'
        jobs.append(dict(stage=name,job_id=None,dependencies=deps,submission=request))
    write_json(root/'jobs.json',jobs)
    if a.submit:
        for job in jobs:
            request=job['submission']
            request=[('--dependency=afterok:'+':'.join(ids[d] for d in job['dependencies'])) if token.startswith('--dependency=') else token for token in request]
            job['submission']=request
            job['job_id']=subprocess.check_output(request,text=True).strip().split(';')[0]
            ids[job['stage']]=job['job_id']
            write_json(root/'jobs.json',jobs)
    print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
