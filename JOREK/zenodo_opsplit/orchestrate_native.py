"""Snapshot and prepare native ES/EM term pretraining, followed by rollout validation.

Default stage after pretraining is a rollout smoke, not a costly 250-epoch
training campaign. --full-rollout prepares GPU training with timed continuation.
"""
import argparse,json,shlex,shutil,subprocess
from pathlib import Path
from orchestrate import PYTHON


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=['es','em'],required=True);p.add_argument('--python',default=PYTHON)
    p.add_argument('--pretrained',type=Path,help='Existing native term checkpoint; skips preparation/pretraining')
    p.add_argument('--full-rollout',action='store_true');p.add_argument('--submit',action='store_true')
    p.add_argument('--dependency',help='Numeric external prerequisite job')
    args=p.parse_args()
    if args.dependency and not args.dependency.isdigit():p.error('Numeric dependency required')
    if not args.manifest.is_file():p.error('Dataset manifest missing')
    if args.pretrained and not args.pretrained.is_file():p.error('Native pretrained checkpoint missing')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);code=root/'code';code.mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,code/path.name)
    shutil.copy2(args.manifest,root/'manifest.json')
    stages=[]
    if args.pretrained is None:
        stages.append(('targets',[args.python,str(code/'prepare_native_terms.py'),'--manifest',str(root/'manifest.json'),
                      '--output',str(root/'targets'),'--case',args.case],False,False))
        stages.append(('term-training',[args.python,str(code/'train_native_terms.py'),'--data',str(root/'targets'),
                      '--output',str(root/'term-training')],True,True))
    pretrained=args.pretrained.resolve() if args.pretrained else root/'term-training/best.pt'
    if args.full_rollout:
        stages.append(('validate-windows',[args.python,str(code/'validate_native_windows.py'),
            '--manifest',str(root/'manifest.json'),'--pretrained',str(pretrained),'--case',args.case,
            '--output',str(root/'window-validation.json')],True,False))
    command=[args.python,str(code/'train_native_rollout.py'),'--manifest',str(root/'manifest.json'),
             '--pretrained',str(pretrained),'--output',str(root/'rollout'),'--case',args.case]
    command+=['--elliptic-device','cuda']
    if not args.full_rollout:command+=['--smoke']
    else:command+=['--max-seconds','79200','--validation-report',str(root/'window-validation.json')]
    stages.append(('rollout',command,True,True))
    previous=args.dependency;jobs=[]
    for name,command,gpu,resumable in stages:
        script=root/f'{name}.sbatch'
        body='#!/bin/bash\nset -euo pipefail\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'
        if name=='rollout' and args.full_rollout:body+='set +e\n'
        if resumable:
            body+='if [ -f '+shlex.quote(str(root/name/'last.pt'))+' ]; then\n'+shlex.join(command+['--resume'])+'\nelse\n'+shlex.join(command)+'\nfi\n'
        else:body+=shlex.join(command)+'\n'
        if name=='rollout' and args.full_rollout:
            body+='training_status=$?\nset -e\nif [ "$training_status" -eq 75 ]; then\n  scontrol requeue "$SLURM_JOB_ID"\n  exit 0\nfi\nexit "$training_status"\n'
        script.write_text(body)
        request=['sbatch','--parsable','--account=fupb2_ukaea_ml','--partition='+('boost_fua_prod' if gpu else 'ptgr_all_serial'),
                 '--nodes=1','--ntasks=1','--cpus-per-task=4','--mem='+('32G' if gpu else '11000M'),
                 '--time='+('24:00:00' if gpu else '01:00:00'),f'--job-name=jorek-native-{args.case}-{name}',f'--output={root}/{name}-%j.log']
        if gpu:request+=['--gres=gpu:1']
        if name=='rollout' and args.full_rollout:request+=['--requeue','--open-mode=append']
        if previous:request+=['--dependency=afterok:'+previous,'--kill-on-invalid-dep=yes']
        request+=[str(script)]
        job=subprocess.check_output(request,text=True).strip().split(';')[0] if args.submit else None
        jobs.append(dict(stage=name,job_id=job,submission=request,script=str(script)))
        previous=job or '<'+name+'-job-id>'
        (root/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
