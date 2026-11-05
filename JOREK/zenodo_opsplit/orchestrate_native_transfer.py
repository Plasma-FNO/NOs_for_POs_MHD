"""Launch a matched EM budget sweep of bounded ES operator corrections.

One fixed, explicitly selected ES source; target seeds vary nested cohorts,
initial nontransferred weights and window order. Each array task audits all
windows, trains with exact continuation, then evaluates untouched holdouts.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess

import torch

from comparison import digest, write_json
from data import select
from orchestrate import PYTHON
from pilot_native_transfer import initialize, verify_source, TRANSFER
from train_es import save_checkpoint

TRAIN_ARMS = ('scratch', 'finetune', 'advection-transfer', 'diffusion-transfer')
SOURCE_POLICY = ('Fixed final ES pilot checkpoint, 8 ES trajectories / 64 updates; '
                 'exploratory transfer, NOT the ES-validation winner (which is zero correction). '
                 'One source shared across target seeds; scratch shares ES interface scales.')


def trials(seeds, budgets):
    return [dict(seed=seed, budget=budget, arm=arm,
                 directory=f'seed-{seed}/budget-{budget}/{arm}')
            for budget in budgets for seed in seeds for arm in TRAIN_ARMS]


def commands(root, trial, python, epochs):
    folder = root/trial['directory']
    initial = root/'initial'/f"seed-{trial['seed']}-{trial['arm']}.pt"
    common = ['--manifest', str(root/'manifest.json'), '--case', 'em',
              '--pretrained', str(initial), '--train-trajectories', str(trial['budget']),
              '--seed', str(trial['seed']), '--substeps', '8']
    audit = [python, str(root/'code/validate_native_windows.py'), *common,
             '--workers', '1', '--output', str(folder/'window-validation.json')]
    train = [python, str(root/'code/train_native_rollout.py'), *common,
             '--validation-report', str(folder/'window-validation.json'),
             '--output', str(folder/'training'), '--epochs', str(epochs),
             '--batch-size', '20', '--elliptic-device', 'cuda', '--checkpoint-steps',
             '--max-seconds', '79200']
    return audit, train


def evaluation_commands(root, folder, python, checkpoint=None):
    result = []
    for horizon in (10, 100):
        command = [python, str(root/'code/evaluate_native_transfer.py'),
                   '--manifest', str(root/'manifest.json'), '--case', 'em',
                   '--split', 'test', '--trajectories', '0', '--seed', '907',
                   '--horizon', str(horizon), '--substeps', '8', '--device', 'cuda',
                   '--output', str(folder/f'test-{horizon}.json')]
        if checkpoint:command += ['--checkpoint', str(checkpoint)]
        result.append(command)
    return result


def script_header():
    return '#!/bin/bash\nset -euo pipefail\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'


def render_trial(root, trial, python, epochs):
    folder = root/trial['directory']
    folder.mkdir(parents=True)
    audit, train = commands(root, trial, python, epochs)
    # Audit and training use separate allocations, so the 22h continuation
    # allowance always leaves 2h for a last batch/checkpoint in a 24h job.
    audited = shlex.quote(str(folder/'audit.done'))
    trained = shlex.quote(str(folder/'training.done'))
    last = shlex.quote(str(folder/'training/last.pt'))
    text = script_header()
    text += f'if [ ! -f {audited} ]; then\n  {shlex.join(audit)}\n  touch {audited}\n'
    text += '  scontrol requeue "$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"\n  exit 0\nfi\n'
    text += f'if [ ! -f {trained} ]; then\n  set +e\n  if [ -f {last} ]; then\n'
    text += f'    {shlex.join(train+["--resume"])}\n  else\n    {shlex.join(train)}\n  fi\n'
    text += '  training_status=$?\n  set -e\n  if [ "$training_status" -eq 75 ]; then\n'
    text += '    scontrol requeue "$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"\n    exit 0\n  fi\n'
    text += f'  if [ "$training_status" -ne 0 ]; then exit "$training_status"; fi\n  touch {trained}\n'
    text += '  scontrol requeue "$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"\n  exit 0\nfi\n'
    # Evaluate both horizons even if one fails; never hide failed predictions.
    text += 'evaluation_status=0\n'
    for command in evaluation_commands(root, folder, python, folder/'training/best.pt'):
        text += shlex.join(command)+' || evaluation_status=1\n'
    text += 'exit "$evaluation_status"\n'
    path = folder/'run.sh';path.write_text(text)
    return path


def array_script(path, scripts):
    body = script_header()+'case "$SLURM_ARRAY_TASK_ID" in\n'
    for index, script in enumerate(scripts):
        body += f'  {index}) exec bash {shlex.quote(str(script))} ;;\n'
    path.write_text(body+'  *) exit 2 ;;\nesac\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'source', 'source-data', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--python', default=PYTHON)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--budgets', type=int, nargs='+', default=[8, 32, 64])
    p.add_argument('--epochs', type=int, default=250)
    p.add_argument('--concurrency', type=int, default=3)
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    if (min(args.budgets+[args.epochs,args.concurrency]) < 1 or min(args.seeds)<0 or
            len(set(args.seeds))!=len(args.seeds) or len(set(args.budgets))!=len(args.budgets)):
        p.error('Positive sizes and unique nonnegative seeds/budgets required')
    torch.set_num_threads(1)
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    source_metadata = verify_source(source, args.source_data, args.manifest)
    if source.get('model_config',{}).get('correction_fraction') is None:
        raise ValueError('A bounded-correction source is required')
    manifest = json.loads(args.manifest.read_text())
    if max(args.budgets)>len(select(manifest,'em','train')):
        raise ValueError('EM budget exceeds the training split')
    root = args.output.resolve();root.mkdir(parents=True, exist_ok=False)
    code = root/'code';code.mkdir();(root/'initial').mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path, code/path.name)
    shutil.copy2(args.manifest, root/'manifest.json')
    shutil.copy2(args.source, root/'source.pt')
    write_json(root/'source-data-manifest.json', source_metadata)
    source_sha = digest(root/'source.pt')
    for seed in args.seeds:
        for arm in TRAIN_ARMS:
            model = initialize(source, arm, seed, 'cpu')
            payload = dict(case='es', phase='residual_pretraining',
                           normalization=source['normalization'], model_config=model.config,
                           state_dict=model.state_dict(), transfer=dict(arm=arm, seed=seed,
                           modules=list(TRANSFER[arm]), source_sha256=source_sha, policy=SOURCE_POLICY))
            save_checkpoint(root/'initial'/f'seed-{seed}-{arm}.pt', payload)
    plan = trials(args.seeds,args.budgets)
    scripts = [render_trial(root, trial, args.python, args.epochs) for trial in plan]
    array_script(root/'training.sbatch', scripts)
    controls = []
    for arm in ('analytic','frozen'):
        folder=root/'controls'/arm;folder.mkdir(parents=True)
        script=folder/'run.sh';body=script_header()+'evaluation_status=0\n'
        for command in evaluation_commands(root, folder, args.python, root/'source.pt' if arm=='frozen' else None):
            body += shlex.join(command)+' || evaluation_status=1\n'
        script.write_text(body+'exit "$evaluation_status"\n');controls.append(script)
    array_script(root/'controls.sbatch', controls)
    (root/'summary.sbatch').write_text(script_header()+shlex.join([
        args.python,str(code/'summarize_native_transfer.py'),str(root)])+'\n')
    for script in scripts+controls+list(root.glob('*.sbatch')):
        subprocess.run(['bash','-n',str(script)],check=True)
    campaign = dict(schema=1, root=str(root), seeds=args.seeds, budgets=args.budgets,
                    epochs=args.epochs, concurrency=args.concurrency, trials=plan,
                    controls=['analytic','frozen'], test_horizons=[10,100],
                    test_ids=[e['id'] for e in select(manifest,'em','test',0,907)],
                    source_policy=SOURCE_POLICY, source_path=str(args.source.resolve()),source_sha256=source_sha,
                    source_correction_training_ids=source['correction_training_ids'],
                    source_correction_validation_ids=source['correction_validation_ids'],
                    source_model_config=source['model_config'],
                    manifest_sha256=digest(root/'manifest.json'),
                    code_sha256={f.name:digest(f) for f in code.glob('*.py')},
                    protocol=dict(train_frames=50,horizon=5,batch_size=20,epochs=args.epochs,substeps=8,
                        optimizer='Adam lr0.001 weight_decay0.0001',scheduler='StepLR50 gamma0.5',
                        clip=1.,anchor='normalized correction L2, weight1',
                        validation='All admissible 49-step validation prefixes; minimum mean normalized primary-state relative L2',
                        initial_checkpoint_eligible=False, windows='All admissible sliding windows each epoch',
                        budget_comparison='Equal epochs, not equal optimizer updates across budgets; arms matched within each seed/budget',
                        test='All 10 held-out families at horizons10/100; validity fixed from release states independently of predictions'))
    write_json(root/'campaign.json',campaign)
    jobs=[]
    for stage, array in [('controls','0-1%1'),('training',f'0-{len(plan)-1}%{args.concurrency}'),('summary',None)]:
        request=['sbatch','--parsable','--account=fupb2_ukaea_ml','--nodes=1','--ntasks=1',
                 '--partition='+('ptgr_all_serial' if stage=='summary' else 'boost_fua_prod'),
                 '--cpus-per-task='+('1' if stage=='summary' else '4'),
                 '--mem='+('2G' if stage=='summary' else '32G'),
                 '--time='+('00:10:00' if stage=='summary' else '24:00:00'),
                 '--job-name=jorek-residual-sweep-'+stage, '--open-mode=append',
                 f'--output={root}/{stage}-%A_%a.log']
        if array:request += ['--gres=gpu:1','--requeue','--array='+array]
        elif args.submit:request += ['--dependency=afterany:'+':'.join(j['job_id'] for j in jobs)]
        request += [str(root/f'{stage}.sbatch')]
        entry=dict(stage=stage,job_id=None,submission=request)
        if args.submit:
            entry['job_id']=subprocess.check_output(request,text=True).strip().split(';')[0]
        jobs.append(entry);write_json(root/'jobs.json',jobs)
    print(json.dumps(dict(root=str(root),trials=len(plan),jobs=jobs),indent=2))


if __name__=='__main__':main()
