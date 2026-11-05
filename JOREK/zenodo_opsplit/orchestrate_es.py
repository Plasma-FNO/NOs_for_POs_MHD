"""Prepare ES/EM comparisons; submit ready methods in order and record the dedicated native workflow."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from orchestrate import PYTHON
from formulation import plan as native_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=['es', 'em'], default='es')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--split-formulation', choices=['jorek', 'effective'], default='jorek')
    parser.add_argument('--account', default='fupb2_ukaea_ml')
    parser.add_argument('--partition', default='boost_fua_prod')
    parser.add_argument('--python', default=PYTHON)
    parser.add_argument('--time', default='24:00:00')
    parser.add_argument('--dependency', help='Optional numeric prerequisite job ID')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.protocol is None:
        args.protocol = Path(__file__).with_name(f'{args.case}_paper_protocol.json')
    cfg = json.loads(args.protocol.read_text())
    if cfg.get('case', args.case) != args.case:
        parser.error('Protocol regime does not match --case')
    if sum(e['case'] == args.case and e['split'] == 'train' for e in manifest['entries']) < (2 if args.smoke else cfg['train_trajectories']):
        parser.error(f'Not enough {args.case.upper()} training trajectories')
    if args.dependency and not args.dependency.isdigit():
        parser.error('Dependency must be a numeric job ID')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    code = output/'code'
    code.mkdir()
    for name in ('train_es.py', 'train.py', 'data.py', 'model.py', 'formulation.py', 'equations.py'):
        shutil.copy2(Path(__file__).with_name(name), code/name)
    shutil.copy2(args.protocol, code/f'{args.case}_paper_protocol.json')
    shutil.copy2(args.manifest, output/'manifest.json')
    plan, previous = [], args.dependency
    for method in ('ar', 'node', 'opsplit'):
        if method == 'opsplit' and args.split_formulation == 'jorek':
            entry = native_plan(args.case)
            entry.update(method=method, job_id=None, prerequisite_job_id=previous)
            plan.append(entry)
            (output/'opsplit-native-workflow.json').write_text(json.dumps(entry, indent=2)+'\n')
            (output/'jobs.json').write_text(json.dumps(plan, indent=2)+'\n')
            continue
        command = [args.python, str(code/'train_es.py'), '--manifest', str(output/'manifest.json'),
                   '--output', str(output/method), '--protocol', str(code/f'{args.case}_paper_protocol.json'), '--method', method, '--case', args.case, '--split-formulation', args.split_formulation]
        if args.smoke:
            command.append('--smoke')
        script = output/f'{method}.sbatch'
        # Resubmitting the script resumes an epoch checkpoint, without overwriting a fresh run.
        script.write_text('#!/bin/bash\nset -euo pipefail\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'
                          + 'if [ -f '+shlex.quote(str(output/method/'last.pt'))+' ]; then\n  '
                          + shlex.join(command+['--resume'])+'\nelse\n  '+shlex.join(command)+'\nfi\n')
        request = ['sbatch', '--parsable', '--account', args.account, '--partition', args.partition,
                   '--job-name', 'jorek-'+args.case+'-'+method+('-smoke' if args.smoke else ''), '--nodes', '1', '--ntasks', '1',
                   '--cpus-per-task', '4', '--gres', 'gpu:1', '--mem', '32G', '--time', args.time,
                   '--output', str(output/f'{method}-%j.log')]
        if previous:
            request += ['--dependency', f'afterok:{previous}', '--kill-on-invalid-dep=yes']
        request.append(str(script))
        entry = dict(method=method, script=str(script), submission=request, job_id=None)
        if args.submit:
            entry['job_id'] = subprocess.check_output(request, text=True).strip().split(';')[0]
        plan.append(entry)
        previous = entry['job_id'] or f'<{method}-job-id>'
        (output/'jobs.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
