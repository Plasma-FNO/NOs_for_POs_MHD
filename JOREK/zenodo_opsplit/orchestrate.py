"""Render or submit a bounded Slurm transfer study after a validated data manifest exists."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

PYTHON = '/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python'


def command(args, output, mode, seed, budget=0, source=None):
    cmd = [args.python, str(Path(__file__).with_name('train.py').resolve()),
           '--manifest', str(args.manifest.resolve()), '--output', str(output),
           '--mode', mode, '--seed', str(seed), '--epochs', str(args.epochs),
           '--steps-per-epoch', str(args.steps_per_epoch), '--eval-horizon', str(args.eval_horizon)]
    if budget:
        cmd += ['--train-trajectories', str(budget)]
    if source:
        cmd += ['--source', str(source)]
    if mode == 'finetune':
        cmd += ['--lr', '0.0003']
    return shlex.join(cmd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', default=PYTHON)
    parser.add_argument('--account', required=True)
    parser.add_argument('--partition', default='boost_fua_prod')
    parser.add_argument('--time', default='02:00:00')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--budgets', type=int, nargs='+', default=[8])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps-per-epoch', type=int, default=64)
    parser.add_argument('--eval-horizon', type=int, default=100)
    parser.add_argument('--dependency', help='Slurm job ID for afterok; manifest may be produced by that job')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if any(n < 1 for n in args.budgets) or len(set(args.seeds)) != len(args.seeds) or len(set(args.budgets)) != len(args.budgets):
        parser.error('Budgets must be positive and seeds/budgets unique')
    if args.dependency and not args.dependency.isdigit():
        parser.error('--dependency must be a numeric job ID')
    if not args.manifest.exists() and not args.dependency:
        parser.error('Manifest must exist, or supply its preparation job dependency')
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
        if max(args.budgets) > manifest['counts']['em']['train']:
            parser.error('Training budget exceeds available EM training trajectories')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = []
    # One GPU per seed, sequential stages: no idle dependent GPU allocation.
    for seed in args.seeds:
        root = args.output / f'seed-{seed}'
        source = root / 'source' / 'best.pt'
        lines = ['#!/bin/bash', 'set -euo pipefail', 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
                 'export PYTHONDONTWRITEBYTECODE=1',
                 command(args, root/'source', 'source', seed),
                 command(args, root/'zero-shot', 'zero-shot', seed, source=source)]
        for budget in args.budgets:
            for mode in ('scratch', 'monolithic', 'frozen', 'finetune'):
                lines.append(command(args, root/f'{mode}-{budget}', mode, seed, budget,
                                     source=source if mode in ('frozen', 'finetune') else None))
        if len(args.seeds) == 1:
            lines.append(shlex.join([args.python, str(Path(__file__).with_name('summarize.py').resolve()), str(args.output)]))
        script = args.output / f'seed-{seed}.sbatch'
        script.write_text('\n'.join(lines) + '\n')
        request = ['sbatch', '--parsable', '--account', args.account, '--partition', args.partition,
                   '--job-name', f'jorek-transfer-{seed}', '--nodes', '1', '--ntasks', '1',
                   '--cpus-per-task', '4', '--gres', 'gpu:1', '--mem', '32G', '--time', args.time,
                   '--output', str(args.output/f'seed-{seed}-%j.log')]
        if args.dependency:
            request += ['--dependency', f'afterok:{args.dependency}', '--kill-on-invalid-dep=yes']
        request += [str(script)]
        entry = {'seed': seed, 'script': str(script), 'submission': request, 'job_id': None}
        if args.submit:
            entry['job_id'] = subprocess.check_output(request, text=True).strip().split(';')[0]
        plan.append(entry)
        (args.output/'jobs.json').write_text(json.dumps(plan, indent=2) + '\n')
    if len(args.seeds) > 1:
        script = args.output / 'summarize.sbatch'
        script.write_text('#!/bin/bash\nset -euo pipefail\n' + shlex.join([
            args.python, str(Path(__file__).with_name('summarize.py').resolve()), str(args.output)]) + '\n')
        request = ['sbatch', '--parsable', '--account', args.account, '--partition', 'ptgr_all_serial',
                   '--job-name', 'jorek-transfer-summary', '--ntasks', '1', '--cpus-per-task', '1',
                   '--mem', '2G', '--time', '00:10:00', '--output', str(args.output/'summary-%j.log')]
        entry = {'stage': 'summary', 'script': str(script), 'job_id': None}
        if args.submit:
            request += ['--dependency', 'afterany:' + ':'.join(p['job_id'] for p in plan)]
        request += [str(script)]
        entry['submission'] = request
        if args.submit:
            entry['job_id'] = subprocess.check_output(request, text=True).strip().split(';')[0]
        plan.append(entry)
        (args.output/'jobs.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
