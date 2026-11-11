"""GPU-only, validation-selected research campaign; preserves the original datasets.

Run from GenericOpsSplit with ``python -m gos.research screen|run --root ROOT``.
Classical screening reads validation trajectories only. Test metrics are produced
after training selects its checkpoint using the original validation split.
"""
import argparse
import copy
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

from .problems import PROBLEMS, get_problem
from .research_models import ETDReference, ResearchModel, SCHEMES
from .train import CONFIG, evaluate, load_data, relative_errors, rollout, train, validation_error


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=float))
    temporary.replace(path)


def device_info():
    if not torch.cuda.is_available():
        raise RuntimeError('This campaign requires a GPU; CPU training is forbidden')
    return dict(requested='cuda', name=torch.cuda.get_device_name(), torch=torch.__version__)


@torch.no_grad()
def timing(model, frames, mu):
    q, mu = frames[:8, 0], mu[:8]
    for _ in range(3):
        model.step(q, mu, model.h)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        model.step(q, mu, model.h)
    torch.cuda.synchronize()
    return dict(batch=len(q), seconds_per_step=(time.perf_counter()-start)/10,
                neural_calls_per_step=0 if model.exact else
                (2*model.substeps if model.closure_each_stage else 1)*
                (2 if getattr(model, 'zero_equilibrium', False) else 1),
                physics_substeps=model.substeps)


def screen(args):
    info, rows = device_info(), []
    names = sorted(PROBLEMS) if args.problem == 'all' else [args.problem]
    suffix = '_etdrk4' if args.reference else ''
    destination = Path(args.root)/f'screen_{args.problem}{suffix}.json'
    for name in names:
        problem = get_problem(name, 'cuda')
        data, scale = load_data(args.root, problem, 'cuda')
        frames, mu = data['val']
        schemes = ('etdrk4',) if args.reference else SCHEMES
        if args.reference and problem.space != 'fourier':
            continue
        for scheme in schemes:
            for substeps in map(int, args.screen_substeps.split(',')):
                model = (ETDReference(problem, scale, 'cuda', substeps=substeps) if args.reference else
                         ResearchModel(problem, scheme, CONFIG[name]['net'], scale, 'cuda',
                                       substeps=substeps, closure=False)).cuda().eval()
                start = time.perf_counter()
                error = validation_error(model, frames, mu, (frames.shape[1]-1)//2)
                row = dict(problem=name, scheme=scheme, substeps=substeps, val=error,
                           seconds=time.perf_counter()-start)
                rows.append(row)
                write_json(destination, dict(device=info, selection_split='val', rows=rows))
                print(json.dumps(row), flush=True)


@torch.no_grad()
def trajectory_metrics(model, data, horizon):
    output = {}
    for name, pair in [('test', data['test'])] + [(f'ood_{mu}', pair) for mu, pair in data['ood'].items()]:
        pred, true = rollout(model, *pair)
        errors = relative_errors(pred, true)[:, horizon-1]
        output[name] = dict(horizon=horizon, errors=errors.cpu().tolist(),
                            mean=float(errors.mean()), median=float(errors.median()),
                            finite_fraction=float(torch.isfinite(errors).float().mean()))
    return output


def run(args):
    info = device_info()
    torch.manual_seed(args.seed)
    problem = get_problem(args.problem, 'cuda')
    data, scale = load_data(args.root, problem, 'cuda')
    cfg = copy.deepcopy(CONFIG[args.problem])
    if args.no_closure and args.linear_mode != 'known':
        raise ValueError('Classical controls require the known operator')
    if args.scheme == 'etdrk4' and not args.no_closure:
        raise ValueError('ETDRK4 is an untrained reference; use --no-closure')
    cfg.update(scheme=args.scheme, substeps=args.substeps, closure=not args.no_closure,
               linear_mode=args.linear_mode, closure_each_stage=args.closure_each_stage,
               dt_mults=[int(s) for s in args.dt_mults.split(',')], lr=args.lr,
               val_every=args.val_every, operator_diagnostics=False,
               skip_jacobian=args.scheme == 'etdrk4', max_step=args.max_step,
               zero_equilibrium=args.zero_equilibrium)
    if args.lr_min is not None:
        cfg['lr_min'] = args.lr_min
    if args.iters is not None:
        cfg['iters'] = args.iters
    if args.window is not None:
        cfg['window'] = args.window
    name = f'physics_{args.scheme}_n{args.substeps}_{args.linear_mode}'
    name += '_exact' if args.no_closure else '_closure'
    name += '_stage' if args.closure_each_stage else ''
    name += '_capped' if args.max_step else ''
    name += '_zero' if args.zero_equilibrium else ''
    name += '_mdt' if cfg['dt_mults'] != [1] else ''
    name += f'_{args.tag}' if args.tag else ''
    directory = Path(args.root)/'runs'/args.problem/f'{name}_s{args.seed}'
    progress = None
    if args.command == 'checkpoint':
        progress = torch.load(directory/'training_state.pt', map_location='cpu', weights_only=False)
        if progress['config'] != cfg:
            raise ValueError(f'Checkpoint configuration differs: {directory}')
        assert 0 < progress['iteration'] <= cfg['iters']
        assert progress['best']['iteration'] <= progress['iteration']
        cfg['planned_iters'], cfg['iters'] = cfg['iters'], progress['iteration']
        name += f'_screen{progress["iteration"]}'
        directory = directory.parent/f'{name}_s{args.seed}'
    if (directory/'result.json').exists():
        previous = json.loads((directory/'result.json').read_text())
        old_config = dict(previous['config'])
        for key in ('max_step', 'zero_equilibrium'):
            old_config.setdefault(key, False)
        comparable = lambda c: {k: v for k, v in c.items() if k not in ('operator_diagnostics', 'skip_jacobian')}
        if comparable(old_config) != comparable(cfg):
            raise ValueError(f'Existing run has a different configuration: {directory}; choose a new --tag')
        print(f'skip {directory}', flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    model = (ETDReference(problem, scale, 'cuda', substeps=args.substeps, max_step=args.max_step) if args.scheme == 'etdrk4' else
             ResearchModel(problem, args.scheme, cfg['net'], scale, 'cuda',
                          substeps=args.substeps, closure=not args.no_closure,
                          linear_mode=args.linear_mode,
                          closure_each_stage=args.closure_each_stage, max_step=args.max_step,
                          zero_equilibrium=args.zero_equilibrium)).cuda()
    print(json.dumps(dict(problem=args.problem, arm=name, device=info, config=cfg)), flush=True)
    start = time.perf_counter()
    if progress is None:
        best, history, training = train(model, data, cfg, lambda text: print(text, flush=True),
                                        checkpoint_path=directory/'training_state.pt',
                                        max_cpu_seconds=args.cpu_budget)
    else:
        best, history = progress['best'], progress['history']
        training = dict(status='screened', seconds=progress['seconds'],
                        cpu_seconds=progress.get('cpu_seconds'), resumes=progress['resumes'],
                        iterations=progress['iteration'], skipped_gradients=progress['skipped_gradients'])
    if training['status'] == 'checkpointed':
        raise SystemExit(75)
    evaluation_start = time.perf_counter()
    model.load_state_dict(best['state'])
    model.eval()
    torch.save(dict(state=best['state'], config=cfg, problem=args.problem, scale=scale), directory/'model.pt')
    # The original theta-step propagator diagnostic does not apply to these RK methods.
    metrics = evaluate(model, problem, data, cfg, 'cuda')
    metrics['operator_note'] = 'Theta-step diagnostic omitted for Runge-Kutta integrator'
    metrics['trajectory_errors'] = trajectory_metrics(model, data, max(cfg['horizons']))
    result = dict(problem=args.problem, arm=name, base_arm='physics_'+args.scheme, seed=args.seed,
                  parameters=sum(p.numel() for p in model.parameters()), config=cfg, device=info,
                  train=dict(training, best_val=best['val'], best_iteration=best['iteration']),
                  history=history, eval=metrics, timing=timing(model, *data['test']),
                  total_seconds=training.get('seconds', evaluation_start-start)+time.perf_counter()-evaluation_start,
                  selection='validation only; historical test set is exploratory')
    write_json(directory/'result.json', result)
    print(json.dumps(dict(arm=name, train=result['train'], test=metrics['test'], long=metrics['long'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['screen', 'run', 'checkpoint'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--problem', choices=['all']+sorted(PROBLEMS), required=True)
    parser.add_argument('--scheme', choices=SCHEMES+('etdrk4',), default='ars222')
    parser.add_argument('--substeps', type=int, default=4)
    parser.add_argument('--max-step', action='store_true', help='cap physics microsteps at training h/substeps; cost scales with macro dt')
    parser.add_argument('--zero-equilibrium', action='store_true', help='force the learned defect to vanish at q=0 (two network calls)')
    parser.add_argument('--screen-substeps', default='1,2,4,8,16')
    parser.add_argument('--reference', action='store_true', help='screen the untrained Fourier ETDRK4 reference')
    parser.add_argument('--no-closure', action='store_true')
    parser.add_argument('--closure-each-stage', action='store_true')
    parser.add_argument('--linear-mode', choices=['known', 'struct', 'free'], default='known')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iters', type=int)
    parser.add_argument('--window', type=int)
    parser.add_argument('--dt-mults', default='1')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-min', type=float, help='cosine schedule floor (legacy default: 1e-5)')
    parser.add_argument('--val-every', type=int, default=500)
    parser.add_argument('--tag', default='')
    parser.add_argument('--cpu-budget', type=float, default=450., help='checkpoint before this process CPU time, then resume in a new CUDA process')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command != 'screen' and args.problem == 'all':
        parser.error('run and checkpoint need one problem')
    if args.cpu_budget <= 0:
        parser.error('--cpu-budget must be positive')
    limit = resource.getrlimit(resource.RLIMIT_CPU)[0]
    if limit != resource.RLIM_INFINITY and args.cpu_budget >= limit:
        parser.error('--cpu-budget must leave room below the process CPU limit')
    if args.command == 'run' and not args.worker:
        command = [sys.executable, '-m', 'gos.research', *sys.argv[1:], '--worker']
        while True:
            status = subprocess.run(command).returncode
            if status != 75:
                raise SystemExit(status)
    torch.set_num_threads(4)
    (screen if args.command == 'screen' else run)(args)


if __name__ == '__main__':
    main()
