"""Freeze candidate checkpoints, then generate and evaluate an independent CUDA holdout.

python -m gos.research_holdout freeze --runs RUN_DIR ... --manifest MANIFEST.json
python -m gos.research_holdout generate --manifest MANIFEST.json --data-root ROOT --holdout-root HOLDOUT
python -m gos.research_holdout evaluate --manifest MANIFEST.json --holdout-root HOLDOUT --out REPORT.json
python -m gos.research_holdout merge --reports REPORT_BURGERS.json REPORT_KDV.json ... --out REPORT.json

Generation is a separate command so model selection can finish before any fresh data exist.
Use --problem on generate/evaluate to bound each process to one PDE while preserving the full manifest.
"""
import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch

from .models import StepModel
from .problems import PROBLEMS, get_problem
from .research import device_info, write_json
from .research_models import ETDReference, ResearchModel
from .train import CONFIG, long_rollout, norms, relative_errors, rollout, spectrum_error


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def freeze(directories, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Refusing to replace frozen selection: {destination}')
    runs, identifiers, configurations = [], set(), {}
    for directory in directories:
        directory = Path(directory).resolve()
        result = json.loads((directory/'result.json').read_text())
        identity = (result['problem'], result['arm'], result['seed'])
        if identity in identifiers:
            raise ValueError(f'Duplicate frozen run identifier: {identity}')
        identifiers.add(identity)
        scientific_config = {key: value for key, value in result['config'].items()
                             if key not in ('operator_diagnostics', 'skip_jacobian')}
        scientific_config.setdefault('lr_min', 1e-5)
        if 'scheme' in scientific_config:
            scientific_config.setdefault('max_step', False)
            scientific_config.setdefault('zero_equilibrium', False)
            scientific_config.setdefault('closure_in_range', False)
        group = identity[:2]
        if group in configurations and configurations[group] != scientific_config:
            raise ValueError(f'Mixed scientific configurations across seeds: {group}')
        configurations[group] = scientific_config
        runs.append(dict(directory=str(directory), problem=result['problem'], arm=result['arm'],
                         seed=result['seed'], config=result['config'],
                         checkpoint_sha256=sha256(directory/'model.pt'),
                         result_sha256=sha256(directory/'result.json')))
    if not runs:
        raise ValueError('Freeze at least one run')
    sources = {name: sha256(Path(__file__).with_name(name)) for name in
               ('models.py', 'operators.py', 'problems.py', 'research_models.py', 'train.py',
                'research.py', 'research_holdout.py', 'research_gate.py')}
    manifest = dict(selection='Frozen before generating the independent holdout', runs=runs, source_sha256=sources)
    write_json(destination, manifest)
    return manifest


def read_frozen(path):
    manifest = json.loads(Path(path).read_text())
    for name, expected in manifest.get('source_sha256', {}).items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Frozen model source changed: {name}')
    for run in manifest['runs']:
        directory = Path(run['directory'])
        for name, field in (('model.pt', 'checkpoint_sha256'), ('result.json', 'result_sha256')):
            if sha256(directory/name) != run[field]:
                raise ValueError(f'Frozen artifact changed: {directory/name}')
    return manifest


def ood_plan(problem, seed, extra_ood):
    extras = [float(value) for value in extra_ood]
    low, high = problem.mu_train
    if extras and low == high:
        raise ValueError('Extra OOD coefficients require a varying-coefficient problem')
    if any(not math.isfinite(value) or value <= 0 or low <= value <= high for value in extras):
        raise ValueError('Extra OOD coefficients must be finite, positive, and outside the training interval')
    if len(set(extras)) != len(extras) or set(extras) & set(problem.mu_ood):
        raise ValueError('Extra OOD coefficients must not duplicate each other or original OOD cases')
    values = list(problem.mu_ood) + extras
    seeds = dict(test=seed, ood={str(value): seed+100*(i+1) for i, value in enumerate(values)})
    planned = {seed, *seeds['ood'].values()}
    if min(planned) < 0 or max(planned) >= 2**64 or planned & {1, 2, 3, 100, 101}:
        raise ValueError('Use valid independent generation seeds for all test and OOD cases')
    return values, seeds


def generate(manifest_path, data_root, holdout_root, test_count=32, ood_count=16, seed=7103,
             problem_name=None, extra_ood=None):
    info = device_info()
    generation_seeds = {seed, seed+100, seed+200}
    if min(test_count, ood_count) < 2 or seed < 0 or generation_seeds & {1, 2, 3, 100, 101}:
        raise ValueError('Use at least two samples per split and an independent generation seed')
    extras = list(extra_ood or [])
    if extras and problem_name is None:
        raise ValueError('--extra-ood requires exactly one selected --problem')
    manifest = read_frozen(manifest_path)
    names = sorted({run['problem'] for run in manifest['runs']})
    if problem_name is not None:
        if problem_name not in names:
            raise ValueError(f'Problem is absent from the frozen selection: {problem_name}')
        names = [problem_name]
    for name in names:
        path = Path(holdout_root)/f'{name}.pt'
        if path.exists():
            raise FileExistsError(f'Refusing to replace an independent holdout: {path}')
    for name in names:
        problem = get_problem(name, 'cuda')
        values, seeds = ood_plan(problem, seed, extras)
        original_path = Path(data_root)/'data'/f'{name}.pt'
        original = torch.load(original_path, map_location='cpu', weights_only=False)
        scale = original['scale']
        del original
        raw = dict(problem=name, scale=scale, h=problem.h, mu_ref=problem.mu_ref,
                   original_data=str(original_path.resolve()),
                   original_data_sha256=sha256(original_path),
                   manifest_sha256=sha256(manifest_path), device=info, seeds=seeds,
                   extra_ood=extras, ood_coefficients=values, ood={})
        raw['test'] = problem.generate(test_count, torch.Generator().manual_seed(seed),
                                       batch=32 if problem.dim == 1 else 8)
        for value in values:
            raw['ood'][float(value)] = problem.generate(
                ood_count, torch.Generator().manual_seed(seeds['ood'][str(value)]),
                ood=value, batch=32 if problem.dim == 1 else 8)
        path = Path(holdout_root)/f'{name}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save(raw, temporary)
        temporary.replace(path)
        print(f'Generated frozen holdout: {path}', flush=True)


def load_model(run, scale):
    directory = Path(run['directory'])
    checkpoint = torch.load(directory/'model.pt', map_location='cuda', weights_only=False)
    result = json.loads((directory/'result.json').read_text())
    if not math.isclose(float(checkpoint['scale']), float(scale), rel_tol=1e-7):
        raise ValueError(f'Checkpoint normalization differs from holdout: {directory}')
    cfg = result['config']
    problem = get_problem(run['problem'], 'cuda')
    if cfg.get('scheme') == 'etdrk4':
        model = ETDReference(problem, scale, 'cuda', substeps=cfg['substeps'],
                             max_step=cfg.get('max_step', False))
    elif 'scheme' in cfg:
        model = ResearchModel(problem, cfg['scheme'], cfg['net'], scale, 'cuda',
                              substeps=cfg['substeps'], closure=cfg['closure'],
                              linear_mode=cfg.get('linear_mode', 'known'),
                              closure_each_stage=cfg.get('closure_each_stage', False),
                              max_step=cfg.get('max_step', False),
                              zero_equilibrium=cfg.get('zero_equilibrium', False),
                              closure_in_range=cfg.get('closure_in_range', False))
    else:
        model = StepModel(problem, result.get('base_arm', checkpoint['arm']), cfg['net'], scale, 'cuda')
    model = model.cuda().eval()
    model.load_state_dict(checkpoint['state'])
    return model, result


def error_summary(errors):
    errors = torch.as_tensor(errors, dtype=torch.float64)
    errors = torch.where(torch.isfinite(errors), errors, float('inf'))
    clipped = errors.clamp(max=2.)
    return dict(errors=errors.tolist(), mean=float(errors.mean()),
                std=float(errors.std()) if len(errors) > 1 else 0.,
                mean_clipped=float(clipped.mean()), std_clipped=float(clipped.std()) if len(errors) > 1 else 0.,
                finite_fraction=float(torch.isfinite(errors).double().mean()))


def step_counts(model, multiple):
    """Count solver substeps and network executions, identifying tangent-free JVP work separately."""
    substeps = getattr(model, 'substeps', 1)
    count = max(1, math.ceil(substeps*multiple-1e-12)) if getattr(model, 'max_step', False) else substeps
    if isinstance(model, ResearchModel):
        calls = 0 if model.exact else (2*count if model.closure_each_stage else 1)
        calls *= 2 if model.zero_equilibrium else 1
        jvps = 0
    elif isinstance(model, ETDReference) or model.exact:
        calls, jvps = 0, 0
    else:
        calls = 2 if model.kind in ('ros2', 'node_rk2') else 1
        jvps = calls if model.tangent_free else 0
        calls += jvps
    return dict(internal_substeps=count, neural_calls_per_macrostep=calls,
                neural_jvps_per_macrostep=jvps)


@torch.no_grad()
def benchmark(model, frames, mu):
    """Post-selection CUDA timing; no inference_mode because tangent-free baselines require JVPs."""
    q, coefficients = frames[:8, 0], mu[:8]
    if not q.is_cuda:
        raise ValueError('Holdout timing requires CUDA tensors')
    output = {}
    for multiple in (1, 4):
        if multiple == 4 and model.kind == 'ar':
            continue
        for _ in range(2):
            model(q, coefficients, 8, dt_mult=multiple)
        seconds = []
        for _ in range(5):
            torch.cuda.synchronize(q.device)
            start = time.perf_counter()
            model(q, coefficients, 8, dt_mult=multiple)
            torch.cuda.synchronize(q.device)
            seconds.append((time.perf_counter()-start)/8)
        output[f'x{multiple}'] = dict(batch=len(q), macrosteps_per_repeat=8, warmups=2, repeats=5,
                                     dt=model.h*multiple, seconds_per_macrostep=mean(seconds),
                                     seconds_per_macrostep_std=stdev(seconds),
                                     repeat_seconds_per_macrostep=seconds,
                                     seconds_per_data_interval=mean(seconds)/multiple,
                                     **step_counts(model, multiple))
    return output


@torch.no_grad()
def evaluate_run(model, raw, result):
    problem = model.problem
    prepare = lambda pair: (pair[0].cuda()/raw['scale'],
                             (pair[1].double()/problem.mu_ref).float().cuda())
    splits = {'test': prepare(raw['test'])}
    splits.update({f'ood_{mu}': prepare(pair) for mu, pair in raw['ood'].items()})
    primary = max(CONFIG[problem.name]['horizons'])
    horizons = sorted(set(CONFIG[problem.name]['horizons'] + ([48, 100, 200] if problem.name == 'ks' else [])))
    output = dict(problem=problem.name, arm=result['arm'], seed=result['seed'], config=result['config'],
                  primary_horizon=primary, parameters=result['parameters'], splits={})
    for split, pair in splits.items():
        output['splits'][split] = {}
        for multiple in (1, 4):
            if multiple != 1 and model.kind == 'ar':
                continue
            pred, true = rollout(model, *pair, dt_mult=multiple, chunk=16)
            errors = relative_errors(pred, true)
            bound = 10*norms(true).amax(1, keepdim=True)
            bounded = torch.isfinite(norms(pred)) & (norms(pred) <= bound)
            metrics = {}
            for horizon in horizons:
                if horizon % multiple == 0 and horizon//multiple <= errors.shape[1]:
                    index = horizon//multiple-1
                    metrics[str(horizon)] = dict(error_summary(errors[:, index].cpu()),
                                                 bounded=float(bounded[:, index].float().mean()))
            if problem.name == 'ks':
                metrics['ks_statistics'] = spectrum_error(pred, true, start=50//multiple)
            output['splits'][split][f'x{multiple}'] = metrics
    frames, mu = splits['test']
    reference = norms(frames[:16]).amax(1)
    output['long'] = long_rollout(model, frames[:16, 0], mu[:16], CONFIG[problem.name]['long'], reference)
    return output


def paired_bootstrap(candidate, baseline, seed=9103, repeats=2000):
    """Paired trajectory bootstrap of clipped mean-error ratios, after averaging training seeds.

    Training-seed means and standard deviations are reported separately; this interval reflects
    held-out trajectory sampling, not uncertainty over all possible training initializations.
    """
    candidate = torch.as_tensor(candidate, dtype=torch.float64)
    baseline = torch.as_tensor(baseline, dtype=torch.float64)
    if candidate.ndim != 2 or baseline.ndim != 2 or candidate.shape[1] != baseline.shape[1]:
        raise ValueError('Supply seed-by-trajectory arrays with matched trajectory counts')
    if candidate.shape[1] < 2 or repeats < 2:
        raise ValueError('Bootstrap needs at least two trajectories and replicates')
    candidate = torch.nan_to_num(candidate, nan=2., posinf=2., neginf=2.).clamp(0., 2.)
    baseline = torch.nan_to_num(baseline, nan=2., posinf=2., neginf=2.).clamp(0., 2.)
    c, b = candidate.mean(0), baseline.mean(0)
    if float(b.mean()) == 0.:
        return dict(ratio=None, ci95=None, reason='zero baseline error')
    indices = torch.randint(len(c), (repeats, len(c)), generator=torch.Generator().manual_seed(seed))
    denominator = b[indices].mean(1)
    ratios = c[indices].mean(1)/denominator
    if not torch.isfinite(ratios).all():
        return dict(ratio=float(c.mean()/b.mean()), ci95=None, reason='zero bootstrap baseline error')
    low, high = torch.quantile(ratios, torch.tensor([.025, .975], dtype=torch.float64))
    return dict(ratio=float(c.mean()/b.mean()), ci95=[float(low), float(high)],
                candidate_seed_means=candidate.mean(1).tolist(), baseline_seed_means=baseline.mean(1).tolist(),
                trajectories=len(c), bootstrap_repeats=repeats,
                interval_scope='paired trajectories after averaging seeds; not training-seed uncertainty')


def compare_runs(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[(run['problem'], run['arm'])].append(run)
    comparisons = []
    for (problem, arm), candidate in groups.items():
        for (other_problem, baseline_arm), baseline in groups.items():
            if problem != other_problem or arm == baseline_arm:
                continue
            for split, multipliers in candidate[0]['splits'].items():
                for multiple, horizons in multipliers.items():
                    for horizon in horizons:
                        if horizon == 'ks_statistics':
                            continue
                        get = lambda run: run['splits'].get(split, {}).get(multiple, {}).get(horizon, {}).get('errors')
                        if any(get(run) is None for run in candidate+baseline):
                            continue
                        common = set(run['seed'] for run in candidate) & set(run['seed'] for run in baseline)
                        pairs = {str(seed): paired_bootstrap(
                            [get(next(run for run in candidate if run['seed'] == seed))],
                            [get(next(run for run in baseline if run['seed'] == seed))]) for seed in sorted(common)}
                        comparisons.append(dict(problem=problem, candidate=arm, baseline=baseline_arm,
                                                split=split, multiple=multiple, horizon=horizon,
                                                aggregate=paired_bootstrap([get(run) for run in candidate],
                                                                           [get(run) for run in baseline]),
                                                matching_seeds=pairs))
    return comparisons


def evaluate(manifest_path, holdout_root, destination, problem_name=None, timing=False):
    info, runs = device_info(), []
    manifest = read_frozen(manifest_path)
    selected = [run for run in manifest['runs'] if problem_name is None or run['problem'] == problem_name]
    if not selected:
        raise ValueError(f'Problem is absent from the frozen selection: {problem_name}')
    holdouts = {}
    for run in selected:
        name = run['problem']
        if name not in holdouts:
            path = Path(holdout_root)/f'{name}.pt'
            raw = torch.load(path, map_location='cpu', weights_only=False)
            if raw['manifest_sha256'] != sha256(manifest_path):
                raise ValueError(f'Holdout was generated for a different frozen selection: {path}')
            holdouts[name] = (raw, sha256(path))
        raw, data_hash = holdouts[name]
        model, result = load_model(run, raw['scale'])
        metrics = evaluate_run(model, raw, result)
        if timing:
            pair = raw['test']
            frames = pair[0].cuda()/raw['scale']
            mu = (pair[1].double()/model.problem.mu_ref).float().cuda()
            metrics['timing'] = benchmark(model, frames, mu)
        metrics.update(checkpoint_sha256=run['checkpoint_sha256'], holdout_sha256=data_hash,
                       generation_seeds=raw['seeds'], extra_ood=raw.get('extra_ood', []),
                       ood_coefficients=raw.get('ood_coefficients', list(raw['ood'])))
        runs.append(metrics)
        report = dict(device=info, manifest=manifest, runs=runs,
                      selection_problem=problem_name, expected_runs=len(selected),
                      total_frozen_runs=len(manifest['runs']),
                      timing_enabled=timing, timing_use='Post-selection inference measurement only',
                      complete_for_problem=problem_name is not None and len(runs) == len(selected),
                      complete=len(runs) == len(manifest['runs']))
        write_json(destination, report)
        print(f"Evaluated {name}/{run['arm']}/seed {run['seed']}", flush=True)
        del model
    report['comparisons'] = compare_runs(runs)
    write_json(destination, report)


def merge_reports(paths, destination):
    """Merge complete PDE reports only; never fill missing runs or average duplicate seeds."""
    if Path(destination).exists():
        raise FileExistsError(f'Refusing to replace a merged holdout report: {destination}')
    manifest, collected, holdouts, devices, timing_enabled = None, {}, {}, [], None
    identity = lambda run: (run['problem'], run['arm'], run['seed'])
    for path in paths:
        report = json.loads(Path(path).read_text())
        if not report.get('complete') and not report.get('complete_for_problem'):
            raise ValueError(f'Incomplete holdout report: {path}')
        measured = report.get('timing_enabled', False)
        if timing_enabled is not None and measured != timing_enabled:
            raise ValueError(f'Mismatched timing flags across PDE reports: {path}')
        timing_enabled = measured
        if manifest is None:
            manifest = report['manifest']
        elif manifest != report['manifest']:
            raise ValueError(f'Mismatched frozen manifests: {path}')
        expected = {identity(run): run for run in manifest['runs']}
        if len(expected) != len(manifest['runs']):
            raise ValueError('Duplicate frozen run identifiers')
        chosen_problem = report.get('selection_problem')
        selected = {key for key in expected if chosen_problem is None or key[0] == chosen_problem}
        if {identity(run) for run in report['runs']} != selected:
            raise ValueError(f'Missing or unexpected runs in completed report: {path}')
        if report.get('device', {}).get('requested') != 'cuda':
            raise ValueError(f'Holdout report lacks CUDA evaluation provenance: {path}')
        if report['device'] not in devices:
            devices.append(report['device'])
        for run in report['runs']:
            key = identity(run)
            if key in collected:
                raise ValueError(f'Duplicate evaluated run: {key}')
            if run.get('checkpoint_sha256') != expected[key]['checkpoint_sha256']:
                raise ValueError(f'Mismatched frozen checkpoint hash: {key}')
            if run['config'] != expected[key]['config']:
                raise ValueError(f'Mismatched frozen configuration: {key}')
            if measured and not run.get('timing'):
                raise ValueError(f'Missing requested inference timing: {key}')
            signature = (run.get('holdout_sha256'), run.get('generation_seeds'))
            if not all(signature):
                raise ValueError(f'Missing holdout provenance: {key}')
            if key[0] in holdouts and holdouts[key[0]] != signature:
                raise ValueError(f'Mismatched holdout data hashes or generation seeds: {key[0]}')
            holdouts[key[0]] = signature
            collected[key] = run
    if manifest is None:
        raise ValueError('Merge at least one holdout report')
    expected = {identity(run) for run in manifest['runs']}
    if set(collected) != expected:
        raise ValueError(f'Missing frozen runs: {sorted(expected-set(collected))}')
    runs = [collected[identity(run)] for run in manifest['runs']]
    merged = dict(device=devices[0], evaluator_devices=devices, manifest=manifest, runs=runs,
                  complete=True, total_frozen_runs=len(runs),
                  timing_enabled=timing_enabled, timing_use='Post-selection inference measurement only',
                  source_reports=[str(Path(path)) for path in paths], comparisons=compare_runs(runs))
    write_json(destination, merged)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze_parser = sub.add_parser('freeze')
    freeze_parser.add_argument('--runs', nargs='+', required=True, type=Path)
    freeze_parser.add_argument('--manifest', required=True, type=Path)
    gen = sub.add_parser('generate')
    gen.add_argument('--manifest', required=True, type=Path)
    gen.add_argument('--data-root', required=True, type=Path)
    gen.add_argument('--holdout-root', required=True, type=Path)
    gen.add_argument('--test-count', type=int, default=32)
    gen.add_argument('--ood-count', type=int, default=16)
    gen.add_argument('--seed', type=int, default=7103)
    gen.add_argument('--problem', choices=sorted(PROBLEMS))
    gen.add_argument('--extra-ood', nargs='+', type=float,
                     help='additional preregistered coefficients for one selected problem; original cases remain')
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--manifest', required=True, type=Path)
    evaluation.add_argument('--holdout-root', required=True, type=Path)
    evaluation.add_argument('--out', required=True, type=Path)
    evaluation.add_argument('--problem', choices=sorted(PROBLEMS))
    evaluation.add_argument('--timing', action='store_true', help='measure post-selection CUDA rollout costs')
    merger = sub.add_parser('merge')
    merger.add_argument('--reports', nargs='+', required=True, type=Path)
    merger.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'freeze':
        freeze(args.runs, args.manifest)
    elif args.command == 'generate':
        generate(args.manifest, args.data_root, args.holdout_root, args.test_count, args.ood_count, args.seed,
                 args.problem, args.extra_ood)
    elif args.command == 'evaluate':
        evaluate(args.manifest, args.holdout_root, args.out, args.problem, args.timing)
    else:
        merge_reports(args.reports, args.out)


if __name__ == '__main__':
    main()
