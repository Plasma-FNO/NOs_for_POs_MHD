"""Compare research runs with the full historical benchmark, preserving budget and seed counts.

python research_compare.py ROOT [ROOT ...] --out results/research_20261110
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def statistics(values):
    if not values or any(value is None for value in values):
        return None
    if any(not math.isfinite(value) for value in values):
        return dict(mean=float('inf'), std=float('inf'), n=len(values))
    return dict(mean=mean(values), std=stdev(values) if len(values) > 1 else 0., n=len(values))


def nested(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def aggregate(roots, baseline):
    groups = defaultdict(list)
    seen = set()
    for root in roots:
        for path in sorted((Path(root)/'runs').glob('*/*/result.json')):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            result = json.loads(path.read_text())
            # Diagnostic switches do not change training or the learned step.
            settings = {key: value for key, value in result['config'].items()
                        if key not in ('operator_diagnostics', 'skip_jacobian')}
            settings.setdefault('lr_min', 1e-5)
            if 'scheme' in settings:
                settings.setdefault('max_step', False)
                settings.setdefault('zero_equilibrium', False)
                settings.setdefault('closure_in_range', False)
            config = json.dumps(settings, sort_keys=True)
            key = (result['problem'], result['arm'], config)
            if any(old['seed'] == result['seed'] for old in groups[key]):
                raise ValueError(f'Duplicate seed/configuration: {path}')
            groups[key].append(dict(result, source=str(path)))
    output = {}
    for (problem, arm, config), runs in groups.items():
        historical = baseline[problem]
        horizon = historical['horizons'][-1]
        old = historical['arms']
        best_arm = min(old, key=lambda name: old[name]['test'][horizon]['mean'])
        best = old[best_arm]['test'][horizon]['mean']
        data_arms = [name for name in old if name == 'ar' or name.startswith('node')]
        data_arm = min(data_arms, key=lambda name: old[name]['test'][horizon]['mean'])
        collect = lambda *keys: statistics([nested(run, *keys) for run in runs])
        primary = collect('eval', 'test', horizon, 'mean_clipped')
        if primary is None:
            raise ValueError(f'{problem}/{arm} has no common primary horizon {horizon}')
        transfer = {}
        for key in ('x2/1', 'x4/1', 'x1/2'):
            common = [h for h in historical['horizons'] if all(
                nested(run, 'eval', 'dt_sweep', key, h, 'mean_clipped') is not None for run in runs)]
            if common:
                transfer[key] = dict(horizon=common[-1], error=collect(
                    'eval', 'dt_sweep', key, common[-1], 'mean_clipped'))
        entry = dict(arm=arm, seeds=sorted(run['seed'] for run in runs), config=json.loads(config),
                     source_configs=[run['config'] for run in runs],
                     sources=[run['source'] for run in runs],
                     statuses=[run['train']['status'] for run in runs],
                     parameters=runs[0]['parameters'], primary=primary,
                     historical_ratio=primary['mean']/best,
                     long_bounded=collect('eval', 'long', 'bounded'), transfer=transfer,
                     ood={mu: collect('eval', 'ood', mu, horizon, 'mean_clipped') for mu in historical['ood']},
                     train_seconds=collect('train', 'seconds'),
                     training=[run.get('train', {}) for run in runs],
                     seconds_per_step=collect('timing', 'seconds_per_step'),
                     timing=[run.get('timing', {}) for run in runs],
                     devices=[run.get('device', {}) for run in runs],
                     best_iterations=[run['train']['best_iteration'] for run in runs],
                     provenance=[run.get('provenance', {}) for run in runs],
                     ks_spectrum=collect('eval', 'ks_statistics', 'log10_error'),
                     ks_energy=collect('eval', 'ks_statistics', 'energy_ratio'))
        if problem not in output:
            output[problem] = dict(horizon=horizon,
                                   historical_best=dict(arm=best_arm, error=old[best_arm]['test'][horizon]),
                                   historical_data_best=dict(arm=data_arm, error=old[data_arm]['test'][horizon]),
                                   runs=[])
        output[problem]['runs'].append(entry)
    for problem in output.values():
        problem['runs'].sort(key=lambda run: run['primary']['mean'])
    return output


def format_stat(value):
    if value is None:
        return '—'
    if not math.isfinite(value['mean']):
        return 'nonfinite'
    return f"{value['mean']:.4g}" + (f" ± {value['std']:.2g}" if value['n'] > 1 else '')


def markdown(summary):
    lines = ['# Generic OpsSplit research comparison', '',
             'Primary errors are mean trajectory relative L2, clipped at 2. Smaller ratios are better. '
             'Ratios use the strongest historical method at the same horizon. Historical comparisons '
             'are exploratory: adaptive use of the old test set requires a fresh holdout before confirmation.', '',
             'A zero-parameter method is a classical solver baseline. Its improvement is numerical, not a learned gain. '
             'Different iteration counts or solver substep counts are different budgets. '
             'One-seed pilots are not confirmed winners.', '']
    lines += ['Training wall times come from concurrently shared GPU runs and resumable worker chunks. '
              'They are descriptive, not controlled speed comparisons. Use the final idle-GPU holdout '
              'timings to compare inference costs; timing-completeness annotations remain in the JSON.', '']
    for name, problem in summary.items():
        best, data = problem['historical_best'], problem['historical_data_best']
        lines += [f"## {name}, H={problem['horizon']}", '',
                  f"Historical best: `{best['arm']}` ({format_stat(best['error'])}); "
                  f"data-only best: `{data['arm']}` ({format_stat(data['error'])}).", '']
        if name == 'ks':
            lines += ['H=50 is within the 100-step training time span. It tests unseen trajectories, '
                      'not extrapolation beyond the training time span.', '']
        lines += ['| Method | Scheme / substeps / closure | Seeds | Iterations | Parameters | Primary error | / historical best | Long bounded | Train seconds | Seconds / step |',
                  '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for run in problem['runs']:
            settings = ' / '.join(str(run['config'].get(key, '—')) for key in ('scheme', 'substeps', 'closure'))
            lines.append(f"| {run['arm']} | {settings} | {','.join(map(str, run['seeds']))} | {run['config']['iters']} | "
                         f"{run['parameters']} | {format_stat(run['primary'])} | {run['historical_ratio']:.3g} | "
                         f"{format_stat(run['long_bounded'])} | {format_stat(run['train_seconds'])} | "
                         f"{format_stat(run['seconds_per_step'])} |")
        mus = list(problem['runs'][0]['ood'])
        lines += ['', '| Method | 2h | 4h | h/2 | ' + ' | '.join('OOD '+mu for mu in mus) + (' |' if mus else ''),
                  '|---|---:|---:|---:|' + '---:|'*len(mus)]
        for run in problem['runs']:
            cells = []
            for key in ('x2/1', 'x4/1', 'x1/2'):
                item = run['transfer'].get(key)
                cells.append(f"{format_stat(item['error'])} (H={item['horizon']})" if item else '—')
            cells.extend(format_stat(run['ood'][mu]) for mu in mus)
            lines.append(f"| {run['arm']} | " + ' | '.join(cells) + ' |')
        lines += ['', 'Configuration and provenance for each row are retained in the accompanying JSON.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--baseline', type=Path, default=Path(__file__).parent/'results'/'summary.json')
    parser.add_argument('--out', type=Path, required=True, help='output stem (.json and .md are added)')
    args = parser.parse_args()
    summary = aggregate(args.roots, json.loads(args.baseline.read_text()))
    if not summary:
        parser.error('No result.json files found under ROOT/runs/PROBLEM/RUN/')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    args.out.with_suffix('.md').write_text(markdown(summary)+'\n')
    print(args.out.with_suffix('.md'))


if __name__ == '__main__':
    main()
