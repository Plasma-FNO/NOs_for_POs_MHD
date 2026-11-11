"""Plot preselected methods on the independent holdout; never rank/select using its errors.

python research_plots.py --report fresh_holdout_results.json --roles selected_roles.json \
    --out results/research_20261110.png
python research_plots.py --self-check

The roles file maps each problem to {readable_label: exact_arm_name}. Freeze this selection
before generating the holdout. Error bars show variation across training seeds, not a confidence
interval over held-out trajectories.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


PROBLEMS = {'burgers': 'Burgers', 'kdv': 'KdV', 'ks': 'Kuramoto–Sivashinsky',
            'ns2d': 'Navier–Stokes 2D', 'burgers_fem': 'Burgers FEM'}


def prepare(report, roles):
    if not report.get('complete'):
        raise ValueError('Plot only a completed independent holdout evaluation')
    if set(roles) != set(PROBLEMS):
        raise ValueError('Roles must explicitly select methods for all five problems')
    panels = {}
    for problem, labels in roles.items():
        if not labels or len(set(labels.values())) != len(labels):
            raise ValueError(f'{problem}: select at least one method without duplicate arm aliases')
        rows, horizon, samples = [], None, None
        for label, arm in labels.items():
            runs = [run for run in report['runs'] if run['problem'] == problem and run['arm'] == arm]
            expected = sorted(run['seed'] for run in report['manifest']['runs']
                              if run['problem'] == problem and run['arm'] == arm)
            actual = sorted(run['seed'] for run in runs)
            if not expected or actual != expected or len(set(actual)) != len(actual):
                raise ValueError(f'{problem}/{arm}: missing, duplicate, or unfrozen training seeds')
            values = []
            for run in runs:
                h = run['primary_horizon']
                metric = run['splits']['test']['x1'][str(h)]
                n = len(metric['errors'])
                if (horizon is not None and h != horizon) or (samples is not None and n != samples):
                    raise ValueError(f'{problem}: methods must use the same horizon and held-out samples')
                horizon, samples = h, n
                value = metric['mean_clipped']
                if not math.isfinite(value) or not 0 <= value <= 2:
                    raise ValueError(f'{problem}/{arm}: invalid clipped primary error')
                values.append(value)
            average = mean(values)
            if average == 0:
                raise ValueError(f'{problem}/{arm}: zero error cannot be represented on a logarithmic axis')
            rows.append(dict(label=label, arm=arm, mean=average,
                             std=stdev(values) if len(values) > 1 else 0., seeds=actual))
        panels[problem] = dict(horizon=horizon, samples=samples, methods=rows)
    return panels


def plot(panels, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(row['label'] for panel in panels.values() for row in panel['methods']))
    colors = {label: plt.get_cmap('tab10')(i % 10) for i, label in enumerate(labels)}
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout='constrained')
    truncated = False
    for axis, (problem, title) in zip(axes.flat, PROBLEMS.items()):
        panel = panels[problem]
        rows = panel['methods']
        lower = min(row['mean'] for row in rows)/4
        upper = max(row['mean']+row['std'] for row in rows)*2
        for x, row in enumerate(rows):
            below = min(row['std'], row['mean']-lower)
            truncated |= below < row['std']
            axis.errorbar(x, row['mean'], yerr=[[below], [row['std']]], fmt='o',
                          color=colors[row['label']], markersize=7, capsize=4,
                          elinewidth=1.4, markeredgecolor='white', markeredgewidth=.7)
        axis.set_yscale('log')
        axis.set_ylim(lower, upper)
        axis.set_xlim(-.55, len(rows)-.45)
        axis.set_xticks(range(len(rows)),
                        [f"{row['label']}\n(n={len(row['seeds'])})" for row in rows],
                        rotation=25, ha='right', fontsize=9)
        axis.set_title(f"{title} · H={panel['horizon']} · {panel['samples']} trajectories", fontsize=11)
        axis.set_ylabel('Relative L2 error, clipped at 2')
        axis.grid(axis='y', which='major', color='.86', linewidth=.7)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    note = axes.flat[-1]
    note.axis('off')
    explanation = (
        'How to read this figure\n\n'
        'Methods were selected before this holdout.\n'
        'Dots: mean error across training seeds.\n'
        'Bars: sample SD across training seeds.\n'
        'n: number of training seeds evaluated.\n'
        'One classical run has zero seed error bar.\n\n'
        'Internal solver stages and precision differ.\n'
        'This is not an equal-runtime comparison.\n'
        'Compare learned methods with their matched\n'
        'classical solver to isolate the value of learning.\n\n'
        'KS H=50 is within the training time span.\n'
        'Later KS statistics, coefficient transfer, and\n'
        'long-rollout boundedness are in the report.'
    )
    if truncated:
        explanation += '\n\nLower SD bars stop at the positive axis limit.'
    note.text(.02, .97, explanation, va='top', fontsize=10, linespacing=1.35)
    figure.suptitle('Independent holdout · preselected method comparisons', fontsize=16)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220, metadata={'Title': 'Generic OpsSplit independent holdout'})
    plt.close(figure)


def self_check():
    # Synthetic values exercise parsing only; no fabricated figure or result file is generated.
    roles = {problem: {'Selected method': 'selected'} for problem in PROBLEMS}
    report = dict(complete=True, runs=[], manifest=dict(runs=[]))
    for problem in PROBLEMS:
        for seed, error in enumerate((.1, .2, .3)):
            identity = dict(problem=problem, arm='selected', seed=seed)
            report['manifest']['runs'].append(identity)
            report['runs'].append(dict(identity, primary_horizon=50, splits={
                'test': {'x1': {'50': dict(errors=[error, error], mean_clipped=error)}}}))
    panels = prepare(report, roles)
    row = panels['burgers']['methods'][0]
    assert math.isclose(row['mean'], .2) and math.isclose(row['std'], .1)
    assert row['seeds'] == [0, 1, 2]
    report['runs'].pop()
    try:
        prepare(report, roles)
    except ValueError as error:
        assert 'seeds' in str(error)
    else:
        raise AssertionError('Incomplete seed sets must fail')
    print('Plot-data self-check passed; no figure generated.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--roles', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not all((args.report, args.roles, args.out)):
        parser.error('--report, --roles, and --out are required')
    panels = prepare(json.loads(args.report.read_text()), json.loads(args.roles.read_text()))
    plot(panels, args.out)
    print(args.out)


if __name__ == '__main__':
    main()
