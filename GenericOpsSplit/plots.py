"""Figures for the generic implicit OpsSplit testbed.

  python plots.py ROOT     -> ROOT/summary/fig_error_vs_horizon.png, fig_operator_recovery.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gos.models import StepModel  # noqa: E402
from gos.problems import get_problem  # noqa: E402
from gos.train import CONFIG  # noqa: E402

# Fixed entity -> categorical slot (validated default palette, light mode); never re-ordered by rank.
COLORS = dict(ar='#2a78d6', node='#eb6834', explicit_known='#1baf7a', imex_exact='#eda100',
              implicit_known='#e87ba4', implicit_known_cn='#008300', implicit_struct='#6250d6', implicit_free='#e34948')
MARKERS = dict(ar='o', node='s', explicit_known='^', imex_exact='v', implicit_known='D', implicit_known_cn='P',
               implicit_struct='*', implicit_free='X')
TITLES = dict(burgers='Burgers (Fourier)', kdv='KdV', ks='Kuramoto-Sivashinsky', ns2d='2D Navier-Stokes',
              burgers_fem='Burgers, Dirichlet FEM')
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def error_vs_horizon(summary, out):
    problems = list(summary)
    fig, axes = plt.subplots(1, len(problems), figsize=(3.3*len(problems), 3.2), constrained_layout=True)
    for ax, p in zip(axes, problems):
        hs = [int(h) for h in summary[p]['horizons']]
        for arm, row in summary[p]['arms'].items():
            ys = [row['test'][str(h)]['mean'] if row['test'][str(h)] else float('nan') for h in hs]
            ys = [min(y, 2.) for y in ys]
            ax.plot(hs, ys, color=COLORS[arm], marker=MARKERS[arm], ms=6, lw=1.6, label=arm)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(top=2.5)
        ax.set_xticks(hs)
        ax.set_xticklabels([str(h) for h in hs])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.axvline(hs[-2], color=MUTED, lw=0.8, ls=':')
        ax.set_title(TITLES[p], fontsize=10, color=INK)
        ax.set_xlabel('rollout steps (dotted: training horizon)', fontsize=8, color=MUTED)
        style(ax)
    axes[0].set_ylabel('test relative L2 error (clipped at 2)', fontsize=8, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=8, fontsize=8, frameon=False)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def load_model(root, problem_name, arm, seed=0):
    path = Path(root)/'runs'/problem_name/f'{arm}_s{seed}'/'model.pt'
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    problem = get_problem(problem_name, 'cpu')
    model = StepModel(problem, ckpt['arm'], CONFIG[problem_name]['net'], ckpt['scale'], 'cpu')
    model.load_state_dict(ckpt['state'])
    return model


OPERATOR_SERIES = [('implicit_known', '-', 'true L'), ('implicit_struct', '-', 'struct'), ('implicit_struct_tf', '--', 'struct, tangent-free N'),
                   ('implicit_free', '-', 'free'), ('implicit_free_tf', '--', 'free, tangent-free N')]


def operator_recovery(root, out):
    panels = [('burgers', 'real', 'Burgers: Re hλ(k)'), ('kdv', 'real', 'KdV: Re hλ(k)'), ('kdv', 'imag', 'KdV: Im hλ(k)'),
              ('ks', 'real', 'KS: Re hλ(k)'), ('burgers_fem', 'conductance', 'FEM: conductance h·c/Δx'),
              ('burgers_fem', 'skew', 'FEM: skew h·s/Δx')]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), constrained_layout=True)
    for ax, (p, part, title) in zip(axes.flat, panels):
        problem = get_problem(p, 'cpu')
        for arm, ls, label in OPERATOR_SERIES:
            model = load_model(root, p, arm)
            if model is None:
                continue
            lin = model.linear
            color = COLORS[arm.replace('_tf', '')]
            with torch.no_grad():
                if problem.space == 'fourier':
                    keep = lin.grid.mask & (lin.grid.ks[0] > 0)
                    k = lin.grid.ks[0][keep]
                    s = (lin.symbol(torch.ones(1))[0]*problem.h)[keep].to(torch.complex128)
                    ax.plot(k, s.real if part == 'real' else s.imag, color=color, ls=ls, lw=1.5, label=label)
                    if arm == 'implicit_known':  # operator whose backward-Euler resolvent equals exp(h L)
                        eff = 1 - torch.exp(-s)
                        y = eff.real if part == 'real' else eff.imag
                        y = torch.where(eff.abs() <= 50, y, torch.full_like(y, float('nan')))
                        ax.plot(k, y, color=MUTED, ls=':', lw=1.5, label='BE-equivalent of exp(hL), |hλ|≤50')
                    ax.set_xlabel('wavenumber k', fontsize=8, color=MUTED)
                    ax.set_yscale('symlog', linthresh=1e-2)
                else:
                    lower, diag, upper = lin.bands(torch.ones(1))
                    unit = lin.grid.dx/problem.h
                    y = 0.5*(upper[0, :-1] + lower[0, 1:]) if part == 'conductance' else 0.5*(upper[0, :-1] - lower[0, 1:])
                    ax.plot(lin.grid.x[:-1] + lin.grid.dx/2, y/unit, color=color, ls=ls, lw=1.5, label=label)
                    ax.set_xlabel('x', fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=10, color=INK)
        style(ax)
    handles, labels = [], []
    for ax in axes.flat:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in labels:
                handles.append(h_)
                labels.append(l_)
    fig.legend(handles, labels, loc='outside lower center', ncol=6, fontsize=8, frameon=False)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def step_transfer(summary, out):
    """Error at the last common horizon against the step multiple used at test time."""
    problems = list(summary)
    fig, axes = plt.subplots(1, len(problems), figsize=(3.3*len(problems), 3.2), constrained_layout=True)
    for ax, p in zip(axes, problems):
        arms = summary[p]['arms']
        H = summary[p]['horizons'][-1]
        for base in ('node', 'implicit_known', 'implicit_struct', 'implicit_free'):
            for suffix, ls in (('', '-'), ('_mdt', '--')):
                row = arms.get(base + suffix)
                if row is None:
                    continue
                pts = [(0.5, row['dt_sweep'].get('x1/2')), (1, dict(error=row['test'][H], horizon=H)),
                       (2, row['dt_sweep'].get('x2/1')), (4, row['dt_sweep'].get('x4/1'))]
                xs = [x for x, v in pts if v and v['error'] and v['horizon'] == H]
                ys = [min(v['error']['mean'], 2.) for x, v in pts if v and v['error'] and v['horizon'] == H]
                ax.plot(xs, ys, color=COLORS[base], ls=ls, marker=MARKERS[base], ms=6, lw=1.5,
                        label=base + (' (trained on h,2h,4h)' if suffix else ''))
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_title(f'{TITLES[p]} (H={H})', fontsize=10, color=INK)
        ax.set_xlabel('test step / training step h', fontsize=8, color=MUTED)
        style(ax)
    axes[0].set_ylabel('test relative L2 error (clipped at 2)', fontsize=8, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=8, frameon=False)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    root = Path(sys.argv[1])
    summary = json.loads((root/'summary'/'summary.json').read_text())
    base = {p: dict(v, arms={a: r for a, r in v['arms'].items() if a in COLORS}) for p, v in summary.items()}
    error_vs_horizon(base, root/'summary'/'fig_error_vs_horizon.png')
    step_transfer(summary, root/'summary'/'fig_step_transfer.png')
    operator_recovery(root, root/'summary'/'fig_operator_recovery.png')
    print('wrote figures to', root/'summary')


if __name__ == '__main__':
    main()
