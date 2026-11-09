"""Aggregate result.json files over seeds into one JSON and markdown tables.

  python summarize.py ROOT [--out ROOT/summary]
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ARM_ORDER = ['ar', 'node', 'explicit_known', 'imex_exact', 'implicit_known', 'implicit_known_cn', 'implicit_struct',
             'implicit_free', 'implicit_struct_tf', 'implicit_free_tf', 'node_mdt', 'implicit_known_mdt',
             'implicit_struct_mdt', 'implicit_free_mdt']
PROBLEM_ORDER = ['burgers', 'kdv', 'ks', 'ns2d', 'burgers_fem']


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < len(values):
        return dict(mean=float('inf'), std=0., n=len(values), nonfinite=len(values)-len(finite))
    m = sum(finite)/len(finite)
    s = (sum((v-m)**2 for v in finite)/max(len(finite)-1, 1))**0.5
    return dict(mean=m, std=s, n=len(values))


def fmt(st, digits=3):
    if st is None:
        return '–'
    if not math.isfinite(st['mean']):
        return 'diverged' if st.get('nonfinite') == st['n'] else f'inf ({st["nonfinite"]}/{st["n"]})'
    if st['n'] > 1:
        return f'{st["mean"]:.{digits}g} ± {st["std"]:.1g}'
    return f'{st["mean"]:.{digits}g}'


def get(d, *keys):
    for k in keys:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--out')
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out or root/'summary')
    out.mkdir(parents=True, exist_ok=True)
    runs = defaultdict(lambda: defaultdict(list))
    for f in sorted((root/'runs').glob('*/*/result.json')):
        r = json.loads(f.read_text())
        runs[r['problem']][r['arm']].append(r)
    summary, md = {}, []
    for problem in [p for p in PROBLEM_ORDER if p in runs]:
        arms = [a for a in ARM_ORDER if a in runs[problem]] + sorted(a for a in runs[problem] if a not in ARM_ORDER)
        any_run = runs[problem][arms[0]][0]
        horizons = [str(h) for h in any_run['config']['horizons']]
        oods = sorted(any_run['eval']['ood'], key=float)
        table = {}
        for arm in arms:
            rs = runs[problem][arm]
            row = dict(seeds=len(rs), status=[r['train']['status'] for r in rs], parameters=rs[0]['parameters'])
            row['test'] = {H: stats([get(r, 'eval', 'test', H, 'mean_clipped') for r in rs]) for H in horizons}
            row['test_bounded_last'] = stats([get(r, 'eval', 'test', horizons[-1], 'bounded') for r in rs])
            row['long_bounded'] = stats([get(r, 'eval', 'long', 'bounded') for r in rs])
            row['jacobian_sigma_max'] = stats([get(r, 'eval', 'jacobian_sigma', 'max') for r in rs])
            row['dt_sweep'] = {}
            for key in ('x2/1', 'x4/1', 'x1/2'):
                last = None
                for H in reversed(horizons):
                    if all(get(r, 'eval', 'dt_sweep', key, H) for r in rs):
                        last = H
                        break
                if last:
                    row['dt_sweep'][key] = dict(horizon=last, error=stats([get(r, 'eval', 'dt_sweep', key, last, 'mean_clipped') for r in rs]))
            row['ood'] = {v: {H: stats([get(r, 'eval', 'ood', v, H, 'mean_clipped') for r in rs]) for H in horizons} for v in oods}
            if problem == 'ks':
                row['ks_spectrum'] = stats([get(r, 'eval', 'ks_statistics', 'log10_error') for r in rs])
                row['ks_energy'] = stats([get(r, 'eval', 'ks_statistics', 'energy_ratio') for r in rs])
            op = [get(r, 'eval', 'operator') for r in rs]
            if op[0]:
                row['operator'] = dict(
                    h_max_re={str(g['mu_t']): stats([o['growth'][i]['h_max_re'] for o in op]) for i, g in enumerate(op[0]['growth'])},
                    recovery={k: stats([o['recovery'].get(k) for o in op]) for k in op[0]['recovery']},
                    propagator={k: stats([get(o, 'propagator', k) for o in op]) for k in (op[0].get('propagator') or {})})
            table[arm] = row
        summary[problem] = dict(horizons=horizons, ood=oods, arms=table)

        md.append(f'## {problem}\n')
        md.append('Test relative L2 error (mean over trajectories, clipped at 2) by rollout horizon; mean ± std over seeds.\n')
        md.append('| arm | ' + ' | '.join(f'H={H}' for H in horizons) + ' | bounded@long | σ_max(DΦ) |')
        md.append('|---|' + '---|'*(len(horizons)+2))
        for arm, row in table.items():
            md.append(f'| {arm} | ' + ' | '.join(fmt(row['test'][H]) for H in horizons)
                      + f' | {fmt(row["long_bounded"], 2)} | {fmt(row["jacobian_sigma_max"])} |')
        md.append('')
        md.append('Step-size transfer (trained at h; error at the longest horizon reachable):\n')
        md.append('| arm | 2h | 4h | h/2 |')
        md.append('|---|---|---|---|')
        for arm, row in table.items():
            if row['dt_sweep']:
                cells = [f'{fmt(row["dt_sweep"][k]["error"])} (H={row["dt_sweep"][k]["horizon"]})' if k in row['dt_sweep'] else '–'
                         for k in ('x2/1', 'x4/1', 'x1/2')]
                md.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
        md.append('')
        if oods:
            md.append(f'OOD coefficient (error at H={horizons[1]} / H={horizons[-1]}):\n')
            md.append('| arm | ' + ' | '.join(oods) + ' |')
            md.append('|---|' + '---|'*len(oods))
            for arm, row in table.items():
                md.append(f'| {arm} | ' + ' | '.join(f'{fmt(row["ood"][v][horizons[1]])} / {fmt(row["ood"][v][horizons[-1]])}' for v in oods) + ' |')
            md.append('')
        if problem == 'ks':
            md.append('KS long-time statistics, steps 50-200 (resolved modes within 6 decades of the peak):\n')
            md.append('| arm | spectrum mean abs log10 ratio | energy ratio |')
            md.append('|---|---|---|')
            for arm, row in table.items():
                md.append(f'| {arm} | {fmt(row["ks_spectrum"])} | {fmt(row["ks_energy"])} |')
            md.append('')
        ops = {a: r['operator'] for a, r in table.items() if 'operator' in r}
        if ops:
            md.append('Implicit operator: h·max Re λ (growth bound) at mu_t = 1, and relative recovery error vs the true operator:\n')
            keys = sorted({k for o in ops.values() for k in o['recovery']})
            md.append('| arm | h·maxRe λ (mu_t=1) | ' + ' | '.join(f'rec {k}' for k in keys) + ' | propagator err (all / low) |')
            md.append('|---|---|' + '---|'*len(keys) + '---|')
            for arm, o in ops.items():
                prop = ' / '.join(fmt(o['propagator'].get(b)) for b in ('all', 'low') if b in o.get('propagator', {}))
                md.append(f'| {arm} | {fmt(o["h_max_re"].get("1.0"))} | ' + ' | '.join(fmt(o['recovery'].get(k)) for k in keys) + f' | {prop or "–"} |')
            md.append('')
    (out/'summary.json').write_text(json.dumps(summary, indent=1))
    (out/'summary.md').write_text('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
