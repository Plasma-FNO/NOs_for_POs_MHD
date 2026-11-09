#!/usr/bin/env python3
"""Freeze full-study evidence and build a verified (at most two-page) ES-to-EM report."""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-transfer-20261104/full-study')
MANIFEST = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/data/jorek_zenodo/full-manifest.json')
_spec = importlib.util.spec_from_file_location('mastu_page_helpers', HERE.parent/'results_one_page/build.py')
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
sha256, page_count = _helpers.sha256, _helpers.page_count
MODES = {'zero-shot': 'Zero-shot', 'frozen': 'Frozen transfer', 'finetune': 'Fine-tuned',
         'scratch': 'Scratch split', 'monolithic': 'Monolithic'}
PAGE_LIMIT = 2


def freeze_evidence():
    manifest = json.loads(MANIFEST.read_text())
    rows, evidence = [], {str(MANIFEST): sha256(MANIFEST)}
    expected_test = sorted(e['id'] for e in manifest['entries'] if e['case'] == 'em' and e['split'] == 'test')
    for seed in range(3):
        for mode in MODES:
            for budget in ([0] if mode == 'zero-shot' else [8, 32, 64]):
                name = mode if not budget else f'{mode}-{budget}'
                directory = RUNS/f'seed-{seed}'/name
                path = directory/'test_metrics.json'
                metrics = json.loads(path.read_text())
                provenance_path = directory/'provenance.json'
                provenance = json.loads(provenance_path.read_text())
                if provenance['manifest_sha256'] != sha256(MANIFEST) or sorted(provenance['test_ids']) != expected_test:
                    raise ValueError(f'Manifest/test set mismatch: {directory}')
                if sorted(t['id'] for t in metrics['trajectories']) != expected_test:
                    raise ValueError(f'Missing test outcomes: {directory}')
                if metrics['fields'][:4] != ['rho', 'T', 'u', 'omega'] or metrics['time_basis'] != 'saved_frame_index':
                    raise ValueError('Unexpected field order or time basis')
                if metrics['completed'] != metrics['total'] or any(len(t['relative_l2_by_frame']) != 100 for t in metrics['trajectories']):
                    raise ValueError('This complete-rollout page requires all outcomes; revise the reporting contract for failures')
                curves = np.asarray([t['relative_l2_by_frame'] for t in metrics['trajectories']])[:, :, :4]
                persistence = np.asarray([t['persistence_relative_l2_by_frame'] for t in metrics['trajectories']])[:, :, :4]
                if not np.isfinite(curves).all():
                    raise ValueError('Nonfinite error values')
                # Training wall time: elapsed seconds at the end of the last epoch (includes per-epoch
                # validation, excludes data statistics and the 100-step test rollout); zero-shot trains nothing.
                history_path = directory/'history.jsonl'
                train_seconds = 0.
                if mode != 'zero-shot':
                    train_seconds = json.loads(history_path.read_text().splitlines()[-1])['elapsed_seconds']
                    evidence[str(history_path)] = sha256(history_path)
                rows.append(dict(seed=seed, mode=mode, budget=budget, completed=metrics['completed'], train_seconds=train_seconds,
                                 total=metrics['total'], mean_curve=curves.mean(axis=(0, 2)).tolist(),
                                 persistence=persistence.mean(axis=(0, 2)).tolist(),
                                 parameters=provenance['parameters'], trainable_parameters=provenance['trainable_parameters'],
                                 epochs=provenance['epochs'], steps_per_epoch=provenance['steps_per_epoch'],
                                 horizon=provenance['horizon'], lr=provenance['lr']))
                evidence[str(path)] = sha256(path)
                evidence[str(provenance_path)] = sha256(provenance_path)
    source_seconds = []
    for seed in range(3):
        path = RUNS/f'seed-{seed}'/'source'/'history.jsonl'
        source_seconds.append(json.loads(path.read_text().splitlines()[-1])['elapsed_seconds'])
        evidence[str(path)] = sha256(path)
    snapshot = dict(schema=1, captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    source_train_seconds=source_seconds, hardware='one NVIDIA H100 GPU and 4 CPUs (boost_fua_prod)',
                    scope='full_data_effective_split_transfer', counts=manifest['counts'], seeds=[0, 1, 2],
                    budgets=[8, 32, 64], metric='mean per-trajectory/per-shared-field R-weighted relative L2; sample SD across seeds',
                    rows=rows, evidence_sha256=evidence)
    temporary = HERE/'results.json.part'
    temporary.write_text(json.dumps(snapshot, indent=2, allow_nan=False)+'\n')
    temporary.replace(HERE/'results.json')
    return snapshot


def load_evidence():
    snapshot = json.loads((HERE/'results.json').read_text())
    if snapshot['schema'] != 1 or snapshot['scope'] != 'full_data_effective_split_transfer' or len(snapshot['rows']) != 39:
        raise ValueError('Expected the full three-seed transfer snapshot')
    for path, expected in snapshot['evidence_sha256'].items():
        if sha256(path) != expected:
            raise ValueError(f'Evidence changed: {path}; explicitly refresh the snapshot')
    return snapshot


def curves(snapshot, mode, budget):
    rows = [row for row in snapshot['rows'] if row['mode'] == mode and row['budget'] == budget]
    if sorted(row['seed'] for row in rows) != [0, 1, 2]:
        raise ValueError('Every comparison must contain the same three seeds')
    return np.asarray([row['mean_curve'] for row in rows])


def make_values(work, snapshot):
    def stats(mode, budget, horizon):
        x = curves(snapshot, mode, budget)[:, horizon-1]
        return f'{x.mean():.3f} \\pm {x.std(ddof=1):.3f}'
    rows = []
    for mode in ('scratch', 'monolithic', 'frozen', 'finetune'):
        rows.append(('Frozen' if mode == 'frozen' else MODES[mode])+' & '+' & '.join('$'+stats(mode, n, 10)+'$' for n in (8, 32, 64))+r' \\')
    persistence = snapshot['rows'][0]['persistence']
    worst = max(row['mean_curve'][99] for row in snapshot['rows'] if row['mode'] == 'finetune' and row['budget'] == 8)

    def setting(key, modes=tuple(MODES)):
        # Method-table values come from run provenance and must agree across every seed and budget.
        found = {row[key] for row in snapshot['rows'] if row['mode'] in modes}
        if len(found) != 1:
            raise ValueError(f'Inconsistent {key} across {modes}: {sorted(found)}')
        return found.pop()

    def scientific(x):
        mantissa, exponent = f'{x:.0e}'.split('e')
        return ('' if mantissa == '1' else mantissa+r'\times ')+f'10^{{{int(exponent)}}}'
    trained = tuple(mode for mode in MODES if mode != 'zero-shot')
    epochs, steps = setting('epochs', trained), setting('steps_per_epoch', trained)
    method = {'Horizon': str(setting('horizon', trained)), 'Epochs': str(epochs),
              'StepsPerEpoch': str(steps), 'Updates': f'{epochs*steps:,}',
              # The ES source has no row of its own: its parameters are the frozen arm's untrainable part.
              'TrainSource': f"{setting('parameters', ('frozen',)) - setting('trainable_parameters', ('frozen',)):,}"}
    if setting('trainable_parameters', ('zero-shot',)) != 0:
        raise ValueError('Zero-shot must not train any parameters')
    if 'source_train_seconds' not in snapshot:
        raise ValueError('Snapshot lacks training times; run build.py --refresh')
    method['SourceMinutes'] = f"{np.mean(snapshot['source_train_seconds'])/60:.1f}"
    for mode in MODES:
        name = ''.join(part.capitalize() for part in mode.split('-'))
        minutes = [row['train_seconds']/60 for row in snapshot['rows'] if row['mode'] == mode]
        method['Minutes'+name] = f'{np.mean(minutes):.1f}'
        method['Train'+name] = f"{setting('trainable_parameters', (mode,)):,}"
        method['Total'+name] = f"{setting('parameters', (mode,)):,}"
        if mode != 'zero-shot':
            method['Lr'+name] = scientific(setting('lr', (mode,)))
    values = {**method, 'ResultsRows':'\n'.join(rows), 'ZeroTen': stats('zero-shot', 0, 10),
              'ZeroHundred': stats('zero-shot', 0, 100), 'PersistenceTen':f'{persistence[9]:.3f}',
              'PersistenceHundred':f'{persistence[99]:.3f}', 'WorstEight':f'{worst:.2e}',
              'SnapshotTime':snapshot['captured_utc'][:16].replace('T', ' ')+' UTC'}
    (work/'experiment_values.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in values.items())+'\n')


def make_figure(path, snapshot):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif', 'font.serif':['cmr10'], 'axes.unicode_minus':False,
                         'axes.formatter.use_mathtext':True, 'font.size':9, 'pdf.fonttype':42, 'mathtext.fontset':'cm'})
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 1.8))
    colors = {'scratch':'#c73c3c','monolithic':'#6b52a2','frozen':'#df7a23','finetune':'#238b45','zero-shot':'#287ab3'}
    for mode in ('scratch', 'monolithic', 'frozen', 'finetune'):
        data = np.array([curves(snapshot, mode, n)[:, 9] for n in (8, 32, 64)])
        axes[0].errorbar([8,32,64], data.mean(axis=1), yerr=data.std(axis=1, ddof=1),
                         color=colors[mode], marker='o', ms=2.5, capsize=2, lw=1, label=MODES[mode])
    for mode in MODES:
        data = curves(snapshot, mode, 0 if mode == 'zero-shot' else 64)
        axes[1].plot(np.arange(1,101), data.mean(axis=0), color=colors[mode], lw=1, label=MODES[mode])
    baseline = snapshot['rows'][0]['persistence']
    axes[0].axhline(baseline[9], color='black', ls='--', lw=.8)
    axes[1].plot(np.arange(1,101), baseline, color='black', ls='--', lw=.8, label='Persistence')
    axes[0].set(title=r'(a) Ten-frame error; mean $\pm$ seed SD', xlabel='EM training trajectories', ylabel=r'$E_{10}$', xticks=[8,32,64])
    axes[1].set(title='(b) Rollouts; 64 EM training trajectories', xlabel='Saved frames ahead', ylabel=r'$E_h$ (log scale)', yscale='log')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.tick_params(labelsize=8, length=2)
        axis.title.set_fontsize(9)
    axes[1].legend(fontsize=6.5, ncol=2, loc='upper left', framealpha=.9)
    fig.subplots_adjust(left=.08, right=.99, top=.86, bottom=.23, wspace=.35)
    fig.savefig(path)
    plt.close(fig)


def build(source=None, output=None):
    source, output = Path(source or HERE/'results_one_page.tex'), Path(output or HERE/'results_one_page.pdf')
    protected = [ROOT/'2602.23113v2.pdf', HERE.parent/'results_one_page/results_one_page.pdf']
    if output.suffix != '.pdf' or output.resolve() in [p.resolve() for p in protected]:
        raise ValueError('Output must be a separate PDF; existing source reports are protected')
    snapshot = load_evidence()
    with tempfile.TemporaryDirectory(prefix='.onepage-', dir=output.parent) as directory:
        work = Path(directory)
        make_values(work, snapshot)
        make_figure(work/'results_figure.pdf', snapshot)
        (work/'companion.tex').write_text(source.read_text())
        result = subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-no-shell-escape','companion.tex'],
                                cwd=work, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError('LaTeX failed; previous PDF preserved.\n'+result.stdout[-3000:])
        log = (work/'companion.log').read_text()
        if re.search(r'Overfull \\[hv]box', log):
            raise RuntimeError('Layout overflow; previous PDF preserved.\n'+log[-4000:])
        pages = page_count(work/'companion.pdf')
        if pages > PAGE_LIMIT:
            raise RuntimeError(f'{PAGE_LIMIT}-PAGE LIMIT: previous PDF preserved')
        os.replace(work/'companion.pdf', output)
    report = dict(pages=pages, page_limit=PAGE_LIMIT, pdf_sha256=sha256(output), tex_sha256=sha256(source),
                  snapshot_sha256=sha256(HERE/'results.json'), evidence_sha256=snapshot['evidence_sha256'],
                  captured_utc=snapshot['captured_utc'], scope=snapshot['scope'])
    output.with_suffix('.build.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='Freeze a new complete three-seed evidence snapshot')
    args = parser.parse_args()
    if args.refresh:
        freeze_evidence()
    build()
    print(f'Built {HERE / "results_one_page.pdf"}: at most {PAGE_LIMIT} pages')
