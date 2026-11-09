#!/usr/bin/env python3
"""Freeze MAST-U OpsSplit evidence and build a verified (at most two-page) methods/results report.

`--refresh` runs the study's own `summarize_implicit.py` over the live campaign folders and
freezes its output, the extra explicit-OpsSplit variants, the native coefficient check and the
SHA-256 of every evaluation file read. A plain build re-verifies those hashes and renders only
the frozen snapshot, so a running campaign cannot silently change a published number.
Incomplete horizons are never averaged; they are shown as pending.

`sha256` and `page_count` are also imported by `../jorek_transfer_one_page/build.py`.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = ROOT/'JOREK/mastu_opsplit'
SUMMARIZER = STUDY/'scripts/summarize_implicit.py'
DATASET = Path('/pitagora_scratch/userexternal/vgopakum/mastu_euler_ensemble100_20261105')
OOD_DATASET = Path('/pitagora_scratch/userexternal/vgopakum/mastu_coefficient_ood10_20261106')
IMPLICIT_ROOT = Path('/pitagora_scratch/userexternal/vgopakum/mastu_implicit_opsplit_20261108')
CAMPAIGN, OOD, COEFFICIENT = IMPLICIT_ROOT/'campaign_v2', IMPLICIT_ROOT/'ood', IMPLICIT_ROOT/'coefficient'
ENSEMBLE = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_ensemble80_comparison_20261106')
GRADIENT = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_gradient_retraining_20261106')
DENSITY = Path('/pitagora_scratch/userexternal/vgopakum/mastu_density_diffusion_seed17_20261106T163039Z')
PAGE_LIMIT = 2
HORIZONS = (16, 32, 96)
FIELDS = ('psi', 'u', 'zj', 'w', 'rho', 'T', 'Vpar')
SCHEMA = 'mastu-opsplit-methods-v1'
# Rows of the results table: (label, test source, expected test seeds, OOD source, expected OOD seeds, training id).
# Sources: ('test'|'ood'|'ood_coefficient', name) from the summarizer, or the extra variants below.
# AR/NODE seeds 18/19 were cancelled (array 2790285); explicit OpsSplit seeds 18/19 come from the
# matched learned-derivative arm of the gradient study, exactly as summarize_implicit.py selects them.
ROWS = [
    ('AR', ('test', 'ar'), (17,), ('ood', 'ar'), (17,), 'ar'),
    ('Euler NODE', ('test', 'node'), (17,), ('ood', 'node'), (17,), 'node'),
    (r'Explicit OpsSplit, learned $\nabla$', ('test', 'opsplit'), (17, 18, 19), ('ood', 'opsplit'), (17,), 'explicit_learned'),
    (r'Explicit OpsSplit, native $\nabla$', ('gradient', 'native'), (17, 18, 19), None, (), 'explicit_native'),
    (r'Explicit OpsSplit $+\,D_\perp\nabla^2\rho$', ('density', 'test'), (17,), ('density', 'ood'), (17,), 'density'),
    (r'Implicit: labels $+$ rollout', ('test', 'implicit'), (17, 18, 19), ('ood', 'implicit'), (17, 18, 19), 'implicit'),
    (r'Implicit: rollout only', ('test', 'implicit_rollout'), (17, 18, 19), ('ood', 'implicit_rollout'), (17, 18, 19), 'implicit_rollout'),
    (r'Implicit: physics only ($N\!=\!0$)', ('test', 'physics'), (17,), ('ood', 'physics'), (17,), 'untrained'),
    (r'\quad$+\,D_\perp$ physics: labels $+$ rollout', None, (), ('ood_coefficient', 'implicit'), (17, 18, 19), None),
    (r'\quad$+\,D_\perp$ physics: rollout only', None, (), ('ood_coefficient', 'implicit_rollout'), (17, 18, 19), None),
    (r'\quad$+\,D_\perp$ physics: physics only', None, (), ('ood_coefficient', 'physics'), (17,), None),
]


def training_histories(name, seed):
    """History files whose per-update wall times make up one training run (same runs as the score columns)."""
    if name in ('ar', 'node'):
        return [ENSEMBLE/f'seed{seed}_{name}'/name/'history.json']
    if name == 'explicit_learned':
        ensemble = ENSEMBLE/f'seed{seed}_opsplit'
        return [ensemble/'opsplit/history.json' if (ensemble/'results.json').exists()
                else GRADIENT/f'seed{seed}_learned/opsplit/history.json']
    if name == 'explicit_native':
        return [GRADIENT/f'seed{seed}_native/opsplit/history.json']
    if name == 'density':
        return [DENSITY/'seed17_opsplit/opsplit/history.json']
    if name == 'implicit':
        return [CAMPAIGN/f'seed{seed}_implicit/implicit/{stage}/history.json' for stage in ('pretrain', 'rollout')]
    if name == 'implicit_rollout':
        return [CAMPAIGN/f'seed{seed}_implicit_rollout/implicit_rollout/rollout/history.json']
    raise ValueError(f'Unknown training run {name}')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def page_count(path):
    result = subprocess.run(['pdfinfo', str(path)], capture_output=True, text=True, check=True, timeout=30)
    match = re.search(r'^Pages:\s+(\d+)\s*$', result.stdout, re.M)
    if match is None:
        raise RuntimeError('Could not verify PDF page count')
    return int(match[1])


def summarizer():
    spec = importlib.util.spec_from_file_location('mastu_summarize_implicit', SUMMARIZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluation_files():
    patterns = [(CAMPAIGN, 'seed*_*/*/evaluation_*_substeps*.json'), (ENSEMBLE, 'seed*_*/*/evaluation_*_substeps*.json'),
                (GRADIENT, 'seed*_learned/opsplit/evaluation_*_substeps*.json'), (OOD, 'seed*_*/evaluation_*_substeps*.json'),
                (COEFFICIENT, 'seed*_*/evaluation_*_substeps*.json'),
                (DENSITY, 'seed17_opsplit/opsplit/evaluation_*_substeps*.json'),
                (DENSITY, 'seed17_opsplit_ood/evaluation_*_substeps*.json')]
    return sorted(str(path) for root, pattern in patterns for path in root.glob(pattern))


def figure_field(path):
    """Element-mean training-mean pressure rho*T on the native mesh, from the campaign's 80-case normalization."""
    import h5py
    normalization = CAMPAIGN/'seed17_implicit/normalization.npz'
    with h5py.File(DATASET/'geometry.h5') as handle:
        basis, dof_map, rz = handle['basis'][()], handle['dof_map'][()], handle['quadrature_rz'][()]
    mean = np.load(normalization)['mean']
    samples = np.einsum('eql,elc->eqc', basis, mean[dof_map][..., 4:6])
    pressure = (samples[..., 0]*samples[..., 1]).mean(1)
    centre = rz.mean(1)
    np.savez_compressed(path, R=centre[:, 0].astype('f4'), Z=centre[:, 1].astype('f4'), pressure=pressure.astype('f4'))
    return normalization


def summarize_consistently(attempts=6):
    """Run the study summarizer while no evaluation file appears or changes.

    Evaluation files are written once, atomically, per trajectory and substep count, so a stable
    file set before and after the run means the summary matches exactly the hashed files.
    """
    module = summarizer()
    for _ in range(attempts):
        before = {path: sha256(path) for path in evaluation_files()}
        with tempfile.TemporaryDirectory(prefix='.refresh-', dir=HERE) as directory:
            output = Path(directory)/'comparison.json'
            subprocess.run([sys.executable, str(SUMMARIZER), '--campaign', str(CAMPAIGN), '--ensemble', str(ENSEMBLE),
                            '--gradient', str(GRADIENT), '--ood', str(OOD), '--coefficient', str(COEFFICIENT),
                            '--output', str(output)], check=True, capture_output=True, text=True, timeout=3600)
            comparison = json.loads(output.read_text())
        density = {'test': module.summarize_model(DENSITY/'seed17_opsplit/opsplit', 20)[0],
                   'ood': module.summarize_model(DENSITY/'seed17_opsplit_ood', 10)[0]}
        if {path: sha256(path) for path in evaluation_files()} == before:
            return comparison, density, before
    raise RuntimeError(f'Evaluation files kept changing during {attempts} refresh attempts; retry later')


def freeze_evidence():
    comparison, density, before = summarize_consistently()
    evidence = dict(before)
    gradient_path = GRADIENT/'comparison.json'
    gradient = json.loads(gradient_path.read_text())
    gradient_rows = {str(row['horizon']): dict(complete=row['complete'], native_change_percent=row['native_change_percent'],
                                               learned=row['variants']['learned']['seed_means'],
                                               native=row['variants']['native']['seed_means'])
                     for row in gradient['comparison'] if row['substeps'] == 1}
    parameters = sorted({row['parameters'] for row in gradient['matching']})
    if gradient['status'] != 'complete' or len(parameters) != 1:
        raise ValueError('Gradient study must be complete with one matched parameter count')
    verify_path = COEFFICIENT/'verify_native.json'
    verify = json.loads(verify_path.read_text())
    rows = [row for state in verify['states'] for row in state['rows']]
    coefficient_check = dict(passed=verify['passed'], ood_trajectories_read=verify['ood_trajectories_read'],
                             forcing=max(row['forcing_error_over_change'] for row in rows),
                             tangent=max(row['tangent_error_over_change'] for row in rows),
                             other_channels=max(row['other_channels_max_change_over_scale'] for row in rows),
                             states=len(verify['states']), multipliers=sorted({row['multiplier'] for row in rows}))
    pretrain = {}
    for seed in (17, 18, 19):
        path = CAMPAIGN/f'seed{seed}_implicit/implicit/pretrain/history.json'
        if path.exists():
            losses = [row['remainder_loss'] for row in json.loads(path.read_text())]
            pretrain[str(seed)] = dict(updates=len(losses), final200=float(np.mean(losses[-200:])))
            evidence[str(path)] = sha256(path)
    # Parameter count of the implicit remainder network: the provenance-checked ensemble count plus the
    # state-dict difference between matched seed-17 checkpoints (only the projection head differs).
    import torch
    sizes = {}
    for name, path in (('ensemble', ENSEMBLE/'seed17_node/node/selected.pt'),
                       ('implicit', CAMPAIGN/'seed17_implicit/implicit/rollout/selected.pt')):
        state = torch.load(path, map_location='cpu', weights_only=False)['model']
        sizes[name] = sum(value.numel() for value in state.values())
        evidence[str(path)] = sha256(path)
    # Training time: summed wall time of the optimizer updates (history rows), counted only for finished
    # runs of 2,000 updates per stage; excludes data verification, operator setup and evaluation.
    training = {}
    for _, _, test_seeds, _, _, name in ROWS:
        if name in (None, 'untrained') or name in training:
            continue
        training[name] = {}
        for seed in test_seeds:
            paths = training_histories(name, seed)
            if not all(path.exists() for path in paths):
                continue
            histories = [json.loads(path.read_text()) for path in paths]
            if any(len(history) != 2000 for history in histories):
                continue
            training[name][str(seed)] = sum(row.get('training_seconds', row.get('seconds')) for history in histories for row in history)
            evidence.update({str(path): sha256(path) for path in paths})
    label_summary = IMPLICIT_ROOT/'labels/summary.json'
    labels = json.loads(label_summary.read_text())
    evidence[str(label_summary)] = sha256(label_summary)
    threads = {name: json.loads(path.read_text())['config']['threads'] for name, path in
               (('explicit', ENSEMBLE/'seed17_node/provenance.json'), ('implicit', CAMPAIGN/'seed17_implicit/provenance.json'))}
    manifest_path, ood_path = DATASET/'manifest.json', OOD_DATASET/'manifest.json'
    manifest, ood_manifest = json.loads(manifest_path.read_text()), json.loads(ood_path.read_text())
    completion = json.loads((DATASET/'completion.json').read_text())
    cases = manifest['cases']
    amplitude = np.abs([case['amplitude'] for case in cases])
    centre, width = np.array([case['center'] for case in cases]), np.array([case['width'] for case in cases])
    import h5py
    with h5py.File(DATASET/'geometry.h5') as handle:
        dofs, elements = len(handle['dof_ids']), len(handle['dof_map'])
    dataset = dict(cases=len(cases), frames=completion['frames_per_case'], interval_s=manifest['observation_interval_seconds'],
                   source_split=manifest['split_counts'], dofs_per_field=dofs, elements=elements,
                   amplitude=[float(amplitude.min()), float(amplitude.max())], taper=cases[0]['taper'],
                   centre_R=[float(centre[:, 0].min()), float(centre[:, 0].max())],
                   centre_Z=[float(centre[:, 1].min()), float(centre[:, 1].max())],
                   width_R=[float(width[:, 0].min()), float(width[:, 0].max())],
                   width_Z=[float(width[:, 1].min()), float(width[:, 1].max())],
                   ood_cases=len(ood_manifest['cases']),
                   ood_multiplier=[min(c['coefficient_multiplier'] for c in ood_manifest['cases']),
                                   max(c['coefficient_multiplier'] for c in ood_manifest['cases'])],
                   ood_baseline=ood_manifest['baseline_coefficient'])
    normalization = figure_field(HERE/'figure_field.npz')
    for path in (manifest_path, ood_path, DATASET/'completion.json', DATASET/'geometry.h5', normalization,
                 gradient_path, verify_path, SUMMARIZER, HERE/'figure_field.npz'):
        evidence[str(path)] = sha256(path)
    snapshot = dict(schema=SCHEMA, captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    comparison=comparison, density=density, gradient=dict(rows=gradient_rows,
                    numerical_failures=gradient['numerical_failures'], audits=gradient['all_audits_passed']),
                    coefficient_check=coefficient_check, pretrain=pretrain, dataset=dataset, training=training,
                    label_export=dict(states=labels['states'], mean_seconds=labels['mean_export_seconds']), threads=threads,
                    parameters=dict(ensemble=parameters[0], implicit=parameters[0]+sizes['implicit']-sizes['ensemble']),
                    evidence_sha256=evidence)
    temporary = HERE/'results.json.part'
    temporary.write_text(json.dumps(snapshot, indent=1, allow_nan=False)+'\n')
    temporary.replace(HERE/'results.json')
    return snapshot


def load_evidence():
    snapshot = json.loads((HERE/'results.json').read_text())
    if snapshot.get('schema') != SCHEMA:
        raise ValueError('Expected a frozen MAST-U methods snapshot; run build.py --refresh')
    for path, expected in snapshot['evidence_sha256'].items():
        if not Path(path).exists() or sha256(path) != expected:
            raise ValueError(f'Evidence changed: {path}; explicitly refresh the snapshot')
    return snapshot


def seed_values(snapshot, source, seeds, horizon):
    """Complete seed means for one table cell, plus partial-trajectory progress for pending seeds."""
    if source is None:
        return None
    kind, name = source
    values, partial = [], []
    for seed in seeds:
        if kind == 'gradient':
            row = snapshot['gradient']['rows'][str(horizon)]
            if row['complete']:
                values.append(row[name][(17, 18, 19).index(seed)])
            continue
        table = snapshot['density'][name] if kind == 'density' else snapshot['comparison'][kind].get(f'{name}/seed{seed}', {})
        cell = table.get(f'{horizon}/1')
        if cell and cell['complete']:
            values.append(cell['mean'])
        else:
            partial.append(f"{cell['trajectories']}/{cell['expected']}" if cell else 'pending')
    return values, partial


def number(x):
    if 0.01 <= x < 100:
        return f'{x:#.2g}' if x < 1 else f'{x:.3g}'
    mantissa, exponent = f'{x:.1e}'.split('e')
    return rf'${mantissa}{{\times}}10^{{{int(exponent)}}}$'


def table_rows(snapshot):
    lines = []
    for label, test, test_seeds, ood, ood_seeds, name in ROWS:
        runs = list(snapshot['training'].get(name, {}).values())
        cells = [label, '0' if name == 'untrained' else f'{np.mean(runs)/3600:.1f}' if runs else
                 '--' if name is None else 'run.']
        for source, seeds in ((test, test_seeds), (ood, ood_seeds)):
            if source is None:
                cells += ['--'] + ['--']*3
                continue
            counts = []
            for horizon in HORIZONS:
                values, partial = seed_values(snapshot, source, seeds, horizon)
                counts.append(len(values))
                if source == ('density', 'ood'):
                    cells.append(r'n.i.')
                elif values:
                    cells.append(number(float(np.mean(values))))
                else:
                    cells.append('run.' if any('/' in p for p in partial) else 'pend.')
            cells.insert(len(cells)-3, f'{min(counts)}/{len(seeds)}')
        lines.append(' & '.join(cells)+r' \\')
    return '\n'.join(lines)


def make_values(work, snapshot):
    comparison, dataset = snapshot['comparison'], snapshot['dataset']
    def pairs(split, h=96):
        rows = [row for key, row in comparison['paired'].items()
                if key.startswith(split+'/') and key.endswith(f'_vs_node/seed17/h{h}') and row is not None]
        return sum(row['left_better'] for row in rows), sum(row['cases'] for row in rows)
    better, total = pairs('test')
    ood_better, ood_total = pairs('ood')
    if better != total or total == 0:
        raise ValueError(f'Headline says every paired test trajectory improves; evidence has {better}/{total}')
    def shares(split, names, fields):
        found = []
        for name in names:
            for seed in (17, 18, 19):
                cell = comparison[split].get(f'{name}/seed{seed}', {}).get('96/1')
                if cell and cell['complete']:
                    found.append(sum(cell['per_field'][f] for f in fields)/sum(cell['per_field'].values()))
        if not found:
            raise ValueError(f'No complete {split} arm for the residual-share statement')
        return f'{100*min(found):.0f}--{100*max(found):.0f}'
    substep = max(abs(comparison['test'][f'{name}/seed17'][f'{h}/4']['mean']/comparison['test'][f'{name}/seed17'][f'{h}/1']['mean']-1)
                  for name in ('node', 'opsplit') for h in HORIZONS)
    def complete_means(name, horizon, seeds=(17, 18, 19)):
        cells = [comparison['test'].get(f'{name}/seed{s}', {}).get(f'{horizon}/1') for s in seeds]
        return [cell['mean'] for cell in cells if cell and cell['complete']]
    # Headline gain: best explicit family (NODE or explicit OpsSplit, seed means) over the worst implicit seed.
    gains = []
    for horizon in HORIZONS:
        implicit = complete_means('implicit', horizon) + complete_means('implicit_rollout', horizon)
        explicit = [np.mean(complete_means(name, horizon)) for name in ('node', 'opsplit') if complete_means(name, horizon)]
        if implicit and explicit:
            gains.append(min(explicit)/max(implicit))
    if not gains:
        raise ValueError('No complete implicit and explicit test horizons for the headline')

    def ood_means(split, names, seeds=(17, 18, 19)):
        return {h: [comparison[split][f'{n}/seed{s}'][f'{h}/1']['mean'] for n in names for s in seeds
                    if comparison[split].get(f'{n}/seed{s}', {}).get(f'{h}/1', {}).get('complete')] for h in HORIZONS}
    fixed, informed = ood_means('ood', ('implicit', 'implicit_rollout')), ood_means('ood_coefficient', ('implicit', 'implicit_rollout'))
    informed_physics = ood_means('ood_coefficient', ('physics',), (17,))
    coefficient_gains = [np.mean(fixed[h])/np.mean(informed[h]) for h in HORIZONS if fixed[h] and informed[h]]
    if not coefficient_gains:
        raise ValueError('No complete coefficient-physics OOD horizon for the headline')
    # Headline: the untrained physics-only model matches the trained arms wherever both are complete.
    matched = 0
    for horizon in HORIZONS:
        for physics, trained in ((informed_physics[horizon], informed[horizon]),
                                 (complete_means('physics', horizon, (17,)),
                                  complete_means('implicit', horizon)+complete_means('implicit_rollout', horizon))):
            if physics and trained:
                matched += 1
                if not 0.5 <= physics[0]/np.mean(trained) <= 2:
                    raise ValueError(f'Physics-only result contradicts the headline at h={horizon}; revise the text')
    if not matched:
        raise ValueError('Headline needs at least one complete physics-only comparison')
    pretrain = [row['final200'] for row in snapshot['pretrain'].values()]
    check = snapshot['coefficient_check']
    if not check['passed'] or check['ood_trajectories_read']:
        raise ValueError('Native coefficient check must pass without reading OOD data')
    interval = dataset['interval_s']
    values = dict(
        Cases=dataset['cases'], TrainCases=dataset['source_split']['train']+dataset['source_split']['validation'],
        TestCases=dataset['source_split']['test'], Frames=dataset['frames'], Intervals=dataset['frames']-1,
        IntervalNs=f'{interval*1e9:.1f}', DurationUs=f"{(dataset['frames']-1)*interval*1e6:.2f}",
        HorizonUs='/'.join(f'{h*interval*1e6:.2f}' for h in HORIZONS),
        DofsPerField=f"{dataset['dofs_per_field']:,}", Unknowns=f"{7*dataset['dofs_per_field']:,}",
        Elements=f"{dataset['elements']:,}",
        AmpMin=f"{dataset['amplitude'][0]:.2f}", AmpMax=f"{dataset['amplitude'][1]:.2f}",
        TaperLo=dataset['taper'][0], TaperHi=dataset['taper'][1],
        CentreR=f"{dataset['centre_R'][0]:.2f}--{dataset['centre_R'][1]:.2f}",
        CentreZ=f"{max(abs(v) for v in dataset['centre_Z']):.2f}",
        WidthR=f"{dataset['width_R'][0]:.2f}--{dataset['width_R'][1]:.2f}",
        WidthZ=f"{dataset['width_Z'][0]:.2f}--{dataset['width_Z'][1]:.2f}",
        OodCases=dataset['ood_cases'], OodMin=f"{dataset['ood_multiplier'][0]:g}", OodMax=f"{dataset['ood_multiplier'][1]:g}",
        LabelHours=f"{snapshot['label_export']['states']*snapshot['label_export']['mean_seconds']/3600:.1f}",
        LabelStates=f"{snapshot['label_export']['states']:,}",
        ThreadsExplicit=snapshot['threads']['explicit'], ThreadsImplicit=snapshot['threads']['implicit'],
        ParamsEnsemble=f"{snapshot['parameters']['ensemble']:,}", ParamsImplicit=f"{snapshot['parameters']['implicit']:,}",
        PairedBetter=better, PairedTotal=total, OodPairedBetter=ood_better, OodPairedTotal=ood_total,
        TestCurrentShare=shares('test', ('implicit', 'implicit_rollout'), ('zj',)),
        OodDensityShare=shares('ood', ('implicit', 'implicit_rollout'), ('rho', 'T')),
        SubstepChange=f'{100*substep:.1f}', PretrainLoss=f'{min(pretrain):.2f}--{max(pretrain):.2f}',
        PretrainSeeds=len(pretrain),
        GradientChange=f"{max(abs(r['native_change_percent']) for r in snapshot['gradient']['rows'].values()):.2f}",
        ImplicitGain=f'{min(gains):.0f}--{max(gains):.0f}',
        CoefGain=f'{min(coefficient_gains):.0f}--{max(coefficient_gains):.0f}',
        CoefForcing=number(check['forcing']), CoefTangent=number(check['tangent']),
        ResultsRows=table_rows(snapshot), SnapshotTime=snapshot['captured_utc'][:16].replace('T', ' ')+' UTC')
    (work/'experiment_values.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k, v in values.items())+'\n')


def series(snapshot, source, seeds):
    points = [seed_values(snapshot, source, seeds, h)[0] for h in HORIZONS]
    if not all(points):
        return None
    return [float(np.mean(p)) for p in points], [min(p) for p in points], [max(p) for p in points]


def make_figure(path, snapshot):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'axes.unicode_minus': False,
                         'axes.formatter.use_mathtext': True, 'font.size': 8, 'pdf.fonttype': 42, 'mathtext.fontset': 'cm'})
    field = np.load(HERE/'figure_field.npz')
    dataset = snapshot['dataset']
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.35), gridspec_kw={'width_ratios': [0.62, 1, 1]})
    axis = axes[0]
    points = axis.scatter(field['R'], field['Z'], c=field['pressure'], s=0.5, cmap='viridis', linewidths=0, rasterized=True)
    (r0, r1), (z0, z1) = dataset['centre_R'], dataset['centre_Z']
    axis.add_patch(Rectangle((r0, z0), r1-r0, z1-z0, fill=False, lw=.9, ls='--', ec='#d62728'))
    axis.set_aspect('equal')
    axis.set_xlabel('R (m)', labelpad=1)
    axis.set_ylabel('Z (m)', labelpad=0)
    axis.xaxis.set_major_locator(MaxNLocator(3))
    axis.tick_params(labelsize=7, length=2)
    axis.set_title(f"(a) Mesh, {dataset['elements']:,} elements", fontsize=8.5)
    bar = fig.colorbar(points, ax=axis, pad=.04, fraction=.09, ticks=MaxNLocator(3))
    bar.ax.tick_params(labelsize=6, length=2)
    styles = [('AR', ('test', 'ar'), ('ood', 'ar'), (17,), (17,), '#7f7f7f', 'o'),
              ('NODE', ('test', 'node'), ('ood', 'node'), (17,), (17,), '#287ab3', 's'),
              ('Explicit OpsSplit', ('test', 'opsplit'), ('ood', 'opsplit'), (17, 18, 19), (17,), '#df7a23', 'D'),
              ('Implicit: labels+rollout', ('test', 'implicit'), ('ood', 'implicit'), (17, 18, 19), (17, 18, 19), '#238b45', '^'),
              ('Implicit: rollout only', ('test', 'implicit_rollout'), ('ood', 'implicit_rollout'), (17, 18, 19), (17, 18, 19), '#74c476', 'v'),
              ('Implicit: physics only', ('test', 'physics'), ('ood', 'physics'), (17,), (17,), '#6b52a2', 'x'),
              (r'Implicit $+D_\perp$ physics', None, ('ood_coefficient', 'implicit'), (), (17, 18, 19), '#c73c3c', '*')]
    ceiling = 30.
    for column, (axis, title) in enumerate(((axes[1], '(b) Held-out test, 20 cases'),
                                             (axes[2], rf"(c) $D_\perp\times${dataset['ood_multiplier'][0]:g} to {dataset['ood_multiplier'][1]:g}, 10 OOD cases"))):
        for label, test, ood, test_seeds, ood_seeds, color, marker in styles:
            source, seeds = (test, test_seeds) if column == 0 else (ood, ood_seeds)
            data = series(snapshot, source, seeds) if source else None
            if data is None:
                continue
            mean, low, high = (np.array(v) for v in data)
            shown = np.minimum(mean, ceiling)
            axis.errorbar(HORIZONS, shown, yerr=[shown-np.minimum(low, shown), np.minimum(high, ceiling)-shown],
                          color=color, marker=marker, ms=3, lw=.9, capsize=1.5, label=label)
            for h, value in zip(HORIZONS, mean):
                if value > ceiling:
                    axis.annotate(f'{label}: {number(value)}', xy=(h, ceiling), xytext=(h-38, ceiling*0.35),
                                  fontsize=6, color=color, arrowprops=dict(arrowstyle='->', lw=.6, color=color))
        axis.set(yscale='log', xticks=HORIZONS, xlabel='Saved intervals rolled out')
        axis.set_ylim(top=ceiling*1.6)
        axis.set_title(title, fontsize=8.5)
        axis.grid(alpha=.2)
        axis.tick_params(labelsize=7, length=2)
    axes[1].set_ylabel('MSE', labelpad=1)
    handles, labels = [], []
    for axis in axes[1:]:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, fontsize=6.5, loc='lower center', bbox_to_anchor=(0.62, 0.0), ncol=4,
               frameon=False, handlelength=1.8, columnspacing=1.2)
    fig.subplots_adjust(left=.02, right=.99, top=.9, bottom=.33, wspace=.42)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def build(source=None, output=None):
    source, output = Path(source or HERE/'results_one_page.tex'), Path(output or HERE/'results_one_page.pdf')
    protected = [ROOT/'2602.23113v2.pdf', HERE.parent/'jorek_transfer_one_page/results_one_page.pdf', source]
    if output.suffix != '.pdf' or output.resolve() in [p.resolve() for p in protected]:
        raise ValueError('Output must be a separate PDF; the original paper and blob report are protected')
    snapshot = load_evidence()
    with tempfile.TemporaryDirectory(prefix='.twopage-', dir=output.parent) as directory:
        work = Path(directory)
        make_values(work, snapshot)
        make_figure(work/'results_figure.pdf', snapshot)
        (work/'companion.tex').write_text(source.read_text())
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-no-shell-escape', 'companion.tex'],
                                cwd=work, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError('LaTeX failed; previous PDF preserved.\n'+result.stdout[-3000:])
        log = (work/'companion.log').read_text(errors='replace')
        if re.search(r'Overfull \\[hv]box', log):
            raise RuntimeError('Layout overflow; previous PDF preserved.\n'+log[-4000:])
        pages = page_count(work/'companion.pdf')
        if pages > PAGE_LIMIT:
            raise RuntimeError(f'{PAGE_LIMIT}-PAGE LIMIT: generated {pages} pages; previous PDF preserved')
        os.replace(work/'companion.pdf', output)
    report = dict(pages=pages, page_limit=PAGE_LIMIT, pdf_sha256=sha256(output), tex_sha256=sha256(source),
                  snapshot_sha256=sha256(HERE/'results.json'), captured_utc=snapshot['captured_utc'], schema=SCHEMA)
    output.with_suffix('.build.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--refresh', action='store_true', help='Freeze a new evidence snapshot from the live campaign')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.refresh:
            freeze_evidence()
        build(output=args.output)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error))
    print(f'Built {args.output or HERE/"results_one_page.pdf"}: at most {PAGE_LIMIT} pages')
