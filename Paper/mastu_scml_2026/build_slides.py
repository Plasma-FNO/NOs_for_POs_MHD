#!/usr/bin/env python3
"""Build evidence-backed MAST-U inserts and a separate updated SCML presentation.

The input PDF is never changed. Movies remain portable MP4 sidecars rather than
PDF-embedded media, which is not supported consistently by presentation viewers.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter

BLUE = '#00558a'
RED = '#bd1d09'
COLORS = {'ar': '#6b7280', 'node': '#cd7200', 'implicit': BLUE,
          'implicit_learned': BLUE, 'implicit_opsplit': BLUE}
WIDTH, HEIGHT = 1920, 1080


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(fig, x, y, value, size=30, color='#111111', weight='normal', **kwargs):
    return fig.text(x, y, value, fontsize=size, color=color, weight=weight,
                    va='top', ha='left', **kwargs)


def slide(title, subtitle=None):
    fig = plt.figure(figsize=(WIDTH/72, HEIGHT/72), dpi=72, facecolor='white')
    text(fig, .05, .94, title, 59, BLUE, 'bold')
    if subtitle:
        text(fig, .052, .835, subtitle, 29, '#444444')
    return fig


def footer(fig, value):
    text(fig, .052, .063, value, 19, '#555555')
    text(fig, .052, .028, 'MAST-U extension: 11 November 2026', 17, '#777777')


def equation_slide():
    fig = slide('Implicit Neural OpsSplit', 'Keep the nonlinear MHD physics; learn a correction anchored at the native reference.')
    text(fig, .055, .715, 'Physical model', 29, RED, 'bold')
    text(fig, .24, .724,
         r'$M_*\dot q=F_{\mathrm{native}}(q)+D_\theta(q)$', 49)
    text(fig, .055, .535, 'Learned defect', 29, RED, 'bold')
    text(fig, .24, .55,
         r'$D_\theta(q)=\mathcal{A}\left[G_\theta(q)-G_\theta(q_*)\right]$', 46)
    text(fig, .24, .432, r'$D_\theta(q_*)=0,\qquad F_{\mathrm{native}}(q_*)\ne0$', 35)
    text(fig, .055, .302,
         'q: native coefficients of (ψ, u, j, ω, ρ, T, V∥)\n'
         'q*: native restart; M*: frozen temporal operator; A: native weak assembly.', 30,
         linespacing=1.65)
    text(fig, .055, .155,
         'Known sources and temperature-dependent transport remain in Fnative.', 29)
    footer(fig, 'Axisymmetric model303 • fixed mesh / equilibrium • baseline coefficients • frozen-mass approximation')
    return fig


def integrator_slide():
    fig = slide('Implicit integration on the native mesh', 'First-order IMEX Euler, with current and vorticity constraints solved at every microstep.')
    text(fig, .055, .71, r'$I(q)=F_*+L_*(q-q_*),\qquad E_\theta(q)=F_{\mathrm{native}}(q)-I(q)+D_\theta(q)$', 39)
    text(fig, .055, .59, 'Interior volume update', 25, RED, 'bold')
    text(fig, .055, .535,
         r'$(M_*-hL_*)\,\delta q=h\,[F_{\mathrm{native}}(q_n)+D_\theta(q_n)]$', 44)
    text(fig, .055, .40, r'$\widetilde q_{n+1}=q_n+\delta q,\qquad q_{n+1}=\mathcal{P}_{q_n}(\widetilde q_{n+1})$', 43)
    text(fig, .055, .255,
         'Coupled algebraic rows: current / vorticity definitions and held boundary values.\n'
         'P: positivity and elliptic guards; nonlinear Mach wall relation re-enforced.\n'
         'h = saved interval / microsteps. The comparison uses two microsteps per interval.',
         29, linespacing=1.7)
    footer(fig, 'L*: frozen native tangent with fixed profiles • M* is never inverted alone • nonlinear physics remains explicit')
    return fig


def comparison_slide(poster, video):
    # The native-mesh movie poster already carries labels, times and equations.
    fig = plt.figure(figsize=(WIDTH/72, HEIGHT/72), dpi=72, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(plt.imread(poster));ax.axis('off')
    fig.text(.994, .995, '11 November 2026', ha='right', va='top', fontsize=15,
             color='#666666', bbox=dict(facecolor='white', edgecolor='none', pad=2))
    return fig


def validate_summary(summary):
    methods = summary['methods']
    if {m['method'] for m in methods} != {'ar', 'node', 'implicit'} or len(summary['cases']) != 2:
        raise ValueError('This presentation requires three methods on the two declared held-out cases')
    if summary.get('missing'):
        raise ValueError('Cannot publish a complete comparison with missing primary runs')
    for method in methods:
        for key in ('mean_rollout_seconds', 'mean_normalized_volume_mse'):
            if not math.isfinite(method[key]) or method[key] <= 0:
                raise ValueError(f'Invalid {key} for {method["method"]}')
    for case in summary['cases']:
        if len(case['methods']) != 3 or any(not m['complete'] or not m['audits_passed'] for m in case['methods']):
            raise ValueError(f'Incomplete or unaudited primary comparison: {case["case_id"]}')
        if any(len(method['timing_samples']) != 3 for method in case['methods']):
            raise ValueError('Primary timing table requires the declared three trials per case and method')
    return methods


def result_slide(summary):
    methods = validate_summary(summary)
    fig = slide('Accuracy and rollout time', 'Two held-out cases • 96 saved intervals • 1.406 μs of plasma evolution • one H100 GPU')
    labels = [{'ar': 'AR', 'node': 'Neural ODE', 'implicit': 'Implicit\nOpsSplit'}[m['method']] for m in methods]
    errors = [m['mean_normalized_volume_mse'] for m in methods]
    seconds = [m['mean_rollout_seconds'] for m in methods]
    colors = [COLORS.get(m['method'], BLUE) for m in methods]
    for bounds, values, title, ylabel in (
        ([.08, .37, .37, .34], errors, 'Prediction error', 'Normalized volume MSE (log scale)'),
        ([.58, .37, .35, .34], seconds, 'Rollout time', 'Seconds per 96-interval trajectory')):
        ax = fig.add_axes(bounds)
        bars = ax.bar(np.arange(3), values, color=colors, width=.6)
        ax.set_xticks(np.arange(3), labels, fontsize=23)
        ax.tick_params(axis='y', labelsize=22)
        ax.set_title(title, fontsize=31, weight='bold', pad=20)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.spines[['top', 'right']].set_visible(False)
        if values is errors:
            ax.set_yscale('log')
            ax.set_ylim(min(values)/2, max(values)*7)
        else:
            ax.set_ylim(0, max(values)*1.3)
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:.4g}', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0, 7), textcoords='offset points', ha='center', fontsize=24)
    native = summary['native_ground_truth']
    native_mean = np.mean([row['wall_seconds'] for row in native])
    text(fig, .055, .205,
         f'Native JOREK reference: {native_mean:,.0f} s ({native_mean/60:.1f} min), historically on 4 CPU MPI ranks.\n'
         'GPU: mean of case medians (3 trials), warmed; guards included; setup / plotting / audits excluded.\n'
         'Native time includes I/O and final checks. Different hardware and timing scope.', 27, linespacing=1.6)
    footer(fig, 'Error: cumulative seven-field volume-weighted normalized MSE, then mean over cases • speed is not accuracy')
    return fig


def matched_control(summary, path):
    report = json.loads(path.read_text())
    if not report['all_audits_passed']:
        raise ValueError('Matched-control evidence did not pass its audits')
    cases = {case['case_id'] for case in summary['cases']}
    values = {}
    for arm, native_arm in (('physics', 'physics'), ('learned', 'frozen_pilot_weights')):
        rows = [row for row in report['rows'] if row['arm'] == native_arm]
        if {row['case_id'] for row in rows} != cases:
            raise ValueError('Matched-control cases differ from the video comparison')
        selected = [next(record for record in row['records'] if record['horizon'] == 96)
                    for row in rows]
        if not all(record['audit_passed'] for record in selected):
            raise ValueError('Matched-control H96 audit failed')
        values[arm] = float(np.mean([record['normalized_volume_mse'] for record in selected]))
        if arm == 'learned':
            for record in selected:
                case = next(case for case in summary['cases'] if case['case_id'] == record['case_id'])
                current = next(method for method in case['methods'] if method['method'] == 'implicit')
                np.testing.assert_allclose(record['normalized_volume_mse'], current['normalized_volume_mse'], rtol=1e-9, atol=1e-13)
    implicit = next(row for row in summary['methods'] if row['method'] == 'implicit')
    np.testing.assert_allclose(values['learned'], implicit['mean_normalized_volume_mse'], rtol=1e-9, atol=1e-13)
    values['reduction_percent'] = 100*(1-values['learned']/values['physics'])
    return dict(path=str(path), sha256=digest(path), **values)


def finding_slide(summary, control):
    methods = validate_summary(summary)
    fig = slide('What the comparison establishes', 'Same native mesh, initial states, normalization and saved times; different model formulations.')
    text(fig, .055, .71, 'Retaining the known physics supplies most of the improvement.', 39, BLUE, 'bold')
    text(fig, .055, .605,
         f'Matched full-physics control: {control["physics"]:.8f} MSE\n'
         f'Adding the anchored learned correction: {control["learned"]:.8f} MSE\n'
         f'Additional reduction: {control["reduction_percent"]:.2f}% across these two cases.', 34, linespacing=1.65)
    text(fig, .055, .365,
         'The implicit model is slower than AR and Neural ODE on this GPU.\n'
         'Its full native nonlinear RHS and coupled solve add cost beyond the shared guards.', 31,
         linespacing=1.65)
    updates = ', '.join(f'{m["label"]}: {m["training_updates"]:,}' for m in methods)
    text(fig, .055, .19, f'Training updates — {updates}', 25)
    footer(fig, '64-update pilot weights; corrected autonomous source implementation • one seed • two existing test cases • no equal-budget claim')
    return fig


def horizon_record(row, horizon):
    records = row.get('horizons', [])
    if isinstance(records, dict):
        value = records.get(str(horizon), records.get(horizon))
        if isinstance(value, dict):
            return value
        return {}
    return next((r for r in records if r.get('horizon') == horizon), {})


def stress_rows(extras):
    rows = []
    for row in extras:
        horizon, microsteps = row.get('horizon'), row.get('microsteps')
        method = {'ar': 'AR', 'node': 'Neural ODE', 'implicit': 'Implicit'}.get(row.get('method'), row.get('method', '?'))
        record = horizon_record(row, min(horizon, 96)) if horizon else {}
        mse = record.get('normalized_volume_mse', row.get('normalized_volume_mse'))
        completed = row.get('completed_steps', row.get('completed_intervals'))
        if isinstance(completed, list):
            completed = min(completed) if completed else None
        status = row.get('status', 'unknown')
        if status in ('completed', 'complete'):
            status = f'Reached H{horizon}'
        else:
            failure = str(row.get('failure', '')).lower()
            status = 'Axis search failed' if 'axis' in failure else 'Failed'
            if completed is not None:
                status += f' after H{completed}'
        score = f'{mse:.5g}' if horizon == 96 and mse is not None else 'No truth beyond H96'
        rows.append([row.get('case_id', ''), method, f'H{horizon} / {microsteps}', status, score])
    return rows


def stress_table(fig, rows, bounds):
    ax = fig.add_axes(bounds);ax.axis('off')
    table = ax.table(cellText=rows, colLabels=['Case', 'Method', 'H / microsteps', 'Outcome', 'MSE / reference'],
                     cellLoc='left', colLoc='left', bbox=[0, 0, 1, 1],
                     colWidths=[.14, .14, .23, .28, .21])
    table.auto_set_font_size(False);table.set_fontsize(21)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if r == 0:
            cell.set_facecolor('#edf3f7');cell.set_text_props(weight='bold', color=BLUE)


def limits_slide(summary):
    fig = slide('How far can we push it?', 'Separate accuracy against native truth from survival beyond the recorded trajectory.')
    text(fig, .055, .72, '24× training window', 39, BLUE, 'bold')
    text(fig, .055, .635,
         'Implicit model: train on 4 saved intervals; evaluate 96 autonomous intervals.\n'
         'This extends the rollout horizon, not the native dataset duration.', 28, linespacing=1.5)
    text(fig, .055, .495, '2× native time step in the main comparison', 39, BLUE, 'bold')
    extras = [row for row in summary.get('extras', []) if row.get('horizon') == 96]
    coarse = [row for row in extras if row['method'] == 'implicit' and row['microsteps'] == 1]
    statement = '1 / 2 / 4 microsteps per saved interval correspond to 4× / 2× / 1× the native step.'
    if {row['case_id'] for row in coarse} == {row['case_id'] for row in summary['cases']}:
        if not all(row['complete'] and row['audits_passed'] for row in coarse):
            raise ValueError('Do not average a failed coarse-step trial into the tradeoff')
        main = next(row for row in summary['methods'] if row['method'] == 'implicit')
        error = np.mean([row['normalized_volume_mse'] for row in coarse])
        seconds = np.mean([row['rollout_seconds'] for row in coarse])
        change = 100*(error/main['mean_normalized_volume_mse']-1)
        speed = main['mean_rollout_seconds']/seconds
        statement = f'One microstep (4× native step): {speed:.2f}× faster, {change:+.1f}% mean MSE over both cases.'
    text(fig, .055, .415,
         '192 implicit microsteps replace 384 native reference steps over 1.406 μs.\n'+statement,
         28, linespacing=1.5)
    rows = stress_rows(extras)
    if rows:
        if len(rows) > 5:
            raise ValueError('More than five H96 refinement results need another slide; no results were hidden')
        stress_table(fig, rows, [.055, .105, .89, .21])
    else:
        text(fig, .055, .22, 'No extra refinement result was supplied; no refinement result is claimed.', 25)
    footer(fig, 'Local axis / X-point search • frozen mass • active guards • baseline coefficients only • no unconditional stability claim')
    return fig


def long_rollout_slides(summary):
    extras = sorted((row for row in summary.get('extras', []) if row.get('horizon', 0) > 96),
                    key=lambda row: (row['case_id'], {'ar': 0, 'node': 1, 'implicit': 2}[row['method']], row['microsteps']))
    if any(row.get('horizon', 0) < 96 for row in summary.get('extras', [])):
        raise ValueError('Unexpected stress horizon; add it explicitly rather than hide it')
    figures = []
    for start in range(0, len(extras), 6):
        group = extras[start:start+6]
        case_ids = {row['case_id'] for row in group}
        if len(case_ids) != 1:
            raise ValueError('Use separate long-rollout slides for different cases')
        fig = slide('Longer rollouts: measure guard dependence',
                    f'{next(iter(case_ids))}: native truth ends at H96. Reaching H384 does not establish physical accuracy.')
        rows = []
        for source, base in zip(group, stress_rows(group)):
            statistics = source['guard_statistics']
            if not statistics:
                raise ValueError('A long-rollout result is missing guard statistics')
            limits = [record['limited_field_steps'] for record in statistics]
            limit_text = str(limits[0]) if len(set(limits)) == 1 else f'{min(limits)}–{max(limits)}'
            total = {2*record['steps'] for record in statistics}
            if len(total) != 1:
                raise ValueError('Trials reached different guard counts; do not average their survival results')
            limit_text += f' / {next(iter(total))}'
            alpha = min(record['minimum_alpha'] for record in statistics)
            rows.append(base[1:4]+[f'{source["rollout_seconds"]:.2f}', limit_text, f'{alpha:.4g}'])
        ax = fig.add_axes([.055, .365, .89, .35]);ax.axis('off')
        table = ax.table(cellText=rows,
                         colLabels=['Method', 'H / microsteps', 'Outcome', 'Seconds', 'Limited / total\nfield-steps', 'Minimum α'],
                         cellLoc='left', colLoc='left', bbox=[0, 0, 1, 1],
                         colWidths=[.14, .20, .20, .12, .22, .12])
        table.auto_set_font_size(False);table.set_fontsize(20)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor('#dddddd')
            if r == 0:
                cell.set_facecolor('#edf3f7');cell.set_text_props(weight='bold', color=BLUE)
        text(fig, .055, .255,
             'H384 requests 5.624 μs: 4× the recorded window, 96× the implicit H4 training window.\n'
             'The positivity limiter blends density / temperature toward the training mean; α = 1 means no limiting.\n'
             'Implicit also checks native axis / X-points, sources and Mach-wall convergence.\n'
             'Guarded survival is not an unconstrained forecast or an accuracy validation.', 27,
             linespacing=1.65)
        footer(fig, 'No truth after H96 • gate sets differ: survival is not a matched stability ranking • one declared stress case')
        figures.append(fig)
    return figures


def write_deck(figures, output, original, summary_path, movie, control=None, poster=None):
    output.mkdir(parents=True, exist_ok=True)
    inserts = output/'MASTU_implicit_OpsSplit_insert.pdf'
    with PdfPages(inserts) as pdf:
        for index, fig in enumerate(figures, 1):
            fig.canvas.draw()
            for label in fig.texts:
                bounds = label.get_window_extent(fig.canvas.get_renderer())
                if not (0 <= bounds.x0 <= bounds.x1 <= WIDTH and 0 <= bounds.y0 <= bounds.y1 <= HEIGHT):
                    raise ValueError(f"Slide {index} text exceeds the canvas: {label.get_text()}")
            pdf.savefig(fig)
            fig.savefig(output/f'slide_{index:02d}.png', dpi=72)
            plt.close(fig)
    source_hash = digest(original)
    source = PdfReader(original)
    added = PdfReader(inserts)
    if len(source.pages) < 3:
        raise ValueError('Expected a presentation with final two conclusion pages')
    writer = PdfWriter()
    insertion = len(source.pages)-2
    for page in source.pages[:insertion]:
        writer.add_page(page)
    for page in added.pages:
        assert abs(float(page.mediabox.width)-WIDTH) < .1
        assert abs(float(page.mediabox.height)-HEIGHT) < .1
        writer.add_page(page)
    for page in source.pages[insertion:]:
        writer.add_page(page)
    writer.add_metadata({'/Title': 'SCML 2026 — Neural Operators and MAST-U Implicit OpsSplit',
                         '/Subject': 'Evidence-backed MAST-U comparison, equations and measured timings'})
    updated = output/'SCML_2026_NOs4POs_with_MASTU.pdf'
    with updated.open('wb') as handle:
        writer.write(handle)
    assert digest(original) == source_hash
    reread = PdfReader(updated)
    assert len(reread.pages) == len(source.pages)+len(added.pages)
    assert all(abs(float(page.mediabox.width)-WIDTH) < .1 and
               abs(float(page.mediabox.height)-HEIGHT) < .1 for page in reread.pages)
    for index, page in enumerate(source.pages):
        new_index = index if index < insertion else index+len(added.pages)
        assert page.extract_text() == reread.pages[new_index].extract_text()
        assert page.get_contents().get_data() == reread.pages[new_index].get_contents().get_data()
    manifest = dict(original_pdf=str(original.resolve()), original_sha256=source_hash,
                    original_pages=len(source.pages), inserted_after_original_page=insertion,
                    inserted_pages=len(added.pages), output_pages=len(reread.pages),
                    insert_pdf=dict(path=str(inserts), sha256=digest(inserts)),
                    updated_pdf=dict(path=str(updated), sha256=digest(updated)),
                    summary=dict(path=str(summary_path), sha256=digest(summary_path)) if summary_path else None,
                    movie=str(movie) if movie else None, movie_sha256=digest(movie) if movie else None,
                    poster=dict(path=str(poster), sha256=digest(poster)) if poster else None,
                    builder_sha256=digest(__file__), matched_control=control,
                    original_text_content_streams_and_file_preserved=True,
                    note='MP4 videos are separate sidecars; PDF contains a static preview.')
    (output/'slides_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--poster', type=Path)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--original', type=Path, default=Path(__file__).resolve().with_name('SCML_2026_NOs4POs.pdf'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--control-report', type=Path, default=Path('/pitagora_scratch/userexternal/vgopakum/mastu_physical_defect_20261111/source_revision_verification/report.json'))
    parser.add_argument('--equations-only', action='store_true', help='Draft two equation slides before measurements finish')
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
    figures = [equation_slide(), integrator_slide()]
    control = None
    if not args.equations_only:
        if not all((args.summary, args.poster, args.video)):
            parser.error('--summary, --poster and --video are required for the complete presentation')
        if not args.video.is_file():
            raise FileNotFoundError(args.video)
        summary = json.loads(args.summary.read_text())
        validate_summary(summary)
        control = matched_control(summary, args.control_report)
        figures.extend([comparison_slide(args.poster, args.video), result_slide(summary),
                        finding_slide(summary, control), limits_slide(summary)])
        figures.extend(long_rollout_slides(summary))
    manifest = write_deck(figures, args.output_dir, args.original, args.summary, args.video, control, args.poster)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
