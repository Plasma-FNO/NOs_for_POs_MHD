"""Render matched native MAST-U trajectories without Cartesian interpolation.

Input NPZ: time_seconds [F], ground_truth/ar/node/implicit [F,D,7] in
physical native coefficients, optional valid_METHOD [F]. JSON: case_id,
restart_path, normalization_scale [7], methods keyed as above. Invalid frames
are blank, never replaced by an earlier forecast or invented ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mastu_opsplit.restart import cubic_basis, read_restart

METHODS = ('ground_truth', 'ar', 'node', 'implicit')
LABELS = ('Ground truth · JOREK', 'Autoregressive', 'Neural ODE', 'Implicit Neural OpsSplit')
FIELDS = (4, 5)
FIELD_NAMES = ('Density', 'Temperature')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def display_mesh(restart, subdivisions=2):
    """Straight display subcells sample the exact curved bicubic element map."""
    geometry, _, metadata = read_restart(restart)
    if metadata['minimum_signed_jacobian_m2'] <= 0:
        raise ValueError('Display requires positive native signed Jacobians')
    with h5py.File(restart) as stream:
        vertex = stream['vertex'][()].T - 1
        sizes = stream['size'][()].transpose(2, 1, 0)
        xy = stream['x'][()][:, :, 0, :].transpose(2, 1, 0)[vertex].reshape(-1, 16, 2)
    n = subdivisions
    s, t = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1), indexing='ij')
    basis = np.einsum('qvd,evd->eqvd', cubic_basis(s.ravel(), t.ravel())[0], sizes).reshape(len(vertex), -1, 16)
    points = np.einsum('eql,elc->eqc', basis, xy)
    corners = np.array([[i*(n+1)+j, (i+1)*(n+1)+j, (i+1)*(n+1)+j+1, i*(n+1)+j+1]
                        for i in range(n) for j in range(n)])
    centers = np.array([[(i+.5)/n, (j+.5)/n] for i in range(n) for j in range(n)])
    center_basis = np.einsum('qvd,evd->eqvd', cubic_basis(centers[:, 0], centers[:, 1])[0], sizes).reshape(len(vertex), -1, 16)
    np.testing.assert_allclose(center_basis.reshape(len(vertex), n*n, 4, 4)[..., 0].sum(-1), 1., atol=1e-13)
    return geometry, points[:, corners].reshape(-1, 4, 2), center_basis


def sample_trajectories(path, metadata, geometry, basis):
    """Sample display centers; compute errors separately using native Q4 dV."""
    with np.load(path, allow_pickle=False) as source:
        times = np.asarray(source['time_seconds'], dtype=float)
        if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError('time_seconds must be finite and strictly increasing')
        scale = np.asarray(metadata['normalization_scale'], dtype=float)
        if scale.shape != (7,) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('normalization_scale must be seven positive training field scales')
        valid = np.stack([np.asarray(source[f'valid_{name}'], dtype=bool) if f'valid_{name}' in source
                          else np.ones(len(times), dtype=bool) for name in METHODS], axis=1)
        if valid.shape != (len(times), 4) or not valid[0].all():
            raise ValueError('All methods require the same valid initial frame')
        values = np.full((len(times), 4, basis.shape[0]*basis.shape[1], 2), np.nan)
        rms = np.full((len(times), 4, 2), np.nan)
        relative = np.full_like(rms, np.nan)
        truth = source['ground_truth']
        expected = (len(times), len(geometry.dof_ids), 7)
        if truth.shape != expected:
            raise ValueError(f'ground_truth shape {truth.shape} differs from {expected}')
        initial = truth[0].copy()
        weights = geometry.volume_weights / geometry.volume_weights.sum()
        truth_q = np.full((len(times), *geometry.basis.shape[:2], 2), np.nan)
        for frame in np.flatnonzero(valid[:, 0]):
            q = truth[frame][:, FIELDS]
            if not np.isfinite(q).all():
                raise ValueError(f'Nonfinite valid ground truth at frame {frame}')
            truth_q[frame] = np.einsum('eql,elc->eqc', geometry.basis, q[geometry.dof_map])
        del truth
        for method, name in enumerate(METHODS):
            states = source[name]
            if states.shape != expected:
                raise ValueError(f'{name} shape {states.shape} differs from {expected}')
            np.testing.assert_allclose(states[0], initial, rtol=1e-12, atol=1e-12,
                                       err_msg=f'{name} does not start from the same native state')
            for frame in np.flatnonzero(valid[:, method]):
                state = states[frame][:, FIELDS]
                if not np.isfinite(state).all():
                    raise ValueError(f'Nonfinite valid {name} frame {frame}')
                local = state[geometry.dof_map]
                values[frame, method] = np.einsum('eql,elc->eqc', basis, local).reshape(-1, 2)
                if valid[frame, 0]:
                    sampled = np.einsum('eql,elc->eqc', geometry.basis, local)
                    difference = sampled - truth_q[frame]
                    absolute = np.sqrt(np.einsum('eq,eqc->c', weights, difference*difference))
                    norm = np.sqrt(np.einsum('eq,eqc->c', weights, truth_q[frame]**2))
                    rms[frame, method] = absolute/scale[list(FIELDS)]
                    relative[frame, method] = np.divide(absolute, norm, out=np.full(2, np.nan), where=norm > 0)
            del states
    return times, valid, values, rms, relative, scale[list(FIELDS)]


def find_encoder(requested, output):
    if requested:
        return str(Path(requested).resolve())
    if shutil.which('ffmpeg'):
        return shutil.which('ffmpeg')
    for parent in [output, *output.parents]:
        candidates = sorted(parent.glob('dependencies/imageio_ffmpeg/binaries/ffmpeg*'))
        if candidates:
            return str(candidates[0])
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError('Supply --ffmpeg pointing to an installed encoder') from error


def timing_label(metadata, name):
    method = metadata.get('methods', {}).get(name, {})
    seconds = method.get('rollout_seconds')
    if seconds is None:
        return 'Reference timing unavailable' if name == 'ground_truth' else 'Timing unavailable'
    scope = method.get('timing_label', 'native CPU run' if name == 'ground_truth' else 'GPU rollout')
    return f'{float(seconds):.3g} s · {scope}'


def view_data(view, values, valid, scale):
    fields = range(values.shape[-1])
    if view == 'fields':
        shown = values
        limits = [(float(np.nanmin(values[:, 0, :, field])), float(np.nanmax(values[:, 0, :, field]))) for field in fields]
        title = 'Plasma fields'
        note = 'Shared limits from all available ground-truth frames; saturated display cells are reported.'
    elif view == 'change':
        shown = values - values[0, 0][None, None]
        limits = [(-float(np.nanmax(np.abs(shown[:, 0, :, field]))), float(np.nanmax(np.abs(shown[:, 0, :, field])))) for field in fields]
        title = 'Change from the common initial plasma'
        note = 'Same initial state; shared limits from ground-truth change over time. Saturation is reported.'
    else:
        shown = (values - values[:, :1])/scale
        shown[:, 0] = (values[:, 0] - values[0, 0])/scale
        limits = [(-float(np.nanmax(np.abs(shown[..., field]))), float(np.nanmax(np.abs(shown[..., field])))) for field in fields]
        title = 'Forecast error / training field scale'
        note = 'Ground-truth column: change / scale. Other columns: (prediction − truth) / scale. Fixed global limits.'
    limits = [(lo, hi) if hi > lo else (lo-1e-12, hi+1e-12) for lo, hi in limits]
    shown_valid = valid.copy()
    if view == 'error':
        shown_valid &= valid[:, :1]
    return shown, shown_valid, limits, title, note


def render_view(args, metadata, geometry, polygons, times, valid, values, rms, scale, view, encoder):
    shown, shown_valid, limits, title_text, note = view_data(view, values, valid, scale)
    case_id = str(metadata['case_id'])
    stem = f'{case_id}_{view}' if args.fields == 'both' else f'{case_id}_{args.fields}_{view}'
    field_names = FIELD_NAMES if args.fields == 'both' else (args.fields.title(),)
    if len(field_names) == 1:
        title_text = f'{field_names[0]} · {title_text.lower()}'
    plt.rcParams.update({'font.size': 13, 'font.family': 'DejaVu Sans', 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(len(field_names), 4, figsize=(19.2, 10.8), dpi=100, squeeze=False)
    fig.subplots_adjust(left=.045, right=.925, top=.84, bottom=.26, hspace=.38, wspace=.35)
    heading = fig.suptitle('', fontsize=22, x=.5, y=.975)
    fig.text(.5, .905, 'Native MAST-U / JOREK · axisymmetric n = 0 · identical initial state', ha='center', fontsize=15)
    artists, metric_text, unavailable = [], [], []
    rmin, zmin = polygons.min(axis=(0, 1))
    rmax, zmax = polygons.max(axis=(0, 1))
    for field, row in enumerate(axes):
        for method, ax in enumerate(row):
            cmap = matplotlib.colormaps['viridis' if view == 'fields' else 'RdBu_r'].copy()
            cmap.set_bad('#eeeeee')
            norm = Normalize(*limits[field]) if view == 'fields' else TwoSlopeNorm(vmin=limits[field][0], vcenter=0, vmax=limits[field][1])
            artist = PolyCollection(polygons, array=shown[0, method, :, field], cmap=cmap, norm=norm,
                                    edgecolors='none', antialiaseds=False, rasterized=True)
            ax.add_collection(artist)
            ax.set(xlim=(rmin-.02, rmax+.02), ylim=(zmin-.03, zmax+.03), aspect='equal', xlabel='R [m]')
            if method == 0:
                ax.set_ylabel(f'{field_names[field]}\nZ [m]')
            ax.tick_params(labelsize=11)
            if field == 0:
                label = metadata.get('methods', {}).get(METHODS[method], {}).get('label', LABELS[method])
                ax.set_title(f'{label}\n{timing_label(metadata, METHODS[method])}', fontsize=14, pad=9)
            metric_y = -.19 if len(field_names) == 2 else -.11
            metric_text.append(ax.text(.5, metric_y, '', ha='center', va='top', transform=ax.transAxes, fontsize=11))
            unavailable.append(ax.text(.5, .5, '', ha='center', va='center', transform=ax.transAxes,
                                       fontsize=13, color='#7a2222', bbox=dict(facecolor='white', edgecolor='none', alpha=.95)))
            artists.append(artist)
        position = row[-1].get_position()
        cax = fig.add_axes([.925, position.y0, .012, position.height])
        bar = fig.colorbar(artists[-1], cax=cax, extend='both' if view != 'error' else 'neither')
        bar.set_label('Simulation units' if view != 'error' else 'Training-scale units', fontsize=12)
        bar.ax.tick_params(labelsize=11)
    fig.text(.5, .12, r'$(M_*-hL_*)\,\Delta q=h\,[F_{\mathrm{native}}(q)+D_\theta(q)]$',
             ha='center', fontsize=20)
    fig.text(.5, .08, r'$D_\theta(q)=\mathcal{A}[G_\theta(q)-G_\theta(q_*)],\quad q^+=\mathcal{P}(q+\Delta q)$' +
             '   ·   IMEX interior stage; native constraints + physical guards',
             ha='center', fontsize=15)
    fig.text(.5, .042, note, ha='center', fontsize=13)
    fig.text(.5, .017, 'RMS error uses native volume quadrature / training field scale. Native CPU and GPU timings are different hardware.',
             ha='center', fontsize=11, color='#444444')
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    target = args.output / f'{stem}.mp4'
    partial = args.output / f'{stem}.partial.mp4'
    command = [encoder, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{width}x{height}',
               '-r', str(args.fps), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast',
               '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)]
    hold = round(args.hold_seconds*args.fps)
    previews, preview_durations, saturation = [], [], np.zeros((len(times), 4, len(field_names)))
    with (args.output / f'{stem}.encoder.log').open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for frame, time in enumerate(times):
                elapsed = (time-times[0])*1e6
                heading.set_text(f'{case_id} · {title_text}\nElapsed {elapsed:.4f} μs · step {frame}/{len(times)-1}')
                for field in range(len(field_names)):
                    for method in range(4):
                        index = field*4+method
                        data = shown[frame, method, :, field]
                        artists[index].set_array(np.ma.masked_invalid(data))
                        is_valid = shown_valid[frame, method]
                        unavailable[index].set_text('' if is_valid else ('Ground truth\nunavailable' if method == 0 else
                                                                       'No reference\nfor error' if view == 'error' and not valid[frame, 0]
                                                                       else 'Rollout stopped\nNo forecast'))
                        if is_valid:
                            fraction = float(np.mean((data < limits[field][0]) | (data > limits[field][1])))
                            saturation[frame, method, field] = fraction
                            error = rms[frame, method, field]
                            caption = ('Reference change / scale' if view == 'error' and method == 0 else
                                       f'RMS error {error:.3g}' if np.isfinite(error) else 'No ground-truth error available')
                            if view != 'error':
                                caption += f'\nDisplay cells outside range {100*fraction:.1f}%'
                            metric_text[index].set_text(caption)
                        else:
                            metric_text[index].set_text('')
                fig.canvas.draw()
                rgba = fig.canvas.buffer_rgba()
                repeat = 1 + (hold if frame in (0, len(times)-1) else 0)
                for _ in range(repeat):
                    process.stdin.write(rgba)
                if frame in (0, len(times)-1):
                    fig.savefig(args.output / f'{stem}_{"initial" if frame == 0 else "final"}.png', dpi=100)
                stride = max(1, round(args.fps))
                if frame % stride == 0 or frame == len(times)-1:
                    preview = Image.frombuffer('RGBA', (width, height), rgba, 'raw', 'RGBA', 0, 1)
                    previews.append(preview.convert('RGB').resize((960, 540), Image.Resampling.LANCZOS).convert('P', palette=Image.Palette.ADAPTIVE))
                    preview_durations.append(round(1000*(stride/args.fps + (args.hold_seconds if frame in (0, len(times)-1) else 0))))
                if frame % 24 == 0:
                    print(f'{stem}: rendered {frame+1}/{len(times)}', flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f'Encoding failed; see {stem}.encoder.log')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            plt.close(fig)
    verified = subprocess.run([encoder, '-v', 'error', '-i', str(partial), '-map', '0:v:0', '-f', 'null', '-',
                               '-progress', 'pipe:1', '-nostats'], capture_output=True, text=True, check=True)
    counts = [int(line.split('=')[1]) for line in verified.stdout.splitlines() if line.startswith('frame=')]
    expected_frames = len(times)+2*hold
    if not counts or counts[-1] != expected_frames:
        raise ValueError(f'Decoded video has {counts[-1] if counts else 0}, expected {expected_frames} frames')
    partial.replace(target)
    gif = args.output / f'{stem}_preview.gif'
    previews[0].save(gif, save_all=True, append_images=previews[1:], duration=preview_durations, loop=0, optimize=True)
    result = dict(view=view, fields=list(field_names), case_id=case_id, trajectory_sha256=args.trajectory_digest,
                  path=str(target), sha256=sha256(target), width=width, height=height,
                  frames=expected_frames, physical_frames=len(times), fps=args.fps, duration_seconds=expected_frames/args.fps,
                  decoded_frames=counts[-1], preview=str(gif), color_limits=limits,
                  saturated_display_cell_fraction=saturation.tolist(), color_policy=note)
    (args.output / f'{stem}_render.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--views', nargs='+', choices=('fields', 'change', 'error'), default=['fields', 'change', 'error'])
    parser.add_argument('--fields', choices=('both', 'density', 'temperature'), default='both')
    parser.add_argument('--fps', type=float, default=8)
    parser.add_argument('--hold-seconds', type=float, default=1)
    parser.add_argument('--subdivisions', type=int, default=2)
    parser.add_argument('--ffmpeg')
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.hold_seconds < 0 or not 1 <= args.subdivisions <= 8:
        parser.error('Require fps > 0, hold-seconds >= 0 and subdivisions in [1,8]')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text())
    restart = Path(metadata['restart_path'])
    if not restart.is_absolute():
        restart = args.metadata.parent / restart
    trajectory_digest, restart_digest = sha256(args.trajectory), sha256(restart)
    args.trajectory_digest = trajectory_digest
    for name, actual in (('trajectory_sha256', trajectory_digest), ('restart_sha256', restart_digest)):
        if name in metadata and metadata[name] != actual:
            raise ValueError(f'Input identity differs from metadata: {name}')
    geometry, polygons, basis = display_mesh(restart, args.subdivisions)
    times, valid, values, rms, relative, scale = sample_trajectories(args.trajectory, metadata, geometry, basis)
    selected = [0, 1] if args.fields == 'both' else [0 if args.fields == 'density' else 1]
    values, rms, relative, scale = values[..., selected], rms[..., selected], relative[..., selected], scale[selected]
    print(f'Sampled {len(times)} frames on {len(polygons)} native display cells', flush=True)
    encoder = find_encoder(args.ffmpeg, args.output)
    records = [render_view(args, metadata, geometry, polygons, times, valid, values, rms, scale, view, encoder) for view in args.views]
    report = dict(status='complete', case_id=metadata['case_id'], trajectory=str(args.trajectory.resolve()), trajectory_sha256=trajectory_digest,
                  metadata_sha256=sha256(args.metadata), restart=str(restart.resolve()), restart_sha256=restart_digest,
                  renderer_sha256=sha256(__file__), fields=[FIELD_NAMES[index] for index in selected], methods=list(METHODS),
                  time_seconds=times.tolist(), valid=valid.tolist(), normalization_scale=scale.tolist(),
                  native_volume_normalized_rms=np.where(np.isfinite(rms), rms, None).tolist(),
                  native_volume_relative_l2=np.where(np.isfinite(relative), relative, None).tolist(),
                  rendering='Native bicubic center samples on curved-element display subcells; no Cartesian interpolation.',
                  field_units='JOREK simulation units; no unverified SI conversion.',
                  extrapolation='Unavailable ground truth or stopped trajectories are blank; errors require valid ground truth.',
                  videos=records)
    report_name = str(metadata['case_id']) + (f'_{args.fields}' if args.fields != 'both' else '')
    report_path = args.output / f'{report_name}_video_report.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    sections = []
    gallery = [json.loads(path.read_text()) for path in sorted(args.output.glob(f'{metadata["case_id"]}*_render.json'))]
    for record in gallery:
        if record.get('trajectory_sha256') != trajectory_digest:
            continue
        name = Path(record['path']).name
        label = ', '.join(record['fields'])+' · '+record['view']
        sections.append(f'<h2>{html.escape(label)}</h2><video controls preload="metadata" width="960" src="{html.escape(name)}"></video>'
                        f'<p><a href="{html.escape(name)}">Download MP4</a> · <a href="{html.escape(Path(record["preview"]).name)}">GIF preview</a></p>')
    (args.output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MAST-U comparison</title>'
        '<style>body{font:18px system-ui;max-width:1100px;margin:40px auto;color:#171717}video{max-width:100%}</style>'
        f'<h1>{html.escape(str(metadata["case_id"]))}: matched MAST-U plasma rollouts</h1>'
        '<p>Native axisymmetric JOREK simulation. Density and temperature use shared scales, fixed over time. '
        'GPU rollout timing is displayed separately from native CPU reference timing.</p>'+''.join(sections)+
        f'<p><a href="{html.escape(report_path.name)}">Rendering and error provenance</a></p>')
    print(json.dumps({'report': str(report_path), 'videos': [record['path'] for record in records]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
