"""Exact native remainder labels N(q) = F(q) - F(q*) - L*(q - q*) for training states.

Each label comes from one native export step of the patched model303 build
(OPSSPLIT_MODE=export), restarted from the dataset's own saved restart. That build
checks, at every quadrature point, that the closed-form physical coefficients
reproduce the untouched native RHS before writing them. Labels keep the 13 active
weak channels of ImplicitOpsSplitModel and are verified against the dataset frame.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess

import h5py
import numpy as np

from .case import namelist_overrides
from .implicit_opsplit import ACTIVE_CHANNELS

RECORD = np.dtype([('vertices', '<i4', (4,)), ('query', '<i4'), ('data', '<f8', (364,))])
FIRST_FRAME_STEP, STEPS_PER_FRAME = 144, 4


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def restart_for_frame(run_root, case_id, frame):
    return Path(run_root)/f'{case_id}_fine'/'work'/f'jorek{FIRST_FRAME_STEP + STEPS_PER_FRAME*frame:06d}.h5'


def plan_exports(manifest_path, frames, *, splits=('train', 'validation')):
    """Every (case, frame) of the pooled 80 training trajectories; restarts are time-verified."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    times = None
    entries = []
    for case in manifest['cases']:
        if case['split'] not in splits:
            continue
        data = manifest_path.parent/case['result']['data']['path']
        if times is None:
            with h5py.File(data) as stream:
                times = stream['time_seconds'][()]
        for frame in frames:
            restart = restart_for_frame(manifest_path.parent/'runs', case['case_id'], frame)
            with h5py.File(restart) as stream:
                seconds = float(stream['t_now'][0])*float(stream['sqrt_mu0_rho0'][0])
                index = int(stream['index_now'][0])
            if abs(seconds-times[frame]) > 1e-15 or index != FIRST_FRAME_STEP + STEPS_PER_FRAME*frame:
                raise ValueError(f'Restart/frame mismatch for {case["case_id"]} frame {frame}')
            entries.append(dict(case_id=case['case_id'], frame=int(frame), restart=str(restart), data=str(data),
                                time_seconds=float(times[frame])))
    return entries


def read_export(files, vertices):
    """Map per-rank export records onto [element, quadrature, 364] in restart element order."""
    lookup = {tuple(v): i for i, v in enumerate(vertices)}
    out = np.full((len(vertices), 16, 364), np.nan)
    for path in files:
        if Path(path).stat().st_size % RECORD.itemsize:
            raise ValueError(f'Truncated native export {path}')
        for row in np.fromfile(path, dtype=RECORD):
            e, q = lookup[tuple(row['vertices'])], int(row['query'])-1
            if not 0 <= q < 16:
                raise ValueError('Invalid quadrature index')
            if np.isfinite(out[e, q]).all() and not np.array_equal(out[e, q], row['data']):
                raise ValueError('Conflicting duplicate native export records')
            out[e, q] = row['data']
    if not np.isfinite(out).all():
        raise ValueError('Missing or nonfinite native export records')
    return out


def remainder_from_export(export, frozen, geometry, state):
    """Return [E*Q, 13] float64 labels and verification metrics against the dataset frame."""
    if not np.allclose(export[..., :2], geometry.quadrature_rz, rtol=0, atol=1e-12):
        raise ValueError('Export quadrature ordering differs from the native geometry')
    local = state[geometry.dof_map]
    values = np.einsum('eql,elc->eqc', geometry.basis, local)
    gradients = np.einsum('eqlr,elc->eqcr', geometry.basis_grad, local)
    jets = export[..., 3:24]
    value_scale = np.abs(values).max(axis=(0, 1)) + 1e-300
    gradient_scale = np.abs(gradients).max(axis=(0, 1)) + 1e-300
    value_error = float((np.abs(jets[..., :7]-values).max(axis=(0, 1))/value_scale).max())
    gradient_error = float(max((np.abs(jets[..., 7:14]-gradients[..., 0]).max(axis=(0, 1))/gradient_scale[:, 0]).max(),
                               (np.abs(jets[..., 14:21]-gradients[..., 1]).max(axis=(0, 1))/gradient_scale[:, 1]).max()))
    if value_error > 1e-9 or gradient_error > 1e-7:
        raise ValueError(f'Exported state differs from dataset frame: {value_error}, {gradient_error}')
    physical = export[..., 34:49]
    if np.any(physical[..., [1, 2]] != 0):
        raise ValueError('Native induction flux channels are nonzero')
    linear = frozen.forcing.reshape(*frozen.forcing.shape[:2], 15) + np.einsum(
        'eqjo,eqj->eqo', frozen.tangent, jets-frozen.jets, optimize=True)
    remainder = physical-linear
    weights = geometry.volume_weights[..., None]
    norms = np.sqrt((physical**2*weights).sum((0, 1))/weights.sum())
    rel = np.sqrt((remainder**2*weights).sum((0, 1))/weights.sum())/np.maximum(norms, 1e-300)
    return (remainder[..., list(ACTIVE_CHANNELS)].reshape(-1, len(ACTIVE_CHANNELS)),
            dict(value_relative_error=value_error, gradient_relative_error=gradient_error,
                 remainder_over_physical_rms=dict(zip(map(str, ACTIVE_CHANNELS), rel[list(ACTIVE_CHANNELS)].tolist()))))


def run_export(restart, work, executable_dir, *, dt=0.0078125, timeout=900, namelist=None):
    """One native export step from a dataset restart; returns the per-rank export files.

    `namelist` adds physics overrides (e.g. {'D_perp(1)': 1e-4}) for coefficient checks; labels use none.
    """
    executable_dir, restart, work = Path(executable_dir), Path(restart), Path(work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with h5py.File(restart) as stream:
        start = int(stream['index_now'][0])
    shutil.copyfile(executable_dir/'eqdsk.dat', work/'eqdsk.dat')
    shutil.copyfile(restart, work/'jorek_restart.h5')
    shutil.copyfile(restart, work/f'jorek{start:06d}.h5')
    overrides = dict(restart=True, regrid=False, tstep_n=dt, nstep_n=1, nout=1, write_ps=False)
    if namelist:
        if set(namelist) & set(overrides):
            raise ValueError('Physics overrides may not change the export control settings')
        overrides.update(namelist)
    (work/'input').write_text(namelist_overrides((executable_dir/'input').read_text(), overrides))
    env = dict(os.environ, OPSSPLIT_MODE='export', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (work/'input').open('rb') as stdin, (work/'stdout.log').open('wb') as stdout, (work/'stderr.log').open('wb') as stderr:
        process = subprocess.Popen(['mpirun', '--bind-to', 'none', '-n', '4', str(executable_dir/'jorek.exe')],
                                   cwd=work, stdin=stdin, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise RuntimeError(f'Native export timed out: {restart}')
    if code != 0:
        tail = (work/'stdout.log').read_text(errors='replace')[-2000:] + (work/'stderr.log').read_text(errors='replace')[-2000:]
        raise RuntimeError(f'Native export failed ({code}) for {restart}: {tail}')
    files = sorted(work.glob('opsplit_export_*.bin'))
    if len(files) != 4:
        raise RuntimeError(f'Expected four rank exports, found {len(files)}')
    return files


def label_scale(paths, weights):
    """Volume-weighted RMS per active channel over all supplied (training) label files."""
    w = weights.ravel()/weights.sum()
    total = np.zeros(len(ACTIVE_CHANNELS))
    for path in paths:
        label = np.load(path).astype(np.float64)
        total += np.einsum('pc,p->c', label**2, w)
    scale = np.sqrt(total/len(paths))
    if not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError('Invalid remainder label scale')
    return scale
