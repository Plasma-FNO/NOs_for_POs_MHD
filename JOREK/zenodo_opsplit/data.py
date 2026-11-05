"""Lazy release reader. Splits are by blob geometry, shared across both regimes."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

FIELDS = ['rho', 'T', 'u', 'omega', 'psi', 'zj']
# Numerical preconditioning only: the release contains no unit attributes.
SCALES = np.array([1e20, 1., 1e3, 1., .1, 1.], dtype=np.float64)


def family(text):
    values = {}
    for line in text.splitlines():
        line = line.split('!')[0].strip().lower()
        match = re.fullmatch(r'(n_jropes|[rzw]_jropes\(\d+\))\s*=\s*([\de.+-]+)', line)
        if match:
            values[match[1]] = float(match[2])
    n = int(values['n_jropes'])
    blobs = sorted(tuple(round(values[f'{key}_jropes({i})'], 8) for key in ('r', 'z', 'w')) for i in range(1, n + 1))
    return hashlib.sha256(json.dumps(blobs).encode()).hexdigest()


def split_for(group):
    bucket = int(group[:8], 16) % 10
    return 'test' if bucket == 0 else 'val' if bucket == 1 else 'train'


def layout(handle, case):
    keys = ['rho', 'T', 'Phi' if case == 'es' else 'u', 'omega']
    if case == 'em':
        keys += ['Psi', 'zj']
    r = np.asarray(handle['R_mesh(nR,nZ)'])
    z = np.asarray(handle['Z_mesh(nR,nZ)'])
    # Release arrays are [time, Z, R], despite the mesh dataset names.
    if not (np.allclose(r, r[0:1, :]) and np.allclose(z, z[:, 0:1])):
        raise ValueError('Expected rectilinear [Z,R] release grid')
    if not (np.all(np.diff(r[0]) > 0) and np.all(np.diff(z[:, 0]) > 0)):
        raise ValueError('Non-increasing coordinates')
    for key in keys:
        if handle[key].shape != (int(handle['n_times'][0]), *r.shape):
            raise ValueError(f'Unexpected field shape: {key}')
    return keys, r[0], z[:, 0]


def read_frames(handle, keys, start, stop):
    x = np.stack([handle[key][start:stop] for key in keys], axis=1)
    x = (x / SCALES[None, :len(keys), None, None]).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite trajectory; refusing silent replacement')
    return x


def inspect_file(path, case):
    with h5py.File(path, 'r') as f:
        keys, r, z = layout(f, case)
        nt = int(f['n_times'][0])
        if nt < 3:
            raise ValueError(f'Too few frames: {path}')
        minimum, maximum = np.full(len(keys), np.inf), np.full(len(keys), -np.inf)
        for start in range(0, nt, 32):
            x = read_frames(f, keys, start, min(start + 32, nt))
            minimum = np.minimum(minimum, x.min(axis=(0, 2, 3)))
            maximum = np.maximum(maximum, x.max(axis=(0, 2, 3)))
        return {'frames': nt, 'shape': [len(z), len(r)], 'r': r.tolist(), 'z': z.tolist(),
                'min_preconditioned': minimum.tolist(), 'max_preconditioned': maximum.tolist()}


def prepare(root, output):
    if output.exists():
        raise FileExistsError(output)
    entries = []
    reference = None
    for case in ('es', 'em'):
        for path in sorted((root / case / 'raw').glob('*.h5')):
            input_path = path.with_name(path.stem.replace('jorek_run', 'jorek_input_run') + '.txt')
            group = family(input_path.read_text())
            info = inspect_file(path, case)
            grid = [info['r'], info['z']]
            if reference is not None and grid != reference:
                raise ValueError(f'Grid differs: {path}')
            reference = grid
            entries.append({'case': case, 'id': path.stem, 'path': str(path.resolve()),
                            'input': str(input_path.resolve()), 'family': group,
                            'split': split_for(group), **info})
    if not entries:
        raise ValueError('No extracted HDF5 files found')
    counts = {case: {s: sum(e['case'] == case and e['split'] == s for e in entries)
                     for s in ('train', 'val', 'test')} for case in ('es', 'em')}
    if any(n == 0 for counts_case in counts.values() for n in counts_case.values()):
        raise ValueError(f'Need nonempty train/val/test in each regime: {counts}')
    manifest = {'schema': 1, 'fields': FIELDS, 'preconditioning': SCALES.tolist(),
                'time_basis': 'saved_frame_index; physical timestamps absent',
                'split_rule': 'SHA256(sorted blob R,Z,width rounded to 8 decimals), first 8 hex mod 10: test=0,val=1,train=2..9',
                'records': {'es': '11099659', 'em': '11099527'}, 'counts': counts, 'entries': entries}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(counts), flush=True)


def select(manifest, case, split, limit=0, seed=0):
    entries = [e for e in manifest['entries'] if e['case'] == case and e['split'] == split]
    # Stable nested subsets for each seed; never sample windows across split boundaries.
    entries.sort(key=lambda e: hashlib.sha256(f"{seed}:{e['family']}:{e['id']}".encode()).hexdigest())
    return entries[:limit] if limit else entries


def statistics(entries):
    count = 0
    total = square = None
    for entry in entries:
        with h5py.File(entry['path'], 'r') as f:
            keys, _, _ = layout(f, entry['case'])
            for start in range(0, entry['frames'], 32):
                x = read_frames(f, keys, start, min(start + 32, entry['frames'])).astype(np.float64)
                sums = x.sum(axis=(0, 2, 3))
                squares = (x*x).sum(axis=(0, 2, 3))
                total = sums if total is None else total + sums
                square = squares if square is None else square + squares
                count += x.shape[0] * x.shape[2] * x.shape[3]
    if count == 0:
        raise ValueError('No training frames')
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean**2, 1e-12))
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


class Windows(Dataset):
    def __init__(self, entries, horizon, stride=1):
        if horizon < 1 or stride < 1:
            raise ValueError('Positive horizon and stride required')
        self.entries, self.horizon = entries, horizon
        self.index = [(i, t) for i, e in enumerate(entries) for t in range(0, e['frames'] - horizon, stride)]
        if not self.index:
            raise ValueError('No windows for the requested horizon')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        row, start = self.index[i]
        entry = self.entries[row]
        with h5py.File(entry['path'], 'r') as f:
            keys, _, _ = layout(f, entry['case'])
            x = read_frames(f, keys, start, start + self.horizon + 1)
        return torch.from_numpy(x)


if __name__ == '__main__':
    from download import DEFAULT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.output)
