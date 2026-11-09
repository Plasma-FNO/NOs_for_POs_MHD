"""Verified, trajectory-disjoint native data windows shared by AR, NODE and OpsSplit."""
from dataclasses import fields
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import sha256_file
from .numerical import NativeGeometry


class EulerDataset:
    """Load shared geometry and training-only normalization once; stream field windows."""
    def __init__(self, manifest):
        self.root = Path(manifest).resolve().parent
        self.manifest = json.loads(Path(manifest).read_text())
        m = self.manifest
        if m.get('schema') != 'mastu-euler-dataset-v1' or not m.get('ready_for_euler_comparison') or m.get('synthetic'):
            raise ValueError('Expected an accepted native Euler dataset')
        self.cases = m['cases']
        for key in ('case_id', 'parent_family_id'):
            if len({c[key] for c in self.cases}) != len(self.cases):
                raise ValueError('Duplicate trajectory or parent family across splits')
        if len({c['initial_check']['sha256'] for c in self.cases}) != len(self.cases):
            raise ValueError('Duplicate initial states across splits')
        if any(c['split'] not in ('train', 'validation', 'test') for c in self.cases):
            raise ValueError('Unknown trajectory split')
        counts = {s: sum(c['split'] == s for c in self.cases) for s in ('train', 'validation', 'test')}
        if counts != m['split_counts'] or not all(counts.values()):
            raise ValueError('Incomplete independent trajectory splits')
        with h5py.File(self.verified_path(m['geometry'])) as f:
            self.geometry = NativeGeometry(**{v.name: f[v.name][()] for v in fields(NativeGeometry)}).validate()
        with np.load(self.verified_path(m['normalization']), allow_pickle=False) as f:
            self.mean = f['mean']; self.scale = f['scale']; self.time_seconds = f['time_seconds']
            fitted = set(f['training_case_ids'].tolist())
        if fitted != {c['case_id'] for c in self.cases if c['split'] == 'train'} or m['normalization']['fitted_split'] != 'train':
            raise ValueError('Normalization must use exactly the training trajectories')
        if self.mean.shape != (len(self.geometry.dof_ids), 7) or self.scale.shape != (7,) or \
           not np.isfinite(self.mean).all() or not np.all(np.isfinite(self.scale) & (self.scale > 0)):
            raise ValueError('Invalid native normalization')
        if len(self.time_seconds) < 2 or not np.all(np.diff(self.time_seconds) > 0) or \
           not np.allclose(np.diff(self.time_seconds), m['observation_interval_seconds'], rtol=1e-10, atol=1e-18):
            raise ValueError('Invalid observation times')
        for case in self.cases:
            if case['result']['status'] != 'accepted':
                raise ValueError('Rejected case in release')
            case['resolved_path'] = self.verified_path(case['result']['data'])
            with h5py.File(case['resolved_path']) as f:
                metadata = json.loads(f['metadata_json'][()] if 'metadata_json' in f else f.attrs['metadata_json'])
                if any(metadata[key] != case[key] for key in ('case_id', 'parent_family_id', 'split')):
                    raise ValueError('Trajectory identity does not match manifest')
                if f['native_state'].shape != (len(self.time_seconds), *self.mean.shape):
                    raise ValueError('Trajectory shape does not match geometry/times')
                np.testing.assert_array_equal(f['time_seconds'][()], self.time_seconds)

    def verified_path(self, record):
        path = (self.root / record['path']).resolve()
        if not path.is_relative_to(self.root) or sha256_file(path) != record['sha256']:
            raise ValueError('Dataset artifact path or hash mismatch')
        return path

    def windows(self, split, horizon, *, stride=1):
        return EulerWindows(self, split, horizon, stride)


class EulerWindows(Dataset):
    def __init__(self, data, split, horizon, stride):
        if split not in ('train', 'validation', 'test') or type(horizon) is not int or horizon < 1 or \
           type(stride) is not int or stride < 1:
            raise ValueError('Invalid split, horizon or observation stride')
        self.data, self.horizon, self.stride = data, horizon, stride
        self.cases = [c for c in data.cases if c['split'] == split]
        self.starts = len(data.time_seconds[::stride]) - horizon
        if self.starts < 1:
            raise ValueError('Rollout horizon exceeds available observations')
        self.observation_interval_seconds = data.manifest['observation_interval_seconds'] * stride

    def __len__(self):
        return len(self.cases) * self.starts

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        case_index, start = divmod(index, self.starts)
        case = self.cases[case_index]
        selection = slice(start*self.stride, (start+self.horizon+1)*self.stride, self.stride)
        with h5py.File(case['resolved_path']) as f:
            state = f['native_state'][selection]
        if not np.isfinite(state).all():
            raise ValueError('Nonfinite native state window')
        # Center in float64 before any neural-network float32 conversion.
        normalized = (state - self.data.mean) / self.data.scale
        return dict(state=torch.from_numpy(normalized), case_id=case['case_id'],
                    time_seconds=torch.from_numpy(self.data.time_seconds[selection].copy()),
                    parameters=torch.ones(1), start_frame=start*self.stride)
