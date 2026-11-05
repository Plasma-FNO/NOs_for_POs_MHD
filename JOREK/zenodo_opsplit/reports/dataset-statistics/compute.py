"""Exact statistics for the source-valid cohorts used by the ES/EM campaigns."""
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

torch.set_num_threads(1)
BASE = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-matched-20261104')
OUT = Path(__file__).resolve().parent
SCALES = [1e20, 1., 1e3, 1., .1, 1.]
NAMES = ['rho', 'T', 'potential', 'omega', 'psi', 'j']


def statistics(values):
    assert values.size and np.isfinite(values).all()
    percentiles = np.percentile(values, [1, 5, 50, 95, 99])
    return dict(count=int(values.size), minimum=float(values.min()),
                maximum=float(values.max()), mean=float(values.mean(dtype=np.float64)),
                std=float(values.std(dtype=np.float64, ddof=0)),
                **dict(zip(['p01', 'p05', 'median', 'p95', 'p99'], map(float, percentiles))))


np.testing.assert_allclose([statistics(np.array([0., 2.]))[k] for k in ('mean', 'std', 'median')], [1, 1, 1])
report = dict(scope='Matched ES100/EM76 campaign cohorts; train/val frames 0..49, test source-valid prefix up to frame 200.',
              weighting='Each eligible frame/grid cell has equal weight; no cylindrical R weighting. Population standard deviation. Exact percentiles, no sampling.',
              precision='Statistics of loader float32 exported values, accumulated in float64. Original-export statistics undo fixed loader preconditioning; not a reread of original higher-precision values.',
              units='The release has no unit attributes. T is treated as eV by the existing adapter. Other fields retain release conventions.',
              cases={})
rows = []
for case in ('es', 'em'):
    folder = BASE / case / 'data'
    meta = json.loads((folder / 'comparison.json').read_text())
    cache = torch.load(folder / 'data.pt', map_location='cpu', weights_only=False, mmap=True)
    selected = {i: set() for i in range(len(meta['entries']['train']))}
    for i, start in meta['windows']:
        selected[i].update(range(start, start + meta['protocol']['horizon'] + 1))
    selection = {'train': {i: sorted(frames) for i, frames in selected.items() if frames},
                 'val': {i: list(range(meta['protocol']['train_frames'])) for i in meta['val_indices']},
                 'test': {i: list(range(min(entry['frames'], meta['protocol']['eval_horizon'] + 1,
                                             meta['test_prefixes'][entry['id']] + 1)))
                          for i, entry in enumerate(meta['entries']['test']) if meta['test_prefixes'][entry['id']] >= 0}}
    info = dict(data=str(folder), saved_data_sha256=meta['data_sha256'], splits={})
    keys = ['rho', 'T', 'Phi' if case == 'es' else 'u', 'omega'] + (['Psi', 'zj'] if case == 'em' else [])
    for split, indices in selection.items():
        cohort = dict(selected_trajectories=len(meta['entries'][split]), included_trajectories=len(indices),
                      frames=sum(map(len, indices.values())),
                      excluded_frames=sum(min(e['frames'], 201 if split == 'test' else 50) for e in meta['entries'][split]) - sum(map(len, indices.values())),
                      trajectory_frames={meta['entries'][split][i]['id']: len(frames) for i, frames in indices.items()}, fields={})
        for field, key in enumerate(keys):
            chunks = []
            for i, frames in indices.items():
                if split == 'test':
                    with h5py.File(meta['entries'][split][i]['path'], 'r') as handle:
                        values = (handle[key][:len(frames)].astype(np.float64) / SCALES[field]).astype(np.float32)
                else:
                    values = cache[split]['exported'][i, frames, field].numpy()
                chunks.append(values.ravel())
            values = np.concatenate(chunks)
            stats = statistics(values)
            center, scale = float(cache['mean'][field]), float(cache['scale'][field])
            if split == 'train':
                np.testing.assert_allclose([(stats['minimum']-center)/scale, (stats['maximum']-center)/scale], [-1, 1], atol=2e-5, rtol=0)
            stats.update(normalization_center=center, normalization_half_range=scale,
                         normalized_minimum=(stats['minimum']-center)/scale,
                         normalized_maximum=(stats['maximum']-center)/scale,
                         nonpositive_count=int(np.count_nonzero(values <= 0)))
            original = {name: value if name == 'count' else value*SCALES[field] for name, value in stats.items()
                        if name in ('count', 'minimum', 'maximum', 'mean', 'std', 'p01', 'p05', 'median', 'p95', 'p99')}
            cohort['fields'][NAMES[field]] = dict(release_key=key, loader_divisor=SCALES[field], exported=stats, original_export=original)
            rows.append(dict(case=case, split=split, field=NAMES[field], release_key=key,
                             loader_divisor=SCALES[field], trajectories=len(indices), frames=cohort['frames'], **stats,
                             **{'original_'+k: v for k,v in original.items() if k != 'count'}))
            print(json.dumps(dict(case=case, split=split, field=NAMES[field], **stats)), flush=True)
            del values, chunks
        info['splits'][split] = cohort
    report['cases'][case] = info
    (OUT / 'statistics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    del cache
with (OUT / 'statistics.csv').open('w') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('Completed; all training extrema reproduce the saved normalizers.', flush=True)
