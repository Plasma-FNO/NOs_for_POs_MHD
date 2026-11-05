"""Frozen data contract and common observed-field scores for the four-arm comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from data import SCALES, layout, select
from native import decode_release, encode_release, common_coefficients, read_input, wall_trace, NativeReference
from native_terms import features, labels
from train_es import range_statistics

ARMS = ('ar', 'node', 'opsplit-scratch', 'opsplit-pretrained')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_fields(entry, frames):
    with h5py.File(entry['path']) as f:
        keys, _, _ = layout(f, entry['case'])
        raw = torch.from_numpy(np.stack([f[k][:frames] for k in keys], axis=1)).double()
    exported = (raw / torch.tensor(SCALES[:len(keys)])[None, :, None, None]).float()
    return exported, decode_release(raw)


def valid_states(exported, q):
    return (torch.isfinite(exported).flatten(1).all(1) & torch.isfinite(q).flatten(1).all(1)
            & (q[:, :2] > 0).flatten(1).all(1))


def window_index(valid, horizon):
    return [s for s in range(len(valid)-horizon) if bool(valid[s:s+horizon+1].all())]


def covered_states(starts, horizon):
    return sorted({t for s in starts for t in range(s, s+horizon+1)})


def reference_for(entry, initial):
    c = common_coefficients(read_input(entry['input']), entry['case'])
    c['viscosity'] = 'old'  # Release-profile inference; same closure in both native arms.
    r, z = torch.tensor(entry['r'], dtype=torch.float64), torch.tensor(entry['z'], dtype=torch.float64)
    return NativeReference(r, z, wall_trace(initial, r, z, c), c)


def decoded_fields(reference, q):
    omega, current = reference.constraints(q)
    raw = encode_release(q, omega[..., 1:-1, 1:-1], current[..., 1:-1, 1:-1])
    return raw / q.new_tensor(SCALES[:raw.shape[-3]])[None, :, None, None]


def normalize(x, mean, scale):
    return (x-mean[:, None, None])/scale[:, None, None]


def frame_sums(prediction, truth, mean, scale, r):
    """Per-field sufficient statistics; preconditioning cancels in decoded relative L2."""
    p, y = normalize(prediction.double(), mean, scale), normalize(truth.double(), mean, scale)
    error = (prediction.double()-truth.double()).square()
    weight = r.to(error)[None, :].expand_as(error[0])
    weight = weight/weight.sum()
    return dict(error=(p-y).square().mean((-2, -1)), energy=y.square().mean((-2, -1)),
                physical_error=(error*weight).sum((-2, -1)),
                physical_energy=(truth.double().square()*weight).sum((-2, -1)))


def scores(sums):
    """Paper eq.13 epsilon=1e-6 in normalized coordinates; cylindrical L2 separate."""
    return dict(nrmse=(sums['error']/(sums['energy']+1e-6)).sqrt().tolist(),
                r_weighted_relative_l2=(sums['physical_error'] /
                    sums['physical_energy'].clamp_min(1e-30)).sqrt().tolist())


def prepare(manifest_path, output, case, protocol_path):
    cfg = json.loads(protocol_path.read_text())
    if cfg.get('case', case) != case:
        raise ValueError('Protocol case mismatch')
    cfg['native_substeps'] = 8
    cfg['notes'] = dict(reference=cfg.get('reference'), shared='All four arms share data windows, five-step decoded rollout loss, epoch sampler, optimizer and validation selection.',
                        adaptation='Native IMEX, constraints and 2/2/1/1 FNO layers; pretraining is an additional JOREK ablation, not an NS-paper protocol claim.')
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text())
    meta = dict(schema=1, case=case, protocol=cfg, arms=list(ARMS), entries={},
                windows=[], excluded_windows=[], excluded_validation=[], test_prefixes={},
                manifest_sha256=digest(manifest_path),
                validity='finite exported/native fields and strictly positive native rho,p at every target and input frame; independent of model predictions',
                normalization='exported fields, train-only minmax over union of valid training windows',
                loss='sum of five relative L2 losses on all normalized exported fields; no term anchor',
                validation='mean 49-step free-rollout relative L2; all common valid validation prefixes',
                term_pretraining='250 extra epochs; FD equation targets, stride5 states belonging to common windows; train-only scales',
                metrics='paper eq13 NRMSE on range-normalized fields; R-weighted decoded relative L2; persistence; EM delta_psi; failure/completion counts',
                native_adaptations='rho,p,u,(psi) with reconstructed omega,j; native IMEX/CFL and wall closure, not the NS Euler split; nominal frame timestamps',
                source_files={})
    bundle = {}
    term_output = output/'terms'; term_output.mkdir()
    term_meta = dict(case=case, phase='term_pretraining', splits={},
                     target_kind='FD approximation of native equation terms; not JOREK-exported residuals',
                     selection='stride5 intersected with shared admissible training-window states or valid evaluation prefixes')
    fit = []
    for split in ('train', 'val', 'test'):
        entries = select(manifest, case, split, cfg['train_trajectories'] if split == 'train' else 0, cfg['seed'])
        if not entries or (split == 'train' and len(entries) != cfg['train_trajectories']):
            raise ValueError('Insufficient independent trajectories')
        meta['entries'][split] = entries
        exports, natives, xs, ys, ids, sample_frames = [], [], [], [], [], []
        for i, entry in enumerate(entries):
            n = cfg['eval_horizon']+1 if split == 'test' else cfg['train_frames']
            if split != 'test' and entry['frames'] < n:
                raise ValueError('Insufficient training/validation prefix: '+entry['id'])
            exported, q = load_fields(entry, min(entry['frames'], n))
            valid = valid_states(exported, q)
            if split == 'train':
                starts = window_index(valid, cfg['horizon']) if bool(valid[0]) else []
                meta['windows'].extend([[i, s] for s in starts])
                meta['excluded_windows'].extend(dict(id=entry['id'], start=s) for s in range(n-cfg['horizon']) if s not in starts)
                allowed = covered_states(starts, cfg['horizon'])
                if allowed: fit.append(exported[allowed])
            else:
                bad = torch.where(~valid)[0].tolist()
                prefix = (bad[0] if bad else len(valid))-1
                allowed = list(range(max(0, prefix+1)))
                if split == 'val' and prefix != cfg['train_frames']-1:
                    meta['excluded_validation'].append(dict(id=entry['id'], invalid_frames=bad))
                    allowed = []
                if split == 'test': meta['test_prefixes'][entry['id']] = prefix
            for path in (entry['path'], entry['input']):
                stat = Path(path).stat()
                meta['source_files'][path] = [stat.st_size, stat.st_mtime_ns]
            if split != 'test':
                exports.append(exported); natives.append(q)
            # Term supervision never uses extrapolation frames, even for diagnostic term tests.
            frames = [t for t in allowed if t < cfg['train_frames'] and t % 5 == 0]
            if frames:
                ref = reference_for(entry, q[:1])
                with torch.no_grad():
                    x, y = features(q[frames]).float(), labels(ref, q[frames]).float()
                if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                    raise ValueError('Nonfinite equation targets: '+entry['id'])
                xs.append(x); ys.append(y); ids.extend([entry['id']]*len(frames)); sample_frames.extend(frames)
        if not xs: raise ValueError('No valid term states in '+split)
        x, y = torch.cat(xs), torch.cat(ys)
        torch.save(dict(x=x, y=y, ids=ids, frames=sample_frames), term_output/f'{split}.pt')
        term_meta['splits'][split] = dict(samples=len(x), ids=[e['id'] for e in entries])
        if split == 'train':
            lo, hi = x.amin((0,2,3)), x.amax((0,2,3))
            norms = dict(input_mean=(lo+hi)/2, input_scale=((hi-lo)/2).clamp_min(1e-20),
                         target_scale=y[:,:,4:-4,4:-4].double().square().mean((0,2,3)).sqrt().float().clamp_min(1e-20))
            torch.save(norms, term_output/'normalization.pt')
        if split != 'test': bundle[split] = dict(exported=torch.stack(exports), q=torch.stack(natives))
        print(json.dumps(dict(case=case, split=split, trajectories=len(entries), term_states=len(x))), flush=True)
    if not meta['windows']: raise ValueError('No admissible training windows')
    rejected = {e['id'] for e in meta['excluded_validation']}
    meta['val_indices'] = [i for i,e in enumerate(meta['entries']['val']) if e['id'] not in rejected]
    if not meta['val_indices']: raise ValueError('No admissible validation prefixes')
    bundle['mean'], bundle['scale'] = range_statistics(torch.cat(fit)[None])
    torch.save(bundle, output/'data.pt')
    meta['data_sha256'] = digest(output/'data.pt')
    meta['term_files_sha256'] = {p.name:digest(p) for p in term_output.glob('*.pt')}
    write_json(output/'comparison.json', meta)
    term_meta['comparison_sha256'] = digest(output/'comparison.json')
    write_json(term_output/'manifest.json', term_meta)
    print(json.dumps(dict(case=case, windows=len(meta['windows']), excluded=len(meta['excluded_windows']),
                          validation=len(meta['val_indices']), output=str(output))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case', choices=['es','em'], required=True); p.add_argument('--protocol', type=Path)
    a = p.parse_args(); torch.set_num_threads(1)
    prepare(a.manifest, a.output, a.case, a.protocol or Path(__file__).with_name(f'{a.case}_paper_protocol.json'))
