"""Train/evaluate ES pretraining and controlled EM transfer experiments offline."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from data import FIELDS, SCALES, Windows, layout, read_frames, select, statistics
from model import SplitFNO


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_source(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = SplitFNO(checkpoint['r'], checkpoint['z'], **checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'])
    if checkpoint['case'] != 'es':
        raise ValueError('Source checkpoint must be electrostatic')
    return model, checkpoint


def rollout_loss(model, frames, mean, std, zero_shot=False):
    frames = (frames - mean[None, None, :, None, None]) / std[None, None, :, None, None]
    x = frames[:, 0]
    loss = x.new_zeros(())
    for t in range(1, frames.shape[1]):
        x = model(x, zero_shot=zero_shot)
        loss = loss + (x - frames[:, t]).square().mean()
    return loss / (frames.shape[1] - 1)


@torch.no_grad()
def evaluate(model, entries, mean, std, horizon, zero_shot=False):
    model.eval()
    result = []
    device = mean.device
    for entry in entries:
        steps = min(horizon, entry['frames'] - 1)
        with h5py.File(entry['path'], 'r') as f:
            keys, _, _ = layout(f, entry['case'])
            truth = torch.from_numpy(read_frames(f, keys, 0, steps + 1)).to(device)
        encoded = (truth - mean[None, :, None, None]) / std[None, :, None, None]
        x = encoded[0:1]
        errors, persistence, relative, persistence_relative = [], [], [], []
        failure = None
        # Toroidal-volume weights, uniform cell areas cancel in normalized averages.
        weight = model.r / model.r.mean()
        for step in range(1, steps + 1):
            x = model(x, zero_shot=zero_shot)
            error = ((x[0]-encoded[step]).double().square()*weight).mean(dim=(-2, -1))
            if not torch.isfinite(x).all() or not torch.isfinite(error).all():
                failure = f'nonfinite_at_frame_{step}'
                break
            base = ((encoded[0]-encoded[step]).double().square()*weight).mean(dim=(-2, -1))
            physical = x[0]*std[:, None, None]+mean[:, None, None]
            denominator = (truth[step].double().square()*weight).sum(dim=(-2, -1)).clamp_min(1e-24)
            rel = (((physical-truth[step]).double().square()*weight).sum(dim=(-2, -1))/denominator).sqrt()
            base_rel = (((truth[0]-truth[step]).double().square()*weight).sum(dim=(-2, -1))/denominator).sqrt()
            errors.append(error.cpu().tolist())
            persistence.append(base.cpu().tolist())
            relative.append(rel.cpu().tolist())
            persistence_relative.append(base_rel.cpu().tolist())
        item = {'id': entry['id'], 'family': entry['family'], 'requested_frames': steps,
                'completed': failure is None, 'failure': failure,
                'standardized_mse_by_frame': errors, 'persistence_standardized_mse_by_frame': persistence,
                'relative_l2_by_frame': relative, 'persistence_relative_l2_by_frame': persistence_relative}
        result.append(item)
    complete = [row for row in result if row['completed']]
    return {'fields': FIELDS[:model.channels], 'time_basis': 'saved_frame_index',
            'zero_shot_magnetic_policy': 'persistence; no learned induction' if zero_shot else None,
            'completed': len(complete), 'total': len(result),
            'mean_standardized_mse_conditional_on_completion': np.mean([np.mean(row['standardized_mse_by_frame'], axis=0) for row in complete], axis=0).tolist() if complete else None,
            'trajectories': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['source', 'zero-shot', 'frozen', 'finetune', 'scratch', 'monolithic'], required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-trajectories', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--steps-per-epoch', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--eval-horizon', type=int, default=100)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--modes', type=int, default=12)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if min(args.epochs, args.steps_per_epoch, args.batch_size, args.horizon, args.eval_horizon) < 1 or args.train_trajectories < 0:
        parser.error('Training lengths must be positive; trajectory limit must be nonnegative')
    transfer = args.mode in ('zero-shot', 'frozen', 'finetune')
    if transfer and args.source is None:
        parser.error('Transfer modes require --source')
    if not transfer and args.source is not None:
        parser.error('--source is only valid for transfer modes')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use Slurm or explicit --device cpu for smoke tests')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    case = 'es' if args.mode == 'source' else 'em'
    train = select(manifest, case, 'train', args.train_trajectories, args.seed)
    val, test = select(manifest, case, 'val'), select(manifest, case, 'test')
    if not train or not val or not test:
        raise ValueError('Nonempty trajectory splits required')
    if args.train_trajectories > len(train):
        raise ValueError('Requested more training trajectories than available')
    channels = 4 if case == 'es' else 6
    source, checkpoint = load_source(args.source) if transfer else (None, None)
    if transfer:
        if checkpoint['manifest_sha256'] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
            raise ValueError('Source and target must use the same audited manifest/family splits')
        config = {key: checkpoint['model_config'][key] for key in ('width', 'modes', 'layers')}
        if args.mode == 'zero-shot':
            # No target statistics or training samples contribute to zero-shot inference.
            mean = torch.cat((checkpoint['mean'], torch.zeros(2)))
            std = torch.cat((checkpoint['std'], torch.ones(2)))
        else:
            mean, std = statistics(train)
            mean[:4], std[:4] = checkpoint['mean'], checkpoint['std']
    else:
        config = dict(width=args.width, modes=args.modes, layers=args.layers)
        mean, std = statistics(train)
    model = SplitFNO(train[0]['r'], train[0]['z'], channels=channels, monolithic=args.mode == 'monolithic', **config)
    if transfer:
        model.transfer(source, freeze=args.mode in ('frozen', 'zero-shot'))
        if args.mode == 'zero-shot':
            model.requires_grad_(False)
    device = torch.device(args.device)
    model, mean, std = model.to(device), mean.to(device), std.to(device)
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    provenance.update(case=case, manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                      source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest() if transfer else None,
                      train_ids=[] if args.mode == 'zero-shot' else [e['id'] for e in train],
                      val_ids=[e['id'] for e in val], test_ids=[e['id'] for e in test],
                      parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      torch_version=torch.__version__, preconditioning=SCALES.tolist(),
                      code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    write_json(args.output / 'provenance.json', provenance)
    started = time.monotonic()
    if args.mode != 'zero-shot':
        windows = Windows(train, args.horizon)
        sampler = RandomSampler(windows, replacement=True, num_samples=args.steps_per_epoch*args.batch_size)
        loader = DataLoader(windows, batch_size=args.batch_size, sampler=sampler)
        validation = DataLoader(Windows(val, args.horizon, stride=max(args.horizon, 20)), batch_size=args.batch_size)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
        best = float('inf')
        for epoch in range(args.epochs):
            model.train()
            train_sum = 0.
            for frames in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = rollout_loss(model, frames.to(device), mean, std)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                train_sum += loss.item()
            model.eval()
            validation_sum, count = 0., 0
            with torch.no_grad():
                for frames in validation:
                    value = rollout_loss(model, frames.to(device), mean, std).item()
                    if not np.isfinite(value):
                        raise FloatingPointError('Nonfinite validation loss')
                    validation_sum += value * len(frames)
                    count += len(frames)
            score = validation_sum / count
            row = dict(epoch=epoch, train_loss=train_sum/len(loader), val_loss=score, elapsed_seconds=time.monotonic()-started)
            with (args.output / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            if score < best:
                best = score
                payload = dict(state_dict=model.state_dict(), model_config=model.config,
                               mean=mean.cpu(), std=std.cpu(), r=model.r.cpu(), z=model.z.cpu(),
                               case=case, epoch=epoch, manifest_sha256=provenance['manifest_sha256'])
                temporary = args.output / 'best.pt.part'
                torch.save(payload, temporary)
                temporary.replace(args.output / 'best.pt')
        payload = torch.load(args.output / 'best.pt', map_location=device, weights_only=False)
        model.load_state_dict(payload['state_dict'])
    metrics = evaluate(model, test, mean, std, args.eval_horizon, zero_shot=args.mode == 'zero-shot')
    metrics['elapsed_seconds'] = time.monotonic() - started
    write_json(args.output / 'test_metrics.json', metrics)
    print(json.dumps({key: value for key, value in metrics.items() if key != 'trajectories'}), flush=True)


if __name__ == '__main__':
    main()
