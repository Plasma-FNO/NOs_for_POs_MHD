"""ES/EM paper-protocol baselines; native JOREK OpsSplit requires the audited solver."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import torch
from torch.utils.data import DataLoader, Dataset

from data import layout, read_frames, select
from model import SplitFNO
from formulation import require_native_ready
from train import evaluate, write_json


def cached_trajectories(entries, frames):
    result = []
    for entry in entries:
        if entry['frames'] < frames:
            raise ValueError(f"{entry['id']} has fewer than {frames} frames")
        with h5py.File(entry['path'], 'r') as handle:
            keys, _, _ = layout(handle, entry['case'])
            result.append(torch.from_numpy(read_frames(handle, keys, 0, frames)))
    return torch.stack(result)


def range_statistics(values):
    lo, hi = values.amin((0, 1, 3, 4)), values.amax((0, 1, 3, 4))
    # Constant channels encode as zero; no test/extrapolation values enter the fit.
    return (lo + hi) / 2, torch.where(hi > lo, (hi - lo) / 2, torch.ones_like(lo))


class CachedWindows(Dataset):
    def __init__(self, values, horizon):
        self.values, self.horizon = values, horizon
        self.starts = values.shape[1] - horizon
        if self.starts < 1:
            raise ValueError('Training frames must exceed rollout horizon')

    def __len__(self):
        return len(self.values) * self.starts

    def __getitem__(self, index):
        trajectory, start = divmod(index, self.starts)
        return self.values[trajectory, start:start + self.horizon + 1]


def relative_lp(prediction, target):
    # Same batch-summed flattened relative L2 as the repository's LpLoss(d=1,p=2).
    return (torch.linalg.vector_norm((prediction-target).flatten(1), dim=1) /
            (torch.linalg.vector_norm(target.flatten(1), dim=1) + 1e-8)).sum()


def sequence_loss(model, frames):
    x, loss = frames[:, 0], frames.new_zeros(())
    for target in frames[:, 1:].unbind(1):
        x = model(x)
        loss = loss + relative_lp(x, target)
    return loss


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.pt.part')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['ar', 'node', 'opsplit'], required=True)
    parser.add_argument('--case', choices=['es', 'em'], default='es')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--split-formulation', choices=['jorek', 'effective'], default='jorek',
                        help='jorek is the requested constrained formulation; effective explicitly selects the legacy pilot')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Full-size FNO/batch, two optimizer steps, one epoch; not a paper result')
    args = parser.parse_args()
    if args.method == 'opsplit' and args.split_formulation == 'jorek':
        require_native_ready(args.case)
    if args.protocol is None:
        args.protocol = Path(__file__).with_name(f'{args.case}_paper_protocol.json')
    cfg = json.loads(args.protocol.read_text())
    if cfg.get('case', args.case) != args.case:
        parser.error('Protocol regime does not match --case')
    if args.smoke:
        cfg.update(epochs=1, train_trajectories=2)
    if min(cfg[key] for key in ('epochs', 'train_trajectories', 'batch_size', 'horizon', 'train_frames')) < 1:
        raise ValueError('Protocol counts must be positive')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use a GPU allocation')
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'])
    manifest = json.loads(args.manifest.read_text())
    train = select(manifest, args.case, 'train', cfg['train_trajectories'], cfg['seed'])
    val, test = select(manifest, args.case, 'val'), select(manifest, args.case, 'test')
    if len(train) != cfg['train_trajectories'] or not val or not test:
        raise ValueError('Insufficient independent trajectory splits')
    if args.smoke:
        val, test = val[:1], test[:1]
    values = cached_trajectories(train, cfg['train_frames'])
    mean, std = range_statistics(values)
    values = (values-mean[None, None, :, None, None])/std[None, None, :, None, None]
    validation = cached_trajectories(val, cfg['train_frames'])
    validation = (validation-mean[None, None, :, None, None])/std[None, None, :, None, None]
    generator = torch.Generator().manual_seed(cfg['seed'])
    loader = DataLoader(CachedWindows(values, cfg['horizon']), batch_size=cfg['batch_size'],
                        shuffle=True, generator=generator)
    valid_loader = DataLoader(validation, batch_size=cfg['batch_size'])
    model = SplitFNO(train[0]['r'], train[0]['z'], channels=4 if args.case == 'es' else 6, width=cfg['width'], modes=cfg['modes'],
                     layers=cfg['layers'], monolithic=args.method != 'opsplit',
                     autoregressive=args.method == 'ar', paper_architecture=True).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg['scheduler_step'], cfg['scheduler_gamma'])
    provenance = dict(method=args.method, case=args.case, protocol=cfg, smoke=args.smoke,
                      formulation=args.split_formulation if args.method == 'opsplit' else 'whole-field baseline',
                      manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                      train_ids=[e['id'] for e in train], val_ids=[e['id'] for e in val], test_ids=[e['id'] for e in test],
                      normalization='fieldwise train-prefix minmax [-1,1]', time_basis='saved_frame_index; dt=1',
                      parameters=sum(p.numel() for p in model.parameters()), torch_version=torch.__version__,
                      code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    start_epoch, best, elapsed = 0, float('inf'), 0.
    if args.resume:
        if json.loads((args.output/'provenance.json').read_text()) != provenance:
            raise ValueError('Resume requires identical data, code and protocol')
        saved = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        generator.set_state(saved['sampler_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if args.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_epoch, best, elapsed = saved['epoch']+1, saved['best_val_loss'], saved['elapsed_seconds']
        # A killed process can leave a history row after its last committed checkpoint.
        history = args.output/'history.jsonl'
        rows = [json.loads(line) for line in history.read_text().splitlines()]
        history.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['epoch'] < start_epoch))
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output/'provenance.json', provenance)
    print(json.dumps(dict(method=args.method, parameters=provenance['parameters'], training_windows=len(loader.dataset),
                          epochs=cfg['epochs'], resume_epoch=start_epoch)), flush=True)
    started = time.monotonic()
    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        train_sum, samples = 0., 0
        lr = optimizer.param_groups[0]['lr']
        for batch, frames in enumerate(loader):
            if args.smoke and batch == 2:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_loss(model, frames.to(args.device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            train_sum += loss.item()
            samples += len(frames)
        model.eval()
        val_sum = 0.
        with torch.no_grad():
            for frames in valid_loader:
                loss = sequence_loss(model, frames.to(args.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite free-rollout validation at epoch {epoch}')
                val_sum += loss.item()
        score = val_sum / (len(validation)*(cfg['train_frames']-1))
        scheduler.step()
        row = dict(epoch=epoch, lr=lr, train_relative_lp=train_sum/(samples*cfg['horizon']),
                   val_relative_lp=score, elapsed_seconds=elapsed+time.monotonic()-started)
        with (args.output/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row), flush=True)
        improved = score < best
        best = min(best, score)
        payload = dict(state_dict=model.state_dict(), model_config=model.config, mean=mean, std=std,
                       r=model.r.cpu(), z=model.z.cpu(), case=args.case, method=args.method, epoch=epoch,
                       manifest_sha256=provenance['manifest_sha256'], normalization=provenance['normalization'])
        if improved:
            save_checkpoint(args.output/'best.pt', payload)
        payload.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), best_val_loss=best,
                       sampler_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
                       elapsed_seconds=row['elapsed_seconds'])
        save_checkpoint(args.output/'last.pt', payload)
    saved = torch.load(args.output/'best.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(saved['state_dict'])
    metrics = evaluate(model, test, mean.to(args.device), std.to(args.device),
                       min(10, cfg['eval_horizon']) if args.smoke else cfg['eval_horizon'])
    metrics.update(method=args.method, normalization=provenance['normalization'],
                   training_last_frame=cfg['train_frames']-1, best_epoch=saved['epoch'],
                   elapsed_seconds=elapsed+time.monotonic()-started,
                   metric_note='Legacy standardized_mse keys mean squared error in range-normalized coordinates here; relative_l2 is decoded and R-weighted.')
    write_json(args.output/'test_metrics.json', metrics)
    print(json.dumps(dict(method=args.method, completed=metrics['completed'], total=metrics['total'])), flush=True)


if __name__ == '__main__':
    main()
