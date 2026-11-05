"""Bounded, matched ES-to-EM native transfer pilot, independent of full campaigns.

Reuses an audited ES operator checkpoint. All trainable arms share source interface
scales, target windows, optimizer and update count. Scratch means random weights;
it shares ES normalization, so this is not an ES-information-free control.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from comparison import digest, write_json, load_fields, reference_for, decoded_fields, valid_states
from data import select
from native_terms import NativeOperators, NAMES, features, labels
from native_rollout import NativeRollout
from train_native_rollout import sequence_objective
from train_native_terms import term_loss
from train_es import range_statistics, save_checkpoint

ARMS = ('scratch', 'finetune', 'advection-transfer', 'diffusion-transfer', 'frozen', 'analytic')
TRANSFER = {'scratch': (), 'finetune': NAMES, 'advection-transfer': NAMES[:2],
            'diffusion-transfer': NAMES[2:], 'frozen': NAMES}


def initialize(source, arm, seed, device):
    if arm == 'analytic':
        return None
    torch.manual_seed(seed)
    model = NativeOperators(**source['normalization'], **source['model_config'])
    if TRANSFER[arm]:
        pretrained = NativeOperators(**source['normalization'], **source['model_config'])
        pretrained.load_state_dict(source['state_dict'])
        model.transfer(pretrained, TRANSFER[arm], freeze=arm == 'frozen')
    return model.to(device)


def verify_source(source, source_data, manifest_path):
    metadata = json.loads((source_data/'manifest.json').read_text())
    if source.get('case') != 'es' or source.get('phase') not in ('term_pretraining','residual_pretraining'):
        raise ValueError('Expected an ES operator-pretraining checkpoint')
    if source['dataset_sha256'] != digest(source_data/'manifest.json'):
        raise ValueError('Source checkpoint and target-cache manifest differ')
    if metadata['manifest_sha256'] != digest(manifest_path):
        raise ValueError('Source and EM pilot must share the family-split manifest')
    manifest = json.loads(manifest_path.read_text())
    allowed = {e['id'] for e in select(manifest, 'es', 'train')}
    if not set(metadata['splits']['train']['ids']) <= allowed:
        raise ValueError('Source pretraining includes nontraining families')
    if source.get('phase') == 'residual_pretraining':
        if not set(source['correction_training_ids']) <= allowed:
            raise ValueError('Source corrections include nontraining families')
        allowed_val = {e['id'] for e in select(manifest,'es','val')}
        if not set(source['correction_validation_ids']) <= allowed_val:
            raise ValueError('Source correction selection includes nonvalidation families')
    return metadata


def load_cohort(entries, frames):
    result = []
    for entry in entries:
        observed, q = load_fields(entry, frames)
        if len(q) != frames or not bool(valid_states(observed, q).all()):
            raise ValueError(f"Pilot cohort contains invalid release states: {entry['id']}")
        result.append((entry, observed, q))
    return result


def make_solver(entry, q, model, device):
    return NativeRollout(reference_for(entry, q[:1]), q[:1], model, device=device)


@torch.no_grad()
def evaluate(cohort, model, device, horizon, substeps):
    if model is not None:
        model.eval()
    rows = []
    for entry, observed, q in cohort:
        solver = make_solver(entry, q, model, device)
        x = q[:1].to(device)
        r = solver.ref.mass.r
        row = dict(id=entry['id'], status='ok', frames=[])
        try:
            for step in range(1, horizon+1):
                x = solver.step(x, solver.ref.c['nominal_frame_dt'], substeps)
                pred = decoded_fields(solver.ref, x)[0]
                truth = observed[step].to(device)
                error = (((pred-truth).square()*r).sum((-2,-1)) /
                         (truth.square()*r).sum((-2,-1)).clamp_min(1e-30)).sqrt()
                persistence = (((observed[0].to(device)-truth).square()*r).sum((-2,-1)) /
                               (truth.square()*r).sum((-2,-1)).clamp_min(1e-30)).sqrt()
                delta_error = None
                if q.shape[1] == 4:
                    delta = q[step,3].to(device)-q[0,3].to(device)
                    delta_error = float((((x[0,3]-q[step,3].to(device)).square()*r).sum() /
                                         (delta.square()*r).sum().clamp_min(1e-30)).sqrt())
                row['frames'].append(dict(frame=step, relative_l2=error.cpu().tolist(),
                    persistence=persistence.cpu().tolist(), delta_psi_relative_l2=delta_error))
            row['limiter'] = dict(solver.limiter_stats)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            row.update(status='failed', error=str(exc), limiter=dict(solver.limiter_stats))
        rows.append(row)
    return rows


def score(rows, frame):
    # A failed trajectory cannot disappear from a favorable aggregate.
    if not rows or any(r['status'] != 'ok' or len(r['frames']) < frame for r in rows):
        return None
    values = [r['frames'][frame-1] for r in rows]
    fields = torch.tensor([r['relative_l2'] for r in values]).double().mean(0)
    return dict(shared_relative_l2=float(fields[:4].mean()), fields=fields.tolist(),
                delta_psi_relative_l2=(sum(r['delta_psi_relative_l2'] for r in values)/len(values)
                                      if values[0]['delta_psi_relative_l2'] is not None else None))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'source', 'source-data', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--budget', type=int, default=8)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--seed', type=int, default=907)
    p.add_argument('--device', default='cuda')
    p.add_argument('--substeps', type=int, default=8)
    args = p.parse_args()
    if min(args.budget, args.epochs, args.substeps) < 1:
        p.error('Positive pilot sizes required')
    torch.set_num_threads(1)
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    source_meta = verify_source(source, args.source_data, args.manifest)
    manifest = json.loads(args.manifest.read_text())
    entries = {s: select(manifest, 'em', s, args.budget if s == 'train' else 2, args.seed)
               for s in ('train', 'val', 'test')}
    if len(entries['train']) != args.budget or any(len(entries[s]) != 2 for s in ('val','test')):
        raise ValueError('Insufficient pilot trajectories')
    # Fixed short prefixes/windows; no test-dependent cohort selection.
    cohorts = {s: load_cohort(v, 7 if s == 'train' else 11) for s,v in entries.items()}
    windows = [(i,s) for i in range(args.budget) for s in (0,4)]
    mean, scale = range_statistics(torch.stack([q for _,_,q in cohorts['train']]))
    mean, scale = mean.to(args.device), scale.to(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(budget=args.budget, epochs=args.epochs, seed=args.seed, horizon=2,
                    windows=windows, substeps=args.substeps, lr=1e-3, weight_decay=1e-4,
                    term_weight=1., batch_size=1, validation_horizon=5, test_horizon=10,
                    correction_fraction=source['model_config'].get('correction_fraction'))
    provenance = dict(protocol=protocol, arms=ARMS,
        source_checkpoint=str(args.source.resolve()), source_sha256=digest(args.source),
        source_training_ids=source_meta['splits']['train']['ids'],
        source_correction_training_ids=source.get('correction_training_ids'),
        manifest_sha256=digest(args.manifest),
        cohorts={s:[e['id'] for e in es] for s,es in entries.items()},
        input_sha256={e['id']:digest(e['input']) for es in entries.values() for e in es},
        interface='Identical ES input/term scales in every arm; scratch has random weights but shares source statistics',
        loss=('Two-step primary-state relative L2 plus normalized correction L2'
              if protocol['correction_fraction'] is not None else
              'Two-step primary-state relative L2 plus independent equation-term MSE'),
        profile='Old viscosity; nominal dt=5; approximate static wall closure; current guarded IMEX',
        limitations='Single seed, eight EM trajectories, short windows; not a full budget sweep or exact JOREK FEM',
        code_sha256={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')})
    write_json(args.output/'provenance.json', provenance)
    summary = dict(protocol=protocol, arms={})
    for arm in ARMS:
        started = time.monotonic()
        model = initialize(source, arm, args.seed, args.device)
        directory = args.output/arm
        directory.mkdir()
        trainable = arm not in ('frozen','analytic')
        best, best_epoch, updates = float('inf'), None, 0
        if trainable:
            optimizer = torch.optim.Adam(model.parameters(), lr=protocol['lr'], weight_decay=protocol['weight_decay'])
            solvers = [make_solver(e,q,model,args.device) for e,_,q in cohorts['train']]
            for epoch in range(args.epochs):
                model.train()
                order = torch.randperm(len(windows), generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
                for index in order:
                    i,start = windows[index]
                    solver = solvers[i]
                    solver.reset_limiter_stats()
                    q = cohorts['train'][i][2][start:start+3].to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = sequence_objective(solver,q,mean,scale,solver.ref.c['nominal_frame_dt'],args.substeps)
                    predicted = model(features(q[:1]).float())
                    with torch.no_grad():
                        target = (torch.zeros_like(predicted) if model.is_residual else
                                  labels(solver.ref,q[:1]).float()/model.target_scale[None,:,None,None])
                    anchor = term_loss(predicted,target)
                    objective = loss+anchor
                    if not torch.isfinite(objective):
                        raise FloatingPointError('Nonfinite pilot loss')
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                    record = dict(arm=arm,epoch=epoch,update=updates,id=cohorts['train'][i][0]['id'],start=start,
                                  rollout_loss=float(loss.detach()),term_loss=float(anchor.detach()),limiter=dict(solver.limiter_stats))
                    with (directory/'history.jsonl').open('a') as f:
                        f.write(json.dumps(record)+'\n')
                    print(json.dumps(record),flush=True)
                validation = evaluate(cohorts['val'],model,args.device,5,args.substeps)
                write_json(directory/f'validation-{epoch}.json',validation)
                metric = score(validation,5)
                if metric is not None and metric['shared_relative_l2'] < best:
                    best,best_epoch = metric['shared_relative_l2'],epoch
                    save_checkpoint(directory/'best.pt',dict(state_dict=model.state_dict(),
                        normalization=source['normalization'],model_config=source['model_config'],
                        case='em',phase='native_transfer_pilot',epoch=epoch,protocol=protocol))
            del solvers,optimizer
            if best_epoch is None:
                raise RuntimeError(f'All pilot validation failed: {arm}')
            model.load_state_dict(torch.load(directory/'best.pt',map_location=args.device,weights_only=False)['state_dict'])
        test = evaluate(cohorts['test'],model,args.device,10,args.substeps)
        write_json(directory/'test.json',test)
        summary['arms'][arm] = dict(updates=updates,best_epoch=best_epoch,
            transferred=list(TRANSFER.get(arm,())),seconds=time.monotonic()-started,
            failures=sum(r['status']!='ok' for r in test),
            scores={str(t):score(test,t) for t in (1,5,10)})
        write_json(args.output/'summary.json',summary)
        print(json.dumps(dict(arm=arm,summary=summary['arms'][arm])),flush=True)
        del model
        gc.collect()
        if args.device.startswith('cuda'):
            from gpu_sparse import clear_cache
            clear_cache()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
