"""Derive an inference-only coefficient-support gate without retraining or changing weights.

python -m gos.research_gate --root ROOT --runs COMPLETED_RUN_DIR ...

The inclusive support is problem.mu_train. The hard gate is generally discontinuous
in the coefficient at its endpoints. Each new `_gated` run retains the source
checkpoint weights and training provenance, and evaluates the new inference rule.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch

from .research import device_info, timing, trajectory_metrics, write_json
from .research_holdout import load_model, sha256
from .train import evaluate, load_data


@torch.no_grad()
def measured_timing(model, frames, mu):
    batches = []
    hook = model.net.register_forward_hook(lambda module, args, output: batches.append(len(args[0])))
    try:
        model.step(frames[:8, 0], mu[:8], model.h)
    finally:
        hook.remove()
    out = timing(model, frames, mu)
    out['neural_calls_per_step'] = len(batches)
    out['neural_sample_evaluations_per_step'] = sum(batches)
    return out


def gate_run(directory, root):
    device = device_info()
    directory = Path(directory).resolve()
    source_path, checkpoint_path = directory/'result.json', directory/'model.pt'
    source = json.loads(source_path.read_text())
    cfg = copy.deepcopy(source['config'])
    if (source['train']['status'] != 'completed' or not cfg.get('closure', False)
            or cfg.get('linear_mode', 'known') != 'known' or cfg.get('closure_in_range', False)):
        raise ValueError('Gate a completed, ungated learned-closure run with known physical operator')
    if source['train'].get('iterations', cfg['iters']) != cfg['iters']:
        raise ValueError('Source training did not complete its configured budget')
    arm = source['arm'] + '_gated'
    destination = directory.parent/f'{arm}_s{source["seed"]}'
    if destination.exists():
        raise FileExistsError(f'Refusing to replace inference variant: {destination}')
    hashes = dict(checkpoint=sha256(checkpoint_path), result=sha256(source_path))
    from .problems import get_problem
    problem = get_problem(source['problem'], 'cuda')
    data, scale = load_data(root, problem, 'cuda')
    model, _ = load_model(dict(directory=str(directory), problem=source['problem']), scale)
    model.closure_in_range = True
    cfg['closure_in_range'] = True
    torch.manual_seed(source['seed'])
    start = time.perf_counter()
    metrics = evaluate(model, problem, data, cfg, 'cuda')
    metrics['operator_note'] = 'Theta-step diagnostic omitted for Runge-Kutta integrator'
    metrics['trajectory_errors'] = trajectory_metrics(model, data, max(cfg['horizons']))
    measured = measured_timing(model, *data['test'])
    measured_ood = {str(value): measured_timing(model, *pair) for value, pair in data['ood'].items()}
    elapsed = time.perf_counter()-start
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = model.state_dict()
    assert state.keys() == checkpoint['state'].keys()
    assert all(torch.equal(state[name].detach().cpu(), value) for name, value in checkpoint['state'].items())
    if sha256(checkpoint_path) != hashes['checkpoint'] or sha256(source_path) != hashes['result']:
        raise ValueError('Source artifacts changed during inference evaluation')
    result = copy.deepcopy(source)
    result.update(arm=arm, config=cfg, eval=metrics, timing=measured, timing_ood=measured_ood,
                  evaluation_device=device, derivation_seconds=elapsed,
                  total_seconds=source['train'].get('seconds', 0.)+elapsed,
                  selection='Inference-only training-support rule; no tuning or retraining in this command')
    result['inference_variant'] = dict(
        source_directory=str(directory), source_arm=source['arm'],
        source_checkpoint_sha256=hashes['checkpoint'], source_result_sha256=hashes['result'],
        source_training_config=source['config'], training_reused=True, weights_unchanged=True,
        checkpoint_config_updated=True, evaluation_seed=source['seed'],
        coefficient_interval_inclusive=list(problem.mu_train),
        normalized_interval_inclusive=[v/problem.mu_ref for v in problem.mu_train],
        rule='Use the learned defect only within the inclusive training coefficient support; use known classical physics elsewhere',
        limitation='The hard gate is generally discontinuous in the coefficient at support boundaries',
        derivation_source_sha256=sha256(__file__))
    assert result['train'] == source['train']
    destination.mkdir()
    checkpoint['config'] = cfg
    torch.save(checkpoint, destination/'model.pt')
    result['inference_variant']['checkpoint_sha256'] = sha256(destination/'model.pt')
    write_json(destination/'result.json', result)
    print(json.dumps(dict(directory=str(destination), source_checkpoint_sha256=hashes['checkpoint'],
                          test=metrics['test'], ood=metrics['ood'], timing=measured,
                          timing_ood=measured_ood)), flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    for directory in args.runs:
        gate_run(directory, args.root)


if __name__ == '__main__':
    main()
