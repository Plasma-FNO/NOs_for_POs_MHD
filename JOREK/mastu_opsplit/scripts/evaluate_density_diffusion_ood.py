#!/usr/bin/env python3
"""Evaluate the final coefficient-aware checkpoint on ten strictly held-out OOD cases."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from mastu_opsplit.coefficient_ood import CoefficientOODDataset, DensityDiffusionEulerModel
from mastu_opsplit.common import config_hash, sha256_file
from mastu_opsplit.ensemble_training import atomic_json, check_config, evaluate, prepare_data


def verify_release(root):
    root = Path(root)
    completion = json.loads((root/'completion.json').read_text())
    manifest = root/'manifest.json'
    if completion.get('status') != 'ood_dataset_ready' or completion.get('cases') != 10 or \
            completion.get('manifest_sha256') != sha256_file(manifest):
        raise ValueError('Require the hash-verified completed ten-case OOD release')
    return manifest


def load_training(training, normalization_sha256):
    training = Path(training)
    provenance = json.loads((training/'provenance.json').read_text())
    config = provenance['config']
    check_config(config)
    if config.get('coefficient_split') != 'density_diffusion' or config.get('diagnostic_only') or \
            config['seed'] != 17 or config['steps'] != 2000 or config['horizon'] != 4 or \
            config['evaluation_horizons'] != [16, 32, 96] or config.get('evaluation_cases') is not None:
        raise ValueError('Require the full-budget seed-17 density-diffusion training run')
    source = Path(__file__).resolve().parents[1]/'src/mastu_opsplit'
    if provenance['source_sha256'] != {path.name: sha256_file(path) for path in source.glob('*.py')}:
        raise ValueError('Training implementation changed')
    normalization = training/'normalization.npz'
    if sha256_file(normalization) != normalization_sha256:
        raise ValueError('Training normalization hash mismatch')
    with np.load(normalization, allow_pickle=False) as saved:
        for name in ('mean', 'scale'):
            if hashlib.sha256(saved[name].tobytes()).hexdigest() != provenance['normalization'][name+'_sha256']:
                raise ValueError('Normalization differs from training provenance')
        if set(saved['training_case_ids'].tolist()) != set(provenance['normalization']['training_case_ids']):
            raise ValueError('Normalization training membership changed')
    checkpoint = torch.load(training/'opsplit/selected.pt', map_location='cpu', weights_only=False)
    if checkpoint['provenance_hash'] != config_hash(provenance) or checkpoint['family'] != 'opsplit' or \
            checkpoint['step'] != config['steps'] or checkpoint['selection'] != 'final fixed-budget update':
        raise ValueError('Require the matching final fixed-budget OpsSplit checkpoint')
    return provenance, checkpoint, normalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--ood-root', type=Path, required=True)
    parser.add_argument('--training-normalization-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = verify_release(args.ood_root)
    provenance, checkpoint, normalization = load_training(args.training, args.training_normalization_sha256)
    config = provenance['config']
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    baseline, space, guard, physical, dense, verification = prepare_data(config)
    data = CoefficientOODDataset(manifest, training_normalization=normalization,
        training_normalization_sha256=args.training_normalization_sha256)
    if len(data.cases) != 10 or data.manifest['parent_manifest']['sha256'] != provenance['manifest_sha256'] or \
            any(not 2 <= case['coefficient_multiplier'] <= 10 for case in data.cases):
        raise ValueError('Require all ten matching test-only coefficient cases in [2,10]')
    for item in fields(type(baseline.geometry)):
        np.testing.assert_array_equal(getattr(data.geometry, item.name), getattr(baseline.geometry, item.name))
    for name in ('mean', 'scale', 'time_seconds'):
        np.testing.assert_array_equal(getattr(data, name), getattr(baseline, name))
    if verification['code_interval'] != provenance['physical_interval']:
        raise ValueError('Physical observation interval changed')
    model = DensityDiffusionEulerModel('opsplit', space, provenance['model_config'],
        mean=data.mean, scale=data.scale, code_interval=provenance['physical_interval'],
        physical_guard=guard, physical_linear=physical)
    model.operator.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    evaluation_provenance = dict(training_provenance_sha256=config_hash(provenance),
        checkpoint_sha256=sha256_file(args.training/'opsplit/selected.pt'),
        training_normalization=str(normalization.resolve()),
        training_normalization_sha256=args.training_normalization_sha256,
        ood_manifest=str(manifest.resolve()), ood_manifest_sha256=sha256_file(manifest),
        evaluator_sha256=sha256_file(__file__), model_class=type(model).__module__+'.'+type(model).__name__,
        training_split_counts=config['split_counts'], ood_split_counts=data.manifest['split_counts'],
        checkpoint_step=checkpoint['step'], normalization_refitted=False, config=config,
        cases=[{key: case[key] for key in ('case_id', 'matched_id_case', 'coefficient_multiplier')}
               for case in data.cases])
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'provenance.json'
    if path.exists() and json.loads(path.read_text()) != evaluation_provenance:
        raise ValueError('OOD evaluation provenance changed; use a fresh output')
    atomic_json(path, evaluation_provenance)
    result = evaluate(model, data, space, guard, dense, config, args.output, config_hash(evaluation_provenance))
    atomic_json(args.output/'metrics.json', result)
    print(json.dumps(dict(audits_passed=result['audits_passed'], aggregates=result['aggregates'])), flush=True)
    if not result['audits_passed'] or any(row['trajectories'] != 10 for row in result['aggregates']):
        raise RuntimeError('Incomplete or failed OOD rollout audits')


if __name__ == '__main__':
    main()
