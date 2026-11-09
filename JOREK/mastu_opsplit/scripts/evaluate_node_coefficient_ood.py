#!/usr/bin/env python3
"""Evaluate the original frozen NODE checkpoint on accepted diffusivity OOD data."""
import argparse
from dataclasses import fields
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from mastu_opsplit import ensemble_training
from mastu_opsplit.common import config_hash, sha256_file
from mastu_opsplit.euler import EulerSplineModel


def load_training(training, normalization_sha256):
    provenance = json.loads((training/'provenance.json').read_text())
    config = provenance['config']
    ensemble_training.check_config(config)
    if config.get('coefficient_split') is not None or config.get('diagnostic_only') or \
            config['seed'] != 17 or config['steps'] != 2000 or config['horizon'] != 4 or \
            config['evaluation_horizons'] != [16, 32, 96] or config.get('evaluation_cases') is not None:
        raise ValueError('Require the original full-budget seed-17 NODE run')
    source = Path(ensemble_training.__file__).parent
    if provenance['source_sha256'] != {path.name: sha256_file(path) for path in source.glob('*.py')}:
        raise ValueError('Use the original frozen training implementation through PYTHONPATH')
    normalization = training/'normalization.npz'
    if sha256_file(normalization) != normalization_sha256:
        raise ValueError('Training normalization hash mismatch')
    with np.load(normalization, allow_pickle=False) as saved:
        for name in ('mean', 'scale'):
            if hashlib.sha256(saved[name].tobytes()).hexdigest() != provenance['normalization'][name+'_sha256']:
                raise ValueError('Normalization differs from training provenance')
        if len(saved['training_case_ids']) != 80 or \
                set(saved['training_case_ids'].tolist()) != set(provenance['normalization']['training_case_ids']):
            raise ValueError('Require exactly the 80 training normalization cases')
    checkpoint = torch.load(training/'node/selected.pt', map_location='cpu', weights_only=False)
    if checkpoint['family'] != 'node' or checkpoint['step'] != 2000 or \
            checkpoint['selection'] != 'final fixed-budget update' or checkpoint['provenance_hash'] != config_hash(provenance):
        raise ValueError('Require the matching final NODE checkpoint')
    return provenance, checkpoint, normalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--loader-study', type=Path, required=True)
    parser.add_argument('--ood-root', type=Path, required=True)
    parser.add_argument('--training-normalization-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    manifest = args.ood_root/'manifest.json'
    completion = json.loads((args.ood_root/'completion.json').read_text())
    release = json.loads(manifest.read_text())
    if completion.get('status') != 'ood_dataset_ready' or completion.get('cases') != 10 or \
            completion.get('manifest_sha256') != sha256_file(manifest) or \
            release.get('schema') != 'mastu-density-diffusivity-ood-v1':
        raise ValueError('Require the accepted ten-case diffusivity OOD release; gamma is not a diffusivity input')
    provenance, checkpoint, normalization = load_training(args.training, args.training_normalization_sha256)
    config = dict(provenance['config'])
    if args.smoke:
        config.update(evaluation_horizons=[4], evaluation_cases=2, diagnostic_only=True)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    loader_path = args.loader_study/'src/mastu_opsplit/coefficient_ood.py'
    spec = importlib.util.spec_from_file_location('mastu_opsplit.frozen_ood_loader', loader_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    baseline, space, guard, physical, dense, verification = ensemble_training.prepare_data(config)
    data = loader.CoefficientOODDataset(manifest, training_normalization=normalization,
        training_normalization_sha256=args.training_normalization_sha256)
    if len(data.cases) != 10 or data.manifest['parent_manifest']['sha256'] != provenance['manifest_sha256'] or \
            any(not 2 <= case['coefficient_multiplier'] <= 10 for case in data.cases):
        raise ValueError('Require the same ten held-out cases and coefficients as OpsSplit')
    for item in fields(type(baseline.geometry)):
        np.testing.assert_array_equal(getattr(data.geometry, item.name), getattr(baseline.geometry, item.name))
    for name in ('mean', 'scale', 'time_seconds'):
        np.testing.assert_array_equal(getattr(data, name), getattr(baseline, name))
    if verification['code_interval'] != provenance['physical_interval']:
        raise ValueError('Physical observation interval changed')
    model = EulerSplineModel('node', space, provenance['model_config'], linear_branch='lspline_continuity',
                             physical_guard=guard, physical_linear=physical)
    model.operator.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    record = dict(training_provenance_sha256=config_hash(provenance),
        checkpoint_sha256=sha256_file(args.training/'node/selected.pt'),
        training_normalization=str(normalization.resolve()), training_normalization_sha256=args.training_normalization_sha256,
        ood_manifest=str(manifest.resolve()), ood_manifest_sha256=sha256_file(manifest),
        evaluator_sha256=sha256_file(__file__), loader_sha256=sha256_file(loader_path),
        model_class=type(model).__module__+'.'+type(model).__name__, family='node',
        config=config, normalization_refitted=False, checkpoint_step=checkpoint['step'],
        parameter_semantics='D_perp / 1e-5 supplied to the existing scalar conditioning input; trained only at 1',
        cases=[{key: case[key] for key in ('case_id', 'matched_id_case', 'coefficient_multiplier')} for case in data.cases])
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'provenance.json'
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError('Evaluation provenance changed; use a fresh output')
    ensemble_training.atomic_json(path, record)
    print(json.dumps(dict(phase='verified', model=record['model_class'], cases=len(data.cases), smoke=args.smoke)), flush=True)
    result = ensemble_training.evaluate(model, data, space, guard, dense, config, args.output, config_hash(record))
    ensemble_training.atomic_json(args.output/'metrics.json', result)
    print(json.dumps(dict(audits_passed=result['audits_passed'], aggregates=result['aggregates'])), flush=True)
    expected_cases = 2 if args.smoke else 10
    if not result['audits_passed'] or any(row['trajectories'] != expected_cases for row in result['aggregates']):
        raise RuntimeError('Incomplete or failed NODE OOD rollout audits')


if __name__ == '__main__':
    main()
