#!/usr/bin/env python3
"""Coefficient-OOD evaluation with the network conditioning held at its training value.

Every training window in the 80/0/20 study supplied conditioning parameter 1, so a
network has never seen any other value. Earlier OOD evaluations passed the D_perp
multiplier (2-10) into that input, which extrapolates FiLM rather than testing the
physics. Here the network always receives 1; the multiplier is metadata only.
No model can see the coefficient except through physics it explicitly contains.

Imports come from the run's own frozen study snapshot and are hash-checked against
its provenance, so AR/NODE/OpsSplit ensemble checkpoints and implicit arms both use
the exact training code. Normalization is the run's saved 80-case statistics.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ENSEMBLE = ('ar', 'node', 'opsplit')
IMPLICIT = ('implicit', 'implicit_rollout', 'physics')


class FixedConditioning:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, name):
        return getattr(self.data, name)

    def windows(self, split, horizon, *, stride=1):
        return FixedWindows(self.data.windows(split, horizon, stride=stride))


class FixedWindows:
    def __init__(self, inner):
        self.inner, self.cases, self.starts = inner, inner.cases, inner.starts

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        sample = self.inner[index]
        sample['withheld_coefficient_parameters'] = sample['parameters']
        sample['parameters'] = torch.ones(1)
        return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='training output, e.g. .../seed17_node')
    parser.add_argument('--family', choices=ENSEMBLE + IMPLICIT, required=True)
    parser.add_argument('--study', type=Path, help='frozen study snapshot; default RUN/../study')
    parser.add_argument('--ood-root', type=Path, required=True)
    parser.add_argument('--loader', type=Path, required=True, help='coefficient_ood.py providing CoefficientOODDataset')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    study = (args.study or args.run.parent/'study').resolve()
    sys.path.insert(0, str(study/'src'))
    from mastu_opsplit import ensemble_training
    from mastu_opsplit.common import config_hash, sha256_file
    provenance = json.loads((args.run/'provenance.json').read_text())
    config = dict(provenance['config'])
    if provenance['source_sha256'] != {p.name: sha256_file(p) for p in (study/'src'/'mastu_opsplit').glob('*.py')}:
        raise ValueError('Study snapshot source differs from the training provenance')
    if config.get('diagnostic_only') or config.get('evaluation_cases') is not None or config.get('coefficient_split'):
        raise ValueError('Require a full-budget, non-coefficient-aware training run')
    if args.smoke:
        config.update(evaluation_horizons=[4], evaluation_cases=2, diagnostic_only=True)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    manifest = args.ood_root/'manifest.json'
    completion = json.loads((args.ood_root/'completion.json').read_text())
    if completion.get('status') != 'ood_dataset_ready' or completion.get('manifest_sha256') != sha256_file(manifest):
        raise ValueError('Require an accepted, hash-matched OOD release')
    spec = importlib.util.spec_from_file_location('mastu_opsplit.frozen_ood_loader', args.loader)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    baseline, space, guard, physical, dense, verification = ensemble_training.prepare_data(config)
    normalization = args.run/'normalization.npz'
    ood = loader.CoefficientOODDataset(manifest, training_normalization=normalization,
                                       training_normalization_sha256=sha256_file(normalization))
    for name in ('mean', 'scale'):
        np.testing.assert_array_equal(getattr(ood, name), getattr(baseline, name))
    if verification['code_interval'] != provenance['physical_interval']:
        raise ValueError('Physical observation interval changed')
    if args.family in ENSEMBLE:
        from mastu_opsplit.euler import EulerSplineModel
        ensemble_training.check_config(config if not args.smoke else provenance['config'])
        model = EulerSplineModel(args.family, space, provenance['model_config'], linear_branch=config['linear_branch'],
                                 physical_guard=guard, physical_linear=physical)
        checkpoint_path = args.run/args.family/'selected.pt'
    else:
        from mastu_opsplit.implicit_opsplit import ImplicitOpsSplitModel
        from mastu_opsplit.implicit_training import build_system
        if provenance['arm'] != args.family:
            raise ValueError('Run arm differs from requested family')
        system, dirichlet = build_system(config, baseline, space, verification, args.run, resume=True)
        if dirichlet != provenance['dirichlet']:
            raise ValueError('Dirichlet set differs from training provenance')
        model = ImplicitOpsSplitModel(space, system, provenance['model_config'], provenance['label_scale'],
                                      physical_guard=guard, physics_only=args.family == 'physics')
        checkpoint_path = None if args.family == 'physics' else args.run/args.family/'rollout'/'selected.pt'
    checkpoint_sha256 = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint['step'] != config['steps'] or checkpoint['provenance_hash'] != config_hash(provenance) or \
                checkpoint['selection'] != 'final fixed-budget update':
            raise ValueError('Require the matching final fixed-budget checkpoint')
        model.operator.load_state_dict(checkpoint['model'], strict=True)
        checkpoint_sha256 = sha256_file(checkpoint_path)
    model.eval()
    record = dict(run=str(args.run.resolve()), family=args.family, study=str(study), checkpoint_sha256=checkpoint_sha256,
                  training_provenance_sha256=config_hash(provenance), ood_manifest_sha256=sha256_file(manifest),
                  loader_sha256=sha256_file(args.loader), evaluator_sha256=sha256_file(__file__),
                  normalization_sha256=sha256_file(normalization), config=config, normalization_refitted=False,
                  conditioning='network receives the training value 1; D_perp multiplier withheld (metadata only)',
                  cases=[{key: case[key] for key in ('case_id', 'matched_id_case', 'coefficient_multiplier')} for case in ood.cases])
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'provenance.json'
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError('Evaluation provenance changed; use a fresh output')
    ensemble_training.atomic_json(path, record)
    result = ensemble_training.evaluate(model, FixedConditioning(ood), space, guard, dense, config, args.output,
                                        config_hash(record))
    ensemble_training.atomic_json(args.output/'metrics.json', result)
    print(json.dumps(dict(audits_passed=result['audits_passed'], aggregates=[
        {k: row[k] for k in ('horizon', 'substeps', 'mean_normalized_volume_mse')} for row in result['aggregates']])), flush=True)
    if not result['audits_passed']:
        raise RuntimeError('Incomplete or failed OOD rollout audits')


if __name__ == '__main__':
    main()
