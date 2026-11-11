#!/usr/bin/env python3
"""Coefficient-OOD evaluation of implicit OpsSplit arms with the exact D_perp(1) physics.

The implicit operator receives the true D_perp(1) of each OOD trajectory through the exact,
natively verified affine dependence of the model303 RHS on D (implicit_coefficient.py,
verify_coefficient_operator.py): F(q*) and L* are updated and refactorised per coefficient,
and the closed-form remainder of the diffusion term is applied explicitly. The network N_theta
always receives its training conditioning value 1, exactly as in the fixed-conditioning OOD
evaluation, and its checkpoint is unchanged. Nothing is fitted or selected on OOD data.

Everything except the coefficient module is imported from the run's frozen study snapshot and
hash-checked against its training provenance.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

IMPLICIT = ('implicit', 'implicit_rollout', 'physics')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='implicit training output, e.g. .../seed17_implicit')
    parser.add_argument('--family', choices=IMPLICIT, required=True)
    parser.add_argument('--study', type=Path, help='frozen study snapshot; default RUN/../study')
    parser.add_argument('--coefficient-module', type=Path, required=True, help='implicit_coefficient.py')
    parser.add_argument('--native-check', type=Path, required=True, help='passed verify_coefficient_operator.py report')
    parser.add_argument('--ood-root', type=Path, required=True)
    parser.add_argument('--loader', type=Path, required=True, help='coefficient_ood.py providing CoefficientOODDataset')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'),
                        help='cuda: torch backend with cuDSS LU (needs a snapshot with gpu_native.py); default cpu')
    args = parser.parse_args()
    study = (args.study or args.run.parent/'study').resolve()
    sys.path.insert(0, str(study/'src'))
    from mastu_opsplit import ensemble_training
    from mastu_opsplit.common import config_hash, sha256_file
    from mastu_opsplit.implicit_opsplit import ImplicitNativeSystem, ImplicitOpsSplitModel, load_frozen_tangent
    from mastu_opsplit.implicit_training import build_system
    provenance = json.loads((args.run/'provenance.json').read_text())
    config = dict(provenance['config'])
    if config.get('physical_defect') is not None:
        raise ValueError('This legacy OOD wrapper does not preserve physical-defect stages; '
                         'use a separately validated profile/system configuration for that coefficient')
    if provenance['source_sha256'] != {p.name: sha256_file(p) for p in (study/'src'/'mastu_opsplit').glob('*.py')}:
        raise ValueError('Study snapshot source differs from the training provenance')
    if provenance['arm'] != args.family:
        raise ValueError('Run arm differs from requested family')
    if config.get('diagnostic_only') or config.get('evaluation_cases') is not None:
        raise ValueError('Require a full-budget training run')
    native_check = json.loads(args.native_check.read_text())
    if not native_check.get('passed'):
        raise ValueError('Native coefficient check did not pass')
    # The coefficient module joins the snapshot package so its relative imports use snapshot code.
    spec = importlib.util.spec_from_file_location('mastu_opsplit.implicit_coefficient', args.coefficient_module)
    coefficient = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = coefficient
    spec.loader.exec_module(coefficient)
    if args.smoke:
        config.update(evaluation_horizons=[4], evaluation_cases=2, diagnostic_only=True)
    if args.device is not None:
        config['device'] = args.device
    device = torch.device(config.get('device', 'cpu'))
    if device.type == 'cuda':
        from mastu_opsplit.gpu_native import configure_cuda_numerics
        configure_cuda_numerics()
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
    if ood.manifest['coefficient_name'] != 'D_perp(1)' or ood.manifest['baseline_coefficient'] != coefficient.BASELINE_D_PERP:
        raise ValueError('Coefficient physics is derived for D_perp(1) about 1e-5 only')
    system, dirichlet = build_system(config, baseline, space, verification, args.run, resume=True)
    if dirichlet != provenance['dirichlet']:
        raise ValueError('Dirichlet set differs from training provenance')
    frozen = load_frozen_tangent(config['tangent'], space.geometry)
    if frozen.sha256 != config['tangent_sha256']:
        raise ValueError('Frozen tangent hash differs from configuration')
    from mastu_opsplit import implicit_training
    if hasattr(implicit_training, 'build_model'):
        base = implicit_training.build_model(config, space, system, provenance['model_config'], provenance['label_scale'],
                                             physical_guard=guard, physics_only=args.family == 'physics')
    else:   # snapshots that predate the tangent correction
        base = ImplicitOpsSplitModel(space, system, provenance['model_config'], provenance['label_scale'],
                                     physical_guard=guard, physics_only=args.family == 'physics')
    checkpoint_sha256 = None
    if args.family != 'physics':
        path = args.run/args.family/'rollout'/'selected.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint['step'] != config['steps'] or checkpoint['provenance_hash'] != config_hash(provenance) or \
                checkpoint['selection'] != 'final fixed-budget update':
            raise ValueError('Require the matching final fixed-budget checkpoint')
        base.operator.load_state_dict(checkpoint['model'], strict=True)
        checkpoint_sha256 = sha256_file(path)

    def factory(multiplier):
        moved = coefficient.coefficient_tangent(frozen, multiplier, F0=config['F0'])
        return ImplicitNativeSystem(space.geometry, moved, baseline.mean, baseline.scale, system.fixed,
                                    code_interval=verification['code_interval'], theta=config['theta'],
                                    F0=config['F0'], relations=system.relations,
                                    **({} if device.type == 'cpu' else dict(backend='torch', device=device)))

    correction = coefficient.DensityDiffusionRemainder(space.geometry, frozen, baseline.mean, baseline.scale,
                                                       F0=config['F0'])
    model = coefficient.CoefficientImplicitModel(base, factory, correction,
                                                 known_multipliers=[case['coefficient_multiplier'] for case in ood.cases])
    model = model.to(device)
    model.eval()
    record = dict(run=str(args.run.resolve()), family=args.family, study=str(study), checkpoint_sha256=checkpoint_sha256,
                  training_provenance_sha256=config_hash(provenance), ood_manifest_sha256=sha256_file(manifest),
                  loader_sha256=sha256_file(args.loader), evaluator_sha256=sha256_file(__file__),
                  coefficient_module_sha256=sha256_file(args.coefficient_module),
                  native_check_sha256=sha256_file(args.native_check), native_check_worst=native_check['worst'],
                  normalization_sha256=sha256_file(normalization), config=config, normalization_refitted=False,
                  conditioning=('network receives the training value 1; implicit physics receives the true D_perp(1) '
                                '(exact affine dependence: frozen F*, L* refactorised per coefficient, explicit closed-form remainder)'),
                  cases=[{key: case[key] for key in ('case_id', 'matched_id_case', 'coefficient_multiplier')} for case in ood.cases])
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'provenance.json'
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError('Evaluation provenance changed; use a fresh output')
    ensemble_training.atomic_json(path, record)
    result = ensemble_training.evaluate(model, ood, space, guard, dense, config, args.output, config_hash(record))
    result['coefficient_systems_built'] = model.systems
    ensemble_training.atomic_json(args.output/'metrics.json', result)
    print(json.dumps(dict(audits_passed=result['audits_passed'], aggregates=[
        {k: row[k] for k in ('horizon', 'substeps', 'mean_normalized_volume_mse')} for row in result['aggregates']])), flush=True)
    if not result['audits_passed']:
        raise RuntimeError('Incomplete or failed OOD rollout audits')


if __name__ == '__main__':
    main()
