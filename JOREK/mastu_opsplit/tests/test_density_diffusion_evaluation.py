import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from mastu_opsplit.common import config_hash, load_config, sha256_file
from mastu_opsplit.ensemble_training import atomic_json as write_json


def evaluator():
    path = Path(__file__).resolve().parents[1]/'scripts/evaluate_density_diffusion_ood.py'
    spec = importlib.util.spec_from_file_location('density_diffusion_evaluation_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ood_completion_gate(tmp_path):
    module = evaluator()
    write_json(tmp_path/'manifest.json', {'cases': []})
    with pytest.raises(FileNotFoundError):
        module.verify_release(tmp_path)
    completion = dict(status='ood_dataset_ready', cases=10, manifest_sha256=sha256_file(tmp_path/'manifest.json'))
    write_json(tmp_path/'completion.json', completion)
    assert module.verify_release(tmp_path) == tmp_path/'manifest.json'
    for update in ({'status': 'running'}, {'cases': 9}, {'manifest_sha256': 'wrong'}):
        write_json(tmp_path/'completion.json', {**completion, **update})
        with pytest.raises(ValueError, match='completed ten-case'):
            module.verify_release(tmp_path)


def test_checkpoint_and_normalization_provenance(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root/'configs/ensemble_density_diffusion.yaml')
    mean, scale = np.zeros((2, 7)), np.ones(7)
    training_ids = [f'case{index}' for index in range(80)]
    normalization = tmp_path/'normalization.npz'
    np.savez(normalization, mean=mean, scale=scale, training_case_ids=training_ids)
    provenance = dict(config=config,
        source_sha256={path.name: sha256_file(path) for path in (root/'src/mastu_opsplit').glob('*.py')},
        normalization=dict(mean_sha256=hashlib.sha256(mean.tobytes()).hexdigest(),
                           scale_sha256=hashlib.sha256(scale.tobytes()).hexdigest(), training_case_ids=training_ids))
    write_json(tmp_path/'provenance.json', provenance)
    (tmp_path/'opsplit').mkdir()
    checkpoint = dict(step=2000, family='opsplit', selection='final fixed-budget update',
                      provenance_hash=config_hash(provenance), model={})
    torch.save(checkpoint, tmp_path/'opsplit/selected.pt')
    module = evaluator()
    loaded, saved, path = module.load_training(tmp_path, sha256_file(normalization))
    assert loaded == provenance and saved == checkpoint and path == normalization
    with pytest.raises(ValueError, match='normalization hash'):
        module.load_training(tmp_path, 'wrong')
    for update in ({'step': 1999}, {'family': 'node'}, {'provenance_hash': 'wrong'}):
        torch.save({**checkpoint, **update}, tmp_path/'opsplit/selected.pt')
        with pytest.raises(ValueError, match='final fixed-budget'):
            module.load_training(tmp_path, sha256_file(normalization))
    provenance['config'].pop('coefficient_split')
    write_json(tmp_path/'provenance.json', provenance)
    with pytest.raises(ValueError, match='density-diffusion training'):
        module.load_training(tmp_path, sha256_file(normalization))


def test_smoke_preserves_production_training_and_model_configuration():
    root = Path(__file__).resolve().parents[1]/'configs'
    production = load_config(root/'ensemble_density_diffusion.yaml')
    smoke = load_config(root/'ensemble_density_diffusion_smoke.yaml')
    differences = {key for key in production.keys() | smoke.keys() if production.get(key) != smoke.get(key)}
    assert differences == {'steps', 'evaluation_horizons', 'evaluation_cases', 'diagnostic_only'}
