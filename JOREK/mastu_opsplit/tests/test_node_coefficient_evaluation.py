import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from mastu_opsplit import ensemble_training
from mastu_opsplit.common import config_hash, load_config, sha256_file


def evaluator():
    path = Path(__file__).resolve().parents[1]/'scripts/evaluate_node_coefficient_ood.py'
    spec = importlib.util.spec_from_file_location('node_coefficient_evaluation_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_node_checkpoint_and_normalization_gates(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root/'configs/ensemble_training.yaml')
    mean, scale = np.zeros((2, 7)), np.ones(7)
    ids = [f'case{index}' for index in range(80)]
    normalization = tmp_path/'normalization.npz'
    np.savez(normalization, mean=mean, scale=scale, training_case_ids=ids)
    provenance = dict(config=config,
        source_sha256={path.name: sha256_file(path) for path in Path(ensemble_training.__file__).parent.glob('*.py')},
        normalization=dict(mean_sha256=hashlib.sha256(mean.tobytes()).hexdigest(),
                           scale_sha256=hashlib.sha256(scale.tobytes()).hexdigest(), training_case_ids=ids))
    ensemble_training.atomic_json(tmp_path/'provenance.json', provenance)
    (tmp_path/'node').mkdir()
    checkpoint = dict(model={}, family='node', step=2000, selection='final fixed-budget update',
                      provenance_hash=config_hash(provenance))
    torch.save(checkpoint, tmp_path/'node/selected.pt')
    module = evaluator()
    assert module.load_training(tmp_path, sha256_file(normalization))[1] == checkpoint
    with pytest.raises(ValueError, match='normalization hash'):
        module.load_training(tmp_path, 'wrong')
    for change in ({'family': 'opsplit'}, {'step': 1999}, {'provenance_hash': 'wrong'}):
        torch.save({**checkpoint, **change}, tmp_path/'node/selected.pt')
        with pytest.raises(ValueError, match='matching final NODE'):
            module.load_training(tmp_path, sha256_file(normalization))
    provenance['source_sha256']['euler.py'] = 'changed'
    ensemble_training.atomic_json(tmp_path/'provenance.json', provenance)
    with pytest.raises(ValueError, match='original frozen'):
        module.load_training(tmp_path, sha256_file(normalization))


def test_gamma_cannot_be_used_as_diffusivity_input(tmp_path, monkeypatch):
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(schema='mastu-gamma-ood-v1')))
    ensemble_training.atomic_json(tmp_path/'completion.json', dict(status='ood_dataset_ready', cases=10,
        manifest_sha256=sha256_file(manifest)))
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--training', str(tmp_path), '--loader-study', str(tmp_path),
        '--ood-root', str(tmp_path), '--training-normalization-sha256', 'unused', '--output', str(tmp_path/'output')])
    with pytest.raises(ValueError, match='gamma is not a diffusivity input'):
        evaluator().main()
