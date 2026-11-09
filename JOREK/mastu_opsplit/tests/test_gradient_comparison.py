import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from mastu_opsplit.common import write_json, load_config
from mastu_opsplit.ensemble_training import check_config


def summary_module():
    spec = importlib.util.spec_from_file_location('gradient_summary', Path(__file__).parents[1]/'scripts/summarize_gradients.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def matched_reports(tmp_path):
    module = summary_module()
    for seed in module.SEEDS:
        for variant, branch in module.BRANCHES.items():
            folder = tmp_path/f'seed{seed}_{variant}'
            config = dict(seed=seed, steps=2000, horizon=4, stride=1, diagnostic_only=False,
                          linear_branch=branch, evaluation_horizons=[16, 32, 96], evaluation_substeps=[1, 2, 4])
            write_json(folder/'provenance.json', dict(config=config, normalization={'training_case_ids': ['train']}))
            np.savez(folder/'normalization.npz', mean=np.zeros((16, 7)), scale=np.ones(7))
            write_json(folder/'opsplit/sampled_windows.json', list(range(2000)))
            score = float(seed-16) * (1. if variant == 'learned' else .75)
            aggregates, records = [], []
            for horizon in config['evaluation_horizons']:
                for substeps in config['evaluation_substeps']:
                    aggregates.append(dict(horizon=horizon, substeps=substeps, complete=True, expected_trajectories=20,
                                           trajectories=20, mean_normalized_volume_mse=score))
                    records.extend(dict(horizon=horizon, substeps=substeps, case_id=f'test{case}', start_frame=0,
                        audit_passed=True, normalized_volume_mse=score, per_field_normalized_mse=dict.fromkeys(module.FIELDS, score))
                        for case in range(20))
            write_json(folder/'opsplit/metrics.json', dict(selected_step=2000, parameters=10, initial_neural_sha256=str(seed),
                evaluation=dict(audits_passed=True, numerical_failures=0, aggregates=aggregates,
                                trajectories=[dict(records=records, failure=None)])))
    return tmp_path, module


def test_gradient_summary_reports_paired_seeds_and_fields(matched_reports):
    root, module = matched_reports
    report = module.summarize(root)
    assert report['status'] == 'complete'
    assert len(report['comparison']) == 9
    for row in report['comparison']:
        assert row['variants']['learned']['mean'] == 2.
        assert row['variants']['native']['mean'] == 1.5
        assert row['variants']['learned']['seed_std'] == 1.
        assert row['native_minus_learned'] == -.5
        assert row['paired_seed_std'] == .25
        assert row['native_change_percent'] == -25.
        assert all(value == 1.5 for value in row['variants']['native']['per_field_normalized_mse'].values())
        assert all(seed['paired_trajectory_std'] == 0 for seed in row['paired_seeds'])


@pytest.mark.parametrize('change', ['windows', 'initialization', 'normalization', 'guards', 'held_out_ids', 'duplicate'])
def test_gradient_summary_rejects_unmatched_experiments(matched_reports, change):
    root, module = matched_reports
    folder = root/'seed18_native'
    if change == 'normalization':
        np.savez(folder/'normalization.npz', mean=np.ones((16, 7)), scale=np.ones(7))
    else:
        path = folder/('opsplit/sampled_windows.json' if change == 'windows' else
                       'provenance.json' if change == 'guards' else 'opsplit/metrics.json')
        data = json.loads(path.read_text())
        if change == 'windows':
            data[0] = -1
        elif change == 'guards':
            data['config']['positivity_penalty'] = .02
        elif change == 'initialization':
            data['initial_neural_sha256'] = 'changed'
        else:
            records = data['evaluation']['trajectories'][0]['records']
            records[0]['case_id'] = 'unexpected' if change == 'held_out_ids' else records[1]['case_id']
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.summarize(root)


@pytest.mark.parametrize('missing', [False, True])
def test_failed_late_horizon_preserves_earlier_comparison(matched_reports, missing):
    root, module = matched_reports
    path = root/'seed17_native/opsplit/metrics.json'
    data = json.loads(path.read_text())
    evaluation = data['evaluation']
    evaluation['audits_passed'] = False
    for aggregate in evaluation['aggregates']:
        if aggregate['horizon'] == 96:
            aggregate.update(complete=False, mean_normalized_volume_mse=None)
    records = evaluation['trajectories'][0]['records']
    if missing:
        evaluation['trajectories'][0]['records'] = [row for row in records if row['horizon'] != 96]
        evaluation['numerical_failures'] = 1
    else:
        for row in records:
            if row['horizon'] == 96:
                row['audit_passed'] = False
    path.write_text(json.dumps(data))
    report = module.summarize(root)
    assert report['status'] == 'failed_audits'
    for row in report['comparison']:
        assert row['complete'] == (row['horizon'] != 96)
        assert row['native_minus_learned'] == (-.5 if row['horizon'] != 96 else None)


def test_missing_training_output_is_not_a_comparison(tmp_path):
    result = summary_module().summarize(tmp_path)
    assert result['status'] == 'incomplete'
    assert result['comparison'] == []
    assert len(result['missing']) == 24


def test_native_continuity_configuration_rejects_silent_coefficient_fallback():
    config = load_config(Path(__file__).parents[1]/'configs/ensemble_training.yaml')
    config['linear_branch'] = 'native_continuity'
    check_config(config)
    config['coefficient_split'] = 'density_diffusion'
    with pytest.raises(ValueError, match='Coefficient-aware'):
        check_config(config)
