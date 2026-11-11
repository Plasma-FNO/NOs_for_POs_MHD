import importlib.util
import json
from pathlib import Path

import pytest

from mastu_opsplit.common import config_hash

spec = importlib.util.spec_from_file_location('ood_summary',
    Path(__file__).parents[1]/'scripts/summarize_conditional_ood.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture_run(tmp_path):
    folder, training = tmp_path/'ood', tmp_path/'training'
    folder.mkdir()
    (training/'conditional').mkdir(parents=True)
    cases = [dict(case_id=f'ood{i}', matched_id_case=f'id{i}', coefficient_multiplier=2+i*8/9) for i in range(10)]
    provenance = dict(config={}, normalization_refitted=False, cases=cases,
                      ood_manifest_sha256='dataset', normalization_sha256='normalization')
    (folder/'provenance.json').write_text(json.dumps(provenance))
    trajectories = []
    for i, case in enumerate(cases):
        rows = [dict(case_id=case['case_id'], horizon=h, start_frame=0, substeps=1, audit_passed=True,
                     normalized_volume_mse=i+1., raw_normalized_volume_mse=i+2.,
                     per_field_normalized_mse={'rho':i+1.}, raw_per_field_normalized_mse={'rho':i+2.},
                     guard_statistics={'limited_field_steps':0, 'minimum_alpha':1.}) for h in summary.HORIZONS]
        trajectories.append(dict(records=rows, provenance_hash=config_hash(provenance), failure=None))
        id_rows = [dict(row, case_id=case['matched_id_case'], normalized_volume_mse=.5) for row in rows]
        (training/'conditional'/f'evaluation_{i:03d}_substeps1.json').write_text(json.dumps(dict(records=id_rows)))
    metrics = dict(trajectories=trajectories, audits_passed=True, numerical_failures=0)
    (folder/'metrics.json').write_text(json.dumps(metrics))
    return folder, training, metrics


def test_summary_pairs_initial_conditions_and_withholds_missing_seeds(tmp_path):
    folder, training, _ = fixture_run(tmp_path)
    report = summary.read_run(folder, training)
    assert report['scores']['96']['mean'] == 5.5
    assert report['scores']['96']['matched_id_mean'] == .5
    assert len(report['scores']['96']['per_coefficient']) == 10
    root = tmp_path/'campaign'
    root.mkdir()
    folder.rename(root/'seed17_conditional_fixed')
    training.rename(tmp_path/'seed17_conditional')
    result = summary.summarize(root, tmp_path)
    assert all(not row['complete'] and row['mean'] is None for row in result['seed_summary'])
    assert result['seed_summary'][0]['completed_seeds'] == 1


@pytest.mark.parametrize('failure', ['audit', 'missing', 'stale'])
def test_summary_refuses_failed_missing_or_stale_evidence(tmp_path, failure):
    folder, training, metrics = fixture_run(tmp_path)
    if failure == 'audit':
        metrics['trajectories'][0]['records'][0]['audit_passed'] = False
    elif failure == 'missing':
        metrics['trajectories'].pop()
    else:
        metrics['trajectories'][0]['provenance_hash'] = 'wrong'
    (folder/'metrics.json').write_text(json.dumps(metrics))
    if failure == 'stale':
        with pytest.raises(ValueError, match='provenance'):
            summary.read_run(folder, training)
    else:
        assert not summary.read_run(folder, training)['scores']['16']['complete']
