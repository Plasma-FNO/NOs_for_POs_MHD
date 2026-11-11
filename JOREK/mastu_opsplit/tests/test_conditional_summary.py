import importlib.util
import json
from pathlib import Path

import pytest

from mastu_opsplit.common import config_hash

spec = importlib.util.spec_from_file_location('conditional_summary',
    Path(__file__).parents[1]/'scripts/summarize_conditional.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture_run(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'cases': [{'case_id': name, 'split': 'test'} for name in ('a', 'b')]}))
    provenance = dict(config=dict(manifest=str(manifest), manifest_sha256='pinned', split_counts={'test':2},
                                 steps=2, horizon=4, stride=1),
                      normalization=dict(mean_sha256='mean', scale_sha256='scale', training_case_ids=['train']))
    run = tmp_path/'campaign/seed17_conditional'
    folder = run/'conditional'
    folder.mkdir(parents=True)
    (run/'results.json').write_text('{}')
    (run/'provenance.json').write_text(json.dumps(provenance))
    (folder/'metrics.json').write_text(json.dumps(dict(selected_step=2, parameters=10,
                                                    evaluation={'audits_passed':True})))
    for index, name in enumerate(('a', 'b')):
        rows = [dict(case_id=name, horizon=h, substeps=1, start_frame=0, audit_passed=True,
                     normalized_volume_mse=.1+index, raw_normalized_volume_mse=.2+index,
                     per_field_normalized_mse=dict.fromkeys(summary.FIELDS, .1+index),
                     elapsed_seconds=1., guard_statistics={'limited_field_steps':0, 'minimum_alpha':1.},
                     query_time_regime='within_training_range') for h in summary.HORIZONS]
        (folder/f'evaluation_{index:03d}_substeps1.json').write_text(json.dumps(
            dict(records=rows, failure=None, provenance_hash=config_hash(provenance))))
    return run, folder, provenance


@pytest.mark.parametrize('problem', ['missing', 'audit', 'nonfinite', 'raw_nonfinite'])
def test_summary_withholds_incomplete_or_invalid_horizons(tmp_path, problem):
    run, folder, provenance = fixture_run(tmp_path)
    path = folder/'evaluation_001_substeps1.json'
    if problem == 'missing':
        path.unlink()
    else:
        report = json.loads(path.read_text())
        for row in report['records']:
            if problem == 'audit': row['audit_passed'] = False
            if problem == 'nonfinite': row['normalized_volume_mse'] = float('nan')
            if problem == 'raw_nonfinite': row['raw_normalized_volume_mse'] = float('nan')
        path.write_text(json.dumps(report))
    result, _ = summary.read_run(run, 'conditional', provenance, ['a', 'b'])
    assert all(not row['complete'] and row['mean'] is None for row in result['scores'].values())


def test_summary_never_calls_one_seed_a_three_seed_result(tmp_path):
    run, _, _ = fixture_run(tmp_path)
    result = summary.summarize(run.parent, tmp_path/'ensemble', tmp_path/'gradient', tmp_path/'implicit')
    valid = result['models']['conditional/seed17']['scores']['16']
    assert valid['complete'] and valid['mean'] == pytest.approx(.6)
    for row in result['seed_summary']:
        assert not row['complete'] and row['mean'] is None and row['seed_std'] is None
        assert row['completed_seeds'] == (1 if row['model'] == 'conditional' else 0)


def test_summary_rejects_mismatched_normalization_and_stale_evaluation(tmp_path):
    run, folder, provenance = fixture_run(tmp_path)
    changed = json.loads(json.dumps(provenance))
    changed['normalization']['scale_sha256'] = 'different'
    report, records = summary.read_run(run, 'conditional', changed, ['a', 'b'])
    assert report['status'] == 'incompatible_data_or_budget' and not records
    path = folder/'evaluation_000_substeps1.json'
    stale = json.loads(path.read_text())
    stale['provenance_hash'] = 'different'
    path.write_text(json.dumps(stale))
    report, records = summary.read_run(run, 'conditional', provenance, ['a', 'b'])
    assert report['status'] == 'evaluation_provenance_mismatch' and not records
