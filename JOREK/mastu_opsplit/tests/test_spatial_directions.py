"""Guards for charged directional refinement and automatic continuation."""
import json

import pytest

from test_spatial_pair import report_module


def test_independent_mesh_controls_and_successive_radial_reference():
    module = report_module('cpu_spatial_directions_20261103.py')
    cases = module.candidates()
    assert cases['radial160']['overrides'] == {'n_flux': 160}
    assert cases['radial320']['reference'] == str(module.BASE / 'radial160')
    assert all(key.startswith('n_ext_block(') for key in cases['wall60']['overrides'])
    assert set(cases['poloidal440']['overrides']) == {'n_tht', 'n_leg', 'n_leg_out', 'n_up_leg', 'n_up_leg_out'}


@pytest.mark.parametrize('label,fallback', [('wall60', 'wall50'), ('poloidal440', 'poloidal330'), ('radial160', None)])
def test_failed_proof_only_starts_declared_fallback(tmp_path, monkeypatch, label, fallback):
    module = report_module('cpu_spatial_directions_20261103.py')
    monkeypatch.setattr(module, 'BASE', tmp_path)
    work = tmp_path / label
    (work / 'probe3').mkdir(parents=True)
    module.write_json(work / 'campaign_case.json', {'jobs': {}})
    module.write_json(work / 'probe3/result.json', {'status': 'failed_validation'})
    monkeypatch.setattr(module, 'verify_campaign', lambda: {})
    started = []
    monkeypatch.setattr(module, 'start_probe', started.append)
    monkeypatch.setattr(module, 'submit', lambda *args: pytest.fail('failed probe must not launch a full pair'))
    module.finish_probe(label, 1)
    assert started == ([] if fallback is None else [fallback])
    assert module.load_case(label)['status'] == 'probe_failed'


@pytest.mark.parametrize('label,met,covered,continuation', [
    ('radial160', False, True, True),
    ('radial160', True, True, False),
    ('radial160', True, False, True),
    ('wall60', False, True, False),
    ('radial320', False, True, False),
])
def test_radial_continuation_requires_failed_metric_or_coverage(tmp_path, monkeypatch, label, met, covered, continuation):
    module = report_module('cpu_spatial_directions_20261103.py')
    monkeypatch.setattr(module, 'BASE', tmp_path)
    monkeypatch.setattr(module, 'REPORTS', tmp_path)
    monkeypatch.setattr(module, 'verify_campaign', lambda: {})
    work = tmp_path / label
    work.mkdir()
    manifest = work / 'analysis_manifest.json'
    manifest.write_text('{}')
    module.write_json(work / 'campaign_case.json', {'jobs': {}, 'analysis_manifest_sha256': module.sha256_file(manifest)})
    output = tmp_path / f'cpu_spatial_directions_{label}_convergence_20261103.json'
    def fake_analysis(*args, **kwargs):
        module.write_json(output, {'candidate_spatial_targets_met_on_covered_volume': met, 'coverage': {'full_wall_coverage': covered}})
    monkeypatch.setattr(module.subprocess, 'run', fake_analysis)
    started = []
    monkeypatch.setattr(module, 'start_probe', started.append)
    module.analyze(label)
    assert started == (['radial320'] if continuation else [])
    assert module.load_case(label)['status'] == ('directional_targets_met' if met and covered else 'directional_targets_not_met')


def test_duplicate_submission_rejected_before_scheduler(tmp_path, monkeypatch):
    module = report_module('cpu_spatial_directions_20261103.py')
    monkeypatch.setattr(module, 'BASE', tmp_path)
    (tmp_path / 'wall60').mkdir()
    module.write_json(tmp_path / 'wall60/campaign_case.json', {'jobs': {'probe': {'job_id': '1'}}})
    with pytest.raises(ValueError, match='already submitted'):
        module.submit('wall60', 'probe', 4, 256, 2, 'false')
