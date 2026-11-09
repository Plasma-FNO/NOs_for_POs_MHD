import importlib.util
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from test_restart import rectangle


def report_module(filename):
    path = Path(__file__).resolve().parents[1] / 'reports' / filename
    spec = importlib.util.spec_from_file_location('spatial_pair_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('change', ['valid', 'boundary', 'unselected', 'amplitude', 'geometry', 'absent'])
def test_refined_initializer_checks(tmp_path, change):
    reference, current = tmp_path / 'reference.h5', tmp_path / 'current.h5'
    rectangle(reference)
    if change == 'boundary':
        with h5py.File(reference, 'r+') as f:
            f['boundary'][...] = 1
    shutil.copyfile(reference, current)
    with h5py.File(current, 'r+') as f:
        values = f['values'][()]
        values[4:6] *= 1.05 if change != 'absent' else 1.
        if change == 'amplitude':
            values[4, 0, 0, 0] *= 1.1
        if change == 'unselected':
            values[2, 0, 0, 0] = 1
        if change == 'geometry':
            f['x'][0, 0, 0, 0] += .1
        f['values'][...] = values
    check = report_module('cpu_spatial_pair_worker_20261103.py').check_initializer
    if change == 'valid':
        result = check(current, reference, [.05, .05], -1., 1.)
        assert result['affected_elements'] == 1
        assert result['maximum_psi_n_control_hull_with_rounding_interval'] < .85
        np.testing.assert_allclose(result['maximum_nodal_relative_perturbations'], .05)
        assert check(reference, reference, [0., 0.], -1., 1.)['affected_elements'] == 0
    else:
        with pytest.raises(ValueError):
            check(current, reference, [.05, .05], -1., 1.)


def test_pressure_response_and_zero_norm_metrics():
    analysis = report_module('cpu_spatial_pair_analysis_20261103.py')
    baseline = np.ones((2, 7))
    perturbed = baseline.copy()
    perturbed[:, 4:6] = [2., 3.]
    response = analysis.pressure_fields(perturbed) - analysis.pressure_fields(baseline)
    np.testing.assert_allclose(response[:, 7], 5.)
    weights = np.array([1., 3.])
    metrics = analysis.field_metrics(2 * response, response, weights, .01)
    assert metrics['pressure']['relative_l2_difference'] == pytest.approx(1.)
    assert not metrics['pressure']['target_met']
    assert metrics['psi']['relative_l2_difference'] is None and metrics['psi']['target_met']
    changed = response.copy()
    changed[:, 0] = 1e-14
    assert not analysis.field_metrics(changed, response, weights, .01)['psi']['target_met']
    with pytest.raises(ValueError, match='positive'):
        analysis.field_metrics(response, response, np.array([1., -1.]), .01)


def test_reused_inverse_map_matches_native_queries_and_rejects_geometry_changes(tmp_path):
    from mastu_opsplit.queries import sample_restart
    analysis = report_module('cpu_spatial_pair_analysis_20261103.py')
    path = tmp_path / 'native.h5'
    rectangle(path)
    points = np.array([[1.2, .4], [1.8, .7], [1.5, 1.5]])
    query = sample_restart(path, points)
    with h5py.File(path) as f:
        geometry = {key: f[key][()] for key in analysis.GEOMETRY_KEYS}
    actual = analysis.sample_fixed_geometry(path, query, geometry)
    np.testing.assert_allclose(actual, query['values'], equal_nan=True, atol=1e-14)
    with h5py.File(path, 'r+') as f:
        values = f['values'][()]
        values[4] *= 2.
        f['values'][...] = values
    actual = analysis.sample_fixed_geometry(path, query, geometry)
    np.testing.assert_allclose(actual[query['mask'], 4], 2 * query['values'][query['mask'], 4])
    assert np.isnan(actual[~query['mask']]).all()
    with h5py.File(path, 'r+') as f:
        f['x'][0, 0, 0, 0] += .1
    with pytest.raises(ValueError, match='geometry changed'):
        analysis.sample_fixed_geometry(path, query, geometry)


def test_complete_spatial_report_detects_full_field_and_paired_response_errors(tmp_path, monkeypatch):
    from mastu_opsplit.common import sha256_file
    analysis = report_module('cpu_spatial_pair_analysis_20261103.py')
    study = tmp_path / 'study'
    (study / 'reports').mkdir(parents=True)
    (study / 'src/mastu_opsplit').mkdir(parents=True)
    shutil.copyfile(analysis.ROOT / 'src/mastu_opsplit/queries.py', study / 'src/mastu_opsplit/queries.py')
    base = tmp_path / 'paired'
    base.mkdir()
    references = {}
    final_time = 1.4059131112280784e-6
    for mesh in ['existing', 'refined']:
        for label in ['baseline48', 'perturbed48']:
            work = (tmp_path / 'existing' if mesh == 'existing' else base) / label
            work.mkdir(parents=True)
            hashes = {}
            for time, name in [('initial', 'jorek000000.h5'), ('final', 'jorek_restart.h5')]:
                path = work / name
                rectangle(path)
                with h5py.File(path, 'r+') as f:
                    f['t_now'][...] = 0. if time == 'initial' else final_time / 1e-6
                    f['index_now'][...] = 0 if time == 'initial' else 48
                    values = f['values'][()]
                    values[4] *= 1.1 if mesh == 'existing' else 1.
                    if label == 'perturbed48':
                        values[4:6] *= 1.05
                    f['values'][...] = values
                hashes[time] = sha256_file(path)
            if mesh == 'existing':
                references[label] = {'work': str(work), **{time + '_sha256': value for time, value in hashes.items()}}
            else:
                (work / 'result.json').write_text(json.dumps({'status': 'completed', **{
                    time: {'restart_sha256': value} for time, value in hashes.items()}}))
    (base / 'manifest.json').write_text(json.dumps({'references': references, 'final_time_s': final_time,
                                                  'field_relative_target': .01, 'integral_relative_target': .02}))
    monkeypatch.setattr(analysis, 'ROOT', study)
    monkeypatch.setattr(analysis.sys, 'argv', [analysis.__file__, str(base)])
    analysis.main()
    result = json.loads((study / 'reports/cpu_spatial_convergence_20261103.json').read_text())
    assert result['status'] == 'spatial_targets_not_met'
    assert result['coverage']['full_wall_coverage']
    assert result['coverage']['covered_refined_volume_fraction'] == pytest.approx(1.)
    for label in ['baseline48', 'perturbed48', 'paired_response']:
        assert result['comparisons']['final'][label]['rho']['relative_l2_difference'] == pytest.approx(.1)
    assert not result['integrals_two_percent_target_met']
    assert not result['full_domain_spatial_convergence_established']
    assert result['scientific_gates_passed'] == []
