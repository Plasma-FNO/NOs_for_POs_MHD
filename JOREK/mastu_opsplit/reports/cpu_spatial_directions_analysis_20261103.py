"""Compare the next frozen spatial pair, including all eight response fields."""
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.numerical import evaluate_native
from mastu_opsplit.queries import sample_restart
from mastu_opsplit.restart import cubic_basis, read_restart

CHANNELS = ('psi', 'u', 'zj', 'w', 'rho', 'T', 'Vpar', 'pressure')
GEOMETRY_KEYS = ('x', 'vertex', 'size', 'index', 'boundary', 'constrained', 'n_sons')


def pressure_fields(values):
    """Form pressure before subtracting the baseline; pressure is rho*T."""
    return np.concatenate((values, (values[:, 4] * values[:, 5])[:, None]), axis=1)


def field_metrics(actual, reference, weights, target):
    if actual.shape != reference.shape or actual.shape != (len(weights), len(CHANNELS)):
        raise ValueError('metric shapes differ')
    if not all(np.isfinite(x).all() for x in [actual, reference, weights]) or (weights <= 0).any():
        raise ValueError('finite fields and positive physical volume weights required')
    if not len(weights):
        raise ValueError('no covered volume')
    error2 = np.einsum('nc,n->c', (actual - reference)**2, weights)
    norm2 = np.einsum('nc,n->c', reference**2, weights)
    return {name: {'relative_l2_difference': float(np.sqrt(error2[i] / norm2[i])) if norm2[i] else None,
                   'absolute_rms_difference': float(np.sqrt(error2[i] / weights.sum())),
                   'reference_rms': float(np.sqrt(norm2[i] / weights.sum())),
                   'target_met': bool(error2[i] <= target**2 * norm2[i])}
            for i, name in enumerate(CHANNELS)}


def sample_fixed_geometry(path, query, expected):
    """Reuse an exact inverse map only after verifying every geometry array."""
    with h5py.File(path) as stream:
        for key in GEOMETRY_KEYS:
            if not np.array_equal(stream[key][()], expected[key]):
                raise ValueError('query-map geometry changed: ' + key)
        nodal = stream['values'][()][:, :, 0].transpose(2, 1, 0)
    if not np.isfinite(nodal).all():
        raise ValueError('nonfinite native coefficients')
    vertex = expected['vertex'].T - 1
    sizes = expected['size'].transpose(2, 1, 0)
    values = np.full((len(query['mask']), 7), np.nan)
    covered = np.flatnonzero(query['mask'])
    for start in range(0, len(covered), 8192):
        ids = covered[start:start + 8192]
        elements = query['element_ids'][ids] - 1
        st = query['local_coordinates'][ids]
        basis = cubic_basis(st[:, 0], st[:, 1])[0] * sizes[elements]
        values[ids] = np.einsum('nvd,nvdc->nc', basis, nodal[vertex[elements]])
    return values


def full_domain_moments(path, expected_time):
    geometry, state, metadata = read_restart(path, quadrature_order=8)
    if abs(metadata['time_seconds'] - expected_time) > 1e-15:
        raise ValueError('moment time differs from the frozen common time')
    fields = evaluate_native(state, geometry)
    if (fields[..., 4:6] <= 0).any():
        raise ValueError('nonpositive sampled density/temperature')
    weights = geometry.volume_weights
    result = {'volume_m3': float(weights.sum()), 'density_integral': float(np.sum(weights * fields[..., 4])),
              'pressure_integral': float(np.sum(weights * fields[..., 4] * fields[..., 5])),
              'minimum_density': float(fields[..., 4].min()), 'minimum_temperature': float(fields[..., 5].min()),
              'metadata': metadata}
    del geometry, state, fields
    gc.collect()
    return result


def main():
    base = Path(sys.argv[1])
    manifest_path = Path(sys.argv[2]) if len(sys.argv) > 2 else base / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    report_path = Path(sys.argv[3]) if len(sys.argv) > 3 else base / 'convergence.json'
    if report_path.exists():
        raise ValueError('spatial comparison already exists')
    paths = {'existing': {}, 'refined': {}}
    for label in ['baseline48', 'perturbed48']:
        reference = manifest['references'][label]
        old = Path(reference['work'])
        new = base / label
        result = json.loads((new / 'result.json').read_text())
        if result['status'] != 'completed':
            raise ValueError('refined run is not validated: ' + label)
        for time, name in [('initial', 'jorek000000.h5'), ('final', 'jorek_restart.h5')]:
            if sha256_file(old / name) != reference[time + '_sha256']:
                raise ValueError('existing reference changed: ' + label + '/' + time)
            if sha256_file(new / name) != result[time]['restart_sha256']:
                raise ValueError('refined output changed: ' + label + '/' + time)
            paths['existing'][label, time] = old / name
            paths['refined'][label, time] = new / name
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'analysis_source_sha256': sha256_file(__file__),
              'queries_source_sha256': sha256_file(ROOT / 'src/mastu_opsplit/queries.py'),
              'manifest_sha256': sha256_file(manifest_path),
              'scope': 'Clean existing/refined paired native fields at common fine-mesh physical q4 queries; q8 full-domain moments.',
              'final_time_s': manifest['final_time_s'], 'steps_both_meshes': 48, 'tstep_code_units_both_meshes': .0625,
              'field_relative_target': manifest['field_relative_target'], 'integral_relative_target': manifest['integral_relative_target'],
              'scientific_gates_passed': [], 'comparisons': {}, 'full_domain_moments_q8': {},
              'limitations': ['Two-resolution consistency at fixed timestep; asymptotic convergence order is not measured.',
                              'Refined-mesh timestep sensitivity is not independently measured.',
                              'Uncovered points are excluded explicitly; no extrapolation is used.',
                              'Full physical balance and autonomous numerical backend remain unvalidated.']}
    if manifest.get('spatial_refinement_scope'):
        report['mesh_refinement_scope']=manifest['spatial_refinement_scope']
        report['limitations'].append(manifest['spatial_refinement_scope'])
    sampled = {'existing': {}, 'refined': {}}
    points, weights, query, old_geometry = None, None, None, None
    for time in ['initial', 'final']:
        expected_time = 0. if time == 'initial' else manifest['final_time_s']
        for label in ['baseline48', 'perturbed48']:
            fine = paths['refined'][label, time]
            geometry, state, metadata = read_restart(fine, quadrature_order=4)
            if abs(metadata['time_seconds'] - expected_time) > 1e-15:
                raise ValueError('refined query times differ')
            values = evaluate_native(state, geometry).reshape(-1, 7)
            if points is None:
                points = geometry.quadrature_rz.reshape(-1, 2).copy()
                weights = geometry.volume_weights.ravel().copy()
                old = paths['existing'][label, time]
                query = sample_restart(old, points)
                if not query['mask'].any():
                    raise ValueError('no common physical queries')
                with h5py.File(old) as stream:
                    old_geometry = {key: stream[key][()] for key in GEOMETRY_KEYS}
                self_query = sample_restart(fine, points[::211])
                if not self_query['mask'].all():
                    raise ValueError('fine native self-query coverage failed')
                self_error = float(np.max(np.abs(self_query['values'] - values[::211])))
                if self_error >= 1e-7:
                    raise ValueError('fine native self-query consistency failed')
                report['native_self_query'] = {'points': len(points[::211]), 'maximum_absolute_field_error': self_error}
                report['coverage'] = {'total_queries': len(points), 'covered_queries': int(query['mask'].sum()),
                    'covered_refined_volume_fraction': float(weights[query['mask']].sum() / weights.sum()),
                    'uncovered_refined_volume_m3': float(weights[~query['mask']].sum()),
                    'uncovered_coordinate_examples_m': points[~query['mask']][:20].tolist(),
                    'maximum_inverse_coordinate_error_m': float(query['coordinate_error_m'][query['mask']].max()),
                    'fallback_points': query['fallback_points'], 'full_wall_coverage': bool(query['mask'].all())}
            elif not (np.array_equal(points, geometry.quadrature_rz.reshape(-1, 2))
                      and np.array_equal(weights, geometry.volume_weights.ravel())):
                raise ValueError('refined comparison geometry changed')
            sampled['refined'][label, time] = pressure_fields(values)
            sampled['existing'][label, time] = pressure_fields(sample_fixed_geometry(paths['existing'][label, time], query, old_geometry))
            del geometry, state, values
            gc.collect()
    mask = query['mask']
    for time in ['initial', 'final']:
        expected_time = 0. if time == 'initial' else manifest['final_time_s']
        comparisons = {}
        for label in ['baseline48', 'perturbed48']:
            comparisons[label] = field_metrics(sampled['existing'][label, time][mask],
                                               sampled['refined'][label, time][mask], weights[mask], manifest['field_relative_target'])
        response = {mesh: sampled[mesh]['perturbed48', time] - sampled[mesh]['baseline48', time] for mesh in sampled}
        comparisons['paired_response'] = field_metrics(response['existing'][mask], response['refined'][mask],
                                                       weights[mask], manifest['field_relative_target'])
        report['comparisons'][time] = comparisons
        moments = {}
        for mesh in paths:
            moments[mesh] = {label: full_domain_moments(paths[mesh][label, time], expected_time)
                             for label in ['baseline48', 'perturbed48']}
        report['full_domain_moments_q8'][time] = moments
        print(json.dumps({'compared': time, 'fields': comparisons}), flush=True)
    integral_checks = {}
    for label in ['baseline48', 'perturbed48']:
        integral_checks[label] = {}
        for name in ['density_integral', 'pressure_integral']:
            a = report['full_domain_moments_q8']['final']['existing'][label][name]
            b = report['full_domain_moments_q8']['final']['refined'][label][name]
            error = abs(a - b) / abs(b)
            integral_checks[label][name] = {'relative_difference': error, 'target_met': error <= manifest['integral_relative_target']}
    report['final_integral_comparisons_q8'] = integral_checks
    final = report['comparisons']['final']
    report['full_fields_one_percent_target_met_on_covered_volume'] = all(
        metric['target_met'] for label in ['baseline48', 'perturbed48'] for metric in final[label].values())
    report['paired_density_temperature_pressure_one_percent_target_met_on_covered_volume'] = all(
        final['paired_response'][name]['target_met'] for name in ['rho', 'T', 'pressure'])
    report['integrals_two_percent_target_met'] = all(metric['target_met'] for item in integral_checks.values() for metric in item.values())
    report['paired_all_fields_one_percent_target_met_on_covered_volume'] = all(metric['target_met'] for metric in final['paired_response'].values())
    report['candidate_spatial_targets_met_on_covered_volume'] = all(report[key] for key in [
        'paired_all_fields_one_percent_target_met_on_covered_volume',
        'full_fields_one_percent_target_met_on_covered_volume',
        'paired_density_temperature_pressure_one_percent_target_met_on_covered_volume', 'integrals_two_percent_target_met'])
    report['full_domain_spatial_convergence_established'] = False
    report['status'] = 'candidate_targets_met_on_covered_volume' if report['candidate_spatial_targets_met_on_covered_volume'] else 'spatial_targets_not_met'
    with report_path.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': report['status'], 'report': str(report_path)}), flush=True)


if __name__ == '__main__':
    main()
