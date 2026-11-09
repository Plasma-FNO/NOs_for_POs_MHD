"""Localize clean two-mesh discrepancies without changing convergence metrics."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from cpu_spatial_pair_analysis_20261103 import (
    ROOT, CHANNELS, GEOMETRY_KEYS, field_metrics, pressure_fields,
    sample_fixed_geometry, sample_restart, read_restart, evaluate_native,
)
from mastu_opsplit.common import sha256_file


def concentration(error, reference, weights, points, regions):
    """Positive weighted energy fractions; no local relative-error floors."""
    energy = error**2 * weights
    total = energy.sum()
    result = {'error_energy': float(total),
              'reference_energy': float(np.sum(reference**2 * weights)),
              'regional_error_energy_fractions': {
                  name: float(energy[mask].sum()/total) if total else 0.
                  for name, mask in regions.items()}}
    order = np.argsort(error**2)[::-1]
    selected = order[np.cumsum(weights[order]) <= .01*weights.sum()]
    result['error_fraction_in_largest_error_one_percent_volume'] = float(energy[selected].sum()/total) if total else 0.
    order = np.argsort(energy)[::-1]
    if total:
        selected = order[:np.searchsorted(np.cumsum(energy[order]), .9*total)+1]
        result['bbox_containing_largest_contributors_90percent_energy_m'] = [
            points[selected].min(axis=0).tolist(), points[selected].max(axis=0).tolist()]
    return result


def main():
    base = Path(sys.argv[1])
    out = Path(sys.argv[2])
    if out.exists():
        raise ValueError('preserve existing diagnostic evidence')
    manifest = json.loads((base/'manifest.json').read_text())
    fine = base/'baseline48'
    old = Path(manifest['references']['baseline48']['work'])
    geometry, state, metadata = read_restart(fine/'jorek000000.h5', quadrature_order=4)
    points = geometry.quadrature_rz.reshape(-1, 2).copy()
    weights = geometry.volume_weights.ravel().copy()
    initial = evaluate_native(state, geometry).reshape(-1, 7)
    psi_n = (initial[:, 0]-manifest['reference_psi_axis'])/(manifest['reference_psi_boundary']-manifest['reference_psi_axis'])
    query = sample_restart(old/'jorek000000.h5', points)
    mask = query['mask']
    with h5py.File(old/'jorek000000.h5') as stream:
        old_geom = {key: stream[key][()] for key in GEOMETRY_KEYS}
    with h5py.File(fine/'jorek000000.h5') as stream:
        fine_geom = {key: stream[key][()] for key in GEOMETRY_KEYS}
    gauss = .5*(np.polynomial.legendre.leggauss(4)[0]+1)
    ss, tt = np.meshgrid(gauss, gauss, indexing='ij')
    fine_query = {'mask': np.ones(len(points), dtype=bool),
                  'element_ids': np.repeat(np.arange(len(geometry.element_ids))+1, 16),
                  'local_coordinates': np.tile(np.column_stack((ss.ravel(), tt.ravel())), (len(geometry.element_ids), 1))}
    check = sample_fixed_geometry(fine/'jorek000000.h5', fine_query, fine_geom)
    if not np.allclose(check, initial, rtol=1e-11, atol=1e-12):
        raise ValueError('fine reused native map ordering differs')
    boundary_elements = np.any(fine_geom['boundary'][fine_geom['vertex'].T-1] != 0, axis=1)
    regions = {'boundary_touching_elements': np.repeat(boundary_elements, 16),
               'central_closed_flux_psi_n_lt_0p9_absZ_lt_1': (psi_n < .9) & (np.abs(points[:, 1]) < 1),
               'separatrix_band_0p9_to_1p1_absZ_lt_1': (psi_n >= .9) & (psi_n <= 1.1) & (np.abs(points[:, 1]) < 1),
               'outer_midplane_psi_n_gt_1p1_absZ_lt_1': (psi_n > 1.1) & (np.abs(points[:, 1]) < 1),
               'divertor_absZ_ge_1': np.abs(points[:, 1]) >= 1,
               'perturbation_box_3width': (np.abs(points[:, 0]-.95)<.24) & (np.abs(points[:, 1])<.48)}
    report = {'source_sha256': sha256_file(__file__), 'manifest_sha256': sha256_file(base/'manifest.json'),
              'field_target': .01, 'integral_target': .02,
              'covered_volume_fraction': float(weights[mask].sum()/weights.sum()),
              'regions_overlap': True,
              'regional_volume_fractions': {name: float(weights[mask & region].sum()/weights[mask].sum()) for name, region in regions.items()},
              'snapshots': {}, 'input_snapshots_sha256': {},
              'limitations': ['Two-mesh localization does not prove asymptotic order or isolate timestep error.',
                              'Current and vorticity are native evolved auxiliary fields, not finite-difference reconstructions.',
                              'Boundary rows and force-balance residuals are not diagnosed by this localization.']}
    final_maps = {}
    for step in range(0, 49, 8):
        sampled = {}
        times = []
        for mesh in ['old', 'fine']:
            for label in ['baseline48', 'perturbed48']:
                folder = Path(manifest['references'][label]['work']) if mesh == 'old' else base/label
                path = folder/f'jorek{step:06d}.h5'
                with h5py.File(path) as stream:
                    times.append(float(stream['t_now'][()].item()))
                report['input_snapshots_sha256'][str(path)] = sha256_file(path)
                sampled[mesh, label] = pressure_fields(sample_fixed_geometry(path, query if mesh == 'old' else fine_query,
                                                                            old_geom if mesh == 'old' else fine_geom))[mask]
        if max(times)-min(times)>1e-15:
            raise ValueError('snapshot times differ')
        comparisons = {}
        for label in ['baseline48', 'perturbed48', 'paired_response']:
            if label == 'paired_response':
                a, b = [sampled[mesh, 'perturbed48']-sampled[mesh, 'baseline48'] for mesh in ['old', 'fine']]
            else:
                a, b = [sampled[mesh, label] for mesh in ['old', 'fine']]
            metrics = field_metrics(a, b, weights[mask], .01)
            for i, name in enumerate(CHANNELS):
                metrics[name].update(concentration(a[:, i]-b[:, i], b[:, i], weights[mask], points[mask],
                                                  {key: value[mask] for key, value in regions.items()}))
                if step == 48 and name in ['u', 'zj', 'w', 'Vpar']:
                    squared = np.zeros(len(points))
                    squared[mask] = (a[:, i]-b[:, i])**2*weights[mask]
                    final_maps[label+'_'+name] = squared.reshape(-1, 16).sum(axis=1)
            comparisons[label] = metrics
        report['snapshots'][str(step)] = {'native_time': times[0], 'comparisons': comparisons}
        print(json.dumps({'step': step, 'full_baseline_percent': {k: 100*v['relative_l2_difference'] if v['relative_l2_difference'] is not None else None for k,v in comparisons['baseline48'].items()},
                          'paired_percent': {k: 100*v['relative_l2_difference'] if v['relative_l2_difference'] is not None else None for k,v in comparisons['paired_response'].items()}}), flush=True)
    np.savez_compressed(out.with_suffix('.npz'), element_centers_m=points.reshape(-1,16,2).mean(axis=1), **final_maps)
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
