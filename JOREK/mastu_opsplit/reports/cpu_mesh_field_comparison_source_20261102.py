"""Compare baseline meshes at common physical queries with explicit coverage."""
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.restart import read_restart
from mastu_opsplit.numerical import evaluate_native
from mastu_opsplit.queries import sample_restart


def main():
    output = ROOT/'reports/cpu_mesh_field_comparison_20261102.json'
    assert not output.exists()
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha256_file(__file__),
        'query_source_sha256': sha256_file(ROOT/'src/mastu_opsplit/queries.py'),
        'scope': 'Unperturbed coarse/fine native fields on fine q4 physical volume queries; uncovered points are reported, not extrapolated. Historical 96-step IO errors remain diagnostic only.',
        'quadrature_order': 4, 'gate_A_passed': False, 'perturbed_mesh_convergence_checked': False,
        'comparisons': {}}
    paths = {'initial': ('cpu_axis_order_equilibrium', 'cpu_fine_mpi4_equilibrium'),
             'final': ('cpu_baseline96', 'cpu_fine_mpi4_baseline96')}
    coordinates = None
    for label, (coarse_name, fine_name) in paths.items():
        coarse, fine = [ROOT/f'runs/{name}_20261102/attempt_1/work/jorek_restart.h5' for name in (coarse_name, fine_name)]
        geometry, state, metadata = read_restart(fine, quadrature_order=4)
        values = evaluate_native(state, geometry).reshape(-1, 7)
        points = geometry.quadrature_rz.reshape(-1, 2)
        weights = geometry.volume_weights.ravel()
        if coordinates is None:
            coordinates = points.copy()
        else:
            assert np.array_equal(coordinates, points)
        if label == 'final':
            check = sample_restart(fine, points[::97])
            assert check['mask'].all()
            error = np.abs(check['values']-values[::97]).max(axis=0)
            assert error.max() < 1e-7
            report['real_native_self_query_check'] = {'points': int(check['mask'].sum()),
                'maximum_absolute_error_by_channel': error.tolist(),
                'maximum_coordinate_error_m': float(check['coordinate_error_m'].max()),
                'fallback_points': check['fallback_points']}
        query = sample_restart(coarse, points)
        covered = query['mask']
        assert covered.any()
        weighted = weights[covered]
        actual, reference = query['values'][covered], values[covered]
        assert np.isfinite(actual).all()
        fields = {}
        for channel, name in enumerate(('psi', 'u', 'zj', 'w', 'rho', 'T', 'Vpar', 'pressure')):
            a, b = (actual[:, channel], reference[:, channel]) if channel < 7 else (actual[:, 4]*actual[:, 5], reference[:, 4]*reference[:, 5])
            norm2 = float(np.sum(weighted*b*b))
            error2 = float(np.sum(weighted*(a-b)**2))
            fields[name] = {'relative_l2_difference': np.sqrt(error2/norm2) if norm2 else None,
                            'absolute_rms_difference': float(np.sqrt(error2/weighted.sum()))}
        report['comparisons'][label] = {'coarse_restart_sha256': sha256_file(coarse),
            'fine_restart_sha256': sha256_file(fine), 'metadata': metadata,
            'total_queries': len(points), 'covered_queries': int(covered.sum()),
            'covered_fine_volume_fraction': float(weights[covered].sum()/weights.sum()),
            'uncovered_fine_volume_m3': float(weights[~covered].sum()),
            'uncovered_examples_m': points[~covered][:20].tolist(),
            'maximum_inverse_coordinate_error_m': float(query['coordinate_error_m'][covered].max()),
            'fallback_points': query['fallback_points'], 'fields_on_covered_volume': fields,
            'full_wall_coverage': bool(covered.all())}
        del geometry, state, values, query, actual, reference
        gc.collect()
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: {'coverage': value['covered_fine_volume_fraction'],
                           'fields': value['fields_on_covered_volume']} for key, value in report['comparisons'].items()}, indent=2))


if __name__ == '__main__':
    main()
