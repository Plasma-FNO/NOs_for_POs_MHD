"""Frozen physical parity for native initial and evolved snapshots."""
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mastu_opsplit.numerical import evaluate_native, evaluate_native_grad
from mastu_opsplit.restart import read_restart
from cpu_cached_rz_recovery_worker_20261103 import rms


def parity(path, reference, protocol):
    with h5py.File(path) as a, h5py.File(reference) as b:
        keys = ['x', 'values', 'vertex', 'size', 'boundary', 'constrained', 'n_sons', 't_now', 'index_now']
        arrays = {key: bool(np.array_equal(a[key][()], b[key][()])) for key in keys}
        initial = all(float(f['t_now'][()].item()) == 0 and int(f['index_now'][()].item()) == 0 for f in [a, b])
        for key in ['xtime', 'energies']:
            if key in a and key in b:
                arrays[key] = bool(np.array_equal(a[key][()], b[key][()]))
            elif not initial or key in a or key in b:
                raise ValueError('missing or inconsistent evolved diagnostic array: ' + key)
    geometry, state, metadata = read_restart(path, quadrature_order=protocol['quadrature_order'])
    expected_geometry, expected, expected_metadata = read_restart(reference, quadrature_order=protocol['quadrature_order'])
    for name in ('dof_map', 'dof_ids', 'basis', 'basis_grad', 'quadrature_rz', 'poloidal_weights', 'bounds', 'boundary_labels'):
        if not np.array_equal(getattr(geometry, name), getattr(expected_geometry, name)):
            raise ValueError('reference geometry changed: ' + name)
    if metadata['steps'] != expected_metadata['steps'] or abs(metadata['time_seconds'] - expected_metadata['time_seconds']) >= 1e-15:
        raise ValueError('reference time/steps differ')
    metrics = {}
    passed = True
    for name, evaluator, gradient in [('field', evaluate_native, False), ('gradient', evaluate_native_grad, True)]:
        current, baseline = evaluator(state, geometry), evaluator(expected, geometry)
        error = rms(current - baseline, geometry.volume_weights, gradient)
        norm = rms(baseline, geometry.volume_weights, gradient)
        limit = protocol['absolute_rms_tolerance_code_units'] + protocol[f'relative_{name}_l2_tolerance'] * norm
        checks = error <= limit
        passed &= bool(checks.all())
        metrics[name] = {'absolute_rms_errors': error.tolist(), 'reference_rms': norm.tolist(),
                         'tolerances': limit.tolist(), 'passed': checks.tolist()}
    if not passed:
        raise ValueError(f'physical parity failed: {metrics}')
    return {'exact_reference_arrays': arrays, 'passed': passed, 'metrics': metrics,
            'snapshot_kind': 'initial' if initial else 'evolved'}
