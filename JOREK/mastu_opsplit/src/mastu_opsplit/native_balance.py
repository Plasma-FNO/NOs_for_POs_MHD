"""Independent native linear-system balance; physical energy closure is separate."""
from pathlib import Path

import numpy as np

from .common import sha256_file
from .native_export import unique_element_records
from .restart import read_restart


def linearized_volume_balance(folder, before, after, export_validation, *, rtol=1e-8, atol=1e-12):
    """Verify A(q) delta_q = RHS(q) on rows away from boundary/axis elements.

    Uses the native exported step matrix, including the actual temporal
    coupling. It never substitutes a diagonal mass or a finite-time field rate.
    """
    if export_validation.get('status') != 'native_selected_export_verified' or export_validation['restart_sha256'] != sha256_file(before):
        raise ValueError('verified same-state native export required')
    geometry, state, metadata = read_restart(before)
    other_geometry, next_state, next_metadata = read_restart(after)
    for name in ['dof_map', 'basis', 'basis_grad', 'quadrature_rz', 'poloidal_weights', 'boundary_labels']:
        if not np.array_equal(getattr(geometry, name), getattr(other_geometry, name)):
            raise ValueError('native solve comparison geometry changed')
    if next_metadata['steps'] != metadata['steps']+1 or next_metadata['time_seconds'] <= metadata['time_seconds']:
        raise ValueError('exactly one native timestep required')
    direction = next_state-state
    action, rhs = np.zeros_like(state), np.zeros_like(state)
    excluded = geometry.boundary_labels != 0
    seen = set()
    files = sorted(Path(folder).glob('opsplit-native-rank-*.bin'))
    if set(map(str, files)) != set(export_validation['export_files']):
        raise ValueError('verified native export file set changed')
    for path in files:
        expected = export_validation['export_files'].get(str(path))
        if expected is None or expected['sha256'] != sha256_file(path):
            raise ValueError('verified native export file identity changed')
    for native in unique_element_records(files):
        e = native['element_id']-1
        if e < 0 or e >= len(geometry.element_ids) or e in seen:
            raise ValueError('invalid native balance element identity')
        seen.add(e)
        ids = geometry.dof_map[e]
        if not np.array_equal(native['dof_ids'], ids+1):
            raise ValueError('native matrix DOF order changed')
        local = 2*np.pi*(native['step_matrix'] @ direction[ids].ravel()).reshape(16, 7)
        np.add.at(action, ids, local)
        np.add.at(rhs, ids, 2*np.pi*native['rhs'].reshape(16, 7))
        if native['axis'].any() or native['boundary'].any():
            excluded[ids] = True
    if seen != set(range(len(geometry.element_ids))) or excluded.all():
        raise ValueError('incomplete native coverage or no untouched interior rows')
    metrics = {}
    for i, name in enumerate(['psi', 'u', 'zj', 'w', 'rho', 'T', 'Vpar']):
        a, b = action[~excluded, i], rhs[~excluded, i]
        error = float(np.linalg.norm(a-b))
        scale = float(np.linalg.norm(a)+np.linalg.norm(b))
        metrics[name] = {'absolute_l2_residual': error, 'component_l2_scale': scale,
                         'component_relative_residual': error/scale if scale else None,
                         'target_met': bool(error <= atol+rtol*scale)}
    passed = all(v['target_met'] for v in metrics.values())
    return {'status': 'interior_linearized_balance_verified' if passed else 'interior_linearized_balance_failed',
            'before_sha256': sha256_file(before), 'after_sha256': sha256_file(after),
            'time_interval_s': next_metadata['time_seconds']-metadata['time_seconds'],
            'included_native_dofs_per_equation': int((~excluded).sum()), 'excluded_native_dofs_per_equation': int(excluded.sum()),
            'metrics': metrics, 'rtol': rtol, 'atol': atol, 'scientific_gates_passed': [],
            'limitations': ['Boundary elements and every axis-touching element are excluded conservatively.',
                            'An algebraic linear-system balance does not establish physical energy/source/flux balance.',
                            'This check does not supply an autonomous retained evaluator or adjoint.']}
