"""Read diagnostic model303 element exports; no production mapping is implied."""
from pathlib import Path
import hashlib

import h5py
import numpy as np

from .common import sha256_file
from .numerical import evaluate_native, evaluate_native_grad, model303_advection_blocks
from .restart import read_restart


def element_records(path):
    """Stream bounded records; reject truncation, unsupported layouts and nonfinite data."""
    with Path(path).open('rb') as stream:
        header = np.fromfile(stream, dtype='<i4', count=8)
        if header.shape != (8,) or tuple(header[[0, 2, 3, 4, 5, 6]]) != (3031601, 7, 16, 4, 8, 1):
            raise ValueError('unsupported native export header')
        q, rank = int(header[1]), int(header[7])
        if q not in (4, 8) or rank < 0:
            raise ValueError('unsupported native quadrature or rank')
        shapes = [('points', (2, q, q)), ('weights', (q, q)), ('values', (7, q, q)),
                  ('gradient', (2, 7, q, q)), ('basis', (16, q, q)),
                  ('basis_gradient', (2, 16, q, q)), ('selected', (2, q, q)),
                  ('selected_weak', (2, 16)), ('rhs', (112,)), ('step_matrix', (112, 112))]
        count = 6 + sum(int(np.prod(shape)) for _, shape in shapes)
        while True:
            identity_bytes = stream.read(33*4)
            if not identity_bytes:
                break
            if len(identity_bytes) != 33*4:
                raise ValueError('truncated native element identity')
            identity = np.frombuffer(identity_bytes, dtype='<i4')
            data = np.fromfile(stream, dtype='<f8', count=count)
            if len(data) != count or not np.isfinite(data).all():
                raise ValueError('truncated or nonfinite native element data')
            record = {'element_id': int(identity[0]), 'record_sha256': hashlib.sha256(identity_bytes + data.tobytes()).hexdigest(),
                      'vertices': identity[1:5],
                      'dof_ids': identity[5:21], 'boundary': identity[21:25],
                      'axis': identity[25:29], 'constrained': identity[29:33],
                      'time_code': data[0], 'tstep': data[1], 'theta': data[2],
                      'zeta': data[3], 'gamma': data[4], 'toroidal_basis': data[5], 'rank': rank}
            offset = 6
            for name, shape in shapes:
                size = int(np.prod(shape))
                record[name] = data[offset:offset+size].reshape(shape, order='F')
                offset += size
            yield record



def unique_element_records(paths, *, audit=None):
    """Count a physical element once, accepting only bit-identical cross-rank copies.

    Native subdomain assembly visits halo elements on several ranks. A repeated
    identity on one rank, or any differing record payload, is still an error.
    """
    seen, ranks, halo_elements = {}, set(), set()
    copies = 0
    for path in paths:
        for native in element_records(path):
            identity, rank = native['element_id'], native['rank']
            ranks.add(rank)
            if identity in seen:
                digest, owners = seen[identity]
                if rank in owners or native['record_sha256'] != digest:
                    raise ValueError('duplicate native element differs or repeats on the same rank')
                owners.add(rank)
                halo_elements.add(identity)
                copies += 1
                continue
            seen[identity] = (native['record_sha256'], {rank})
            yield native
    if audit is not None:
        audit.update(identical_halo_copies=copies, halo_elements=len(halo_elements), ranks=sorted(ranks))


def validate_export(folder, restart, *, rtol=1e-8, atol=1e-12):
    """Compare all native elements against an independently evaluated saved state.

    Native RHS and matrices precede axis transforms and modified boundary rows.
    They remain diagnostic objects, distinct from an accepted training exchange.
    """
    files = sorted(Path(folder).glob('opsplit-native-rank-*.bin'))
    if not files:
        raise ValueError('native export files are missing')
    first = next(element_records(files[0]), None)
    if first is None:
        raise ValueError('empty native export')
    q = first['weights'].shape[0]
    geometry, state, metadata = read_restart(restart, quadrature_order=q)
    expected_values = evaluate_native(state, geometry)
    expected_gradient = evaluate_native_grad(state, geometry)
    with h5py.File(restart) as stream:
        vertices = stream['vertex'][()].T
        boundary = stream['boundary'][()]
        norm = float(stream['sqrt_mu0_rho0'][()].item())
    metrics, seen = {}, set()
    selected_global = np.zeros((len(geometry.dof_ids), 2))
    total_global = np.zeros_like(selected_global)

    def compare(name, actual, expected):
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError('invalid native comparison: ' + name)
        error = actual - expected
        entry = metrics.setdefault(name, {'error2': 0., 'reference2': 0., 'maximum_absolute_error': 0.})
        entry['error2'] += float(np.sum(error*error))
        entry['reference2'] += float(np.sum(expected*expected))
        entry['maximum_absolute_error'] = max(entry['maximum_absolute_error'], float(np.max(np.abs(error))))

    overlap = {}
    for native in unique_element_records(files, audit=overlap):
        e = native['element_id'] - 1
        if e < 0 or e >= len(vertices) or e in seen:
            raise ValueError('duplicate or invalid native element identity')
        seen.add(e)
        if not np.array_equal(native['vertices'], vertices[e]) or not np.array_equal(native['dof_ids'], geometry.dof_map[e]+1):
            raise ValueError('native element connectivity or derivative DOF mapping differs')
        if not np.array_equal(native['boundary'], boundary[vertices[e]-1]) or native['constrained'].any():
            raise ValueError('native boundary identity or elimination mapping differs')
        if abs(native['time_code']*norm - metadata['time_seconds']) > 1e-15 or native['tstep'] <= 0 or native['zeta'] != 0 or native['toroidal_basis'] != 1:
            raise ValueError('native time/history/normalization differs')
        values = native['values'].transpose(1, 2, 0).reshape(-1, 7)
        gradient = native['gradient'].transpose(2, 3, 1, 0).reshape(-1, 7, 2)
        points = native['points'].transpose(1, 2, 0).reshape(-1, 2)
        basis = native['basis'].transpose(1, 2, 0).reshape(-1, 16)
        basis_gradient = native['basis_gradient'].transpose(2, 3, 1, 0).reshape(-1, 16, 2)
        weights = native['weights'].reshape(-1)
        if (weights <= 0).any() or (values[:, 4:6] <= 0).any():
            raise ValueError('nonpositive native weights/density/temperature')
        for name, actual, expected in [
            ('coordinates', points, geometry.quadrature_rz[e]),
            ('poloidal_weights', weights, geometry.poloidal_weights[e]),
            ('basis', basis, geometry.basis[e]), ('basis_gradient', basis_gradient, geometry.basis_grad[e]),
            ('state', values, expected_values[e]), ('state_gradient', gradient, expected_gradient[e])]:
            compare(name, actual, expected)
        selected = native['selected'].transpose(1, 2, 0).reshape(-1, 2)
        independent = model303_advection_blocks(expected_values[e], expected_gradient[e], geometry.quadrature_rz[e, :, 0], native['gamma'])
        compare('strong_selected_terms', selected, independent)
        physical_weights = 2*np.pi*points[:, 0]*weights
        weak = 2*np.pi*native['selected_weak'].T
        compare('independent_selected_weak_assembly', weak, np.einsum('ql,qc,q->lc', basis, independent, physical_weights))
        np.add.at(selected_global, geometry.dof_map[e], weak)
        # The exporter asserts zero history and TG before this conversion.
        total = 2*np.pi*native['rhs'].reshape(16, 7)[:, [4, 5]]/native['tstep']
        np.add.at(total_global, geometry.dof_map[e], total)
    if seen != set(range(len(vertices))):
        raise ValueError('native export does not cover every physical element')
    for name, metric in metrics.items():
        error, reference = np.sqrt(metric.pop('error2')), np.sqrt(metric.pop('reference2'))
        metric.update(absolute_l2_error=float(error), reference_l2_norm=float(reference),
                      relative_l2_error=float(error/reference) if reference else None,
                      passed=bool(error <= atol + rtol*reference))
        if not metric['passed']:
            raise ValueError(f'native export mismatch {name}: {metric}')
    interior = geometry.boundary_labels == 0
    strengths = {}
    for i, name in enumerate(['density', 'pressure']):
        a = float(np.linalg.norm(selected_global[interior, i]))
        b = float(np.linalg.norm(total_global[interior, i]))
        strengths[name] = {'selected_weak_l2': a, 'total_volume_weak_l2': b,
                           'ratio': a/b if b else None}
    return {'status': 'native_selected_export_verified', 'restart_sha256': sha256_file(restart),
            'export_files': {str(p): {'sha256': sha256_file(p), 'bytes': p.stat().st_size} for p in files},
            'elements_verified': len(seen), 'ranks': overlap['ranks'], 'halo_overlap': overlap, 'quadrature_order': q,
            'metrics': metrics, 'term_strength_coefficient_norms': strengths,
            'mapping_status': 'provisional', 'scientific_gates_passed': [],
            'limitations': ['First pre-update frame only; algebraic and differential rows remain distinct.',
                            'Native element RHS/matrix precede axis transforms and modified boundary rows.',
                            'Term strengths use coefficient norms, not physical L2 field norms.',
                            'Autonomous retained evaluation, temporal/constraint solves and adjoint remain unverified.']}
