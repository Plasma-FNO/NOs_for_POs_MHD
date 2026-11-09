"""Diagnostic interpolation of unrefined, axisymmetric model303 restart files.

Basis conventions follow pinned JOREK elements/mod_basisfunctions.f90 (LGPL-3.0-or-later).
This reader does not export native residuals, matrix constraints, or patch identities.
"""
import h5py
import numpy as np

from .common import sha256_file
from .numerical import NativeGeometry, evaluate_native


def cubic_basis(s, t):
    """Return JOREK's [point, vertex, value/s/t/st] basis and local derivatives."""
    s, t = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(t, dtype=float))
    if not np.isfinite(s).all() or not np.isfinite(t).all() or ((s < 0) | (s > 1) | (t < 0) | (t > 1)).any():
        raise ValueError('element coordinates must be finite and in [0,1]')

    def one(x):
        values = np.stack(((1-x)**2*(1+2*x), 3*(1-x)**2*x,
                           x*x*(3-2*x), 3*(1-x)*x*x), axis=-1)
        gradients = np.stack((6*x*(x-1), 3*(x-1)*(3*x-1),
                              -6*x*(x-1), -3*x*(3*x-2)), axis=-1)
        return values.reshape(*x.shape, 2, 2), gradients.reshape(*x.shape, 2, 2)

    hs, ds = one(s)
    ht, dt = one(t)
    result = []
    for a, b in ((hs, ht), (ds, ht), (hs, dt)):
        result.append(np.stack([np.stack((a[..., i, 0]*b[..., j, 0],
                                          a[..., i, 1]*b[..., j, 0],
                                          a[..., i, 0]*b[..., j, 1],
                                          a[..., i, 1]*b[..., j, 1]), axis=-1)
                                for i, j in ((0, 0), (1, 0), (1, 1), (0, 1))], axis=-2))
    return tuple(result)


def read_restart(path, *, quadrature_order=4):
    """Return diagnostic geometry, canonical native coefficients, and native time.

    Native shared node/derivative IDs are checked and condensed. Unknown patch
    IDs are -1; physical constraint/boundary rows still need the native backend.
    """
    if type(quadrature_order) is not int or not 2 <= quadrature_order <= 16:
        raise ValueError('diagnostic quadrature order must be an integer in [2,16]')
    with h5py.File(path) as f:
        def scalar(name):
            return np.asarray(f[name]).item()
        expected = {'jorek_model': 303, 'n_tor': 1, 'n_coord_tor': 1, 'n_order': 3,
                    'n_degrees': 4, 'n_var': 7, 'n_dim': 2}
        if any(scalar(name) != value for name, value in expected.items()):
            raise ValueError('reader requires axisymmetric bicubic model303')
        n, e = scalar('n_nodes'), scalar('n_elements')
        if n <= 0 or e <= 0:
            raise ValueError('empty native mesh')
        if np.any(f['n_sons'][()] != 0) or np.any(f['constrained'][()] != b'F'):
            raise ValueError('refined or constrained meshes require the native elimination map')
        vertex = f['vertex'][()].T
        sizes = f['size'][()].transpose(2, 1, 0)
        coordinates = f['x'][()]
        values = f['values'][()]
        indices = f['index'][()].T
        boundary = f['boundary'][()]
        time_norm, code_time = scalar('sqrt_mu0_rho0'), scalar('t_now')
        steps = scalar('index_now')
    shapes = ((vertex, (e, 4)), (sizes, (e, 4, 4)), (coordinates, (2, 4, 1, n)),
              (values, (7, 4, 1, n)), (boundary, (n,)), (indices, (n, 4)))
    if any(array.shape != shape or not np.isfinite(array).all() for array, shape in shapes):
        raise ValueError('invalid native restart arrays')
    if not np.issubdtype(vertex.dtype, np.integer) or vertex.min() < 1 or vertex.max() > n:
        raise ValueError('invalid native vertex IDs')
    if not np.isfinite([time_norm, code_time]).all() or time_norm <= 0 or code_time < 0:
        raise ValueError('invalid native time normalization')
    vertex = vertex - 1
    expanded = values[:, :, 0, :].transpose(2, 1, 0).reshape(n*4, 7)
    if not np.issubdtype(indices.dtype, np.integer) or indices.min() < 1:
        raise ValueError('native DOF indices must be positive integers')
    dof_ids, first, inverse = np.unique(indices.reshape(-1), return_index=True, return_inverse=True)
    if not np.array_equal(dof_ids, np.arange(1, len(dof_ids)+1)):
        raise ValueError('native DOF indices must be contiguous')
    coefficients = expanded[first]
    tolerance = 64*np.finfo(float).eps
    if not np.allclose(expanded, coefficients[inverse], rtol=tolerance, atol=tolerance):
        raise ValueError('native aliases have inconsistent field coefficients')
    expanded_boundary = np.repeat(boundary, 4)
    if not np.array_equal(expanded_boundary, expanded_boundary[first][inverse]):
        raise ValueError('native aliases have conflicting boundary labels')
    xy = coordinates[:, :, 0, :].transpose(2, 1, 0)
    points, weights = np.polynomial.legendre.leggauss(quadrature_order)
    points, weights = (points+1)/2, weights/2
    s, t = np.meshgrid(points, points, indexing='ij')
    h, hs, ht = cubic_basis(s.ravel(), t.ravel())
    basis, bs, bt = [np.einsum('qvd,evd->eqvd', item, sizes).reshape(e, -1, 16)
                     for item in (h, hs, ht)]
    local_xy = xy[vertex].reshape(e, 16, 2)
    rz, rs, rt = [np.einsum('eql,elc->eqc', item, local_xy) for item in (basis, bs, bt)]
    jac = rs[..., 0]*rt[..., 1] - rt[..., 0]*rs[..., 1]
    scale = np.linalg.norm(rs, axis=-1)*np.linalg.norm(rt, axis=-1)
    if (np.abs(jac) <= 64*np.finfo(float).eps*scale).any() or ((jac.min(axis=1) < 0) & (jac.max(axis=1) > 0)).any():
        raise ValueError('singular or folded native geometry')
    grad = np.stack(((bs*rt[..., 1, None]-bt*rs[..., 1, None])/jac[..., None],
                     (bt*rs[..., 0, None]-bs*rt[..., 0, None])/jac[..., None]), axis=-1)
    # The Bezier control hull bounds the full curved element independently of quadrature order.
    jets = xy[vertex]*sizes[..., None]
    control = np.stack((jets[:, :, 0], jets[:, :, 0]+jets[:, :, 1],
                        jets[:, :, 0]+jets[:, :, 2], jets.sum(axis=2)), axis=2)
    bounds = np.stack((control.min(axis=(0, 1, 2)), control.max(axis=(0, 1, 2))), axis=-1)
    dof_map = inverse[(vertex[..., None]*4+np.arange(4)).reshape(e, 16)]
    ids = np.arange(e*quadrature_order**2)
    geometry = NativeGeometry(dof_map, basis, grad, rz,
                              np.abs(jac)*np.outer(weights, weights).ravel(), bounds,
                              dof_ids, np.arange(1, e+1), np.full(e, -1),
                              expanded_boundary[first], ids, ids.copy()).validate()
    return geometry, coefficients, {'time_seconds': code_time*time_norm, 'steps': steps,
                                    'quadrature_order': quadrature_order,
                                    'mapping_status': 'diagnostic_restart_only',
                                    'expanded_nodal_dofs': n*4, 'canonical_native_dofs': len(dof_ids),
                                    'native_dof_aliases_verified': True,
                                    'minimum_signed_jacobian_m2': float(jac.min()),
                                    'negative_jacobian_samples': int((jac < 0).sum()),
                                    'patch_ids_available': False, 'native_constraints_available': False}


def diagnose_restart(path, reference, *, quadrature_order=4):
    """Measure full-wall-domain field drift; sources and fluxes are not balance-checked."""
    geometry, state, metadata = read_restart(path, quadrature_order=quadrature_order)
    reference_geometry, reference_state, _ = read_restart(reference, quadrature_order=quadrature_order)
    for name in ('dof_map', 'basis', 'basis_grad', 'quadrature_rz', 'poloidal_weights', 'boundary_labels'):
        if not np.array_equal(getattr(geometry, name), getattr(reference_geometry, name)):
            raise ValueError('restart comparison requires the same physical mesh')
    current, initial = evaluate_native(state, geometry), evaluate_native(reference_state, geometry)
    weights = geometry.volume_weights

    def integral(value):
        return float(np.sum(weights*value))

    def rms(value):
        return float(np.sqrt(integral(value*value)/weights.sum()))

    fields = {}
    for index, name in enumerate(('psi', 'u', 'zj', 'w', 'rho', 'T', 'Vpar')):
        error, norm = rms(current[..., index]-initial[..., index]), rms(initial[..., index])
        fields[name] = {'min': float(current[..., index].min()),
                        'max': float(current[..., index].max()),
                        'absolute_rms_drift': error, 'relative_l2_drift': error/norm if norm else None}
    moments = {}
    for name, first, last in (('density', initial[..., 4], current[..., 4]),
                              ('pressure', initial[..., 4]*initial[..., 5], current[..., 4]*current[..., 5])):
        before, after = integral(first), integral(last)
        moments[name] = {'initial': before, 'final': after,
                         'relative_change': (after-before)/before if before else None}
    return {'synthetic': False, 'scope': 'native restart diagnostics; scientific gates remain open',
            'restart_sha256': sha256_file(path), 'reference_sha256': sha256_file(reference),
            'diagnostic_source_sha256': sha256_file(__file__), 'metadata': metadata,
            'domain': 'full wall-extended mesh, including exterior and private-flux regions',
            'elements': len(geometry.element_ids), 'quadrature_points': len(geometry.query_ids),
            'volume_m3': float(weights.sum()), 'fields': fields, 'moments_code_units': moments,
            'positive_at_sampled_quadrature': bool((current[..., 4:6] > 0).all()),
            'balance_checked': False, 'constraints_checked': False}
