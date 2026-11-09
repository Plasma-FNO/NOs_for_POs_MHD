"""Diagnostic physical queries of native bicubic restarts, with explicit coverage."""
import h5py
import numpy as np
from scipy.spatial import cKDTree

from .restart import cubic_basis, read_restart


def sample_restart(path, points, *, chunk_size=2048):
    """Invert curved elements and evaluate native fields at physical R,Z points.

    Outside or unresolved points have mask=False, element_id=-1 and NaN values.
    This is FE evaluation, not a nearest-node approximation or native solver.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError('queries require nonempty finite [point, R/Z] coordinates')
    if (points[:, 0] <= 0).any() or type(chunk_size) is not int or not 1 <= chunk_size <= 65536:
        raise ValueError('positive radius and integer chunk size in [1,65536] required')
    geometry, state, _ = read_restart(path, quadrature_order=4)
    with h5py.File(path) as stream:
        xy = stream['x'][()][:, :, 0, :].transpose(2, 1, 0)
        vertex = stream['vertex'][()].T-1
        sizes = stream['size'][()].transpose(2, 1, 0)
    jets = xy[vertex]*sizes[..., None]
    control = np.stack((jets[:, :, 0], jets[:, :, 0]+jets[:, :, 1],
                        jets[:, :, 0]+jets[:, :, 2], jets.sum(axis=2)), axis=2)
    lower, upper = control.min(axis=(1, 2)), control.max(axis=(1, 2))
    tree = cKDTree((lower+upper)/2)
    max_radius = float(np.linalg.norm((upper-lower)/2, axis=-1).max())
    tolerance = 1e-10
    local_xy = jets.reshape(-1, 16, 2)
    result = np.full((len(points), state.shape[1]), np.nan)
    elements = np.full(len(points), -1, dtype=int)
    local = np.full((len(points), 2), np.nan)
    errors = np.full(len(points), np.nan)
    fallback_points = 0

    def try_candidates(query_ids, element_ids):
        if not len(query_ids):
            return
        eligible = ((elements[query_ids] == -1)
                    & (points[query_ids] >= lower[element_ids]-tolerance).all(axis=1)
                    & (points[query_ids] <= upper[element_ids]+tolerance).all(axis=1))
        query_ids, element_ids = query_ids[eligible], element_ids[eligible]
        if not len(query_ids):
            return
        st = np.full((len(query_ids), 2), .5)
        coordinates = local_xy[element_ids]
        # Projected Newton may fail on a difficult element; coverage remains explicit.
        for _ in range(30):
            h, hs, ht = [basis.reshape(-1, 16) for basis in cubic_basis(st[:, 0], st[:, 1])]
            position, rs, rt = [np.einsum('kl,klc->kc', basis, coordinates) for basis in (h, hs, ht)]
            residual = position-points[query_ids]
            error = np.linalg.norm(residual, axis=1)
            active = error > tolerance
            if not active.any():
                break
            determinant = rs[:, 0]*rt[:, 1]-rt[:, 0]*rs[:, 1]
            stable = np.abs(determinant) > 64*np.finfo(float).eps*np.linalg.norm(rs, axis=1)*np.linalg.norm(rt, axis=1)
            active &= stable
            if not active.any():
                break
            step = np.zeros_like(st)
            step[active, 0] = (residual[active, 0]*rt[active, 1]-residual[active, 1]*rt[active, 0])/determinant[active]
            step[active, 1] = (rs[active, 0]*residual[active, 1]-rs[active, 1]*residual[active, 0])/determinant[active]
            updated = np.clip(st-step, 0, 1)
            if np.array_equal(updated, st):
                break
            st = updated
        h = cubic_basis(st[:, 0], st[:, 1])[0].reshape(-1, 16)
        error = np.linalg.norm(np.einsum('kl,klc->kc', h, coordinates)-points[query_ids], axis=1)
        accepted = np.flatnonzero(error <= tolerance)
        if not len(accepted):
            return
        # Shared-edge queries can have two valid elements; use the first exact map.
        _, first = np.unique(query_ids[accepted], return_index=True)
        accepted = accepted[first]
        q, e = query_ids[accepted], element_ids[accepted]
        basis = h[accepted]*sizes[e].reshape(-1, 16)
        result[q] = np.einsum('kl,klc->kc', basis, state[geometry.dof_map[e]])
        elements[q], local[q], errors[q] = e+1, st[accepted], error[accepted]

    for start in range(0, len(points), chunk_size):
        ids = np.arange(start, min(start+chunk_size, len(points)))
        nearest = tree.query(points[ids], k=min(32, len(lower)))[1].reshape(len(ids), -1)
        try_candidates(np.repeat(ids, nearest.shape[1]), nearest.ravel())
        missing = ids[elements[ids] == -1]
        fallback_points += len(missing)
        if len(missing):
            candidates = tree.query_ball_point(points[missing], max_radius+tolerance)
            counts = np.array([len(candidate) for candidate in candidates])
            if counts.sum():
                try_candidates(np.repeat(missing, counts), np.concatenate(candidates).astype(int))
    return {'values': result, 'mask': elements != -1, 'element_ids': elements,
            'local_coordinates': local, 'coordinate_error_m': errors,
            'fallback_points': fallback_points}
