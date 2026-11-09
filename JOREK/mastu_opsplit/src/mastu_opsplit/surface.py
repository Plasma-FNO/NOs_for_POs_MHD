"""Provisional native boundary quadrature; no modified solver rows are inferred."""
import h5py
import numpy as np

from .restart import cubic_basis, read_restart


def restart_boundary_quadrature(path, *, order=4):
    """Evaluate exterior native edges, including explicit unlabelled-edge coverage.

    Canonical endpoint IDs identify shared edges. Degenerate axis edges carry
    zero area; boundary and axis elimination still require the native backend.
    """
    if type(order) is not int or not 2 <= order <= 16:
        raise ValueError('boundary quadrature order must be an integer in [2,16]')
    geometry, state, _ = read_restart(path, quadrature_order=2)
    with h5py.File(path) as stream:
        vertex = stream['vertex'][()].T-1
        xy = stream['x'][()][:, :, 0, :].transpose(2, 1, 0)
        size = stream['size'][()].transpose(2, 1, 0)
        labels = stream['boundary'][()]
        ids = stream['index'][0]-1
    pairs = np.stack((vertex, np.roll(vertex, -1, axis=1)), axis=-1)
    keys = np.sort(ids[pairs].reshape(-1, 2), axis=1)
    # Axis endpoints have one value ID and zero physical edge measure.
    noncollapsed = keys[:, 0] != keys[:, 1]
    collapsed = np.flatnonzero(~noncollapsed)
    if len(collapsed):
        e, side = collapsed//4, collapsed % 4
        g = np.linspace(0, 1, 4)
        s = np.stack((g, np.ones(4), 1-g, np.zeros(4)))[side]
        t = np.stack((np.zeros(4), g, np.ones(4), 1-g))[side]
        basis = cubic_basis(s, t)[0].reshape(-1, 4, 16)*size[e].reshape(-1, 1, 16)
        positions = np.einsum('eql,elc->eqc', basis, xy[vertex[e]].reshape(-1, 16, 2))
        if np.max(np.linalg.norm(positions-positions[:, :1], axis=-1)) > 1e-12:
            raise ValueError('coincident endpoint IDs have a noncollapsed physical edge')
    _, inverse, counts = np.unique(keys[noncollapsed], axis=0, return_inverse=True, return_counts=True)
    if (counts > 2).any():
        raise ValueError('nonmanifold canonical edges require the native boundary map')
    boundary = np.flatnonzero(noncollapsed)[counts[inverse] == 1]
    if not len(boundary):
        raise ValueError('no exterior edges located')
    elements, sides = boundary//4, boundary % 4
    points, weights = np.polynomial.legendre.leggauss(order)
    points, weights = (points+1)/2, weights/2
    s = np.stack((points, np.ones(order), 1-points, np.zeros(order)))[sides]
    t = np.stack((np.zeros(order), points, np.ones(order), 1-points))[sides]
    h, hs, ht = [basis.reshape(len(boundary), order, 16)*size[elements].reshape(-1, 1, 16)
                 for basis in cubic_basis(s, t)]
    local_xy = xy[vertex[elements]].reshape(-1, 16, 2)
    rz, rs, rt = [np.einsum('eql,elc->eqc', basis, local_xy) for basis in (h, hs, ht)]
    jacobian = rs[..., 0]*rt[..., 1]-rt[..., 0]*rs[..., 1]
    scale = np.linalg.norm(rs, axis=-1)*np.linalg.norm(rt, axis=-1)
    if (np.abs(jacobian) <= 64*np.finfo(float).eps*scale).any():
        raise ValueError('singular exterior geometry requires native boundary treatment')
    tangents = np.where((sides % 2 == 0)[:, None, None], rs, rt)
    tangents *= np.where(sides < 2, 1., -1.)[:, None, None]
    lengths = np.linalg.norm(tangents, axis=-1)
    if (lengths <= 0).any() or (rz[..., 0] <= 0).any():
        raise ValueError('invalid exterior edge measure')
    normals = np.stack((tangents[..., 1], -tangents[..., 0]), axis=-1)
    normals *= np.sign(jacobian)[..., None]/lengths[..., None]
    gradient_basis = np.stack(((hs*rt[..., 1, None]-ht*rs[..., 1, None])/jacobian[..., None],
                               (ht*rs[..., 0, None]-hs*rt[..., 0, None])/jacobian[..., None]), axis=-1)
    local_state = state[geometry.dof_map[elements]]
    values = np.einsum('eql,elc->eqc', h, local_state)
    gradients = np.einsum('eqlr,elc->eqcr', gradient_basis, local_state)
    edge_labels = labels[pairs.reshape(-1, 2)[boundary]]
    known = (edge_labels != 0).all(axis=1)
    surface_weights = 2*np.pi*rz[..., 0]*lengths*weights
    return {'points': rz, 'values': values, 'gradients': gradients, 'normals': normals,
            'surface_weights': surface_weights, 'edge_labels': edge_labels,
            'known_boundary_edges': known, 'element_ids': elements+1, 'local_sides': sides,
            'collapsed_axis_edges': int((~noncollapsed).sum()),
            'unknown_edge_count': int((~known).sum()),
            'unknown_surface_area_m2': float(surface_weights[~known].sum())}
