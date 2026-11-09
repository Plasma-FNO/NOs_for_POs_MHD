import h5py
import numpy as np
import pytest

from mastu_opsplit.queries import sample_restart
from test_restart import rectangle


def test_exact_curved_native_queries_and_explicit_exterior(tmp_path):
    path = tmp_path/'curved.h5'
    rectangle(path)
    # R=1+s, Z=t+.2*s*(1-s); rho=2+R+Z, represented by native derivative DOFs.
    with h5py.File(path, 'r+') as stream:
        xy = stream['x'][()]
        radius = xy[0, 0, 0]
        xy[1, 1, 0] = .2*(3-2*radius)
        stream['x'][...] = xy
        values = stream['values'][()]
        values[4, 1, 0] = 1+xy[1, 1, 0]
        stream['values'][...] = values
    rng = np.random.default_rng(54)
    st = np.concatenate((rng.random((100, 2)), [[0, 0], [1, 1], [0, .5], [1, .5]]))
    points = np.stack((1+st[:, 0], st[:, 1]+.2*st[:, 0]*(1-st[:, 0])), axis=1)
    points = np.concatenate((points, [[.9, .4], [2.1, .4], [1.5, -.1], [1.5, 1.2]]))
    query = sample_restart(path, points, chunk_size=13)
    assert query['mask'][:-4].all() and not query['mask'][-4:].any()
    np.testing.assert_allclose(query['values'][:-4, 4], 2+points[:-4].sum(axis=1), atol=2e-10)
    np.testing.assert_allclose(query['local_coordinates'][:-4], st, atol=2e-10)
    assert np.isnan(query['values'][-4:]).all()
    assert (query['element_ids'][-4:] == -1).all()
    assert np.nanmax(query['coordinate_error_m']) <= 1e-10
    other = sample_restart(path, points, chunk_size=108)
    np.testing.assert_allclose(other['values'], query['values'], equal_nan=True)


def test_elongated_element_found_beyond_nearest_centers(tmp_path):
    path = tmp_path/'elongated.h5'
    rectangle(path)
    boxes = [(1., 0., 1., 100.)]+[(1.49, .2+.02*i, .02, .01) for i in range(40)]
    count = len(boxes)*4
    xy = np.zeros((2, 4, 1, count))
    values = np.zeros((7, 4, 1, count))
    for i, (r, z, width, height) in enumerate(boxes):
        nodes = slice(i*4, i*4+4)
        xy[:, 0, 0, nodes] = np.array([[r, z], [r+width, z],
                                      [r+width, z+height], [r, z+height]]).T
        xy[0, 1, 0, nodes], xy[1, 2, 0, nodes] = width, height
        values[4, 0, 0, nodes] = 2+xy[:, 0, 0, nodes].sum(axis=0)
        values[4, 1, 0, nodes], values[4, 2, 0, nodes] = width, height
        values[5, 0, 0, nodes] = 1
    with h5py.File(path, 'r+') as stream:
        sizes = stream['size'][()]
        arrays = {'x': xy, 'values': values, 'size': np.repeat(sizes, len(boxes), axis=2),
                  'vertex': np.arange(1, count+1).reshape(-1, 4).T,
                  'index': np.arange(1, count*4+1).reshape(count, 4).T,
                  'boundary': np.zeros(count, dtype=int), 'constrained': np.full(count, b'F'),
                  'n_sons': np.zeros(len(boxes), dtype=int), 'n_nodes': count, 'n_elements': len(boxes)}
        for key, value in arrays.items():
            del stream[key]
            stream[key] = value
    result = sample_restart(path, [[1.5, .1]])
    assert result['mask'].all() and result['fallback_points'] == 1
    assert result['element_ids'][0] == 1
    np.testing.assert_allclose(result['values'][0, 4], 3.6, atol=1e-12)


@pytest.mark.parametrize('points', [[], [[1, float('nan')]], [[0, 1]], [[1, 2, 3]]])
def test_invalid_physical_queries_are_rejected(tmp_path, points):
    with pytest.raises(ValueError):
        sample_restart(tmp_path/'unused.h5', points)
