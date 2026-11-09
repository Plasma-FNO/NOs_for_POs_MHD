import h5py
import numpy as np
import pytest

from mastu_opsplit.surface import restart_boundary_quadrature
from test_restart import rectangle


def test_curved_boundary_normals_and_axisymmetric_divergence_theorem(tmp_path):
    path = tmp_path/'curved-boundary.h5'
    rectangle(path)
    with h5py.File(path, 'r+') as stream:
        xy = stream['x'][()]
        radius = xy[0, 0, 0]
        xy[1, 1, 0] = .2*(3-2*radius)
        stream['x'][...] = xy
        values = stream['values'][()]
        values[4, 1, 0] = 1+xy[1, 1, 0]
        stream['values'][...] = values
        stream['boundary'][...] = 1
    result = restart_boundary_quadrature(path, order=8)
    assert result['known_boundary_edges'].all() and result['unknown_edge_count'] == 0
    np.testing.assert_allclose(np.linalg.norm(result['normals'], axis=-1), 1., atol=1e-14)
    np.testing.assert_allclose(result['values'][..., 4], 2+result['points'].sum(axis=-1), atol=1e-13)
    np.testing.assert_allclose(result['gradients'][..., 4, :], 1., atol=1e-13)
    flux = np.sum(result['surface_weights']*np.sum(result['points']*result['normals'], axis=-1))
    assert flux == pytest.approx(9*np.pi, rel=1e-13)
    # A constant axial field has zero net flux, checking both orientations.
    assert np.sum(result['surface_weights']*result['normals'][..., 1]) == pytest.approx(0, abs=1e-13)


def test_unlabelled_exterior_is_reported(tmp_path):
    path = tmp_path/'unlabelled.h5'
    rectangle(path)
    result = restart_boundary_quadrature(path)
    assert result['unknown_edge_count'] == 4 and not result['known_boundary_edges'].any()
    assert result['unknown_surface_area_m2'] == pytest.approx(12*np.pi)
