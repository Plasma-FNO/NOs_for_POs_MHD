"""Independent polynomial roots and gradients, including moving X-point height."""
import h5py
import numpy as np
import pytest
import torch

from mastu_opsplit.native_critical import NativeCritical
from mastu_opsplit.restart import read_restart


def fixture(tmp_path, source_saddle=False):
    path = tmp_path/'reference.h5'
    # Three distinct patches: one elliptic minimum and two hyperbolic saddles.
    centers = [(1., 0.), (.5, -1.), (.5, 1.)]
    fluxes = [-1., -.1, -.2]
    if source_saddle:
        centers.append((1.02, 0.))
        fluxes.append(-.8)
    nodes, elements = 4*len(centers), len(centers)
    values, coordinates = np.zeros((7, 4, 1, nodes)), np.zeros((2, 4, 1, nodes))
    width = .2
    for patch, (R, Z) in enumerate(centers):
        for vertex, (s, t) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
            node = 4*patch+vertex
            ss, tt = 1-2*s, 1-2*t
            coordinates[:, 0, 0, node] = [R+width*(s-.4), Z+width*(t-.6)]
            coordinates[0, 1, 0, node] = width*ss/3
            coordinates[1, 2, 0, node] = width*tt/3
            sign = 1 if patch == 0 else -1
            values[0, :, 0, node] = [fluxes[patch]+(s-.4)**2+sign*(t-.6)**2,
                                     ss*2*(s-.4)/3, tt*2*sign*(t-.6)/3, 0.]
            if patch == 3:
                values[0, :, 0, node] *= .1
                values[0, 0, 0, node] += .9*fluxes[patch]
            values[4:6, 0, 0, node] = 1.
    with h5py.File(path, 'w') as f:
        for k, v in dict(jorek_model=303, n_tor=1, n_coord_tor=1, n_order=3,
                         n_degrees=4, n_var=7, n_dim=2, n_nodes=nodes, n_elements=elements,
                         sqrt_mu0_rho0=1., t_now=1., index_now=1).items():
            f[k] = [v]
        f['vertex'] = np.arange(1, nodes+1).reshape(elements, 4).T
        f['index'] = np.arange(1, 4*nodes+1).reshape(nodes, 4).T
        f['size'] = np.ones((4, 4, elements))
        f['x'], f['values'] = coordinates, values
        f['boundary'], f['n_sons'] = np.zeros(nodes, int), np.zeros(nodes, int)
        f['constrained'] = np.full(nodes, b'F')
        f['R_axis_t'], f['Z_axis_t'] = [1.], [0.]
        f['R_xpoint_t'], f['Z_xpoint_t'] = [[.5], [.5]], [[-1.], [1.]]
        f['psi_axis_t'], f['psi_bnd_t'] = [-1.], [-.2]
    geometry, state, _ = read_restart(path)
    return path, geometry, torch.from_numpy(state)[None]


def test_native_roots_and_flux_boundary_choice(tmp_path):
    path, geometry, state = fixture(tmp_path)
    tracker = NativeCritical(geometry, path)
    result = tracker(state)
    torch.testing.assert_close(result['psi_axis'], torch.tensor([-1.], dtype=torch.float64))
    torch.testing.assert_close(result['source_psi_axis'], result['psi_axis'])
    torch.testing.assert_close(result['psi_bnd'], torch.tensor([-.2], dtype=torch.float64))
    torch.testing.assert_close(result['Z_xpoint'], torch.tensor([[-1., 1.]], dtype=torch.float64))
    # Independent calls and batching do not use prior trajectories/history.
    batched = tracker(state.repeat(2, 1, 1))
    torch.testing.assert_close(batched['psi_axis'], result['psi_axis'].expand(2))


def test_moving_xpoint_gradient_matches_finite_difference(tmp_path):
    path, geometry, state = fixture(tmp_path)
    tracker = NativeCritical(geometry, path, search_radius=.15)
    # Perturb psi by epsilon*Z on every patch; saddle Z shifts by width^2*epsilon/2.
    with h5py.File(path) as f:
        z_coefficients = torch.from_numpy(f['x'][()][1, :, 0, :].T.reshape(-1))
    epsilon = torch.tensor(.05, dtype=torch.float64, requires_grad=True)
    flux = state[..., 0] + epsilon*z_coefficients
    height = tracker(flux)['Z_xpoint'].sum()
    derivative, = torch.autograd.grad(height, epsilon)
    torch.testing.assert_close(derivative, torch.tensor(.04, dtype=torch.float64), atol=1e-10, rtol=1e-9)
    def value(e):
        return tracker(state[..., 0]+e*z_coefficients)['Z_xpoint'].sum()
    finite = (value(.050001)-value(.049999))/2e-6
    torch.testing.assert_close(derivative, finite, atol=1e-8, rtol=1e-7)


def test_missing_roots_rejected(tmp_path):
    path, geometry, state = fixture(tmp_path)
    tracker = NativeCritical(geometry, path)
    with pytest.raises(FloatingPointError, match='tracking failed'):
        tracker(torch.zeros_like(state))
    with pytest.raises(ValueError, match='float64'):
        tracker(state.float())
    changed = geometry.dof_map.copy()
    geometry.dof_map[0, 0] = 47
    with pytest.raises(ValueError, match='DOF maps'):
        NativeCritical(geometry, path)
    geometry.dof_map[:] = changed


def test_native_source_saddle_does_not_replace_qualified_minimum(tmp_path):
    path, geometry, state = fixture(tmp_path, source_saddle=True)
    tracker = NativeCritical(geometry, path)
    result = tracker(state)
    torch.testing.assert_close(result['psi_axis'], torch.tensor([-1.], dtype=torch.float64))
    torch.testing.assert_close(result['source_psi_axis'], torch.tensor([-.8], dtype=torch.float64))
    # A source-compatible saddle cannot satisfy the independent minimum gate.
    state[:, :16, 0] = 0.
    with pytest.raises(FloatingPointError, match='tracking failed'):
        tracker(state)


def test_native_source_flux_derivative_and_batching(tmp_path):
    path, geometry, state = fixture(tmp_path, source_saddle=True)
    tracker = NativeCritical(geometry, path)
    shift = torch.zeros_like(state[..., 0])
    shift[:, 48::4] = 1.
    epsilon = torch.tensor(.03, dtype=torch.float64, requires_grad=True)
    value = tracker(state[..., 0]+epsilon*shift)['source_psi_axis']
    derivative, = torch.autograd.grad(value.sum(), epsilon)
    torch.testing.assert_close(derivative, torch.ones_like(epsilon))
    finite = (tracker(state[..., 0]+.030001*shift)['source_psi_axis']-
              tracker(state[..., 0]+.029999*shift)['source_psi_axis'])/2e-6
    torch.testing.assert_close(derivative, finite.squeeze(), atol=1e-9, rtol=1e-8)
    batched = tracker(torch.cat((state[..., 0], state[..., 0]+.03*shift)))
    torch.testing.assert_close(batched['source_psi_axis'], torch.tensor([-.8, -.77], dtype=torch.float64))
