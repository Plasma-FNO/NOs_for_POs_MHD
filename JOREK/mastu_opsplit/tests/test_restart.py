import ctypes
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np
import pytest

from mastu_opsplit.numerical import evaluate_native, evaluate_native_grad
from mastu_opsplit.restart import cubic_basis, diagnose_restart, read_restart


def rectangle(path):
    xy = np.array([[1., 0.], [2., 0.], [2., 1.], [1., 1.]])
    coordinates = np.zeros((2, 4, 1, 4))
    coordinates[:, 0, 0] = xy.T
    coordinates[0, 1, 0] = 1
    coordinates[1, 2, 0] = 1
    values = np.zeros((7, 4, 1, 4))
    values[4, 0, 0] = 2+xy[:, 0]+xy[:, 1]
    values[4, 1:3, 0] = 1
    values[5, 0, 0] = 1
    sizes = np.array([[1, 1/3, 1/3, 1/9], [1, -1/3, 1/3, -1/9],
                      [1, -1/3, -1/3, 1/9], [1, 1/3, -1/3, -1/9]])
    with h5py.File(path, 'w') as f:
        scalars = {'jorek_model': 303, 'n_tor': 1, 'n_coord_tor': 1, 'n_order': 3,
                   'n_degrees': 4, 'n_var': 7, 'n_dim': 2, 'n_nodes': 4,
                   'n_elements': 1, 'sqrt_mu0_rho0': 1e-6, 't_now': 3., 'index_now': 3}
        arrays = {'vertex': np.arange(1, 5)[:, None], 'size': sizes.T[:, :, None],
                  'x': coordinates, 'values': values, 'boundary': np.zeros(4, dtype=int),
                  'index': np.arange(1, 17).reshape(4, 4).T,
                  'constrained': np.full(4, b'F'), 'n_sons': np.zeros(1, dtype=int)}
        for name, value in {**scalars, **arrays}.items():
            f[name] = value


def test_native_linear_field_gradients_and_cylindrical_volume(tmp_path):
    path = tmp_path/'restart.h5'
    rectangle(path)
    geometry, state, metadata = read_restart(path)
    fields, gradients = evaluate_native(state, geometry), evaluate_native_grad(state, geometry)
    np.testing.assert_allclose(fields[..., 4], 2+geometry.quadrature_rz.sum(axis=-1), atol=1e-14)
    np.testing.assert_allclose(gradients[..., 4, :], 1., atol=2e-14)
    np.testing.assert_allclose(fields[..., 5], 1., atol=1e-14)
    assert geometry.poloidal_weights.sum() == pytest.approx(1.)
    assert geometry.volume_weights.sum() == pytest.approx(3*np.pi)
    assert metadata['time_seconds'] == pytest.approx(3e-6)
    assert metadata['mapping_status'] == 'diagnostic_restart_only'
    assert metadata['native_dof_aliases_verified'] and metadata['negative_jacobian_samples'] == 0
    np.testing.assert_array_equal(geometry.bounds, read_restart(path, quadrature_order=8)[0].bounds)


def test_model303_current_and_vorticity_weak_constraints(tmp_path):
    from mastu_opsplit.numerical import model303_constraint_blocks
    path = tmp_path/'constraints.h5'
    rectangle(path)
    with h5py.File(path, 'r+') as stream:
        radius = stream['x'][0, 0, 0, :]
        values = stream['values'][()]
        values[:4] = 0
        values[0, 0, 0] = radius**3
        values[0, 1, 0] = 3*radius**2
        values[1, 0, 0] = radius**2
        values[1, 1, 0] = 2*radius
        values[2, 0, 0] = 3*radius
        values[2, 1, 0] = 3
        values[3, 0, 0] = 4
        stream['values'][...] = values
    geometry, state, _ = read_restart(path, quadrature_order=8)
    mass, gradient = model303_constraint_blocks(state, geometry)
    # Mixed-derivative test functions vanish on every edge of this single element.
    interior_tests = np.arange(3, 16, 4)
    np.testing.assert_allclose((mass+gradient)[interior_tests], 0, atol=3e-14)
    assert np.linalg.norm(mass[interior_tests]) > .01
    state[:, 2] *= 2
    perturbed_mass, perturbed_gradient = model303_constraint_blocks(state, geometry)
    np.testing.assert_allclose((perturbed_mass+perturbed_gradient)[interior_tests, 0],
                               mass[interior_tests, 0], atol=3e-14)


def test_native_aliases_condense_without_changing_fields_and_reject_inconsistency(tmp_path):
    path = tmp_path/'restart.h5'
    rectangle(path)
    with h5py.File(path, 'r+') as f:
        indices = f['index'][()].T.copy().reshape(-1)
        indices[5] = indices[1]
        indices[indices > 6] -= 1
        f['index'][...] = indices.reshape(4, 4).T
    geometry, state, metadata = read_restart(path)
    assert len(state) == 15 and metadata['canonical_native_dofs'] == 15
    fields = evaluate_native(state, geometry)
    np.testing.assert_allclose(fields[..., 4], 2+geometry.quadrature_rz.sum(axis=-1), atol=1e-14)
    with h5py.File(path, 'r+') as f:
        f['values'][4, 1, 0, 1] += .1
    with pytest.raises(ValueError, match='inconsistent field'):
        read_restart(path)


@pytest.mark.parametrize('invalid', ['constrained', 'vertex', 'n_tor', 'x'])
def test_unsupported_or_invalid_restart_is_rejected(tmp_path, invalid):
    path = tmp_path/'restart.h5'
    rectangle(path)
    with h5py.File(path, 'r+') as f:
        if invalid == 'constrained':
            f[invalid][0] = b'T'
        elif invalid == 'vertex':
            f[invalid][0, 0] = 0
        elif invalid == 'n_tor':
            f[invalid][()] = 3
        else:
            f[invalid][...] = 0
    with pytest.raises(ValueError):
        read_restart(path)


def test_basis_matches_pinned_fortran(tmp_path):
    from mastu_opsplit.common import load_config
    config = load_config(Path(__file__).resolve().parents[1]/'configs/case.yaml')
    source = Path(config['jorek']['source'])/'elements/mod_basisfunctions.f90'
    compiler = shutil.which('gfortran')
    if compiler is None or not source.exists():
        pytest.skip('Pinned source and Fortran compiler needed for independent parity check')
    (tmp_path/'parameters.f90').write_text('module mod_parameters\ninteger, parameter :: n_order=3, n_degrees=4\nend module\n')
    (tmp_path/'wrapper.f90').write_text('''subroutine sample(s,t,out) bind(C)
use iso_c_binding
use mod_basisfunctions
real(c_double), value :: s,t
real(c_double) :: out(16,3), h(4,4), hs(4,4), ht(4,4)
integer :: v,d,k
call basisfunctions(s,t,h,hs,ht)
do v=1,4
  do d=1,4
    k=4*(v-1)+d
    out(k,:)=[h(v,d),hs(v,d),ht(v,d)]
  enddo
enddo
end subroutine
''')
    library = tmp_path/'basis.so'
    subprocess.run([compiler, '-shared', '-fPIC', '-ffree-line-length-none',
                    'parameters.f90', str(source), 'wrapper.f90', '-o', str(library)],
                   cwd=tmp_path, check=True, capture_output=True)
    function = ctypes.CDLL(str(library)).sample
    function.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
    for s, t in np.vstack(([[0., 0.], [1., 1.]], np.random.default_rng(7).random((20, 2)))):
        output = np.empty((3, 4, 4))
        function(s, t, output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        np.testing.assert_allclose(np.asarray(cubic_basis(s, t)), output, atol=3e-15)


def test_diagnostic_distinguishes_drift_from_balance_and_rejects_changed_mesh(tmp_path):
    reference, path = tmp_path/'reference.h5', tmp_path/'current.h5'
    rectangle(reference)
    rectangle(path)
    with h5py.File(path, 'r+') as f:
        f['values'][4, :, :, :] *= 1.1
    report = diagnose_restart(path, reference)
    assert report['fields']['rho']['relative_l2_drift'] == pytest.approx(.1)
    assert report['moments_code_units']['pressure']['relative_change'] == pytest.approx(.1)
    assert report['positive_at_sampled_quadrature']
    assert not report['balance_checked'] and not report['constraints_checked']
    with h5py.File(path, 'r+') as f:
        f['x'][0, 0, 0, :] += .1
    with pytest.raises(ValueError, match='same physical mesh'):
        diagnose_restart(path, reference)


def test_diagnostic_cli_dry_run_and_exclusive_output(tmp_path):
    import json
    import sys
    source = Path(__file__).resolve().parents[1]/'scripts/diagnose_restart.py'
    restart, output = tmp_path/'restart.h5', tmp_path/'report.json'
    rectangle(restart)
    command = [sys.executable, str(source), '--input', str(restart), '--reference', str(restart),
               '--output', str(output)]
    subprocess.run([*command, '--dry-run'], check=True, capture_output=True)
    assert not output.exists()
    subprocess.run(command, check=True, capture_output=True)
    assert json.loads(output.read_text())['positive_at_sampled_quadrature']
    before = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == before
