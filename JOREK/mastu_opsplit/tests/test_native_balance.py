import shutil

import h5py
import numpy as np

from mastu_opsplit.native_balance import linearized_volume_balance
from mastu_opsplit.native_export import validate_export
from mastu_opsplit.numerical import evaluate_native, evaluate_native_grad
from mastu_opsplit.restart import read_restart
from test_restart import rectangle


def test_native_matrix_balance_uses_canonical_channel_order_and_reports_failure(tmp_path):
    before, after = tmp_path / 'before.h5', tmp_path / 'after.h5'
    rectangle(before)
    geometry, state, _ = read_restart(before)
    direction = np.arange(112, dtype=float).reshape(16, 7)*1e-7
    shutil.copyfile(before, after)
    with h5py.File(after, 'r+') as stream:
        stream['values'][...] += direction.reshape(4, 4, 7).transpose(2, 1, 0)[:, :, None, :]
        stream['index_now'][...] = 4
        stream['t_now'][...] = 3.25
    fields, gradient = evaluate_native(state, geometry), evaluate_native_grad(state, geometry)
    matrix = 2*np.eye(112)+.01*np.triu(np.ones((112, 112)), 1)
    path = tmp_path / 'opsplit-native-rank-000000.bin'

    def write_export(bad_rhs=False):
        rhs = matrix @ direction.ravel()
        if bad_rhs:
            rhs[5] += .001
        arrays = [geometry.quadrature_rz[0].reshape(4, 4, 2).transpose(2, 0, 1),
                  geometry.poloidal_weights[0].reshape(4, 4),
                  fields[0].reshape(4, 4, 7).transpose(2, 0, 1),
                  gradient[0].reshape(4, 4, 7, 2).transpose(3, 2, 0, 1),
                  geometry.basis[0].reshape(4, 4, 16).transpose(2, 0, 1),
                  geometry.basis_grad[0].reshape(4, 4, 16, 2).transpose(3, 2, 0, 1),
                  np.zeros((2, 4, 4)), np.zeros((2, 16)), rhs, matrix]
        with path.open('wb') as stream:
            stream.write(np.array([3031601, 4, 7, 16, 4, 8, 1, 0], dtype='<i4').tobytes())
            stream.write(np.array([1, 1, 2, 3, 4, *range(1, 17), *([0]*12)], dtype='<i4').tobytes())
            stream.write(np.array([3., .25, .5, 0., 5/3, 1.], dtype='<f8').tobytes())
            for array in arrays:
                stream.write(np.asarray(array, dtype='<f8').tobytes(order='F'))

    write_export()
    validation = validate_export(tmp_path, before)
    result = linearized_volume_balance(tmp_path, before, after, validation)
    assert result['status'] == 'interior_linearized_balance_verified'
    assert result['included_native_dofs_per_equation'] == 16
    assert result['scientific_gates_passed'] == []
    write_export(bad_rhs=True)
    validation = validate_export(tmp_path, before)
    failed = linearized_volume_balance(tmp_path, before, after, validation)
    assert failed['status'] == 'interior_linearized_balance_failed'
    assert not failed['metrics']['T']['target_met']
