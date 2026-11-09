import h5py
import numpy as np
import pytest

from mastu_opsplit.native_patches import recover_patch_ids
from test_restart import rectangle


def test_patch_membership_requires_exact_every_element_identity(tmp_path):
    core = tmp_path / 'grid_no_patch.h5'
    wall = tmp_path / 'grid_patch1.h5'
    final = tmp_path / 'final.h5'
    rectangle(core)
    rectangle(wall)
    with h5py.File(wall, 'r+') as stream:
        values = stream['x'][()]
        values[0, 0, 0] += 1
        stream['x'][...] = values
    with h5py.File(core) as a, h5py.File(wall) as b, h5py.File(final, 'w') as stream:
        stream['x'] = np.concatenate([a['x'][()], b['x'][()]], axis=-1)
        stream['vertex'] = np.concatenate([a['vertex'][()], b['vertex'][()]+4], axis=1)
    ids, report = recover_patch_ids(final, tmp_path, wall_blocks=1)
    np.testing.assert_array_equal(ids, [0, 1])
    assert report['exact_oriented_corner_identity'] and report['elements'] == 2
    assert report['scientific_gates_passed'] == []
    with h5py.File(final, 'r+') as stream:
        stream['x'][0, 0, 0, -1] += 1e-8
    with pytest.raises(ValueError, match='identity changed'):
        recover_patch_ids(final, tmp_path, wall_blocks=1)
