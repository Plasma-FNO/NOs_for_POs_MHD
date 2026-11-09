"""Recover native construction-group membership without guessing from coordinates."""
from pathlib import Path

import h5py
import numpy as np

from .common import sha256_file


def _corners(path):
    with h5py.File(path) as stream:
        vertices = stream['vertex'][()].T - 1
        coordinates = stream['x'][()][:, 0, 0, :].T
        if vertices.ndim != 2 or vertices.shape[1] != 4 or not len(vertices):
            raise ValueError('invalid native patch element layout')
        if vertices.min() < 0 or vertices.max() >= len(coordinates) or not np.isfinite(coordinates).all():
            raise ValueError('invalid native patch coordinates/connectivity')
        return coordinates[vertices]


def recover_patch_ids(restart, construction_folder, wall_blocks=11):
    """Verify native append order using exact oriented corners of every element.

    Group 0 is the central/divertor assembly; 1..N are actual wall-extension
    blocks. Final geometry supplies the basis: joining can rescale tangents.
    """
    if type(wall_blocks) is not int or not 1 <= wall_blocks <= 99:
        raise ValueError('a positive native wall-block count is required')
    construction_folder = Path(construction_folder)
    files = [construction_folder / 'grid_no_patch.h5'] + [construction_folder / f'grid_patch{i}.h5' for i in range(1, wall_blocks+1)]
    final = _corners(restart)
    patch_ids = np.full(len(final), -1, dtype=np.int32)
    groups, start = {}, 0
    for label, path in enumerate(files):
        corners = _corners(path)
        stop = start + len(corners)
        if stop > len(final) or not np.array_equal(corners, final[start:stop]):
            raise ValueError('native patch append-order/corner identity changed: ' + str(path))
        patch_ids[start:stop] = label
        groups[str(label)] = {'name': 'central_and_divertor' if label == 0 else f'wall_extension_{label}',
                              'first_element_id': start+1, 'last_element_id': stop,
                              'elements': len(corners), 'source': str(path), 'sha256': sha256_file(path)}
        start = stop
    if start != len(final) or (patch_ids < 0).any():
        raise ValueError('native patch files do not cover every final element')
    return patch_ids, {'status': 'native_construction_group_identity_verified', 'elements': len(final),
                       'restart_sha256': sha256_file(restart), 'groups': groups, 'exact_oriented_corner_identity': True,
                       'physical_basis_source': 'Final native restart, after patch joining and tangent rescaling.',
                       'limitation': 'Group 0 combines central and divertor subpatches; native constraint elimination is separate.',
                       'scientific_gates_passed': []}
