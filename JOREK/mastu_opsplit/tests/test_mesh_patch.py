"""Independent checks of shared interior attachment propagation in the isolated patch."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


def test_shared_attachments_preserve_wall_endpoints_and_distinct_coordinates(tmp_path):
    compiler = shutil.which('gfortran')
    if compiler is None:
        pytest.skip('Fortran compiler unavailable')
    patch = Path(__file__).resolve().parents[1]/'patches/0002-grid-shared-patch-attachments.patch'
    additions = '\n'.join(line[1:] for line in patch.read_text().splitlines()
                          if line.startswith('+') and not line.startswith('+++'))
    start = additions.index('pure subroutine opsplit_share_patch_attachment(')
    helper = additions[start:additions.index('end subroutine opsplit_share_patch_attachment')
                       + len('end subroutine opsplit_share_patch_attachment')]
    source, library = tmp_path/'helper.f90', tmp_path/'helper.so'
    source.write_text(helper)
    subprocess.run([compiler, '-shared', '-fPIC', '-fcheck=all', str(source), '-o', str(library)], check=True)
    function = ctypes.CDLL(str(library)).opsplit_share_patch_attachment_
    real_pointer, int_pointer = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
    function.argtypes = [real_pointer]*4 + [int_pointer]*2 + [real_pointer]*2
    old_r, old_z, new_r, new_z = [np.array([x]) for x in (1.0238, -.8151, 1.02058, -.81807)]
    r = np.array([old_r[0], old_r[0], old_r[0], old_r[0]+1e-12, 2., old_r[0]])
    z = np.array([old_z[0], old_z[0], old_z[0], old_z[0], old_z[0], old_z[0]+1e-12])
    counts = np.array([3, 1, 0, 3, 3, 3], dtype=np.int32)
    before_r, before_z = r.copy(), z.copy()

    def call(size):
        n = np.array([size], dtype=np.int32)
        function(*(a.ctypes.data_as(real_pointer) for a in (old_r, old_z, new_r, new_z)),
                 n.ctypes.data_as(int_pointer), counts.ctypes.data_as(int_pointer),
                 r.ctypes.data_as(real_pointer), z.ctypes.data_as(real_pointer))

    call(0)
    np.testing.assert_array_equal(r, before_r)
    np.testing.assert_array_equal(z, before_z)
    call(len(r))
    assert r[0] == new_r[0] and z[0] == new_z[0]
    np.testing.assert_array_equal(r[1:], before_r[1:])
    np.testing.assert_array_equal(z[1:], before_z[1:])
