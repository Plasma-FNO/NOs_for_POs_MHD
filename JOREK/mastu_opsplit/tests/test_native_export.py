import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from mastu_opsplit.native_export import element_records
from mastu_opsplit.numerical import model303_advection_blocks


@pytest.mark.parametrize('contents', [b'', np.array([3031601, 4, 7, 16, 4, 8, 1, 0], dtype='<i4').tobytes()+b'bad'])
def test_native_export_rejects_truncated_records(tmp_path, contents):
    path = tmp_path / 'broken.bin'
    path.write_bytes(contents)
    with pytest.raises(ValueError, match='header|truncated'):
        list(element_records(path))


def test_exported_fortran_selected_blocks_match_independent_physical_algebra(tmp_path):
    compiler = shutil.which('gfortran')
    if compiler is None:
        pytest.skip('Fortran compiler unavailable')
    root = Path(__file__).resolve().parents[1]
    declarations = (root / 'native/opsplit_export_declarations.f90').read_text()
    point = (root / 'native/opsplit_export_point.f90').read_text()
    program = f'''subroutine check(values,gradient,radius,result) bind(C)
use iso_c_binding
implicit none
integer, parameter :: n_gauss=1,n_degrees=4
real(c_double), intent(in) :: values(7), gradient(2,7), radius
real(c_double), intent(out) :: result(4)
integer :: i,j,ms,mt
real*8 :: xjac,wst,BigR,y_g(1,1),v,v_x,v_y,GAMMA
real*8 :: ps0,u0,zj0,w0,r0,T0,Vpar0
real*8 :: ps0_x,ps0_y,u0_x,u0_y,zj0_x,zj0_y,w0_x,w0_y,r0_x,r0_y,T0_x,T0_y,Vpar0_x,Vpar0_y
real*8 :: ps0_s,ps0_t,u0_s,u0_t,r0_s,r0_t,T0_s,T0_t,Vpar0_s,Vpar0_t
{declarations}
i=1;j=1;ms=1;mt=1;xjac=1.d0;wst=.7d0;BigR=radius;y_g=0.d0
v=.3d0;v_x=.03d0;v_y=-.04d0;GAMMA=5.d0/3.d0
ps0=values(1);u0=values(2);zj0=values(3);w0=values(4);r0=values(5);T0=values(6);Vpar0=values(7)
ps0_x=gradient(1,1);ps0_y=gradient(2,1);u0_x=gradient(1,2);u0_y=gradient(2,2)
zj0_x=gradient(1,3);zj0_y=gradient(2,3);w0_x=gradient(1,4);w0_y=gradient(2,4)
r0_x=gradient(1,5);r0_y=gradient(2,5);T0_x=gradient(1,6);T0_y=gradient(2,6)
Vpar0_x=gradient(1,7);Vpar0_y=gradient(2,7)
ps0_s=ps0_x;ps0_t=ps0_y;u0_s=u0_x;u0_t=u0_y;r0_s=r0_x;r0_t=r0_y
T0_s=T0_x;T0_t=T0_y;Vpar0_s=Vpar0_x;Vpar0_t=Vpar0_y
opsplit_export_active=.true.;opsplit_weak=0.d0
{point}
result(1:2)=opsplit_selected(:,1,1)
result(3:4)=opsplit_weak(:,1)
end subroutine
'''
    source, library = tmp_path / 'selected.f90', tmp_path / 'selected.so'
    source.write_text(program)
    subprocess.run([compiler, '-shared', '-fPIC', '-fcheck=all', '-ffree-line-length-none', str(source), '-o', str(library)], check=True)
    function = ctypes.CDLL(str(library)).check
    pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [pointer]*4
    rng = np.random.default_rng(4821)
    for _ in range(32):
        state = rng.normal(size=7)
        state[4:6] = np.exp(state[4:6])
        gradient = rng.normal(size=(7, 2))
        radius = np.array(rng.uniform(.3, 1.7))
        result = np.empty(4)
        function(*(array.ctypes.data_as(pointer) for array in (state, gradient, radius, result)))
        expected = model303_advection_blocks(state, gradient, radius, 5/3)
        np.testing.assert_allclose(result[:2], expected, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(result[2:], expected*.3*radius*.7, rtol=2e-14, atol=2e-14)


def test_instrumentation_preserves_native_rhs_and_matrix_expressions(tmp_path):
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('prepare_native_export', root / 'reports/cpu_native_export_prepare_20261103.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent = module.PARENT
    for relative in ['models/model303/mod_elt_matrix_fft.f90', 'matrix/construct_matrix_mod.f90', 'Makefile']:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / relative, target)
    original = (tmp_path / 'models/model303/mod_elt_matrix_fft.f90').read_text()
    module.instrument(tmp_path)
    assert '$(CGDEP) $(wildcard models/$(MODEL)/opsplit_export_*.inc)' in (tmp_path / 'Makefile').read_text()
    changed = (tmp_path / 'models/model303/mod_elt_matrix_fft.f90').read_text()
    assert changed.index('#include "opsplit_export_finish.inc"') < changed.index('if (.NOT. use_fft) return')
    stripped = '\n'.join(line for line in changed.splitlines() if not line.startswith('#include "opsplit_export_'))
    assert ''.join(stripped.split()) == ''.join(original.split())


def write_record_fixture(path, rank, *, element=1, changed_matrix=False, twice=False):
    header = np.array([3031601, 4, 7, 16, 4, 8, 1, rank], dtype='<i4')
    identity = np.zeros(33, dtype='<i4')
    identity[0] = element
    data = np.zeros(6 + 74*16 + 32 + 112 + 112*112, dtype='<f8')
    if changed_matrix:
        data[-1] = 1e-15
    record = identity.tobytes() + data.tobytes()
    path.write_bytes(header.tobytes() + record*(2 if twice else 1))


def test_native_halo_copies_are_counted_once_only_when_bit_identical(tmp_path):
    from mastu_opsplit.native_export import unique_element_records
    files = [tmp_path / f'rank{i}.bin' for i in range(3)]
    write_record_fixture(files[0], 0)
    write_record_fixture(files[1], 1)
    write_record_fixture(files[2], 2, element=2)
    audit = {}
    assert [r['element_id'] for r in unique_element_records(files, audit=audit)] == [1, 2]
    assert audit == {'identical_halo_copies': 1, 'halo_elements': 1, 'ranks': [0, 1, 2]}


@pytest.mark.parametrize('same_rank,changed', [(True, False), (False, True)])
def test_native_halo_conflicts_remain_errors(tmp_path, same_rank, changed):
    from mastu_opsplit.native_export import unique_element_records
    a, b = tmp_path / 'a.bin', tmp_path / 'b.bin'
    write_record_fixture(a, 0)
    write_record_fixture(b, 0 if same_rank else 1, changed_matrix=changed)
    with pytest.raises(ValueError, match='differs|same rank'):
        list(unique_element_records([a, b]))


def test_duplicate_within_one_native_file_is_rejected(tmp_path):
    from mastu_opsplit.native_export import unique_element_records
    path = tmp_path / 'a.bin'
    write_record_fixture(path, 0, twice=True)
    with pytest.raises(ValueError, match='same rank'):
        list(unique_element_records([path]))
