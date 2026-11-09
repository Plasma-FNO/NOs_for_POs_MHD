"""Compare the diagnostic resistivity directly with pinned model303 assembly."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


def test_model303_diagnostic_resistivity_matches_native_branches(tmp_path):
    compiler = shutil.which('gfortran')
    if compiler is None:
        pytest.skip('Fortran compiler unavailable')
    root = Path(__file__).resolve().parents[1]
    source = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/src/jorek/src')
    text = (source/'models/model303/mod_elt_matrix_fft.f90').read_text()
    native = text.split('! --- Temperature dependent resistivity', 1)[1].split('! --- Eta for ohmic heating', 1)[0]
    patch = (root/'patches/0005-model303-energy-diagnostics.patch').read_text()
    additions = '\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
    diagnostic = additions.split('! OPSPLIT_MODEL303_RESISTIVITY_BEGIN', 1)[1].split('! OPSPLIT_MODEL303_RESISTIVITY_END', 1)[0]
    correction = (source/'models/corr_neg_include.f90').read_text()
    correction = correction[correction.index('real*8 function corr_neg_temp1('):]
    correction = correction[:correction.index('end function corr_neg_temp1')+len('end function corr_neg_temp1')]
    wrappers = []
    for name, body in [('native', native), ('diagnostic', diagnostic)]:
        wrappers.append(f'''
subroutine {name}(raw, coefficient, cap, floor_value, enabled, axis, result) bind(C)
  use iso_c_binding
  real(c_double), intent(in) :: raw, coefficient, cap, floor_value
  integer(c_int), intent(in) :: enabled, axis
  real(c_double), intent(out) :: result
  real*8 :: T0, eta, T_max_eta, T_min, T_0, eta_T, deta_dT, d2eta_d2T
  logical :: eta_T_dependent, xpoint2, xpoint
  T0=raw; eta=coefficient; T_max_eta=cap; T_min=floor_value; T_0=.018d0
  eta_T_dependent=enabled/=0; xpoint2=axis/=0
  xpoint=xpoint2
  {body}
  result=eta_T
end subroutine
''')
    program = '''module phys_module
  real*8, parameter :: T_1=1d-4, T_min_neg=-1d12, corr_neg_temp_coef(2)=[.5d0,.5d0]
end module
module fixture
  interface corr_neg_temp
    module procedure corr_neg_temp1
  end interface
contains
''' + correction + '\n' + '\n'.join(wrappers) + '\nend module\n'
    path, library = tmp_path/'resistivity.f90', tmp_path/'resistivity.so'
    path.write_text(program)
    subprocess.run([compiler, '-cpp', '-shared', '-fPIC', '-fcheck=all', '-ffree-line-length-none',
                    str(path), '-o', str(library)], cwd=tmp_path, check=True)
    functions = ctypes.CDLL(str(library))
    double, integer = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
    for function in [functions.native, functions.diagnostic]:
        function.argtypes = [double]*4+[integer]*2+[double]
    comparisons = 0
    for temperature in [-.001, 1e-6, 1e-4, .01, .018, 1.]:
        for coefficient in [0., 1e-5]:
            for cap in [.001, .1]:
                for enabled in [0, 1]:
                    for axis in [0, 1]:
                        args = [ctypes.c_double(value) for value in [temperature, coefficient, cap, 1e-4]]
                        args += [ctypes.c_int(enabled), ctypes.c_int(axis)]
                        actual, expected = ctypes.c_double(), ctypes.c_double()
                        functions.native(*(ctypes.byref(value) for value in args), ctypes.byref(expected))
                        functions.diagnostic(*(ctypes.byref(value) for value in args), ctypes.byref(actual))
                        np.testing.assert_allclose(actual.value, expected.value, rtol=1e-14, atol=0)
                        comparisons += 1
    assert comparisons == 96
