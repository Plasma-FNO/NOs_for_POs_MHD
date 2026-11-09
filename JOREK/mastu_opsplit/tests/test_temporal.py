import ctypes
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from mastu_opsplit.numerical import evaluate_native, evaluate_native_grad
from mastu_opsplit.restart import read_restart
from mastu_opsplit.temporal import corrected_density, temporal_constraint_action
from test_restart import rectangle


def test_positive_low_density_correction_remains_active():
    values = np.array([.002, .01, .1])
    result = corrected_density(values, rho_1=.01, rho_min_neg=-1e12, coefficients=[.5, .5])
    assert result[0] > values[0]
    np.testing.assert_array_equal(result[1:], values[1:])


def test_volume_temporal_action_matches_compiled_native_blocks(tmp_path):
    compiler = shutil.which('gfortran')
    if compiler is None:
        pytest.skip('Fortran compiler unavailable')
    parent = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/restart_capacity_recovery_20261103/source')
    text = (parent / 'models/model303/mod_elt_matrix_fft.f90').read_text()
    names = ['11', '22', '31', '33', '42', '44', '55', '65', '66', '71', '77']
    expressions = []
    for name in names:
        matches = re.findall(r'^\s*amat_'+name+r'\s*=\s*(.*)$', text, flags=re.M)
        # First standalone assignment is the exact tstep-independent native term.
        expression = next(x for x in matches if not x.startswith('0.d0') and not x.startswith('amat_'))
        expressions.append('amat_'+name+' = '+expression.split('&')[0])
    correction = (parent / 'models/corr_neg_include.f90').read_text()
    correction = correction[correction.index('real*8 function corr_neg_dens1('):]
    correction = correction[:correction.index('end function corr_neg_dens1')+len('end function corr_neg_dens1')]
    program = '''module phys_module
real*8, parameter :: rho_1=.01d0, rho_min_neg=-1d12, corr_neg_dens_coef(2)=[.5d0,.5d0]
end module
module fixture
contains
''' + correction + '''
subroutine kernel(test, current, delta, radius, weight, result) bind(C)
use iso_c_binding
implicit none
real(c_double), intent(in) :: test(3), current(7,3), delta(7,3),radius,weight
real(c_double), intent(out) :: result(7)
real*8 :: v,v_x,v_y,ps0_x,ps0_y,r0,r0_corr,T0,Vpar0,BigR,F0,xjac,zeta
real*8 :: psi,psi_x,psi_y,u_x,u_y,zj,w,rho,T,Vpar
real*8 :: amat_11,amat_22,amat_31,amat_33,amat_42,amat_44,amat_55,amat_65,amat_66,amat_71,amat_77
v=test(1);v_x=test(2);v_y=test(3);BigR=radius;F0=.64d0;xjac=weight;zeta=0.d0
r0=current(5,1);r0_corr=corr_neg_dens1(r0);T0=current(6,1);Vpar0=current(7,1)
ps0_x=current(1,2);ps0_y=current(1,3)
psi=delta(1,1);psi_x=delta(1,2);psi_y=delta(1,3);u_x=delta(2,2);u_y=delta(2,3)
zj=delta(3,1);w=delta(4,1);rho=delta(5,1);T=delta(6,1);Vpar=delta(7,1)
'''+'\n'.join(expressions)+'''
result=[amat_11,amat_22,amat_31+amat_33,amat_42+amat_44,amat_55,amat_65+amat_66,amat_71+amat_77]
end subroutine
end module
'''
    source, library = tmp_path / 'mass.f90', tmp_path / 'mass.so'
    source.write_text(program)
    subprocess.run([compiler, '-cpp', '-shared', '-fPIC', '-fcheck=all', '-ffree-line-length-none', str(source), '-o', str(library)], cwd=tmp_path, check=True)
    function = ctypes.CDLL(str(library)).kernel
    pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [pointer]*6
    path = tmp_path / 'state.h5'
    rectangle(path)
    geometry, state, _ = read_restart(path)
    rng = np.random.default_rng(9921)
    state = rng.normal(scale=.001, size=state.shape)
    state[::4, 4] = .002
    state[:, 4][np.arange(len(state)) % 4 != 0] *= .01
    state[::4, 5] = 1.
    state[::4, 6] = .1
    increment = rng.normal(scale=.01, size=state.shape)
    fields, gradient = evaluate_native(state, geometry), evaluate_native_grad(state, geometry)
    delta, delta_gradient = evaluate_native(increment, geometry), evaluate_native_grad(increment, geometry)
    expected = np.zeros_like(state)
    for q in range(geometry.basis.shape[1]):
        current = np.asfortranarray(np.column_stack([fields[0, q], gradient[0, q]]))
        direction = np.asfortranarray(np.column_stack([delta[0, q], delta_gradient[0, q]]))
        radius = np.array(geometry.quadrature_rz[0, q, 0])
        weight = np.array(geometry.poloidal_weights[0, q])
        for l, index in enumerate(geometry.dof_map[0]):
            test = np.array([geometry.basis[0, q, l], *geometry.basis_grad[0, q, l]])
            result = np.empty(7)
            function(*(a.ctypes.data_as(pointer) for a in [test, current, direction, radius, weight, result]))
            expected[index] += 2*np.pi*result
    actual = temporal_constraint_action(state, increment, geometry, F0=.64, rho_1=.01, rho_min_neg=-1e12, density_coefficients=[.5, .5])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-14)
    assert np.linalg.norm(actual[:, 5]) > 0 and np.linalg.norm(actual[:, 6]) > 0
