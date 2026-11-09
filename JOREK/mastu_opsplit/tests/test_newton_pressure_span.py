"""Finite-difference the patched native GS profile Jacobian, including axis/boundary columns."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


@pytest.fixture(scope='module')
def native_columns(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    parent = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/fast_mesh_20261103/source160')
    work = tmp_path_factory.mktemp('gs_pressure_span')
    (work / 'models').mkdir()
    path = work / 'models/element_matrix_GS_perturbation.f90'
    shutil.copyfile(parent / 'models/element_matrix_GS_perturbation.f90', path)
    subprocess.run(['patch', '--batch', '-p1', '-i', str(root / 'patches/0024-preserve-normalized-pressure-in-newton-gs.patch')], cwd=work, check=True, capture_output=True)
    source = path.read_text()
    span = source[source.index('    pressure_span_derivative ='):source.index('    do i=1,n_vertex_max')]
    matrix = source[source.index('            ELM(index_ij,index_kl) ='):source.index('         enddo\n       enddo')]
    driver = '''program check
implicit none
real*8 :: ELM(1,1),ELM_axis(1,1),ELM_bnd(1,1),x_g(1,1),factor(1,1)
real*8 :: psi_axis_kl(1,1),psi_bnd_kl(1,1),psi_x,v_x,psi_y,v_y
real*8 :: pprime_fact,fact_newton,pressure_span_derivative,psi_axis,psi_bnd,delta_psi_GS
real*8 :: zn,zT,dn_dpsi,dT_dpsi,ddn_dpsi,ddT_dpsi,dFFprime_dpsi
real*8 :: psi_norm,v,psi,xjac,wst,fact_private
integer :: index_ij,index_kl,ms,mt,k,l
read(*,*) psi_axis,psi_bnd,delta_psi_GS,psi_norm,zn,zT,dn_dpsi,dT_dpsi,ddn_dpsi,ddT_dpsi,dFFprime_dpsi
index_ij=1;index_kl=1;ms=1;mt=1;k=1;l=1
ELM=0;ELM_axis=0;ELM_bnd=0;factor=0
psi_x=0;v_x=0;psi_y=0;v_y=0
x_g=1.2d0;v=1;psi=1;xjac=1;wst=1;fact_private=1;fact_newton=1
psi_axis_kl=1;psi_bnd_kl=1
pprime_fact=1
if(delta_psi_GS<10000.d0) pprime_fact=(psi_bnd-psi_axis)/delta_psi_GS
''' + span + matrix + '''
print *,ELM(1,1),ELM_axis(1,1),ELM_bnd(1,1)
end program
'''
    (work / 'check.f90').write_text(driver)
    subprocess.run(['gfortran', '-O0', '-fcheck=all', '-ffree-line-length-none', str(work / 'check.f90'), '-o', str(work / 'check')], cwd=work, check=True, capture_output=True)

    def evaluate(psi, axis, boundary, fixed):
        d = boundary - axis
        x = (psi - axis) / d
        n, t = 1 - .2*x + .05*x*x, .01*(1 - .8*x + .3*x*x)
        args = [axis, boundary, fixed, x, n, t, (-.2+.1*x)/d, .01*(-.8+.6*x)/d,
                .1/d**2, .006/d**2, -.3*x/d]
        output = subprocess.check_output([str(work / 'check')], input=' '.join(map(str, args))+'\n', text=True)
        return np.fromstring(output, sep=' ')
    return evaluate


def residual_profile(psi, axis, boundary, fixed):
    d = boundary - axis
    x = (psi-axis)/d
    n, t = 1-.2*x+.05*x*x, .01*(1-.8*x+.3*x*x)
    dp = n*.01*(-.8+.6*x) + (-.2+.1*x)*t
    return .15*(1-x*x)/1.2 - 1.2*dp/(fixed if fixed < 10000 else d)


@pytest.mark.parametrize('fraction', [.1, .5, .95])
@pytest.mark.parametrize('fixed', [1e30, .09])
def test_native_profile_columns_match_finite_differences(native_columns, fraction, fixed):
    axis, boundary = -.103, -.012
    q = np.array([axis+fraction*(boundary-axis), axis, boundary])
    expected = []
    for i in range(3):
        h = 1e-6 * (boundary-axis)
        plus, minus = q.copy(), q.copy()
        plus[i] += h
        minus[i] -= h
        expected.append(-(residual_profile(*plus, fixed)-residual_profile(*minus, fixed))/(2*h))
    np.testing.assert_allclose(native_columns(*q, fixed), expected, rtol=2e-8, atol=1e-9)
