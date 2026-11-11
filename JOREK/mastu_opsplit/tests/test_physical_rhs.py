import ctypes
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from mastu_opsplit.implicit_opsplit import ACTIVE_CHANNELS, FrozenTangent, WeakAssembler
from mastu_opsplit.physical_rhs import NativeProfiles, PhysicalRemainder, physical_coefficients
from test_implicit_opsplit import jets as native_jets
from test_physical_guard import fixture


PROFILE_NAMES = ('F0', 'gamma', 'eta', 'mu', 'etah', 'D', 'K', 'Dpar', 'Kpar', 'jsource', 'Sp', 'Sh', 'muv')


def profiles(values, normalized):
    return {**dict(zip(PROFILE_NAMES, values[..., :13].unbind(-1))),
            'vsource': values[..., 13:15], 'omegasource': values[..., 15:17], 'normalized': normalized}


@pytest.fixture(scope='module')
def native_kernel(tmp_path_factory):
    """Compile the existing native real RHS and complex-step tangent unmodified."""
    compiler = shutil.which('gfortran')
    if compiler is None:
        pytest.skip('Fortran compiler unavailable')
    text = (Path(__file__).parents[1]/'native/full_split_procedures.f90').read_text()
    routines = []
    for name in ('opsplit_linear', 'opsplit_physical', 'opsplit_linear_complex', 'opsplit_physical_complex', 'opsplit_tangent'):
        routines.append(re.search(r'^pure subroutine '+name+r'\(.*?^end subroutine$', text, flags=re.S|re.M)[0])
    source = 'module native_fixture\ncontains\n'+'\n'.join(routines)+'''
subroutine evaluate(jets,r,p,normalized,c,jac) bind(C)
use iso_c_binding
implicit none
real(c_double), intent(in) :: jets(21),r,p(17)
integer(c_int), intent(in) :: normalized
real(c_double), intent(out) :: c(3,5),jac(15,21)
call opsplit_physical(jets(1:7),jets(8:14),jets(15:21),r,p(1),p(2),p(3),p(4),p(5),p(6),p(7), &
                     p(8),p(9),p(10),p(11),p(12),p(13),normalized/=0,p(14:15),p(16:17),c)
end subroutine
end module
'''
    # Native RHS has a second output a(10), and its tangent uses the same frozen profiles.
    source = source.replace('real(c_double), intent(out) :: c(3,5),jac(15,21)',
                            'real(c_double), intent(out) :: c(3,5),jac(15,21)\nreal(c_double) :: a(10)')
    source = source.replace('p(16:17),c)\n', 'p(16:17),c,a)\n'+
                            'call opsplit_tangent(jets(1:7),jets(8:14),jets(15:21),r,p(1),p(2),p(3),p(4),p(5),p(6),p(7), &\n'+
                            'p(8),p(9),p(10),p(11),p(12),p(13),normalized/=0,p(14:15),p(16:17),jac)\n')
    work = tmp_path_factory.mktemp('physical_native')
    (work/'check.f90').write_text(source)
    subprocess.run([compiler, '-shared', '-fPIC', '-fcheck=all', '-ffree-line-length-none',
                    str(work/'check.f90'), '-o', str(work/'check.so')], check=True, capture_output=True)
    function = ctypes.CDLL(str(work/'check.so')).evaluate
    pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [pointer]*3+[ctypes.POINTER(ctypes.c_int)]+[pointer]*2

    def evaluate(jet, radius, profile, normalized):
        arguments = [np.ascontiguousarray(a, dtype=np.float64) for a in (jet, radius, profile)]
        output, jacobian = np.empty(15), np.empty((21, 15))
        flag = ctypes.c_int(int(normalized))
        function(*(a.ctypes.data_as(pointer) for a in arguments), ctypes.byref(flag),
                 output.ctypes.data_as(pointer), jacobian.ctypes.data_as(pointer))
        return output, jacobian
    return evaluate


@pytest.mark.parametrize('normalized', [False, True])
def test_coefficients_and_autograd_match_native_complex_step(native_kernel, normalized):
    rng = np.random.default_rng(17)
    jet = rng.normal(size=(7, 21))
    radius = rng.uniform(.3, 2., size=7)
    p = rng.uniform(.01, 1., size=(7, 17))
    p[:, 0], p[:, 1] = .64, 5/3
    expected, jacobian = zip(*(native_kernel(j, r, t, normalized) for j, r, t in zip(jet, radius, p)))
    x = torch.tensor(jet, requires_grad=True)
    result = physical_coefficients(x, torch.tensor(radius), profiles(torch.tensor(p), normalized))
    np.testing.assert_allclose(result.detach(), expected, rtol=2e-14, atol=2e-14)
    derivative = torch.stack([torch.autograd.grad(result[:, k].sum(), x, retain_graph=True)[0]
                              for k in range(15)], dim=-1)
    np.testing.assert_allclose(derivative, jacobian, rtol=5e-14, atol=3e-14)


class FixedProfiles(nn.Module):
    def __init__(self, values, normalized=True):
        super().__init__()
        self.register_buffer('values', values)
        self.normalized = normalized

    def forward(self, jets, *, physical_state=None):
        return profiles(self.values, self.normalized)


def test_split_recombines_physics_retains_reference_and_differentiates(tmp_path, native_kernel):
    geometry, _, reference = fixture(tmp_path)
    # The native fixture has positive nodal density and temperature.
    rng = np.random.default_rng(3)
    values, gradients, reference_jets = native_jets(reference, geometry)
    npoints = reference_jets[..., 0].size
    profile = rng.uniform(.001, .02, size=(npoints, 17))
    profile[:, 0], profile[:, 1] = .64, 5/3
    physical, tangent = zip(*(native_kernel(j, r, p, True) for j, r, p in
                              zip(reference_jets.reshape(-1, 21), geometry.quadrature_rz[..., 0].ravel(), profile)))
    e, q = values.shape[:2]
    frozen = FrozenTangent(np.array(tangent).reshape(e, q, 21, 15), np.array(physical).reshape(e, q, 5, 3),
                           values, gradients, 'synthetic-native', 'memory')
    # Deliberately choose a normalization mean different from q*: zero state is not the reference.
    mean, scale = reference+.001, np.arange(1, 8)*.1
    model = PhysicalRemainder(geometry, frozen, mean, scale, FixedProfiles(torch.tensor(profile)))
    ystar = torch.tensor((reference-mean)/scale)[None]
    torch.testing.assert_close(model.pointwise(ystar), torch.zeros(1, npoints, 13, dtype=torch.float64), atol=2e-13, rtol=0)
    state = (ystar+torch.tensor(rng.normal(size=(2, *reference.shape)))*1e-4).requires_grad_()
    full = physical_coefficients(model.jets(state), model.radius, model.profiles(model.jets(state)))
    torch.testing.assert_close(model.pointwise(state)+model.linear_pointwise(model.jets(state))[..., list(ACTIVE_CHANNELS)],
                               full[..., list(ACTIVE_CHANNELS)], atol=1e-14, rtol=1e-14)
    expected = WeakAssembler(geometry).numpy(full.detach().numpy()[..., list(ACTIVE_CHANNELS)])
    np.testing.assert_allclose((model(state)+model.linear_forcing(state)).detach(), expected, rtol=1e-12, atol=1e-12)
    model(state).square().mean().backward()
    assert torch.isfinite(state.grad).all() and state.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='configured physical coefficient'):
        model(state, torch.tensor([[2.]]))
    bad = state.detach().clone()
    bad[:, ::4, 5] -= 1e5
    with pytest.raises(FloatingPointError, match='positive density/temperature'):
        model(bad)


def pinned_physics():
    return dict(f0=.64, gamma=5/3, eta=1e-5, visco=1e-5, eta_ohmic=0., d_perp=1e-5, zk_perp=1e-5,
                d_par=0., zk_par=1., zk_par_max=1e20, visco_par=1e-5, eta_t_dependent=True,
                visco_t_dependent=True, zkpar_t_dependent=True, t_0=.018, t_1=.0001, t_min_neg=-1e12,
                corr_neg_temp_coef=[.5, .5], t_max_eta=1e99, t_max_eta_ohm=1e99, t_min=1e-20,
                t_min_zkpar=1e-20, rho_0=1., rho_1=.01, rho_coef=[-.192, .199, -.09, .023, .99],
                t_coef=[-.37, -1.135, .664, .025, .989], ff_0=1.4, ff_1=0.,
                ff_coef=[-1.2, .2, 0., .005, 1., .0003, .985, .03, 0.], particlesource=1e-5,
                particlesource_psin=100., particlesource_sig=.1, heatsource=1e-7, heatsource_psin=1.,
                heatsource_sig=.1, normalized_velocity_profile=True)


@pytest.fixture(scope='module')
def native_profile_kernel(tmp_path_factory):
    """Compile pinned native profile functions and actual transport statements."""
    compiler = shutil.which('gfortran')
    parent = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_full_opsplit_20261104/build/source/models')
    if compiler is None or not (parent/'density.f90').exists():
        pytest.skip('Pinned native analytic profile source/compiler unavailable')
    p = pinned_physics()
    integer_names = 'LOWER_XPOINT=1,UPPER_XPOINT=2,num_rho_len=2,num_T_len=2,num_ffprime_len=2'
    declarations = ['module phys_module', 'implicit none', 'integer :: '+integer_names]
    # Read-only helper routines contain inactive numerical-profile branches; provide their declared arrays too.
    declarations += ['logical :: num_rho=.false.,num_T=.false.,num_ffprime=.false.,freeboundary_equil=.false.']
    for key in ('rho_0', 'rho_1', 't_0', 't_1', 'ff_0', 'ff_1', 'particlesource', 'particlesource_psin',
                'particlesource_sig', 'heatsource', 'heatsource_psin', 'heatsource_sig', 't_min_neg',
                'eta', 'visco', 'eta_ohmic', 't_max_eta', 't_max_eta_ohm', 't_min', 'zk_par', 'zk_par_max', 't_min_zkpar'):
        declarations.append(f'real*8 :: {key}={float(p[key]):.17e}_8')
    for key in ('rho_coef', 't_coef', 'ff_coef', 'corr_neg_temp_coef'):
        declarations.append(f'real*8 :: {key}({len(p[key])})=['+','.join(f'{x:.17e}_8' for x in p[key])+']')
    declarations += ['logical :: eta_T_dependent=.true.,visco_T_dependent=.true.,ZKpar_T_dependent=.true.',
                     'real*8 :: edgeparticlesource=0,edgeparticlesource_psin=1,edgeparticlesource_sig=1',
                     'real*8 :: heatsource_gauss(5)=0,heatsource_gauss_psin(5)=0,heatsource_gauss_sig(5)=1',
                     'real*8 :: particlesource_gauss(5)=0,particlesource_gauss_psin(5)=0,particlesource_gauss_sig(5)=1']
    for name in ('rho', 'T', 'ffprime'):
        declarations.append('real*8 :: '+','.join(f'num_{name}_{suffix}(2)=0' for suffix in ('x', 'y0', 'y1', 'y2', 'y3')))
    declarations += ['end module', 'module vacuum', 'real*8 :: current_FB_fact=1', 'end module',
                     'module mod_F_profile', 'end module', 'module native_profiles_fixture', 'contains']
    native = '\n'.join((parent/f'{name}.f90').read_text() for name in ('density', 'temperature', 'ffprime'))
    sources = (parent/'model303/mod_sources.f90').read_text()
    native += '\n'+re.search(r'^subroutine sources\(.*?^end subroutine sources$', sources, flags=re.S|re.M)[0]
    correction = (parent/'corr_neg_include.f90').read_text()
    native += '\n'+re.search(r'^real\*8 function corr_neg_temp1\(.*?^end function corr_neg_temp1', correction, flags=re.S|re.M)[0]
    matrix = (parent/'model303/mod_elt_matrix_fft.f90').read_text()
    transport = matrix[matrix.index('          ! --- Temperature dependent resistivity'):matrix.index('          ! --- Diamagnetic viscosity')]
    wrapper = '''
subroutine evaluate_profiles(point,diag,out) bind(C)
use iso_c_binding
use phys_module
implicit none
real(c_double),intent(in) :: point(4),diag(4)
real(c_double),intent(out) :: out(6)
real*8 :: density_value,density_derivative,temperature_value,temperature_derivative,ff
real*8 :: a,b,c,d,e,f,g,h,BigR,Z,psi,T0,T0_corr,psi_axis,psi_bnd,Z_xpoint(2)
real*8 :: eta_T,deta_dT,d2eta_d2T,eta_T_ohm,deta_dT_ohm,visco_T,dvisco_dT,d2visco_dT2,ZKpar_T,dZKpar_dT
logical :: xpoint2
BigR=point(1);Z=point(2);psi=point(3);T0=point(4);T0_corr=corr_neg_temp1(T0)
psi_axis=diag(1);psi_bnd=diag(2);Z_xpoint=diag(3:4);xpoint2=.true.
call density(.true.,3,Z,Z_xpoint,psi,psi_axis,psi_bnd,density_value,density_derivative,a,b,c,d,e,f,g)
call temperature(.true.,3,Z,Z_xpoint,psi,psi_axis,psi_bnd,temperature_value,temperature_derivative,a,b,c,d,e,f,g)
call FFprime(.true.,3,Z,Z_xpoint,psi,psi_axis,psi_bnd,ff,a,b,c,d,e,.true.)
out(1)=ff-BigR**2*(density_value*temperature_derivative+density_derivative*temperature_value)
call sources(.true.,3,Z,Z_xpoint,psi,psi_axis,psi_bnd,out(2),out(3))
'''+transport+'''
out(4:6)=[eta_T,visco_T,ZKpar_T]
end subroutine
end module
'''
    work = tmp_path_factory.mktemp('native_profiles')
    (work/'check.f90').write_text('\n'.join(declarations)+'\n'+native+'\n'+wrapper)
    compiled = subprocess.run([compiler, '-cpp', '-shared', '-fPIC', '-fcheck=all', '-ffree-line-length-none',
                               str(work/'check.f90'), '-o', str(work/'check.so')], cwd=work, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    function = ctypes.CDLL(str(work/'check.so')).evaluate_profiles
    pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [pointer]*3

    def evaluate(point, diagnostic):
        arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in (point, diagnostic)]
        output = np.empty(6)
        function(*(a.ctypes.data_as(pointer) for a in (*arrays, output)))
        return output
    return evaluate


@pytest.mark.parametrize('separate_source_axis', [False, True])
def test_native_source_and_temperature_profiles_match_values_and_derivatives(native_profile_kernel, separate_source_axis):
    rng = np.random.default_rng(91)
    radius, height = rng.uniform(.3, 1.4, size=(2, 12))
    height = np.linspace(-1.6, 1.6, 12)
    axis, boundary, lower, upper = -.103, -.012, -1.245, 1.245
    psi = axis+(boundary-axis)*np.linspace(-.2, 2.2, 12)
    temperature = np.array([.00003, .00008, .0002, .0005, .001, .004, .01, .018, .025, .04, .07, .12])
    points = np.stack((radius, height, psi, temperature), -1)
    diagnostics = np.array([axis, boundary, lower, upper])
    expected = np.array([native_profile_kernel(point, diagnostics) for point in points])
    model = NativeProfiles(SimpleNamespace(quadrature_rz=points[None, :, :2]), pinned_physics(), nn.Identity())
    jet = torch.zeros(1, 12, 21, dtype=torch.float64)
    jet[0, :, 0], jet[0, :, 5] = torch.tensor(psi), torch.tensor(temperature)
    jet.requires_grad_()
    diag = dict(psi_axis=torch.tensor([axis]), psi_bnd=torch.tensor([boundary]), Z_xpoint=torch.tensor([[lower, upper]]))
    diag = {k: v.double() for k, v in diag.items()}
    # Construct directly in float64: float32 diagnostic rounding changes steep source profiles measurably.
    diag = dict(psi_axis=torch.tensor([axis], dtype=torch.float64), psi_bnd=torch.tensor([boundary], dtype=torch.float64),
                Z_xpoint=torch.tensor([[lower, upper]], dtype=torch.float64))
    if separate_source_axis:
        diag['source_psi_axis'] = diag['psi_axis']
        diag['psi_axis'] = diag['psi_axis']+.001
    result = model.from_diagnostics(jet, diag)
    values = torch.stack([result[key] for key in ('jsource', 'Sp', 'Sh', 'eta', 'mu', 'Kpar')], -1)[0]
    np.testing.assert_allclose(values.detach(), expected, rtol=5e-13, atol=5e-14)
    # Native finite differences independently check derivatives through the transported profiles.
    for column, input_index in ((0, 2), (5, 3)):
        step = np.maximum(np.abs(points[:, input_index]), 1e-4)*1e-6
        shifted = points.copy()
        shifted[:, input_index] += step
        plus = np.array([native_profile_kernel(point, diagnostics) for point in shifted])
        shifted[:, input_index] -= 2*step
        minus = np.array([native_profile_kernel(point, diagnostics) for point in shifted])
        derivative = torch.stack([torch.autograd.grad(values[:, k].sum(), jet, retain_graph=True)[0][0, :, column]
                                  for k in range(6)], -1)
        np.testing.assert_allclose(derivative, (plus-minus)/(2*step[:, None]), rtol=1e-5, atol=2e-8)
