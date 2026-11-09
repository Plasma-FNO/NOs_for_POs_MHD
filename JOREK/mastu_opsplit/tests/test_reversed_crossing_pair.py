"""Compiled native reversed-pair correction, healthy no-op and atomic rollback."""
from pathlib import Path
import subprocess
import numpy as np
import pytest

@pytest.fixture(scope='module')
def reversed_pair(tmp_path_factory):
    work=tmp_path_factory.mktemp('reversed_pair')
    root=Path(__file__).resolve().parents[1]
    cub=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/restart_capacity_recovery_20261103/source/elements/cub1D.f90')
    # Curved native FE geometry and nonlinear flux, with analytically known roots.
    stub='''module data_structure
 type type_node_list
 integer :: dummy
 end type
 type type_element_list
 integer :: dummy
 end type
end module
module mod_interp
 use data_structure
contains
 subroutine interp_RZ(nodes,elements,ielm,s,t,R,Rs,Rt,Z,Zs,Zt)
 type(type_node_list) :: nodes
 type(type_element_list) :: elements
 integer :: ielm
 real*8 :: s,t,R,Rs,Rt,Z,Zs,Zt
 R=1+s+.1d0*s*t; Rs=1+.1d0*t; Rt=.1d0*s
 Z=t+.05d0*s*s; Zs=.1d0*s; Zt=1
 end subroutine
 subroutine interp(nodes,elements,ielm,ivar,iharm,s,t,P,Ps,Pt)
 type(type_node_list) :: nodes
 type(type_element_list) :: elements
 integer :: ielm,ivar,iharm
 real*8 :: s,t,P,Ps,Pt
 P=s*s+t*t+.1d0*s*t; Ps=2*s+.1d0*t; Pt=2*t+.1d0*s
 end subroutine
end module
'''
    driver='''program check
 use data_structure
 implicit none
 type(type_node_list) :: nodes
 type(type_element_list) :: elements
 real*8 :: s(2),t(2),tau(2),rr(2),zz(2),target(2),rc(4),zc(4)
 integer :: fail
 read(*,*) s,t,tau,target
 rr=[1.2d0,1.3d0];zz=[.2d0,.3d0]
 rc=[1.42d0,0.d0,1.42d0,0.d0];zc=[0.d0,.6d0,1.2d0,.6d0]
 call opsplit_refine_reversed_pair(nodes,elements,[1,1],target,rc,zc,s,t,tau,rr,zz,fail)
 print *,fail,s,t,tau,rr,zz
end program
'''
    text=stub+(root/'native/opsplit_refine_core_crossing.f90').read_text()+(root/'native/opsplit_refine_reversed_pair.f90').read_text()+cub.read_text()+driver
    (work/'check.f90').write_text(text)
    subprocess.run(['gfortran','-O0','-fcheck=all','-ffree-line-length-none',str(work/'check.f90'),'-o',str(work/'check')],cwd=work,check=True,capture_output=True)
    def run(targets=(.57,.58),tau=(.1,-.1)):
        original=np.array([.4,.4,.6,.6,*tau,1.2,1.3,.2,.3])
        output=subprocess.check_output([str(work/'check')],input=' '.join(map(str,[*original[:6],*targets]))+'\n',text=True).split()
        return int(output[0]),np.array(list(map(float,output[1:]))),original
    return run

def test_reversed_intersections_recover_exact_flux_order(reversed_pair):
    status,a,_=reversed_pair()
    assert status==0 and a[5]>a[4]
    s,t=a[:2],a[2:4]
    np.testing.assert_allclose(s*s+t*t+.1*s*t,[.57,.58],atol=1e-12,rtol=0)
    np.testing.assert_allclose(a[6:8],1.42,atol=1e-11,rtol=0)
    np.testing.assert_allclose(a[8:],t+.05*s*s,atol=1e-11,rtol=0)

def test_ordered_pair_is_bitwise_unchanged(reversed_pair):
    status,a,b=reversed_pair(tau=(-.1,.1))
    assert status==0
    np.testing.assert_array_equal(a,b)

@pytest.mark.parametrize('targets',[(.57,5.),(.58,.57),(.57,.57)])
def test_unavailable_or_nonordered_exact_roots_roll_back_both(reversed_pair,targets):
    status,a,b=reversed_pair(targets=targets)
    assert status!=0
    np.testing.assert_array_equal(a,b)
