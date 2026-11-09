"""Reject the opposite side of a magnetic-axis root on an extended polar curve."""
from pathlib import Path
import subprocess
import numpy as np
import pytest

@pytest.fixture(scope='module')
def crossing(tmp_path_factory):
    work=tmp_path_factory.mktemp('bounded_core')
    root=Path(__file__).resolve().parents[1]
    source=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/restart_capacity_recovery_20261103/source')
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
 R=s; Rs=1; Rt=0; Z=t; Zs=0; Zt=1
 end subroutine
 subroutine interp(nodes,elements,ielm,ivar,iharm,s,t,P,Ps,Pt)
 type(type_node_list) :: nodes
 type(type_element_list) :: elements
 integer :: ielm,ivar,iharm
 real*8 :: s,t,P,Ps,Pt
 P=(s-.5d0)**2+(t-.5d0)**2; Ps=2*(s-.5d0); Pt=2*(t-.5d0)
 end subroutine
end module
'''
    driver='''program check
 use data_structure
 implicit none
 type(type_node_list) :: nodes
 type(type_element_list) :: elements
 real*8 :: s,t,tau,R,Z,target,rc(4),zc(4)
 integer :: fail
 read(*,*) s,target
 t=.5d0; tau=4*s-3
 rc=[.5d0,.25d0,1.d0,.25d0]; zc=[.5d0,0.d0,.5d0,0.d0]
 call opsplit_refine_bounded_core_crossing(nodes,elements,1,target,rc,zc,s,t,tau,R,Z,fail)
 print *,fail,s,t,tau
end program
'''
    text=stub+(root/'native/opsplit_refine_bounded_core_crossing.f90').read_text()+(source/'elements/cub1D.f90').read_text()+driver
    (work/'check.f90').write_text(text)
    subprocess.run(['gfortran','-O0','-fcheck=all','-ffree-line-length-none',str(work/'check.f90'),'-o',str(work/'check')],cwd=work,check=True,capture_output=True)
    def run(s,target):
        row=subprocess.check_output([str(work/'check')],input=f'{s} {target}\n',text=True).split()
        return int(row[0]),np.array(list(map(float,row[1:])))
    return run

@pytest.mark.parametrize('radius',[.003,.03,.2,.49])
def test_correct_axis_branch_is_accurate(crossing,radius):
    status,q=crossing(.5+radius*.95,radius**2)
    assert status==0
    np.testing.assert_allclose(q[:2],[.5+radius,.5],rtol=0,atol=2e-10)
    assert -1 <= q[2] <= 1

@pytest.mark.parametrize('radius',[.003,.03,.2])
def test_opposite_branch_in_same_parent_is_rejected(crossing,radius):
    status,q=crossing(.5-radius*.95,radius**2)
    assert status!=0
    assert 0<q[0]<.5 and q[2]<-1
