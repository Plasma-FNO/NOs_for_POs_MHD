"""Compile the continuation guard with native Newton iteration and analytic saddle pairs."""
from pathlib import Path
import subprocess
import numpy as np
import pytest


@pytest.fixture(scope='module')
def continue_root(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    parent = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/normalized_newton_20261103/source')
    work = tmp_path_factory.mktemp('xpoint_continuation')
    stub = '''module data_structure
 type type_node_list
 integer :: dummy
 end type
 type element_type
 integer :: neighbours(4)=0
 end type
 type type_element_list
 integer :: n_elements=1
 type(element_type) :: element(1)
 end type
end module
module mod_interp
 use data_structure
 integer :: mode=0
 real*8 :: a=.49d0,b=.51d0
contains
 subroutine interp(nodes,elements,ielm,ivar,iharm,s,t,P,Ps,Pt,Pst,Pss,Ptt)
 type(type_node_list) :: nodes
 type(type_element_list) :: elements
 integer :: ielm,ivar,iharm
 real*8 :: s,t,P,Ps,Pt,Pst,Pss,Ptt
 P=(s-a)*(s-b)*(t-.5d0);Ps=(2*s-a-b)*(t-.5d0);Pt=(s-a)*(s-b)
 Pst=2*s-a-b;Pss=2*(t-.5d0);Ptt=0
 if(mode==1) then
 P=(s-.5d0)**2+(t-.5d0)**2;Ps=2*(s-.5d0);Pt=2*(t-.5d0)
 Pss=2;Ptt=2;Pst=0
 endif
 end subroutine
 subroutine interp_RZ(nodes,elements,ielm,s,t,R,Rs,Rt,Z,Zs,Zt)
 type(type_node_list) :: nodes
 type(type_element_list) :: elements
 integer :: ielm
 real*8 :: s,t,R,Rs,Rt,Z,Zs,Zt
 R=1+s;Z=1+t;Rs=1;Rt=0;Zs=0;Zt=1
 end subroutine
end module
subroutine find_RZ(nodes,elements,Rf,Zf,R,Z,e,s,t,fail)
 use data_structure
 implicit none
 type(type_node_list) :: nodes
 type(type_element_list) :: elements
 real*8 :: Rf,Zf,R,Z,s,t
 integer :: e,fail
 R=Rf;Z=Zf;s=R-1;t=Z-1;e=1;fail=0
 if(min(s,t)<0 .or. max(s,t)>1) fail=1
end subroutine
'''
    driver = '''program check
 use data_structure
 use mod_interp, only: mode,a,b
 implicit none
 type(type_node_list) :: nodes
 type(type_element_list) :: elements
 real*8 :: previous(2),radius,s,t
 integer :: side,e
 logical :: found
 read(*,*) previous,side,radius,mode,a,b
 call opsplit_continue_xpoint(nodes,elements,previous,1.d0,0.d0,radius,side,e,s,t,found)
 print *,merge(1,0,found),e,s,t
end program
'''
    source = stub + (parent/'solvers/mnewtax.f90').read_text() + (root/'native/opsplit_continue_xpoint.f90').read_text() + driver
    (work/'check.f90').write_text(source)
    subprocess.run(['gfortran','-O0','-fcheck=all','-ffree-line-length-none',str(work/'check.f90'),'-o',str(work/'check')],cwd=work,check=True,capture_output=True)
    def run(previous, side=1, radius=.1, mode=0, a=.49, b=.51):
        args=[*previous,side,radius,mode,a,b]
        output=subprocess.check_output([str(work/'check')],input=' '.join(map(str,args))+'\n',text=True)
        return np.fromstring(output,sep=' ')
    return run


@pytest.mark.parametrize('previous,expected', [(1.491,.49),(1.509,.51)])
def test_continuation_keeps_nearby_saddle_branch(continue_root,previous,expected):
    row=continue_root([previous,1.5])
    assert row[0]==1
    np.testing.assert_allclose(row[2:],[expected,.5],rtol=0,atol=1e-10)


def test_continuation_follows_a_moving_root(continue_root):
    row=continue_root([1.49,1.5],a=.491,b=.511)
    assert row[0]==1
    np.testing.assert_allclose(row[2:],[.491,.5],rtol=0,atol=1e-10)


@pytest.mark.parametrize('kwargs', [dict(mode=1),dict(side=-1),dict(radius=.001),dict(a=1.1,b=1.2)])
def test_invalid_continuation_defers_to_global_search(continue_root,kwargs):
    assert continue_root([1.48,1.5],**kwargs)[0]==0
