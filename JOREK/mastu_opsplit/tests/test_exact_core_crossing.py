"""Check native intersection refinement against a curved FE map and exact flux."""
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture(scope='module')
def exact_crossing(tmp_path_factory):
    work=tmp_path_factory.mktemp('exact_crossing')
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
 real*8 :: s,t,tau,R,Z,target,rc(4),zc(4),radius
 integer :: fail
 read(*,*) s,t,target,radius
 tau=0.d0
 rc=[radius,0.d0,radius,0.d0]; zc=[0.d0,.6d0,1.2d0,.6d0]
 call opsplit_refine_core_crossing(nodes,elements,1,target,rc,zc,s,t,tau,R,Z,fail)
 print *,fail,s,t,tau,R,Z
end program
'''
    text=stub+(root/'native/opsplit_refine_core_crossing.f90').read_text()+(source/'elements/cub1D.f90').read_text()+driver
    (work/'check.f90').write_text(text)
    subprocess.run(['gfortran','-O0','-fcheck=all','-ffree-line-length-none',str(work/'check.f90'),'-o',str(work/'check')],cwd=work,check=True,capture_output=True)
    def run(s,t,target,radius):
        a=subprocess.check_output([str(work/'check')],input=f'{s} {t} {target} {radius}\n',text=True).split()
        return int(a[0]),np.array(list(map(float,a[1:])))
    return run


@pytest.mark.parametrize('s,t',[(.2,.3),(.45,.62),(.9,.8)])
def test_exact_bicubic_flux_and_curve_intersection(exact_crossing,s,t):
    target=s*s+t*t+.1*s*t
    radius=1+s+.1*s*t
    status,values=exact_crossing(s+.02,t-.03,target,radius)
    assert status==0
    np.testing.assert_allclose(values[:2],[s,t],rtol=0,atol=2e-11)
    np.testing.assert_allclose(values[3:],[radius,t+.05*s*s],rtol=0,atol=2e-11)


def test_refinement_rejects_root_outside_parent(exact_crossing):
    s,t=1.01,.4
    assert exact_crossing(.99,t,s*s+t*t+.1*s*t,1+s+.1*s*t)[0]!=0


def test_adjacent_targets_keep_flux_order(exact_crossing):
    targets=np.linspace(.57,.58,9)
    roots=[]
    for target in targets:
        status,q=exact_crossing(.4,.6,target,1.42)
        assert status==0
        roots.append(q[1])
    assert np.all(np.diff(roots)>0)
