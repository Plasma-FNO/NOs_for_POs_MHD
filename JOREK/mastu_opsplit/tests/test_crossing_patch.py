"""Exercise the native crossing routine on finite and extrapolated segments."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/restart_capacity_recovery_20261103/source')


@pytest.fixture(scope='module')
def crossing(tmp_path_factory):
    work = tmp_path_factory.mktemp('crossing')
    name = 'grids/grid_utils/find_crossing_on_surface_piece.f90'
    target = work / name
    target.parent.mkdir(parents=True)
    shutil.copyfile(SOURCE / name, target)
    old = target.read_text().replace('find_crossing_on_surface_piece', 'crossing_old')
    subprocess.run(['patch', '--batch', '-p1', '-i', str(ROOT / 'patches/0019-bound-native-contour-crossings.patch')], cwd=work, check=True, capture_output=True)
    stub = '''module data_structure
 type type_node_list
 integer :: dummy
 end type
 type type_element_list
 integer :: dummy
 end type
 type type_surface
 real*8 :: s(4,1), t(4,1)
 integer :: elm(1)
 end type
end module
module phys_module
 real*8 :: R_geo=1.d0, surface_cross_tol=1.005d0
end module
module mod_interp
 use data_structure
contains
 subroutine interp_RZ(nodes,elements,ielm,r,s,x,xr,xs,xrs,xrr,xss,y,yr,ys,yrs,yrr,yss)
 type(type_node_list) :: nodes
 type(type_element_list) :: elements
 integer :: ielm
 real*8 :: r,s,x,xr,xs,xrs,xrr,xss,y,yr,ys,yrs,yrr,yss
 x=r; xr=1; xs=0; xrs=0; xrr=0; xss=0
 y=s; yr=0; ys=1; yrs=0; yrr=0; yss=0
 end subroutine
end module
'''
    driver = '''program test_crossing
 use data_structure
 implicit none
 type(type_node_list) :: nodes
 type(type_element_list) :: elements
 type(type_surface) :: surface
 real*8 :: rc(4),zc(4),r,z,s,t,tht,target,endpoint
 integer :: fail,mode
 read(*,*) mode,target,endpoint
 surface%elm=1
 surface%s(:,1)=(/0.d0,endpoint/2,endpoint,endpoint/2/)
 surface%t(:,1)=(/.5d0,0.d0,.5d0,0.d0/)
 rc=(/target,0.d0,target,0.d0/)
 zc=(/0.d0,.5d0,1.d0,.5d0/)
 r=-99; z=-99; s=-99; t=-99; tht=-99; fail=-1
 if(mode==0) then
 call crossing_old(nodes,elements,surface,1,rc,zc,r,z,s,t,tht,fail,.true.)
 else
 call find_crossing_on_surface_piece(nodes,elements,surface,1,rc,zc,r,z,s,t,tht,fail,.true.)
 endif
 print *,fail,r,z,s,t,tht
end program
'''
    program = work / 'test.f90'
    program.write_text(stub + old + target.read_text() + (SOURCE / 'elements/cub1D.f90').read_text() + driver)
    binary = work / 'test'
    subprocess.run(['gfortran', '-O0', '-fcheck=all', '-ffree-line-length-none', str(program), '-o', str(binary)], cwd=work, check=True, capture_output=True)

    def run(mode, target, endpoint=.5):
        output = subprocess.check_output([str(binary)], input=f'{mode} {target} {endpoint}\n', text=True)
        values = output.split()
        return int(values[0]), list(map(float, values[1:]))
    return run


@pytest.mark.parametrize('position', [.001, .1, .25, .499, .5])
def test_native_in_segment_crossings_preserved(crossing, position):
    before = crossing(0, position)
    after = crossing(1, position)
    assert before[0] == after[0] == 0
    assert after[1][:2] == pytest.approx([position, .5], abs=1e-12)
    assert after[1] == pytest.approx(before[1], abs=1e-12)


def test_native_extrapolated_root_rejected(crossing):
    # The old routine accepts t_flux=1.004, outside this surface piece.
    assert crossing(0, .501)[0] == 0
    assert crossing(1, .501)[0] != 0


def test_parent_element_extrapolation_rejected_before_caller_clamps(crossing):
    before = crossing(0, 1.001, endpoint=1.)
    assert before[0] == 0 and before[1][2] > 1.
    # define_central_grid clamps this parent coordinate, moving the mesh point.
    assert crossing(1, 1.001, endpoint=1.)[0] != 0
