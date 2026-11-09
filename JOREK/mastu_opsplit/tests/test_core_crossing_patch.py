"""Native reproduction: skip an extrapolated core root and find its true element."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/restart_capacity_recovery_20261103/source')


@pytest.fixture(scope='module')
def core_crossing(tmp_path_factory):
    work = tmp_path_factory.mktemp('core_crossing')
    name = 'grids/grid_utils/find_crossing.f90'
    target = work / name
    target.parent.mkdir(parents=True)
    shutil.copyfile(SOURCE / name, target)
    old = target.read_text().replace('subroutine find_crossing(', 'subroutine crossing_old(')
    subprocess.run(['patch', '--batch', '-p1', '-i', str(ROOT / 'patches/0020-keep-core-crossings-inside-parent-elements.patch')], cwd=work, check=True, capture_output=True)
    stub = '''module data_structure
 type type_node_list
 integer :: dummy
 end type
 type type_element_list
 integer :: dummy
 end type
 type type_surface
 integer :: n_pieces,elm(2)
 real*8 :: s(4,2),t(4,2)
 end type
 type type_surface_list
 type(type_surface) :: flux_surfaces(1)
 end type
end module
module phys_module
 real*8 :: R_geo=1.d0, surface_cross_tol=1.005d0
 integer :: n_flux=2
end module
module mod_interp
 use data_structure
contains
 subroutine interp_RZ(nodes,elements,ielm,r,s,x,xr,xs,xrs,xrr,xss,y,yr,ys,yrs,yrr,yss)
 type(type_node_list) :: nodes
 type(type_element_list) :: elements
 integer :: ielm
 real*8 :: r,s,x,xr,xs,xrs,xrr,xss,y,yr,ys,yrs,yrr,yss
 x=ielm-1+r; xr=1; xs=0; xrs=0; xrr=0; xss=0
 y=s; yr=0; ys=1; yrs=0; yrr=0; yss=0
 end subroutine
end module
'''
    driver = '''program test_crossing
 use data_structure
 use phys_module
 implicit none
 type(type_node_list) :: nodes
 type(type_element_list) :: elements
 type(type_surface_list) :: surfaces
 real*8 :: rc(4),zc(4),r,z,s,t,tht,target
 integer :: fail,mode,ielm,k,fast
 read(*,*) mode,n_flux,target,fast
 surfaces%flux_surfaces(1)%n_pieces=2
 surfaces%flux_surfaces(1)%elm=(/1,2/)
 do k=1,2
 surfaces%flux_surfaces(1)%s(:,k)=(/0.d0,.5d0,1.d0,.5d0/)
 surfaces%flux_surfaces(1)%t(:,k)=(/.5d0,0.d0,.5d0,0.d0/)
 enddo
 rc=(/target,0.d0,target,0.d0/)
 zc=(/0.d0,.5d0,1.d0,.5d0/)
 r=-99;z=-99;s=-99;t=-99;tht=-99;fail=-1;ielm=-1
 if(mode==0) then
 call crossing_old(nodes,elements,surfaces,1,rc,zc,r,z,ielm,s,t,tht,fail,fast==1)
 else
 call find_crossing(nodes,elements,surfaces,1,rc,zc,r,z,ielm,s,t,tht,fail,fast==1)
 endif
 print *,fail,ielm,r,z,s,t,tht
end program
'''
    program = stub + old + target.read_text() + (SOURCE / 'grids/grid_utils/find_crossing_on_surface_piece.f90').read_text() + (SOURCE / 'elements/cub1D.f90').read_text() + driver
    (work / 'test.f90').write_text(program)
    binary = work / 'test'
    subprocess.run(['gfortran', '-O0', '-fcheck=all', '-ffree-line-length-none', str(work / 'test.f90'), '-o', str(binary)], cwd=work, check=True, capture_output=True)

    def run(mode, n_flux, position, fast=1):
        out = subprocess.check_output([str(binary)], input=f'{mode} {n_flux} {position} {fast}\n', text=True).split()
        return int(out[0]), int(out[1]), list(map(float, out[2:]))
    return run


def test_closed_core_finds_valid_neighbor_instead_of_clamping(core_crossing):
    old = core_crossing(0, 2, 1.001)
    new = core_crossing(1, 2, 1.001)
    assert old[0] == new[0] == 0
    assert old[1] == 1 and old[2][2] > 1.
    assert new[1] == 2 and new[2][2] == pytest.approx(.001, abs=1e-12)
    assert new[2][:2] == pytest.approx([1.001, .5], abs=1e-12)


@pytest.mark.parametrize('n_flux,fast,position', [(1, 1, 1.001), (2, 0, 1.001), (2, 1, .2), (2, 1, 1.5)])
def test_exterior_and_valid_crossings_unchanged(core_crossing, n_flux, fast, position):
    old = core_crossing(0, n_flux, position, fast)
    new = core_crossing(1, n_flux, position, fast)
    assert old[:2] == new[:2]
    assert old[0] == 0
    assert new[2] == pytest.approx(old[2], abs=1e-12)
