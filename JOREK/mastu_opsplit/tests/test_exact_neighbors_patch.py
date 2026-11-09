"""Compile both native algorithms and compare their complete neighbor arrays."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def native_neighbors(tmp_path_factory):
    work=tmp_path_factory.mktemp('neighbors')
    root=Path(__file__).resolve().parents[1]
    source=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/restart_capacity_recovery_20261103/source/grids/grid_utils/update_neighbours_basic.f90')
    target=work/'grids/grid_utils/update_neighbours_basic.f90';target.parent.mkdir(parents=True)
    shutil.copyfile(source,target)
    original=target.read_text()
    subprocess.run(['patch','--batch','-p1','-i',str(root/'patches/0015-preserve-exact-neighbours-with-incidence-search.patch')],cwd=work,check=True,capture_output=True)
    updated=target.read_text()
    stub='''module data_structure
 integer, parameter :: n_vertex_max=4
 type type_element
  integer :: vertex(4),neighbours(4)
 end type
 type type_element_list
  integer :: n_elements
  type(type_element),allocatable :: element(:)
 end type
 type type_node_list
  integer :: n_nodes=0
 end type
end module
module mod_parameters
end module
'''
    driver='''program compare_neighbors
 use data_structure
 implicit none
 type(type_element_list) :: old,new
 type(type_node_list) :: nodes
 integer :: e,i
 read(*,*) e
 old%n_elements=e;new%n_elements=e
 allocate(old%element(e),new%element(e))
 do i=1,e
  read(*,*) old%element(i)%vertex
  old%element(i)%neighbours=(/i,-i,7,0/)
 enddo
 new%element=old%element
 call update_neighbours_old(old,nodes)
 call update_neighbours_new(new,nodes)
 do i=1,e
  if(any(old%element(i)%neighbours /= new%element(i)%neighbours)) stop 9
 enddo
 print *, 'EXACT'
end program
'''
    program=stub+original.replace('update_neighbours_basic','update_neighbours_old').replace('are_elements_neighbours','are_elements_neighbours_old')+updated.replace('update_neighbours_basic','update_neighbours_new').replace('are_elements_neighbours','are_elements_neighbours_new')+driver
    (work/'compare.f90').write_text(program)
    executable=work/'compare'
    subprocess.run([shutil.which('gfortran'),'-O2','-fopenmp','-fcheck=all','-ffree-line-length-none',str(work/'compare.f90'),'-o',str(executable)],cwd=work,check=True,capture_output=True)
    return executable


@pytest.mark.parametrize('case',['empty','structured','permuted','degenerate','nonmanifold','random'])
def test_exact_native_neighbors(native_neighbors,case):
    rng=np.random.default_rng(617)
    grid=np.array([[i+9*j,i+1+9*j,i+1+9*(j+1),i+9*(j+1)] for j in range(7) for i in range(8)])+1
    if case=='empty':
        elements=np.empty((0,4),dtype=int)
    elif case=='structured':
        elements=grid
    elif case=='permuted':
        elements=grid[rng.permutation(len(grid))]
        elements=rng.permutation(90)[elements]+11
    elif case=='degenerate':
        elements=np.array([[1,1,2,3],[1,2,3,4],[3,2,2,1],[0,0,-1,-2]])
    elif case=='nonmanifold':
        elements=np.array([[1,2,3,4],[2,1,5,6],[1,2,7,8],[2,1,9,10]])
    else:
        elements=rng.integers(-5,30,(100,4))
    payload=str(len(elements))+'\n'+'\n'.join(' '.join(map(str,row)) for row in elements)+'\n'
    result=subprocess.run([str(native_neighbors)],input=payload,text=True,capture_output=True,timeout=10,env={'OMP_NUM_THREADS':'1'})
    assert result.returncode==0,result.stdout+result.stderr
    assert 'EXACT' in result.stdout
