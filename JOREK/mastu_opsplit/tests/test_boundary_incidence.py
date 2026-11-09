"""Compile old/new boundary labeling and compare whole label arrays on real meshes."""
import importlib.util
from pathlib import Path
import subprocess
import time
import h5py
import numpy as np
import pytest

@pytest.fixture(scope='module')
def boundary_labelers(tmp_path_factory):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('boundary_incidence',root/'native/boundary_incidence.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    base=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/scaling8_capacity400_20261104/source')
    text=(base/'grids/grid_utils/update_boundary_types.f90').read_text()
    text=text[:text.index('end subroutine adjacent_elements')+len('end subroutine adjacent_elements')]
    stub='''module constants
 integer,parameter :: DOUBLE_NULL=3
end module
module mod_parameters
end module
module phys_module
 integer :: xcase=3
end module
module data_structure
 type type_node
 integer :: boundary=0
 real*8 :: x(1,4,2)=0
 end type
 type type_node_list
 integer :: n_nodes
 type(type_node),allocatable :: node(:)
 end type
 type type_element
 integer :: vertex(4)
 end type
 type type_element_list
 integer :: n_elements
 type(type_element),allocatable :: element(:)
 end type
end module
'''
    driver='''program check
 use data_structure
 implicit none
 type(type_node_list) :: nodes
 type(type_element_list) :: elements
 integer :: n,e,across,i
 read(*,*) n,e,across
 nodes%n_nodes=n;elements%n_elements=e
 allocate(nodes%node(n),elements%element(e))
 do i=1,e
 read(*,*) elements%element(i)%vertex
 enddo
 call update_boundary_types(elements,nodes,across)
 print *,'RESULT'
 print *,nodes%node%boundary
end program
'''
    work=tmp_path_factory.mktemp('boundary_incidence');executables=[]
    for name,source in [('old',text),('cached',module.replace_boundary_incidence(text))]:
        folder=work/name;folder.mkdir();p=folder/'check.f90';p.write_text(stub+source+'\n'+driver)
        subprocess.run(['gfortran','-O2','-fcheck=all','-ffree-line-length-none',str(p),'-o',str(folder/'check')],cwd=folder,check=True,capture_output=True)
        executables.append(folder/'check')
    def compare(vertices,across=0):
        vertices=np.asarray(vertices,dtype=int);n=int(vertices.max())
        inp=f'{n} {len(vertices)} {across}\n'+'\n'.join(' '.join(map(str,row)) for row in vertices)+'\n'
        outputs=[];times=[]
        for exe in executables:
            start=time.monotonic();out=subprocess.check_output([str(exe)],input=inp,text=True,timeout=120);times.append(time.monotonic()-start)
            outputs.append(np.fromstring(out.split('RESULT')[-1],sep=' ',dtype=int))
        assert len(outputs[0])==n
        np.testing.assert_array_equal(*outputs)
        return times
    return compare

def rectangle(nx,ny,hole=False):
    ids=np.arange((nx+1)*(ny+1)).reshape(ny+1,nx+1)+1
    return np.array([[ids[j,i],ids[j,i+1],ids[j+1,i+1],ids[j+1,i]] for j in range(ny) for i in range(nx) if not(hole and nx//3<i<2*nx//3 and ny//3<j<2*ny//3)])

@pytest.mark.parametrize('hole',[False,True])
def test_rectangular_grid_and_internal_hole_labels(boundary_labelers,hole):
    boundary_labelers(rectangle(20,18,hole))

def test_shuffled_elements_and_repeated_corner_vertices(boundary_labelers):
    v=rectangle(12,10);v[0,0]=v[0,1];np.random.default_rng(4).shuffle(v)
    boundary_labelers(v)

@pytest.mark.parametrize('name',['grid_patch4.h5','grid_patch11.h5'])
def test_saved_balanced_wall_patches_keep_exact_boundary_labels(boundary_labelers,name):
    base=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/combined_balanced2_attachment_20261104/probe3')
    with h5py.File(base/name) as f:v=f['vertex'][()].T
    boundary_labelers(v)
