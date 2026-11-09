"""Compile the repair against native basis functions; audit actual failed meshes."""
from pathlib import Path
import subprocess
import h5py
import numpy as np
import pytest
from mastu_opsplit.restart import cubic_basis

B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')

@pytest.fixture(scope='module')
def repair(tmp_path_factory,request):
    work=tmp_path_factory.mktemp('corner_handle')
    root=Path(__file__).resolve().parents[1]
    stub='''module mod_parameters
 integer, parameter :: n_order=3,n_degrees=4
end module
module data_structure
 type node_type
 real*8 :: x(1,4,2)
 end type
 type element_type
 integer :: vertex(4)
 real*8 :: size(4,4)
 end type
 type type_node_list
 integer :: n_nodes
 type(node_type),allocatable :: node(:)
 end type
 type type_element_list
 integer :: n_elements
 type(element_type),allocatable :: element(:)
 end type
end module
'''
    driver='''program check
 use data_structure
 implicit none
 type(type_node_list) :: nodes
 type(type_element_list) :: elements
 integer :: i
 read(*,*) nodes%n_nodes,elements%n_elements
 allocate(nodes%node(nodes%n_nodes),elements%element(elements%n_elements))
 do i=1,nodes%n_nodes
 read(*,*) nodes%node(i)%x
 enddo
 do i=1,elements%n_elements
 read(*,*) elements%element(i)%vertex,elements%element(i)%size
 enddo
 call opsplit_limit_corner_handle(nodes,elements)
 open(10,file='sizes.txt',status='replace')
 do i=1,elements%n_elements
 write(10,'(16ES25.16)') elements%element(i)%size
 enddo
end program
'''
    routine=getattr(request,'param','opsplit_limit_corner_handle')
    driver=driver.replace('opsplit_limit_corner_handle',routine)
    (work/'check.f90').write_text(stub+(B/'mumps_scaling8_20261104/source/elements/mod_basisfunctions.f90').read_text()+
        (root/f'native/{routine}.f90').read_text()+driver)
    subprocess.run(['gfortran','-O0','-fcheck=all','-ffree-line-length-none',str(work/'check.f90'),'-o',str(work/'check')],cwd=work,check=True,capture_output=True)
    def run(x,v,s):
        # Integer vertices need integer lexical input to Fortran.
        text=f'{len(x)} {len(v)}\n'+'\n'.join(' '.join(map(str,z.T.ravel())) for z in x)+'\n'+\
            '\n'.join(' '.join(map(str,a+1))+' '+' '.join(map(str,z.T.ravel())) for a,z in zip(v,s))+'\n'
        subprocess.run([str(work/'check')],input=text,text=True,cwd=work,check=True,capture_output=True)
        return np.loadtxt(work/'sizes.txt').reshape(-1,4,4).transpose(0,2,1)
    return run

def load(case):
    with h5py.File(B/case/'jorek_grid.h5') as f:
        return f['x'][:,:,0,:].transpose(2,1,0),f['vertex'][()].T-1,f['size'][()].transpose(2,1,0)

def minimum(x,v,s):
    q,_=np.polynomial.legendre.leggauss(32)
    a,b=np.meshgrid((q+1)/2,(q+1)/2,indexing='ij')
    _,hs,ht=cubic_basis(a.ravel(),b.ravel())
    jets=x[v]*s[...,None]
    rs,rt=[np.einsum('qvd,evdc->eqc',h,jets) for h in (hs,ht)]
    return (rs[...,0]*rt[...,1]-rs[...,1]*rt[...,0]).min()

@pytest.mark.parametrize('case', ['combined_wall160_20261104/probe3','scaling8_wall120_20261104/probe3'])
def test_real_fold_repaired_without_boundary_change(repair,case):
    x,v,s=load(case)
    out=repair(x,v,s)
    changed=np.flatnonzero((out!=s).any(axis=(1,2)))
    assert len(changed)==2
    assert minimum(x,v[changed],s[changed])<0
    assert minimum(x,v[changed],out[changed])>0
    # Every physical boundary edge has identical endpoint value/tangent data.
    edges={}
    for e,vs in enumerate(v):
        for side in range(4):
            edges.setdefault(tuple(sorted((vs[side],vs[(side+1)%4]))),[]).append((e,side))
    for entries in edges.values():
        if len(entries)!=1:
            continue
        e,side=entries[0];d=1+side%2
        np.testing.assert_array_equal(out[e,[side,(side+1)%4]][:,[0,d]],s[e,[side,(side+1)%4]][:,[0,d]])
    # Both incident representations of the altered interior edge remain identical.
    shared=set(v[changed[0]])&set(v[changed[1]])
    assert len(shared)==2
    for node in shared:
        rows=[]
        for e in changed:
            iv=int(np.flatnonzero(v[e]==node)[0]);iv2=int(np.flatnonzero(np.isin(v[e],list(shared))&(v[e]!=node))[0])
            side=iv if (iv+1)%4==iv2 else iv2
            rows.append(out[e,iv,1+side%2])
        assert rows[0]==rows[1]

def test_healthy_reference_is_unchanged(repair):
    x,v,s=load('combined_refined_solver_20261103/wall80')
    np.testing.assert_array_equal(repair(x,v,s),s)

def test_unrepairable_collapsed_geometry_rolls_back(repair):
    x,v,s=load('combined_wall160_20261104/probe3')
    x[:]=0
    np.testing.assert_array_equal(repair(x,v,s),s)
