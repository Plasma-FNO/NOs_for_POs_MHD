"""Compile the local chart repair and audit real failed/reference patch geometries."""
from pathlib import Path
import subprocess
import h5py
import numpy as np
import pytest
from mastu_opsplit.restart import cubic_basis

B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')

@pytest.fixture(scope='module')
def center_patch(tmp_path_factory):
    root=Path(__file__).resolve().parents[1];work=tmp_path_factory.mktemp('patch_tangent')
    stub='''module data_structure
 type node_type
 real*8 :: x(1,4,2)
 end type
 type type_node_list
 type(node_type), allocatable :: node(:)
 end type
end module
'''
    driver='''program check
 use data_structure
 implicit none
 type(type_node_list) :: nodes
 real*8, allocatable :: R(:,:),Z(:,:)
 integer :: nr,np,start,k,i,j,s_side
 read(*,*) nr,np,start,s_side
 allocate(nodes%node(start+nr*np),R(nr,np),Z(nr,np))
 do k=1,size(nodes%node)
 read(*,*) nodes%node(k)%x
 enddo
 do i=1,nr
 do j=1,np
 k=start+(i-1)*np+j
 R(i,j)=nodes%node(k)%x(1,1,1);Z(i,j)=nodes%node(k)%x(1,1,2)
 enddo
 enddo
 call opsplit_center_patch_tangents(nodes,start,nr,np,R,Z,s_side==1)
 do k=1,size(nodes%node)
 write(*,'(8ES25.16)') nodes%node(k)%x
 enddo
end program
'''
    (work/'check.f90').write_text(stub+(root/'native/opsplit_center_patch_tangents.f90').read_text()+driver)
    subprocess.run(['gfortran','-O0','-fcheck=all',str(work/'check.f90'),'-o',str(work/'check')],check=True,capture_output=True)
    def run(xy,v):
        start=int(v.min());edges=[abs(int(v[0,j]-v[0,0])) for j in (1,3)]
        npol=max(edges);nr=(len(xy)-start)//npol
        header=f'{nr} {npol} {start} {int(edges[0]==npol)}\n'
        data='\n'.join(' '.join(map(str,x.T.ravel())) for x in xy)
        out=subprocess.check_output([str(work/'check')],input=header+data+'\n',text=True)
        return np.fromstring(out,sep=' ').reshape(-1,2,4).transpose(0,2,1),nr,npol,start
    return run


def load_patch(case,patch):
    with h5py.File(B/case/f'grid_patch{patch}.h5') as f:
        return f['x'][()][:,:,0,:].transpose(2,1,0),f['vertex'][()].T-1,f['size'][()].transpose(2,1,0)


def minimum_jacobian(x,v,size):
    q,_=np.polynomial.legendre.leggauss(8);s,t=np.meshgrid((q+1)/2,(q+1)/2,indexing='ij')
    _,hs,ht=cubic_basis(s.ravel(),t.ravel());jets=x[v]*size[...,None]
    rs,rt=[np.einsum('qvd,evdc->eqc',h,jets) for h in (hs,ht)]
    return (rs[...,0]*rt[...,1]-rs[...,1]*rt[...,0]).min()


def resize(x,v,size):
    size=size.copy();c=x[v,0]
    se=np.stack([c[:,1]-c[:,0],c[:,1]-c[:,0],c[:,2]-c[:,3],c[:,2]-c[:,3]],axis=1)
    te=np.stack([c[:,3]-c[:,0],c[:,2]-c[:,1],c[:,2]-c[:,1],c[:,3]-c[:,0]],axis=1)
    for dof,edge,ends in [(1,se,[1,-1,-1,1]),(2,te,[1,1,-1,-1])]:
        size[:,:,dof]=abs(size[:,:,dof])*np.sign(np.sum(edge*x[v,dof],axis=-1))*ends
    size[:,:,3]=size[:,:,1]*size[:,:,2]
    return size


@pytest.mark.parametrize('case,patch', [('wall80_newton_20261103/wall80',10),('exact_poloidal_crossing_20261103/poloidal440',1)])
def test_real_folded_patch_has_positive_geometry_and_fixed_boundaries(center_patch,case,patch):
    x,v,size=load_patch(case,patch);assert minimum_jacobian(x,v,size)<0
    y,nr,np_,start=center_patch(x,v)
    assert minimum_jacobian(y,v,resize(y,v,size))>0
    np.testing.assert_array_equal(y[:,0],x[:,0])
    rows=np.arange(start,len(x)).reshape(nr,np_)
    boundary=np.unique(np.r_[rows[0],rows[-1],rows[:,0],rows[:,-1],np.arange(start)])
    np.testing.assert_array_equal(y[boundary],x[boundary])


@pytest.mark.parametrize('patch',range(1,12))
def test_original_reference_patch_is_exactly_unchanged(center_patch,patch):
    x,v,_=load_patch('normalized_newton_20261103/control1',patch)
    y,*_=center_patch(x,v)
    np.testing.assert_array_equal(y,x)


def test_degenerate_chord_leaves_whole_patch_unchanged(center_patch):
    x,v,_=load_patch('wall80_newton_20261103/wall80',10)
    start=int(v.min());stride=max(abs(int(v[0,j]-v[0,0])) for j in (1,3))
    x[start+stride+1,0]=x[start+1,0]
    y,*_=center_patch(x,v)
    np.testing.assert_array_equal(y,x)
