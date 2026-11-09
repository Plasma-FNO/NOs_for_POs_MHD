"""Native guard identifies the actual concave mesh cells and preserves healthy meshes."""
import ctypes
from pathlib import Path
import subprocess
import h5py
import numpy as np
import pytest

@pytest.fixture(scope='module')
def concave(tmp_path_factory):
    root=Path(__file__).resolve().parents[1];w=tmp_path_factory.mktemp('concave')
    driver='''subroutine check(n,rr,zz,flags) bind(C)
 use iso_c_binding
 implicit none
 integer(c_int),value :: n
 real(c_double) :: rr(4,n),zz(4,n)
 integer(c_int) :: flags(n)
 logical,external :: opsplit_quad_is_concave
 integer :: i
 do i=1,n
 flags(i)=0
 if(opsplit_quad_is_concave(rr(:,i),zz(:,i))) flags(i)=1
 enddo
end subroutine
'''
    p=w/'check.f90';p.write_text((root/'native/opsplit_quad_is_concave.f90').read_text()+driver)
    subprocess.run(['gfortran','-O2','-fcheck=all','-fPIC','-shared',str(p),'-o',str(w/'check.so')],cwd=w,check=True,capture_output=True)
    lib=ctypes.CDLL(str(w/'check.so'));fn=lib.check
    arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS');out=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fn.argtypes=[ctypes.c_int,arr,arr,out]
    def check(path):
        with h5py.File(path) as f:xy=f['x'][:,0,0,:].T;v=f['vertex'][()].T-1
        q=xy[v];flags=np.empty(len(v),np.int32);fn(len(v),np.asfortranarray(q[...,0].T),np.asfortranarray(q[...,1].T),flags)
        return np.flatnonzero(flags)+1
    return check

B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
@pytest.mark.parametrize('path',['mumps_scaling8_20261104/control3/jorek000000.h5','scaling8_wall80_20261104/probe3/jorek000000.h5','scaling8_poloidal880_20261104/probe3/jorek000000.h5'])
def test_healthy_reference_meshes_do_not_trigger(concave,path):
    assert not len(concave(B/path))

def test_saved_balanced_mesh_identifies_all_eight_concave_cells(concave):
    np.testing.assert_array_equal(concave(B/'combined_balanced2_attachment_20261104/probe3/jorek_grid.h5'),[75630,75631,90053,90054,106612,106613,107248,107249])
