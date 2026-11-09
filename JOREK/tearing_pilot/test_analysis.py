"""Checks that diagnostic norms see physical fields and toroidal quadrature."""
import h5py
import numpy as np
from pathlib import Path
from analyze import mesh, values, norm


def test_constant_and_toroidal_field():
    path=Path(__file__).parent/'snapshot/begin.h5'
    with h5py.File(path) as original, h5py.File('test-memory.h5','w',driver='core',backing_store=False) as f:
        for key in original:original.copy(key,f)
        g=mesh(f)
        v=np.zeros_like(f['values'][()]);v[:,0,0,:]=2.;v[:,0,1,:]=3.;v[:,0,2,:]=4.
        f['values'][...]=v
        physical=values(f,g)
    np.testing.assert_allclose(physical[:,0],2.,atol=1e-13)
    np.testing.assert_allclose(physical[:,1],3.,atol=1e-13)
    np.testing.assert_allclose(physical[:,2],4.,atol=1e-13)
    np.testing.assert_allclose(norm(physical,g['volume']),np.sqrt(4.+(9.+16.)/2))
    assert abs(g['volume'].sum()-2*np.pi**2*10)<.1


def test_nodal_basis_reconstruction():
    from mastu_opsplit.restart import cubic_basis
    h,_,_=cubic_basis(np.array([0.,1.,1.,0.]),np.array([0.,0.,1.,1.]))
    np.testing.assert_array_equal(h[:,:,0],np.eye(4))
    np.testing.assert_array_equal(h[:,:,1:],0.)


def test_reference_physical_point_matching():
    from check_reference import match_physical_points
    with h5py.File(Path(__file__).parent/'snapshot/begin.h5') as f:
        original=mesh(f,4)
    shifted={**original,'x':original['x'].copy()}
    shifted['x'][0,0,0,:]+=1e-7
    matched,error=match_physical_points(shifted,original['rz'])
    assert error<1e-12
    np.testing.assert_allclose(matched['rz'],original['rz'],rtol=0,atol=1e-12)


def test_convergence_detects_dynamic_error_behind_large_background():
    from analyze_growth import compare
    a=np.zeros((4,6,3,1,1));a[:,:,0]=1000000
    a[:,:,1,0,0]=np.arange(1,5)[:,None]
    b=a.copy();b[:,:,1]*=1.1
    result=compare(a,b,np.ones((1,1)))
    for field in result.values():
        assert field['full_field_relative_l2_error']<1e-6
        assert field['n1_relative_l2_error']>.09
        assert field['error_over_temporal_std']>.2
