"""The X-point-adjacent repair must preserve healthy grids and boundary jets."""
import numpy as np
import pytest
from test_corner_handle import repair,load,minimum

pytestmark=pytest.mark.parametrize('repair',['opsplit_limit_xpoint_handle'],indirect=True)

def test_real_divertor_fold_and_all_affected_cells(repair):
    x,v,s=load('combined_divertor160_20261104/probe3')
    out=repair(x,v,s)
    changed=np.flatnonzero((out!=s).any(axis=(1,2)))
    assert len(changed)==2
    assert minimum(x,v[changed],s[changed])<0
    modified_nodes=set()
    for e in changed:
        for iv in np.flatnonzero((s[e]!=out[e]).any(axis=1)):
            modified_nodes.add(v[e,iv])
            assert v[e,iv]>=8
            # Exactly the derivative and mixed factors change, by the first trial.
            nz=s[e,iv]!=0
            assert np.isclose(np.min(out[e,iv,nz]/s[e,iv,nz]),.8)
    assert len(modified_nodes)==1
    n=next(iter(modified_nodes));incident=np.flatnonzero((v==n).any(axis=1))
    assert len(incident)==4
    assert minimum(x,v[incident],out[incident])>0
    np.testing.assert_array_equal(out[:,:,0],s[:,:,0])
    # Both representations of the shared handle change together; no boundary edge changes.
    for e in changed:
        iv=int(np.flatnonzero(v[e]==n)[0]);d=int(np.flatnonzero(out[e,iv,1:3]!=s[e,iv,1:3])[0])+1
        side=iv if iv%2==d-1 else (iv-1)%4
        edge=v[e,[side,(side+1)%4]]
        assert np.count_nonzero(np.isin(v,edge).sum(axis=1)==2)==2
    np.testing.assert_array_equal(repair(x,v,out),out)

@pytest.mark.parametrize('case',['mumps_scaling8_20261104/control3','scaling8_poloidal880_20261104/probe3','scaling8_wall80_20261104/probe3'])
def test_healthy_reference_remains_exact(repair,case):
    x,v,s=load(case)
    np.testing.assert_array_equal(repair(x,v,s),s)

def test_failed_trial_restores_sizes(repair):
    x,v,s=load('combined_divertor160_20261104/probe3');x[:]=0
    np.testing.assert_array_equal(repair(x,v,s),s)
