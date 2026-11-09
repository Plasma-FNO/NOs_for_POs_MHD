from dataclasses import replace
import numpy as np
import pytest
import torch

from mastu_opsplit.euler import NativeSplineSpace, EulerSplineModel, field_loss
from mastu_opsplit.physical_guard import EllipticPair, PhysicalGuard, QuadratureSampler, bernstein_basis
from mastu_opsplit.restart import read_restart
from test_restart import rectangle
from test_learning import configuration


def fixture(tmp_path):
    rectangle(tmp_path/'mesh.h5')
    g,q,_=read_restart(tmp_path/'mesh.h5')
    q8,_,_=read_restart(tmp_path/'mesh.h5',quadrature_order=8)
    labels=np.ones(16,dtype=int);labels[3::4]=0
    return replace(g,boundary_labels=labels),q8,q


def test_elliptic_closure_residual_and_exact_adjoint(tmp_path):
    g,_,_=fixture(tmp_path)
    for current in (True,False):
        pair=EllipticPair(g,current)
        x=torch.randn(2,16,2,dtype=torch.float64,requires_grad=True)
        out=pair(x)
        p,a=out.detach().numpy()[...,0].T,out.detach().numpy()[...,1].T
        np.testing.assert_allclose(pair.stiff@p+pair.mass@a,0,atol=1e-12)
        torch.testing.assert_close(out[:,pair.boundary,0],x[:,pair.boundary,0])
        torch.testing.assert_close(out[...,1],x[...,1])
        assert torch.autograd.gradcheck(pair,(x,),eps=1e-5)


def test_coefficient_limiter_and_elliptic_closure_are_differentiable(tmp_path):
    g,q8,mean=fixture(tmp_path)
    guard=PhysicalGuard(g,q8,mean,np.ones(7))
    previous=torch.zeros(1,16,7,dtype=torch.float64)
    proposal=torch.randn_like(previous)*.1
    proposal[:,::4,4:6]=-10
    proposal.requires_grad_()
    out=guard(previous,proposal)
    physical=out*guard.scale+guard.mean
    assert (guard.sampler(physical[...,4:6])>guard.floor).all()
    assert guard.statistics['limited_field_steps']==2
    assert guard.statistics['minimum_alpha']<1
    for pair,pi,ai in ((guard.current,0,2),(guard.vorticity,1,3)):
        p=physical.detach().numpy()[0]
        np.testing.assert_allclose(pair.stiff@p[:,pi]+pair.mass@p[:,ai],0,atol=1e-12)
    torch.testing.assert_close(physical[:,guard.fixed_boundary_values,:2],guard.mean[None,guard.fixed_boundary_values,:2])
    (out.square().mean()+.001*guard.last_penalty).backward()
    assert torch.isfinite(proposal.grad).all() and proposal.grad.abs().sum()>0
    # No correction to an already valid physical state; no future truth required.
    torch.testing.assert_close(guard(previous,previous),previous,atol=1e-12,rtol=0)


def test_guarded_zero_initialized_split_matches_node_and_all_families_train(tmp_path):
    torch.set_num_threads(1)
    g,q8,mean=fixture(tmp_path)
    space=NativeSplineSpace(g)
    guard=PhysicalGuard(g,q8,mean,np.ones(7))
    models={}
    for family in ('ar','node','opsplit'):
        torch.manual_seed(11)
        model=EulerSplineModel(family,space,configuration(),physical_guard=guard,linear_initialization='zero')
        state=torch.zeros(1,16,7,dtype=torch.float64,requires_grad=True)
        out=model(state,torch.ones(1,1))
        field_loss(out,state+.01,space).backward()
        assert torch.isfinite(out).all() and torch.isfinite(state.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        models[family]=out.detach()
    torch.testing.assert_close(models['opsplit'],models['node'],rtol=0,atol=0)


@pytest.mark.parametrize('linear_branch',['lspline_analytic','lspline_continuity'])
def test_guarded_trainer_exports_positive_constrained_rollouts(tmp_path,monkeypatch,linear_branch):
    from mastu_opsplit import euler_training as training
    g,q8,mean=fixture(tmp_path)
    rng=np.random.default_rng(17)
    trend=rng.normal(size=mean.shape)*1e-4
    trend[::4,:2]=0
    data=np.stack([np.stack([mean+trend*(i+.2*branch) for i in range(49)]) for branch in range(2)])
    provenance={'files':[[{'path':str(tmp_path/'mesh.h5')}]],'reference_max_dynamic_error':0.}
    monkeypatch.setattr(training,'load_pilot',lambda config:(g,data,np.arange(49)*1e-8,provenance))
    result=training.run_pilot({'purpose':'chronological_diagnostic','steps':1,'validate_every':1,
                               'seed':17,'threads':1,'learning_rate':.001,'architecture':{},
                               'physical_guard':True,'linear_initialization':'zero',
                               'linear_branch':linear_branch,'gamma':5/3},configuration(),tmp_path/'pilot')
    for metrics in result.values():
        for case in metrics['test_by_branch']:
            assert case['q8_positive']
            assert max(case['max_rollout_current_vorticity_defect'])<1e-8
            assert case['guard_statistics']['steps']==16
            assert case['positive_in_limiter_basis']


def test_bernstein_limiter_catches_negativity_missed_by_gauss_points(tmp_path):
    g,q8,mean=fixture(tmp_path)
    guard=PhysicalGuard(g,q8,mean,np.ones(7))
    physical=mean.copy()
    physical[:,4]=0
    physical[::4,4]=np.array([-.01,.99,.99,-.01])
    physical[1::4,4]=1
    raw=torch.from_numpy(physical)[None]
    assert QuadratureSampler(q8)(raw[...,4:6]).min()>0
    assert guard.sampler(raw[...,4:6]).min()<0
    out=guard(torch.zeros_like(raw),raw-guard.mean)+guard.mean
    assert guard.sampler(out[...,4:6]).min()>0
    dense,_,_=read_restart(tmp_path/'mesh.h5',quadrature_order=16)
    assert QuadratureSampler(dense)(out[...,4:6]).min()>0
    assert out[:,::4,4:6].min()>0


def test_subdivided_bernstein_map_reconstructs_native_polynomial(tmp_path):
    from mastu_opsplit.restart import cubic_basis
    g,_,_=fixture(tmp_path)
    basis=bernstein_basis(g)
    rng=np.random.default_rng(9);q=rng.normal(size=16)
    controls=(basis[0]@q).reshape(4,4,4)
    sizes=np.array([[1,1/3,1/3,1/9],[1,-1/3,1/3,-1/9],
                    [1,-1/3,-1/3,1/9],[1,1/3,-1/3,-1/9]])
    for i in range(2):
        for j in range(2):
            s,t=rng.random(2)
            def b(x):return np.array([(1-x)**3,3*x*(1-x)**2,3*x*x*(1-x),x**3])
            expected=((cubic_basis((i+s)/2,(j+t)/2)[0]*sizes).reshape(16)@q)
            actual=b(s)@controls[2*i+j]@b(t)
            np.testing.assert_allclose(actual,expected,atol=1e-13)
