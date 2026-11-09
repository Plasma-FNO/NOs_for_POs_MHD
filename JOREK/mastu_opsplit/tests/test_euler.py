import numpy as np
import pytest
import torch
from mastu_opsplit.euler import NativeSplineSpace, EulerSplineModel, field_loss
from mastu_opsplit.restart import read_restart
from test_restart import rectangle
from test_learning import configuration


def test_projection_adjoint_and_diffusion_energy(tmp_path):
    rectangle(tmp_path/'mesh.h5')
    geometry,_,_=read_restart(tmp_path/'mesh.h5',quadrature_order=5)
    space=NativeSplineSpace(geometry)
    torch.manual_seed(8)
    x=torch.randn(1,16,2,dtype=torch.float64,requires_grad=True)
    torch.testing.assert_close(space.project(space.sample(x)),x,rtol=1e-9,atol=1e-9)
    for operation in (space.sample,space.laplace):
        # The maps are exactly linear; a larger finite-difference step reduces solve roundoff.
        assert torch.autograd.gradcheck(operation,(x,),eps=1e-3)
    points=torch.randn(1,25,2,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(space.project,(points,),eps=1e-3)
    q=x.detach().numpy()[0]
    lap=space.laplace(x).detach().numpy()[0]
    np.testing.assert_allclose(np.sum(q*(space.mass@lap)), -np.sum(q*(space.stiffness@q)),rtol=1e-10)
    advanced=q+1.8/space.spectral_bound*lap
    assert np.sum(advanced*(space.mass@advanced))<=np.sum(q*(space.mass@q))
    constant=torch.zeros_like(x);constant[:,::4,:]=1
    torch.testing.assert_close(space.laplace(constant),torch.zeros_like(x),atol=1e-9,rtol=0)


def test_all_families_train_through_native_rollout(tmp_path):
    torch.set_num_threads(1)
    rectangle(tmp_path/'mesh.h5')
    geometry,_,_=read_restart(tmp_path/'mesh.h5')
    space=NativeSplineSpace(geometry)
    for family in ('ar','node','opsplit'):
        torch.manual_seed(11)
        model=EulerSplineModel(family,space,configuration())
        x=torch.randn(1,16,7,dtype=torch.float64,requires_grad=True)
        parameters=torch.ones(1,1)
        y=model(model(x,parameters),parameters)
        field_loss(y,x*.8,space).backward()
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if family=='ar':
            with pytest.raises(ValueError,match='no substep'):model(x,parameters,substeps=2)
        else:
            y2=model(x,parameters,substeps=2)
            assert y2.shape==x.shape and torch.isfinite(y2).all()


def test_normalization_excludes_held_out_data(tmp_path):
    from mastu_opsplit.euler_training import normalize_training
    rectangle(tmp_path/'mesh.h5')
    geometry,_,_=read_restart(tmp_path/'mesh.h5')
    space=NativeSplineSpace(geometry)
    data=np.random.default_rng(4).normal(size=(2,8,16,7))
    mean,scale=normalize_training(data,space,4)
    data[:,4:]=1e20
    after=normalize_training(data,space,4)
    np.testing.assert_array_equal(mean,after[0]);np.testing.assert_array_equal(scale,after[1])


def test_trainer_runs_all_families_and_keeps_splits_separate(tmp_path,monkeypatch):
    from mastu_opsplit import euler_training as training
    rectangle(tmp_path/'mesh.h5')
    geometry,state,_=read_restart(tmp_path/'mesh.h5')
    rng=np.random.default_rng(3)
    trend=rng.normal(size=state.shape)*.001
    data=np.stack([np.stack([state+trend*(i+branch*.1) for i in range(49)]) for branch in range(2)])
    provenance={'files':[[{'path':str(tmp_path/'mesh.h5')}]],'reference_max_dynamic_error':0.}
    monkeypatch.setattr(training,'load_pilot',lambda config:(geometry,data,np.arange(49)*1e-8,provenance))
    config={'purpose':'chronological_diagnostic','steps':1,'seed':17,'validate_every':1,
            'learning_rate':.001,'threads':1,'architecture':{}}
    result=training.run_pilot(config,configuration(),tmp_path/'pilot')
    assert set(result)=={'ar','node','opsplit'}
    for family,metrics in result.items():
        assert metrics['best_step']==1 and metrics['independent_trajectory_test'] is False
        assert len(metrics['test_by_branch'])==2
        assert (tmp_path/'pilot'/family/'best.pt').is_file()
    import json
    recorded=json.loads((tmp_path/'pilot/provenance.json').read_text())
    assert recorded['split']=={'train':[0,24],'validation':[24,32],'test':[32,49]}
    assert recorded['linear_branch']=='lspline_analytic'
    assert all('persistence_by_branch' not in m for m in result.values())


def test_lspline_derivatives_match_upstream_synthesis_and_physical_differences():
    from mastu_opsplit.learning import FieldSplineOperator
    from mastu_opsplit.spline_derivatives import SplineDerivatives
    import importlib
    torch.set_num_threads(1)
    bounds=torch.tensor([[1.,3.],[-2.,4.]],dtype=torch.float64)
    operator=FieldSplineOperator(2,2,bounds,configuration(),condition_transport=False).double()
    backbone=operator.backbone
    # Include truncated boundary stencils, knots, and interior points.
    points=torch.tensor([[1.01,-1.99],[1.13,3.92],[1.67,.37],[2.,1.],[2.99,3.99]],dtype=torch.float64)
    coefficients=torch.randn(2,2,64,dtype=torch.float64,requires_grad=True)
    derivative=SplineDerivatives(points,bounds,backbone._get_grid_axes())
    upstream=importlib.import_module('Models.learnable_spline_fno')
    def synthesis(query):
        coords=operator.normalized_coordinates(query,2)
        return upstream._decode_gather(backbone._analytic_stencil(coords),coefficients).transpose(1,2)
    torch.testing.assert_close(derivative(coefficients),synthesis(points),rtol=1e-12,atol=1e-12)
    for d in range(2):
        shift=torch.zeros_like(points);shift[:,d]=1e-4
        first=(synthesis(points+shift)-synthesis(points-shift))/(2e-4)
        second=(synthesis(points+shift)-2*synthesis(points)+synthesis(points-shift))/1e-8
        order=[0,0];order[d]=1
        torch.testing.assert_close(derivative(coefficients,order),first,rtol=2e-6,atol=2e-6)
        order[d]=2
        torch.testing.assert_close(derivative(coefficients,order),second,rtol=2e-5,atol=2e-5)
    constant=torch.ones_like(coefficients)
    torch.testing.assert_close(derivative.laplace(constant),torch.zeros(2,5,2,dtype=torch.float64),atol=1e-12,rtol=0)
    assert torch.autograd.gradcheck(derivative.laplace,(coefficients,),eps=1e-5)


def test_lspline_linear_branch_is_linear_and_trains_scatter(tmp_path,monkeypatch):
    rectangle(tmp_path/'mesh.h5')
    geometry,_,_=read_restart(tmp_path/'mesh.h5')
    space=NativeSplineSpace(geometry)
    model=EulerSplineModel('opsplit',space,configuration())
    a=torch.randn(1,16,7,dtype=torch.float64,requires_grad=True)
    b=torch.randn_like(a)
    # The native stiffness and nonlinear FNO must not be used in this branch.
    def forbidden(*args,**kwargs):raise AssertionError('Wrong derivative path')
    monkeypatch.setattr(space,'laplace',forbidden)
    monkeypatch.setattr(model.operator.backbone.fno_blocks,'forward',forbidden)
    torch.testing.assert_close(model.linear_rhs(2*a-3*b),2*model.linear_rhs(a)-3*model.linear_rhs(b),rtol=3e-4,atol=1e-5)
    field_loss(model.linear_rhs(a),a*.1,space).backward()
    assert torch.isfinite(a.grad).all()
    gradients=[p.grad for p in model.operator.backbone.scatter_kernels.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients)>0


def test_constraint_audit_detects_inconsistent_auxiliary_fields(tmp_path):
    import importlib.util
    from pathlib import Path
    from dataclasses import replace
    path=Path(__file__).parents[1]/'scripts/assess_euler_pilot.py'
    spec=importlib.util.spec_from_file_location('euler_audit',path)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    rectangle(tmp_path/'mesh.h5')
    geometry,_,_=read_restart(tmp_path/'mesh.h5',quadrature_order=8)
    labels=np.ones(16,dtype=int);labels[3::4]=0
    geometry=replace(geometry,boundary_labels=labels)
    q=np.zeros((16,7));radius=np.array([1.,2.,2.,1.])
    q[::4,0]=radius**3;q[1::4,0]=3*radius**2
    q[::4,1]=radius**2;q[1::4,1]=2*radius
    q[::4,2]=3*radius;q[1::4,2]=3;q[::4,3]=4
    assert max(audit.constraints(q,geometry).values())<1e-12
    q[:,2:4]*=2
    assert min(audit.constraints(q,geometry).values())>.1
