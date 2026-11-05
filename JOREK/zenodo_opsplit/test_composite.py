import json
import pytest
import torch
from composite import InteriorDiffusion, CompositeSplit, positive_increment
from test_comparison import fixture
from train_comparison import Experiment, audit, train, require_audit, require_composite_stability


def grid(n=16):
    return torch.linspace(9.501,10.499,n,dtype=torch.float64),torch.linspace(-.499,.499,n,dtype=torch.float64)


@pytest.mark.parametrize('sign',[1,-1])
def test_implicit_solve_matches_independent_dense_system_and_gradient(sign):
    r,z=grid();s=4;dt=5.;c=1e-5
    step=InteriorDiffusion(r,z,c,dt,sign)
    x=torch.rand(2,len(z),len(r),dtype=torch.float64,requires_grad=True)
    y=step(x)
    nr,nz=len(r)-2*s,len(z)-2*s;dr,dz=r[1]-r[0],z[1]-z[0]
    a=torch.eye(nr*nz,dtype=torch.float64);b=x[:,s:-s,s:-s].clone()
    for j in range(nz):
        for i in range(nr):
            index=j*nr+i;radius=r[s+i]
            weights=[(0,-1,1/dr**2-sign/(2*dr*radius)),(0,1,1/dr**2+sign/(2*dr*radius)),(-1,0,1/dz**2),(1,0,1/dz**2)]
            a[index,index]+=dt*c*(2/dr**2+2/dz**2)
            for dj,di,w in weights:
                jj,ii=j+dj,i+di
                if 0<=jj<nz and 0<=ii<nr:a[index,jj*nr+ii]-=dt*c*w
                else:b[:,j,i]+=dt*c*w*x[:,s+jj,s+ii]
    expected=torch.linalg.solve(a,b.flatten(1).T).T.reshape(2,nz,nr)
    torch.testing.assert_close(y[:,s:-s,s:-s],expected,rtol=1e-11,atol=1e-12)
    actual_grad=torch.autograd.grad(y[:,s:-s,s:-s].square().sum(),x,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected.square().sum(),x)[0]
    torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-10,atol=1e-11)
    assert y.min()>0
    mask=torch.ones_like(y,dtype=torch.bool);mask[:,s:-s,s:-s]=False
    torch.testing.assert_close(y[mask],x[mask],rtol=0,atol=0)


def test_stiff_diffusion_damps_checkerboard_and_preserves_constant():
    r,z=grid(100);step=InteriorDiffusion(r,z,1e-5,5.,-1)
    constant=torch.ones(1,100,100,dtype=torch.float64)
    torch.testing.assert_close(step(constant),constant,rtol=1e-11,atol=1e-11)
    checker=torch.zeros_like(constant);ii,jj=torch.meshgrid(torch.arange(92),torch.arange(92),indexing='ij')
    checker[:,4:-4,4:-4]=(-1.)**(ii+jj)
    assert step(checker).norm()<.3*checker.norm()


@pytest.mark.parametrize('channels',[4,6])
def test_auxiliary_defect_preserved_without_neural_rates(channels):
    r,z=grid();model=CompositeSplit(r,z,torch.zeros(channels),torch.ones(channels),channels,width=4,modes=3,layers=2)
    torch.manual_seed(7);x=.01*torch.randn(2,channels,16,16,dtype=torch.float64);x[:,:2]+=1
    y=model(x);a=x.double()*model.native_factor[None,:,None,None];b=y.double()*model.native_factor[None,:,None,None]
    for i,before,after in zip([3,5],model.constraints(a),model.constraints(b)):
        torch.testing.assert_close(b[:,i]-after,a[:,i]-before,atol=2e-8,rtol=1e-5)
    torch.testing.assert_close(y[:,:,0],x[:,:,0],atol=1e-7,rtol=1e-6)
    y.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_transfer_keeps_source_interface_and_target_linear_physics():
    r,z=grid();source=CompositeSplit(r,z,torch.zeros(4),torch.ones(4),4,width=4,modes=3,layers=2)
    target=CompositeSplit(r,z,torch.ones(6),torch.full((6,),2.),6,width=4,modes=3,layers=2)
    target.transfer(source,freeze=True)
    torch.testing.assert_close(target.fluid_mean,source.fluid_mean)
    torch.testing.assert_close(target.fluid_scale,source.fluid_scale)
    assert not any(p.requires_grad for p in target.fluid.parameters())
    assert all(p.requires_grad for p in target.magnetic.parameters())
    assert target.resistive.step>0
    clone=CompositeSplit(r,z,torch.ones(6),torch.full((6,),2.),6,width=4,modes=3,layers=2)
    clone.load_state_dict(target.state_dict())
    torch.testing.assert_close(clone(torch.ones(1,6,16,16)),target(torch.ones(1,6,16,16)))


def test_matched_trainer_smoke_audit_and_resume(tmp_path):
    data=fixture(tmp_path)
    ex=Experiment(data,'composite','cpu')
    windows=ex.meta['windows'][:2];actual=ex.batch_loss(windows)
    expected=sum(float(ex.objective('train',i,s,2)) for i,s in windows)
    assert actual==pytest.approx(expected,rel=1e-5)
    assert audit(ex,tmp_path/'audit.json')
    require_audit(ex,tmp_path/'audit.json')
    train(Experiment(data,'composite','cpu'),tmp_path/'smoke',smoke=True)
    train(Experiment(data,'composite','cpu'),tmp_path/'smoke',smoke=True,resume=True)
    saved=torch.load(tmp_path/'smoke/last.pt',weights_only=False)
    assert saved['model_config']['formulation']=='composite-imex-auxiliary-defect'


@pytest.mark.parametrize('case,stability_epochs',[('es',None),('em',None),('em',110)])
def test_launcher_gates_and_preserves_shared_data(tmp_path,monkeypatch,case,stability_epochs):
    import orchestrate_composite
    data=fixture(tmp_path,case=case,epochs=111)
    if stability_epochs is not None:
        meta=json.loads((data/'comparison.json').read_text())
        meta['protocol']['composite_stability_epochs']=stability_epochs
        (data/'comparison.json').write_text(json.dumps(meta))
    before=(data/'comparison.json').read_bytes()
    output=tmp_path/'campaign'
    monkeypatch.setattr('sys.argv',['orchestrate_composite.py','--data',str(data),'--output',str(output)])
    orchestrate_composite.main()
    jobs=json.loads((output/'jobs.json').read_text())
    assert [(j['stage'],j['dependency']) for j in jobs]==[('smoke',None),('audit','smoke'),('probe','audit'),('train','probe')]
    assert '--audit-report' in (output/'train.sbatch').read_text()
    assert '--resume' in (output/'train.sbatch').read_text()
    assert (data/'comparison.json').read_bytes()==before
    assert (output/'code/composite.py').exists()
    assert '--stop-after-epoch '+str(stability_epochs or (26 if case=='em' else 12)) in (output/'probe.sbatch').read_text()
    assert 'stability.json' in (output/'train.sbatch').read_text()


def test_positive_updates_preserve_positivity_and_have_gradients():
    r,z=grid();model=CompositeSplit(r,z,torch.zeros(4),torch.ones(4),4,width=4,modes=3,layers=2)
    with torch.no_grad():model.fluid.projection.fcs[-1].bias[:2].fill_(-4.)
    x=torch.ones(2,4,16,16);x[:,2:]=0
    y=model(x)
    expected=positive_increment(torch.tensor(1.),torch.tensor(-4.),torch.tensor(1.))
    torch.testing.assert_close(y[:,:2],torch.full_like(y[:,:2],expected),rtol=1e-5,atol=1e-8)
    y[:,:2].sum().backward()
    assert torch.all(model.fluid.projection.fcs[-1].bias.grad[:2]>0)
    x[:,0]=0
    with pytest.raises(FloatingPointError,match='input'):model(x)


def test_positive_coordinate_avoids_exponential_feedback_and_bounds_rate_gradient():
    value=torch.tensor([1e-12,1.,2303.,4e9],dtype=torch.float64)
    reference=torch.tensor(.3,dtype=torch.float64)
    torch.testing.assert_close(positive_increment(value,torch.zeros_like(value),reference),value,rtol=1e-12,atol=0)
    rate=torch.tensor([1.,88.,182418.,286153.],dtype=torch.float64,requires_grad=True)
    result=positive_increment(value,rate,reference)
    assert torch.isfinite(result).all() and torch.all(result>0)
    # Softplus is 1-Lipschitz; even the observed runaway rates give bounded increments.
    assert torch.all(result-value<=reference*rate+1e-6)
    gradient=torch.autograd.grad(result.sum(),rate)[0]
    assert torch.all(gradient>0) and torch.all(gradient<=reference)


def test_validation_failure_saves_exact_weights_and_trajectory(tmp_path):
    from unittest.mock import patch
    data=fixture(tmp_path);ex=Experiment(data,'composite','cpu')
    original=ex.model.forward
    def fail_during_validation(x):
        if not ex.model.training:raise FloatingPointError('test failure')
        return original(x)
    with patch.object(ex.model,'forward',side_effect=fail_during_validation):
        with pytest.raises(RuntimeError,match='val trajectory val, rollout step 1'):
            train(ex,tmp_path/'failed',smoke=True)
    report=json.loads((tmp_path/'failed/failure.json').read_text())
    assert report['stage']=='validation' and report['epoch']==0
    saved=torch.load(tmp_path/'failed/failed-validation.pt',weights_only=False)
    for name,value in ex.model.state_dict().items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(saved['state_dict'][name],value)


@pytest.mark.parametrize('case,epochs',[('es',12),('em',26),('em',28)])
def test_bounded_optimizer_check_resumes_same_campaign(tmp_path,case,epochs):
    data=fixture(tmp_path,case=case,epochs=epochs+1);output=tmp_path/'train'
    if epochs==28:
        meta=json.loads((data/'comparison.json').read_text())
        meta['protocol']['composite_stability_epochs']=epochs
        (data/'comparison.json').write_text(json.dumps(meta))
    ex=Experiment(data,'composite','cpu')
    train(ex,output,stop_after_epoch=epochs)
    report=json.loads((output/'stability.json').read_text())
    assert report['passed'] and report['completed_epochs']==epochs
    assert report['validation_horizon']==6 and report['validation_trajectories']==1
    assert not (output/'test_metrics.json').exists()
    require_composite_stability(ex,output)
    changed=dict(report,completed_epochs=epochs-1)
    (output/'stability.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='checked optimizer epochs'):require_composite_stability(ex,output)
    changed=dict(report,checkpoint_sha256='changed')
    (output/'stability.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='Changed checked checkpoint'):require_composite_stability(ex,output)
    (output/'stability.json').write_text(json.dumps(report))
    train(Experiment(data,'composite','cpu'),output,resume=True,train_only=True)
    history=[json.loads(row) for row in (output/'history.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in history]==list(range(epochs+1))
    require_composite_stability(ex,output)


@pytest.mark.parametrize('channels',[4,6])
def test_carried_physical_temperature_survives_affine_cancellation_and_gradients(channels):
    r,z=grid();mean=torch.ones(channels,dtype=torch.float64);scale=torch.ones_like(mean)
    model=CompositeSplit(r,z,mean,scale,channels,width=4,modes=3,layers=2,substeps=2)
    with torch.no_grad():model.fluid.projection.fcs[-1].bias[1]=-1.
    initial=torch.ones(1,channels,16,16,dtype=torch.float64)
    initial[:,1]=1e-20;initial[:,2:]=0;initial.requires_grad_()
    # The old recurrent representation loses this field before the first step.
    assert torch.all(((initial-mean[None,:,None,None])/scale[None,:,None,None]*scale[None,:,None,None]+mean[None,:,None,None])[:,1]==0)
    state=initial
    for _ in range(49):state=model(state)
    assert state.dtype==torch.float64 and torch.all(state[:,1]>0)
    torch.testing.assert_close(state[:,1],initial[:,1]*torch.exp(torch.tensor(-49.,dtype=torch.float64)),rtol=1e-11,atol=0)
    gradient=torch.autograd.grad(state[:,1].sum(),initial)[0]
    assert torch.isfinite(gradient).all() and torch.all(gradient[:,1]>0)
