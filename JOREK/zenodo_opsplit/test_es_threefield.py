import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import es_threefield as es
import train_comparison
from test_comparison import fixture
from comparison import digest
from composite import CompositeSplit


def small(method='opsplit',opsplit_update='additive'):
    r=torch.linspace(9.501,10.499,16,dtype=torch.float64)
    z=torch.linspace(-.499,.499,16,dtype=torch.float64)
    mean=torch.tensor([1.3,26.8,.107]);scale=torch.tensor([1.298,26.667,1.612])
    return es.ThreeFieldModel(r,z,mean,scale,method,width=4,modes=3,layers=2,opsplit_update=opsplit_update)


@pytest.mark.parametrize('method',['ar','node','opsplit'])
def test_three_inputs_three_outputs_and_no_native_state(method):
    model=small(method);x=torch.rand(2,3,16,16,dtype=torch.float64)+.1
    with patch('native.decode_release',side_effect=AssertionError('native conversion called')):
        y=model(x)
    assert y.shape==x.shape and y.dtype==torch.float64
    assert model.config['native_state_conversion'] is False
    with pytest.raises(ValueError,match='batch,3'):
        model(torch.cat((x,x[:,:1]),1))


def test_baseline_rules_in_preconditioned_coordinates():
    ar=small('ar');node=small('node');node.load_state_dict(ar.state_dict())
    x=torch.rand(2,3,16,16,dtype=torch.float64)+.1
    torch.testing.assert_close(node(x),x+ar(x)-ar.mean[None,:,None,None],rtol=1e-12,atol=1e-12)
    z=ar.normalize(x)
    torch.testing.assert_close(ar.normalize(ar(x)),ar.network(z.float()).double(),rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('diffusion',[False,True])
def test_additive_opsplit_matches_node_then_density_diffusion_and_gradients(diffusion):
    node=small('node');split=small()
    split.network.load_state_dict(node.network.state_dict())
    if not diffusion:split.particle.step=0.
    x=(torch.randn(2,3,16,16,dtype=torch.float64)*.2).requires_grad_()
    actual=split(x);candidate=node(x)
    expected=torch.stack((split.particle(candidate[:,0]),candidate[:,1],candidate[:,2]),1)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    ga=torch.autograd.grad(actual.square().mean(),[x,*split.network.parameters()],retain_graph=True)
    ge=torch.autograd.grad(expected.square().mean(),[x,*node.network.parameters()])
    for a,e in zip(ga,ge):torch.testing.assert_close(a,e,rtol=0,atol=0)


def test_additive_has_no_hidden_positivity_projection_and_softplus_remains_available():
    additive=small();legacy=small(opsplit_update='softplus')
    x=torch.ones(1,3,16,16,dtype=torch.float64)
    with torch.no_grad():
        additive.network.projection.fcs[-1].bias.fill_(-2.)
    legacy.network.load_state_dict(additive.network.state_dict())
    assert (additive(x)[:,:2]<0).all()
    assert (legacy(x)[:,:2]>0).all()
    negative=-x
    assert torch.isfinite(additive(negative)).all()
    with pytest.raises(FloatingPointError,match='Nonpositive OpsSplit input'):legacy(negative)


def test_update_map_is_part_of_checkpoint_identity(tmp_path):
    data=fixture(tmp_path,epochs=1)
    old=es.ThreeFieldExperiment(data,'opsplit','cpu',opsplit_update='softplus')
    new=es.ThreeFieldExperiment(data,'opsplit','cpu')
    assert old.cfg['opsplit_update']=='softplus' and new.cfg['opsplit_update']=='additive'
    assert old.identity!=new.identity
    train_comparison.train(old,tmp_path/'legacy',train_only=True)
    with pytest.raises(ValueError,match='identity|Identity|provenance|Changed'):
        train_comparison.train(new,tmp_path/'legacy',resume=True,train_only=True)


def test_launcher_can_isolate_two_additive_opsplit_runs(tmp_path):
    from launch_es_threefield import prepare
    data=fixture(tmp_path)
    root=tmp_path/'campaign'
    jobs=prepare(Path(es.__file__).parent,data,root,100.,{},arms=['opsplit','opsplit-vorticity'])
    assert len(jobs)==10
    setup=json.loads((root/'setup.json').read_text())
    assert setup['arms']==['opsplit','opsplit-vorticity']
    assert setup['protocol']['opsplit_update']=='additive'
    for job in jobs:
        assert '--opsplit-update additive' in Path(job['script']).read_text()
        for dependency in job['dependencies']:
            assert dependency in [prior['stage'] for prior in jobs[:jobs.index(job)]]


def test_laplacian_analytic_and_existing_constraint_parity():
    m=small();zz,rr=torch.meshgrid(m.z,m.r,indexing='ij')
    actual=es.laplacian(rr**4+zz**4,m.r,m.z)
    expected=(16*rr**2+12*zz**2)[4:-4,4:-4]
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-8)
    ref=CompositeSplit(m.r,m.z,torch.cat((m.mean,torch.zeros(1))),torch.cat((m.scale,torch.ones(1))),width=4,modes=3,layers=2)
    x=torch.rand(2,4,16,16,dtype=torch.float64)+.1
    q=x*ref.native_factor[None,:,None,None]
    torch.testing.assert_close(m.omega(x[:,:3]),ref.constraints(q)[0][:,4:-4,4:-4],rtol=1e-11,atol=1e-12)


def test_omega_gradient_and_no_boundary_placeholder_loss():
    m=small();x=torch.rand(2,3,16,16,dtype=torch.float64)+.1
    target=torch.randn(2,8,8,dtype=torch.float64)
    out=m.omega(m(x));assert out.shape==target.shape
    (out-target).square().mean().backward()
    gradient=m.network.projection.fcs[-1].weight.grad
    assert gradient[2].abs().max()>0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    phi=torch.randn(1,16,16,dtype=torch.float64,requires_grad=True);direction=torch.randn_like(phi)
    def objective(p):return es.laplacian(p,m.r,m.z).square().mean()
    exact=(torch.autograd.grad(objective(phi),phi)[0]*direction).sum()
    epsilon=1e-6
    approx=(objective(phi.detach()+epsilon*direction)-objective(phi.detach()-epsilon*direction))/(2*epsilon)
    torch.testing.assert_close(exact,approx,rtol=1e-7,atol=1e-5)


def test_zero_heads_preserve_primary_physics_and_tiny_temperature():
    m=small();x=torch.ones(1,3,16,16,dtype=torch.float64);x[:,1]=1e-20;x[:,2]=.02
    y=m(x)
    torch.testing.assert_close(y,x,rtol=1e-11,atol=1e-25)
    assert torch.all(y[:,:2]>0)
    assert m.particle.step==es.DIFFUSION_PER_FRAME


@pytest.mark.parametrize('arm',es.ARMS)
def test_cached_pipeline_uses_only_primary_inputs_and_no_native_decoder(tmp_path,arm):
    data=fixture(tmp_path)
    with patch('native.decode_release',side_effect=AssertionError('native decode called')):
        ex=es.ThreeFieldExperiment(data,arm,'cpu')
        source=es.load_release(ex.meta['entries']['test'][0],3)
        p,a=ex.losses(ex.values['train']['exported'][:1,:3])
    assert source.shape[1]==4 and ex.model.config['channels']==3
    assert torch.isfinite(p+a)
    assert ex.weight==(100. if arm=='opsplit-vorticity' else 0.)
    changed=ex.values['train']['exported'][:1,:3].clone();changed[:,:,3]+=100
    if not ex.weight:
        p2,a2=ex.losses(changed)
        torch.testing.assert_close(p,p2,rtol=0,atol=0);assert a2==0


def test_weighted_batch_gradients_and_shared_validation_objective(tmp_path):
    data=fixture(tmp_path)
    ex=es.ThreeFieldExperiment(data,'opsplit-vorticity','cpu',omega_weight=3.)
    windows=ex.meta['windows'][:2]
    returned=ex.batch_loss(windows)
    actual=[p.grad.clone() if p.grad is not None else None for p in ex.model.parameters()]
    ex.model.zero_grad(set_to_none=True)
    frames=torch.stack([ex.values['train']['exported'][i,s:s+3] for i,s in windows])
    primary,aux=ex.losses(frames);(primary+3*aux).backward()
    assert returned==pytest.approx(float(primary),rel=1e-7)
    for a,p in zip(actual,ex.model.parameters()):
        if a is not None:torch.testing.assert_close(a,p.grad,rtol=1e-7,atol=1e-8)
    plain=es.ThreeFieldExperiment(data,'opsplit','cpu')
    assert es.initialization_hash(plain.model)==es.initialization_hash(ex.model)
    assert plain.validation()[0]==pytest.approx(ex.validation()[0],rel=1e-12)


def test_exact_resume_and_evaluation(tmp_path,monkeypatch):
    data=fixture(tmp_path,epochs=2)
    train_comparison.train(es.ThreeFieldExperiment(data,'node','cpu'),tmp_path/'continuous',train_only=True)
    save=train_comparison.save_checkpoint
    def interrupted(path,payload):
        save(path,payload)
        if path.name=='last.pt' and payload['next_epoch']==0:raise InterruptedError('test allocation end')
    with monkeypatch.context() as m:
        m.setattr(train_comparison,'save_checkpoint',interrupted)
        with pytest.raises(InterruptedError):
            train_comparison.train(es.ThreeFieldExperiment(data,'node','cpu'),tmp_path/'resumed',train_only=True)
    ex=es.ThreeFieldExperiment(data,'node','cpu')
    train_comparison.train(ex,tmp_path/'resumed',resume=True,train_only=True)
    first=torch.load(tmp_path/'continuous/last.pt',weights_only=False)
    second=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    for key,value in first['state_dict'].items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(value,second['state_dict'][key],rtol=0,atol=0)
    es.evaluate(ex,tmp_path/'evaluation.json')
    result=json.loads((tmp_path/'evaluation.json').read_text())
    assert result['fields']==['rho','T','potential'] and result['omega_crop']==4 and result['complete']
    assert len(result['horizons']['7']['mean_nrmse'])==3
    assert result['horizons']['7']['mean_derived_omega_r_weighted_relative_l2'] is not None
    es.evaluate(ex,tmp_path/'evaluation.json')


def test_failure_and_partial_resume_are_not_survivor_averages(tmp_path):
    data=fixture(tmp_path);ex=es.ThreeFieldExperiment(data,'ar','cpu')
    with patch.object(ex.model,'forward',side_effect=FloatingPointError('forced failure')):
        es.evaluate(ex,tmp_path/'failed.json')
    report=json.loads((tmp_path/'failed.json').read_text())
    assert report['horizons']['7']['failed']==1
    assert report['horizons']['7']['mean_nrmse'] is None
    with pytest.raises(SystemExit) as stopped:
        es.evaluate(ex,tmp_path/'partial.json',deadline=1)
    assert stopped.value.code==75
    es.evaluate(ex,tmp_path/'partial.json')
    assert json.loads((tmp_path/'partial.json').read_text())['complete']


def test_opsplit_audit_probe_and_continuation_gate(tmp_path):
    data=fixture(tmp_path,epochs=2);ex=es.ThreeFieldExperiment(data,'opsplit-vorticity','cpu')
    assert es.audit(ex,tmp_path/'audit.json')
    es.require_audit(ex,tmp_path/'audit.json')
    train_comparison.train(ex,tmp_path/'train',stop_after_epoch=2,train_only=True)
    es.require_composite_stability(ex,tmp_path/'train')
    stale=json.loads((tmp_path/'audit.json').read_text());stale['identity']['arm']='wrong'
    (tmp_path/'audit.json').write_text(json.dumps(stale))
    with pytest.raises(ValueError,match='exhaustive audit'):es.require_audit(ex,tmp_path/'audit.json')
