import torch
import pytest
from native_terms import NativeOperators, NAMES
from pilot_native_transfer import initialize, score, verify_source
from comparison import digest, write_json


def test_selective_transfer_preserves_common_random_control_and_freezes():
    source = NativeOperators(torch.zeros(4),torch.ones(4),torch.ones(5),width=4,modes=4,layers=(1,1,1,1))
    checkpoint = dict(normalization=dict(input_mean=source.input_mean,input_scale=source.input_scale,target_scale=source.target_scale),
                      model_config=source.config,state_dict=source.state_dict())
    scratch = initialize(checkpoint,'scratch',7,'cpu')
    for arm,names in [('advection-transfer',NAMES[:2]),('diffusion-transfer',NAMES[2:]),('frozen',NAMES)]:
        model = initialize(checkpoint,arm,7,'cpu')
        for name in NAMES:
            expected = source if name in names else scratch
            for a,b in zip(model.blocks[name].parameters(),expected.blocks[name].parameters()):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
                assert a.requires_grad == (arm!='frozen')


def test_failed_rollout_is_not_silently_removed():
    row = dict(status='ok',frames=[dict(relative_l2=[1]*6,delta_psi_relative_l2=2)])
    assert score([row],1)['shared_relative_l2']==1
    assert score([row,dict(status='failed',frames=[])],1) is None
    assert score([row],5) is None


def test_source_provenance_rejects_wrong_split(tmp_path):
    manifest=tmp_path/'data.json'
    write_json(manifest,dict(entries=[dict(id='source',case='es',split='train',family='a')]))
    source_data=tmp_path/'terms';source_data.mkdir()
    metadata=dict(manifest_sha256=digest(manifest),splits=dict(train=dict(ids=['source'])))
    write_json(source_data/'manifest.json',metadata)
    checkpoint=dict(case='es',phase='term_pretraining',dataset_sha256=digest(source_data/'manifest.json'))
    verify_source(checkpoint,source_data,manifest)
    metadata['splits']['train']['ids']=['held-out']
    write_json(source_data/'manifest.json',metadata)
    checkpoint['dataset_sha256']=digest(source_data/'manifest.json')
    with pytest.raises(ValueError,match='nontraining'):
        verify_source(checkpoint,source_data,manifest)


def test_residual_zero_initialization_bound_roundtrip_and_gradient():
    norms=dict(input_mean=torch.zeros(4),input_scale=torch.ones(4),target_scale=torch.arange(1,6).float())
    model=NativeOperators(**norms,width=4,modes=4,layers=(1,1,1,1),correction_fraction=.1)
    x=torch.randn(1,4,12,12)
    torch.testing.assert_close(model(x),torch.zeros(1,5,12,12),atol=0,rtol=0)
    model(x).sum().backward()
    assert all(block.projection.fcs[-1].bias.grad.abs().sum()>0 for block in model.blocks.values())
    with torch.no_grad():
        for block in model.blocks.values():block.projection.fcs[-1].bias.fill_(100.)
    torch.testing.assert_close(model(x),torch.full((1,5,12,12),.1))
    clone=NativeOperators(**norms,**model.config)
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone(x),model(x))
    old=NativeOperators(**norms,width=4,modes=4,layers=(1,1,1,1))
    with pytest.raises(ValueError,match='semantics'):clone.transfer(old,NAMES)


@pytest.mark.parametrize('case',['es','em'])
def test_zero_corrections_recover_analytic_solver_and_train(case):
    from test_native import grid,coefficients
    from native import NativeReference,wall_trace
    from native_rollout import NativeRollout
    r,z,rr,zz=grid(11);c=coefficients('em');c.update(viscosity='old',conservative=case=='em')
    shape=torch.sin(torch.pi*(rr-9.5))*torch.sin(torch.pi*(zz+.5))
    q=torch.stack((c['rho_wall']+shape*.001,c['rho_wall']*c['T_wall']+shape*1e-8,shape*1e-7))[None]
    if case=='em':q=torch.cat((q,(.1+shape*1e-5)[None,None]),1)
    ref=NativeReference(r,z,wall_trace(q,r,z,c),c)
    norms=dict(input_mean=torch.zeros(4),input_scale=torch.ones(4),target_scale=torch.tensor([1e-4,1e-9,1e-5,1.,1e-4]))
    model=NativeOperators(**norms,width=4,modes=4,layers=(1,1,1,1),correction_fraction=.1)
    learned=NativeRollout(ref,q,model)
    analytic=NativeRollout(ref,q)
    expected=analytic.step(q,1.,2)
    actual=learned.step(q,1.,2)
    torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-9)
    assert learned.limiter_stats['energy_limited']==0
    (actual/actual.detach().abs().clamp_min(1e-10)).sum().backward()
    for block in model.blocks.values():
        g=block.projection.fcs[-1].bias.grad
        assert torch.isfinite(g).all() and g.abs().sum()>0


def test_residual_selective_transfer_copies_learned_heads():
    norms=dict(input_mean=torch.zeros(4),input_scale=torch.ones(4),target_scale=torch.ones(5))
    source=NativeOperators(**norms,width=4,modes=4,layers=(1,1,1,1),correction_fraction=.1)
    with torch.no_grad():
        for block in source.blocks.values():block.projection.fcs[-1].bias.fill_(.3)
    payload=dict(normalization=norms,model_config=source.config,state_dict=source.state_dict())
    x=torch.randn(1,4,12,12)
    for arm,channels in [('advection-transfer',slice(0,3)),('diffusion-transfer',slice(3,5))]:
        target=initialize(payload,arm,907,'cpu')
        expected=torch.zeros(1,5,12,12);expected[:,channels]=.1*torch.tanh(torch.tensor(.3))
        torch.testing.assert_close(target(x),expected)
