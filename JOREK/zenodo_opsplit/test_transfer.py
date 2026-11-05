import json

import pytest
import torch

from composite import CompositeSplit
from prepare_transfer import load_fluid_transfer, prepare
from test_comparison import fixture
from train_comparison import Experiment, train


def test_transfer_preserves_fluid_and_target_physics_through_training_resume(tmp_path):
    es_root, em_root = tmp_path/'es', tmp_path/'em'
    es_root.mkdir();em_root.mkdir()
    es = fixture(es_root, case='es', epochs=111)
    em = fixture(em_root, case='em', epochs=111)
    source = Experiment(es, 'composite', 'cpu')
    target_dt = Experiment(em, 'composite', 'cpu').model.config['frame_dt']
    # Synthetic ES/EM input fixtures use different time conventions; match the source interface explicitly.
    old = source.model
    source.model = CompositeSplit(old.r,old.z,old.mean,old.scale,channels=4,width=4,modes=4,layers=6,
                                  diffusion=old.config['diffusion'],resistivity=0.,frame_dt=target_dt)
    with torch.no_grad():
        source.model.fluid.projection.fcs[-1].bias[:2] = .01
    checkpoint = tmp_path/'es-best.pt'
    legacy_config = dict(source.model.config)
    legacy_config.pop('state_representation')
    torch.save(dict(state_dict=source.model.state_dict(),model_config=legacy_config,identity=source.identity,next_epoch=70,best=.04),checkpoint)
    # The fixture's geometry labels overlap; refusing that is part of the transfer contract.
    with pytest.raises(ValueError, match='overlaps held-out'):
        prepare(es, checkpoint, em, tmp_path/'rejected')
    meta = json.loads((em/'comparison.json').read_text())
    for entries in meta['entries'].values():
        for entry in entries:entry['family']='em-'+entry['family']
    (em/'comparison.json').write_text(json.dumps(meta))
    before = (em/'comparison.json').read_bytes()
    output = tmp_path/'transfer-data'
    prepare(es, checkpoint, em, output)
    assert (em/'comparison.json').read_bytes()==before
    assert (output/'data.pt').resolve()==em/'data.pt'
    target = Experiment(output,'composite','cpu')
    baseline = Experiment(em,'composite','cpu')
    for name,value in source.model.fluid.state_dict().items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(target.model.fluid.state_dict()[name],value,rtol=0,atol=0)
    torch.testing.assert_close(target.model.fluid_mean,source.model.fluid_mean,rtol=0,atol=0)
    torch.testing.assert_close(target.model.fluid_scale,source.model.fluid_scale,rtol=0,atol=0)
    torch.testing.assert_close(target.model.mean,baseline.model.mean,rtol=0,atol=0)
    for name,value in baseline.model.resistive.state_dict().items():
        torch.testing.assert_close(target.model.resistive.state_dict()[name],value,rtol=0,atol=0)
    assert not any(p.requires_grad for p in target.model.fluid.parameters())
    assert all(p.requires_grad for p in target.model.magnetic.parameters())
    x=target.values['val']['exported'][0,:1].double()
    normalized=((x-target.model.mean[None,:,None,None])/target.model.scale[None,:,None,None]).float()
    assert torch.count_nonzero(target.model.magnetic(normalized))==0
    train(target,tmp_path/'smoke',smoke=True)
    saved=torch.load(tmp_path/'smoke/last.pt',weights_only=False)
    for name,value in source.model.fluid.state_dict().items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(saved['state_dict']['fluid.'+name],value,rtol=0,atol=0)
    assert len(saved['optimizer']['param_groups'][0]['params'])==len(list(target.model.magnetic.parameters()))
    assert torch.count_nonzero(target.model.magnetic.projection.fcs[-1].weight)>0
    resumed=Experiment(output,'composite','cpu')
    train(resumed,tmp_path/'smoke',smoke=True,resume=True)
    assert not any(p.requires_grad for p in resumed.model.fluid.parameters())
    for name,value in source.model.fluid.state_dict().items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(resumed.model.fluid.state_dict()[name],value,rtol=0,atol=0)
    with pytest.raises(ValueError,match='Changed ES transfer checkpoint'):
        load_fluid_transfer(baseline.model,dict(target.cfg['composite_transfer'],sha256='wrong'))
