import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from comparison import (prepare, valid_states, window_index, covered_states, frame_sums, scores,
                        decoded_fields, reference_for, digest)
from native import encode_release
from train_comparison import Experiment, train, audit, require_audit, evaluate
from test_native import grid


def fixture(tmp_path, case='es', epochs=1):
    r,z,rr,zz=grid(12);entries=[]
    for split in ('train','val','test'):
        q=torch.zeros(8,3 if case=='es' else 4,12,12,dtype=torch.float64)
        q[:,0]=.05;q[:,1]=.05*5e-6
        q[:,0]+=1e-5*torch.arange(8)[:,None,None]
        raw=encode_release(q,torch.zeros_like(q[:,0]),torch.zeros_like(q[:,0])).numpy()
        path=tmp_path/(split+'.h5')
        with h5py.File(path,'w') as f:
            f['R_mesh(nR,nZ)']=rr.numpy();f['Z_mesh(nR,nZ)']=zz.numpy();f['n_times']=[8]
            keys=['rho','T','Phi' if case=='es' else 'u','omega']+(['Psi','zj'] if case=='em' else [])
            for i,key in enumerate(keys):f[key]=raw[:,i]
        inputs=tmp_path/(split+'.txt')
        inputs.write_text('tstep=0.00001\nrho_0=0.05\nt_0='+('0.05' if case=='es' else '0.000005')+'\nd_perp=0.0001\nzk_perp=0.0001\nvisco=0.001\n')
        entries.append(dict(id=split,case=case,split=split,family=split,path=str(path),input=str(inputs),frames=8,r=r.tolist(),z=z.tolist()))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(entries=entries)))
    cfg=json.loads(Path(__file__).with_name(case+'_paper_protocol.json').read_text())
    cfg.update(epochs=epochs,train_trajectories=1,train_frames=7,horizon=2,batch_size=2,width=4,modes=4,layers=6,eval_horizon=7)
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(cfg))
    prepare(manifest,tmp_path/'shared',case,protocol)
    return tmp_path/'shared'


def test_invalid_state_removes_overlapping_windows_for_every_arm():
    x=torch.ones(9,4,2,2);q=torch.ones(9,3,2,2)
    q[4,1,0,0]=0
    valid=valid_states(x,q)
    assert window_index(valid,2)==[0,1,5,6]
    assert covered_states([0,5],2)==[0,1,2,5,6,7]
    x[1,3,0,0]=float('nan')
    assert window_index(valid_states(x,q),2)==[5,6]


@pytest.mark.parametrize('case',['es','em'])
def test_common_data_scratch_and_decoding(tmp_path,case):
    data=fixture(tmp_path,case)
    ex=Experiment(data,'opsplit-scratch','cpu')
    assert len(ex.meta['windows'])==5
    q=ex.values['train']['q'][0,:1].clone().requires_grad_()
    ref=reference_for(ex.meta['entries']['train'][0],q.detach())
    decoded=decoded_fields(ref,q)
    torch.testing.assert_close(decoded[:,:3].float(),ex.values['train']['exported'][0,:1,:3])
    decoded.square().sum().backward();assert torch.isfinite(q.grad).all()
    assert ex.pretraining is None
    with pytest.raises(ValueError,match='cannot load'):
        Experiment(data,'opsplit-scratch','cpu',Path('irrelevant.pt'))
    saved=dict(phase='term_pretraining',case=case,dataset_sha256=digest(data/'terms/manifest.json'),
               model_config=ex.model.config,normalization=torch.load(data/'terms/normalization.pt',weights_only=True),
               state_dict=ex.model.state_dict(),epoch=249)
    checkpoint=tmp_path/'pretrained.pt';torch.save(saved,checkpoint)
    pretrained=Experiment(data,'opsplit-pretrained','cpu',checkpoint)
    assert pretrained.identity['initialization_sha256']==ex.identity['initialization_sha256']
    # Forward and backward through the same common decoded objective in both native initializations.
    loss=ex.objective('train',0,0,2);loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in ex.model.parameters())


def test_shared_baseline_batch_loss_matches_serial_objective(tmp_path):
    data=fixture(tmp_path)
    ex=Experiment(data,'node','cpu')
    windows=ex.meta['windows'][:2]
    actual=ex.batch_loss(windows)
    expected=sum(float(ex.objective('train',i,s,2)) for i,s in windows)
    assert actual==pytest.approx(expected,rel=1e-5)


def test_paper_metric_and_cylindrical_metric():
    truth=torch.ones(4,3,3);pred=2*truth
    value=scores(frame_sums(pred,truth,torch.zeros(4),torch.ones(4),torch.arange(1,4)))
    assert value['nrmse']==pytest.approx([1/(1+1e-6)**.5]*4)
    assert value['r_weighted_relative_l2']==pytest.approx([1.]*4)


def test_resume_matches_uninterrupted(tmp_path,monkeypatch):
    import train_comparison
    data=fixture(tmp_path,epochs=2)
    train(Experiment(data,'ar','cpu'),tmp_path/'continuous')
    save=train_comparison.save_checkpoint
    def interrupted(path,payload):
        save(path,payload)
        if path.name=='last.pt' and payload['next_epoch']==0:
            raise InterruptedError('allocation ended')
    with monkeypatch.context() as m:
        m.setattr(train_comparison,'save_checkpoint',interrupted)
        with pytest.raises(InterruptedError):train(Experiment(data,'ar','cpu'),tmp_path/'resumed')
    train(Experiment(data,'ar','cpu'),tmp_path/'resumed',resume=True)
    a=torch.load(tmp_path/'continuous/last.pt',weights_only=False)
    b=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    for key,value in a['state_dict'].items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(value,b['state_dict'][key],rtol=0,atol=0)
    assert a['scheduler']==b['scheduler']


def test_failed_rollout_is_not_dropped_from_metrics(tmp_path,monkeypatch):
    data=fixture(tmp_path);ex=Experiment(data,'node','cpu')
    def failed(*args):
        raise FloatingPointError('model instability')
        yield
    monkeypatch.setattr(ex,'predictions',failed)
    evaluate(ex,tmp_path/'metrics.json')
    report=json.loads((tmp_path/'metrics.json').read_text())
    assert report['horizons']['7']['failed']==1
    assert report['horizons']['7']['mean_nrmse'] is None


def test_audit_gate_rejects_smoke_and_changed_identity(tmp_path):
    data=fixture(tmp_path);ex=Experiment(data,'ar','cpu');path=tmp_path/'audit.json'
    assert audit(ex,path,smoke=True)
    with pytest.raises(ValueError,match='exhaustive'):require_audit(ex,path)
    assert audit(ex,path)
    require_audit(ex,path)
    ex.identity['initialization_sha256']='changed'
    with pytest.raises(ValueError,match='exhaustive'):require_audit(ex,path)


def test_launcher_has_independent_native_gates(tmp_path,monkeypatch):
    import orchestrate_comparison
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    monkeypatch.setattr('sys.argv',['orchestrate_comparison.py','--manifest',str(manifest),
                                   '--output',str(tmp_path/'campaign'),'--case','es'])
    orchestrate_comparison.main()
    jobs={j['stage']:j for j in json.loads((tmp_path/'campaign/jobs.json').read_text())}
    assert jobs['ar']['dependencies']==['ar-smoke']
    assert jobs['node']['dependencies']==['node-smoke']
    assert jobs['opsplit-scratch']['dependencies']==['opsplit-scratch-audit']
    assert jobs['opsplit-pretrained']['dependencies']==['opsplit-pretrained-audit']
    assert jobs['opsplit-pretrained-smoke']['dependencies']==['terms']
    assert '--pretrained' not in (tmp_path/'campaign/opsplit-scratch.sbatch').read_text()
    assert '--pretrained' in (tmp_path/'campaign/opsplit-pretrained.sbatch').read_text()


def test_normalization_does_not_use_validation_or_test(tmp_path):
    data=fixture(tmp_path);before=torch.load(data/'data.pt',weights_only=False)
    for split in ('val','test'):
        with h5py.File(tmp_path/(split+'.h5'),'r+') as f:f['rho'][:]=f['rho'][:]*100
    prepare(tmp_path/'manifest.json',tmp_path/'changed','es',tmp_path/'protocol.json')
    after=torch.load(tmp_path/'changed/data.pt',weights_only=False)
    torch.testing.assert_close(before['mean'],after['mean'],rtol=0,atol=0)
    torch.testing.assert_close(before['scale'],after['scale'],rtol=0,atol=0)


def test_parallel_audit_matches_serial_coverage(tmp_path):
    data=fixture(tmp_path);ex=Experiment(data,'ar','cpu')
    assert audit(ex,tmp_path/'serial.json')
    assert audit(ex,tmp_path/'parallel.json',workers=2)
    serial=json.loads((tmp_path/'serial.json').read_text())
    parallel=json.loads((tmp_path/'parallel.json').read_text())
    for key in ('identity','windows','gradient_trajectories','validation_trajectories','failures','trajectories'):
        assert serial[key]==parallel[key]


def test_audit_resume_preserves_coverage_and_identity(tmp_path):
    data=fixture(tmp_path);ex=Experiment(data,'ar','cpu');path=tmp_path/'partial.json'
    assert not audit(ex,path,max_seconds=1e-12)
    partial=json.loads(path.read_text())
    assert not partial['completed'] and not partial['passed']
    with pytest.raises(ValueError):require_audit(ex,path)
    assert audit(ex,path,resume=True)
    require_audit(ex,path)
    assert audit(ex,tmp_path/'fresh.json')
    resumed=json.loads(path.read_text());fresh=json.loads((tmp_path/'fresh.json').read_text())
    assert resumed['trajectories']==fresh['trajectories']
    changed=Experiment(data,'node','cpu')
    with pytest.raises(ValueError,match='identical'):audit(changed,path,resume=True)


@pytest.mark.parametrize('gpus',[1,4])
def test_relaunch_reuses_matched_data_and_resumes_audits(tmp_path,monkeypatch,gpus):
    import orchestrate_comparison
    data=fixture(tmp_path);source=tmp_path/'existing';source.mkdir()
    (source/'data').symlink_to(data,target_is_directory=True)
    (source/'terms').mkdir();(source/'terms/best.pt').write_bytes(b'checkpoint existence fixture')
    (source/'manifest.json').write_text('{}');(source/'protocol.json').write_text('{}')
    output=tmp_path/'relaunch'
    monkeypatch.setattr('sys.argv',['orchestrate_comparison.py','--native-from',str(source),'--case','es','--output',str(output),'--gpus',str(gpus)])
    orchestrate_comparison.main()
    jobs=json.loads((output/'jobs.json').read_text())
    assert len(jobs)==(6 if gpus==1 else 8) and all(j['stage'].startswith('opsplit') for j in jobs)
    body=(output/'opsplit-scratch-audit.sbatch').read_text()
    assert str(source/'data') in body and '--audit' in body and '--resume' in body
    assert str(output/'opsplit-scratch-audit.json') in body and '--max-seconds 79200' in body
    assert str(source/'terms/best.pt') in (output/'opsplit-pretrained.sbatch').read_text()
    assert jobs[2]['dependencies']==['opsplit-scratch-audit']
    if gpus>1:
        import subprocess
        assert '--audit-workers 4 --audit-gpus 4' in body
        training=(output/'opsplit-scratch.sbatch').read_text()
        assert '--nproc_per_node 4' in training and '--train-only' in training
        assert 'requeue.json' in training and '"$status" -eq 0' in training
        assert jobs[3]['dependencies']==['opsplit-scratch']
        assert '--gres=gpu:1' in jobs[3]['submission']
        assert '--gres=gpu:4' in jobs[2]['submission']
        for script in output.glob('*.sbatch'):subprocess.run(['bash','-n',str(script)],check=True)



def test_parallel_audit_does_not_enqueue_after_deadline(tmp_path):
    data=fixture(tmp_path);ex=Experiment(data,'ar','cpu');path=tmp_path/'partial-parallel.json'
    assert not audit(ex,path,workers=2,max_seconds=1e-12)
    report=json.loads(path.read_text())
    assert report['windows']==0 and not report['completed'] and not report['passed']
    assert audit(ex,path,workers=2,resume=True)
    require_audit(ex,path)
