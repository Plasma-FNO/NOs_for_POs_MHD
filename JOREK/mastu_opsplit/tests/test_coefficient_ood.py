import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from mastu_opsplit.common import sha256_file as digest, write_json as write
from mastu_opsplit.coefficient_ood import CoefficientOODDataset, DensityDiffusionEulerModel
from mastu_opsplit.continuity import FrozenContinuity
from mastu_opsplit.euler import NativeSplineSpace, field_loss
from mastu_opsplit.restart import read_restart
from test_euler_dataset import fixture_release
from test_learning import configuration
from test_restart import rectangle


def builder():
    path=Path(__file__).resolve().parents[1]/'scripts/build_coefficient_ood.py'
    spec=importlib.util.spec_from_file_location('ood_builder_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_sampling_only_held_out_cases_and_real_coefficient_shift():
    parent={'cases':[dict(case_id=f'case{i}',split='test' if i>=80 else 'train',
                          parent_family_id=f'parent{i}',initial_check={'sha256':str(i)}) for i in range(100)]}
    cases=builder().sample_cases(parent)
    assert len(cases)==10 and len({c['matched_id_case'] for c in cases})==10
    assert cases==builder().sample_cases(parent)
    assert all(int(c['matched_id_case'][4:])>=80 and c['split']=='test' for c in cases)
    factors=[c['coefficient_multiplier'] for c in cases]
    assert min(factors)==2 and max(factors)==10 and len(set(factors))==10
    assert all(c['pde_coefficients']=={'D_perp(1)':1e-5*c['coefficient_multiplier']} for c in cases)
    strong=builder().sample_cases(parent,multipliers=[20,40,70,100])
    assert [c['coefficient_multiplier'] for c in strong]==[20,40,70,100]
    assert len({c['matched_id_case'] for c in strong})==4
    assert not {c['case_id'] for c in cases} & {c['case_id'] for c in strong}
    for invalid in ([20,20],[0,20],[float('nan')],[101],[]):
        with pytest.raises(ValueError,match='multipliers'):
            builder().sample_cases(parent,multipliers=invalid)


@pytest.mark.parametrize('multiplier',[3.,100.,'gamma'])
def test_ood_uses_parent_normalization_and_returns_coefficient(tmp_path,multiplier):
    gamma_ood = multiplier == 'gamma'
    if gamma_ood:multiplier = (2/3)/(5/3)
    parent_root=tmp_path/'parent';parent_root.mkdir();fixture_release(parent_root)
    parent=json.loads((parent_root/'manifest.json').read_text())
    root=tmp_path/'ood';root.mkdir();(root/'cases').mkdir()
    for name in ('geometry.h5','normalization.npz'):shutil.copyfile(parent_root/name,root/name)
    shutil.copyfile(parent_root/'manifest.json',root/'parent_manifest.json')
    source=parent['cases'][-1]
    case=dict(source,case_id='ood0',matched_id_case=source['case_id'],coefficient_multiplier=multiplier,pde_coefficients={'D_perp(1)':1e-5*multiplier})
    if gamma_ood:case['pde_coefficients']={'D_perp(1)':1e-5, 'gamma':2/3}
    path=root/'cases/ood0.h5';shutil.copyfile(parent_root/source['result']['data']['path'],path)
    with h5py.File(path,'r+') as f:
        del f['metadata_json'];f.create_dataset('metadata_json',data=json.dumps(case),dtype=h5py.string_dtype('utf-8'))
    case['result']=dict(status='accepted',data=dict(path='cases/ood0.h5',sha256=digest(path)))
    manifest=dict(schema='mastu-density-diffusivity-ood-v1',ready_for_evaluation=True,cases=[case],
        split_counts={'train':0,'validation':0,'test':1},coefficient_multiplier_range=[2.,multiplier],geometry=parent['geometry'],normalization=parent['normalization'],
        parent_manifest=dict(path='parent_manifest.json',sha256=digest(root/'parent_manifest.json')),
        observation_interval_seconds=parent['observation_interval_seconds'])
    if gamma_ood:manifest['schema']='mastu-gamma-ood-v1'
    write(root/'manifest.json',manifest)
    data=CoefficientOODDataset(root/'manifest.json')
    with np.load(parent_root/'normalization.npz') as f:np.testing.assert_array_equal(data.mean,f['mean'])
    sample=data.windows('test',1)[0]
    torch.testing.assert_close(sample['parameters'],torch.tensor([1.,2/3] if gamma_ood else [multiplier]))
    assert sample['matched_id_case']==source['case_id']
    with pytest.raises(ValueError,match='reserved'):data.windows('train',1)
    normalization=tmp_path/'checkpoint-normalization.npz'
    with np.load(parent_root/'normalization.npz') as f:
        payload={k:f[k] for k in f.files}
    payload['mean']=payload['mean']+2
    payload['training_case_ids']=np.array([c['case_id'] for c in parent['cases'] if c['split']!='test'])
    np.savez(normalization,**payload)
    alternate=CoefficientOODDataset(root/'manifest.json',training_normalization=normalization,
                                     training_normalization_sha256=digest(normalization))
    np.testing.assert_array_equal(alternate.mean,payload['mean'])
    with pytest.raises(ValueError,match='hash'):
        CoefficientOODDataset(root/'manifest.json',training_normalization=normalization,training_normalization_sha256='wrong')
    payload['training_case_ids']=np.array([c['case_id'] for c in parent['cases']])
    np.savez(normalization,**payload)
    with pytest.raises(ValueError,match='exactly parent'):
        CoefficientOODDataset(root/'manifest.json',training_normalization=normalization,
                              training_normalization_sha256=digest(normalization))
    manifest['cases'][0]['matched_id_case']=parent['cases'][0]['case_id']
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='held-out'):CoefficientOODDataset(root/'manifest.json')


def test_explicit_density_coefficient_scaling_and_gradients(tmp_path):
    torch.set_num_threads(1)
    rectangle(tmp_path/'mesh.h5');geometry,mean,_=read_restart(tmp_path/'mesh.h5')
    space=NativeSplineSpace(geometry);scale=np.ones(7)
    physical=FrozenContinuity(geometry,mean,scale,.03125)
    model=DensityDiffusionEulerModel('opsplit',space,configuration(),mean=mean,scale=scale,
                                    code_interval=.03125,physical_linear=physical)
    state=torch.randn(1,len(mean),7,dtype=torch.float64,requires_grad=True)
    one=model.density_rhs(state,torch.ones(1,1))
    ten=model.density_rhs(state,torch.full((1,1),10.))
    torch.testing.assert_close(ten,10*one,atol=1e-12,rtol=1e-9)
    assert one[...,4].abs().max()>0 and one[...,[0,1,2,3,5,6]].count_nonzero()==0
    one.square().sum().backward()
    assert torch.isfinite(state.grad).all() and state.grad[...,4].abs().sum()>0
    with pytest.raises(ValueError,match='positive'):model.density_rhs(state,torch.zeros(1,1))
    # Check the cylindrical term and pseudotime factor with rho/scale=R^2+Z^2.
    class PolynomialDerivatives(torch.nn.Module):
        def laplace(self,c):return torch.full((1,len(model.coordinates),1),4.,dtype=c.dtype)
        def forward(self,c,orders):return 2*model.coordinates[None,:,0:1].to(c)
    model.derivatives=PolynomialDerivatives()
    # The radial plus vertical second derivatives are 2+2, radial first/R adds 2.
    got=space.sample(model.density_rhs(state.detach(),torch.full((1,1),3.)))
    torch.testing.assert_close(got[...,4],torch.full_like(got[...,4],6*3*1e-5*.03125),atol=1e-12,rtol=1e-7)


@pytest.mark.parametrize('family',['ar','node','opsplit'])
def test_coefficient_variant_rollout_backward_all_families(tmp_path,family):
    torch.set_num_threads(1)
    rectangle(tmp_path/'mesh.h5');geometry,mean,_=read_restart(tmp_path/'mesh.h5')
    space=NativeSplineSpace(geometry);scale=np.ones(7)
    model=DensityDiffusionEulerModel(family,space,configuration(),mean=mean,scale=scale,code_interval=.03125,
                                    physical_linear=FrozenContinuity(geometry,mean,scale,.03125))
    state=torch.randn(1,len(mean),7,dtype=torch.float64,requires_grad=True)
    out=model(state,torch.tensor([[2.]]));field_loss(out,state,space).backward()
    assert torch.isfinite(out).all() and torch.isfinite(state.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_runtime_parameter_echo_and_failed_release(tmp_path):
    b=builder();work=tmp_path/'work';work.mkdir()
    (tmp_path/'stdout.log').write_text(' D_perp = 2.1235E-05 0.0 0.0 99.0 99.0 0.0\n')
    b.verify_runtime_coefficient(work,2.123456e-5)
    with pytest.raises(ValueError,match='Runtime'):b.verify_runtime_coefficient(work,1e-5)
    write(tmp_path/'plan.json',dict(cases=[{'case_id':'missing'}]))
    with pytest.raises(ValueError,match='Incomplete'):b.finalize(tmp_path)
    assert not (tmp_path/'manifest.json').exists()


@pytest.mark.parametrize('refine',[False,True])
def test_runs_same_initial_coefficient_template_and_matched_id_controls(tmp_path,monkeypatch,refine):
    b=builder();(tmp_path/'cases').mkdir();(tmp_path/'prepared'/'ood').mkdir(parents=True)
    (tmp_path/'initial.h5').write_bytes(b'initial');(tmp_path/'prepared/ood/case.json').write_text('{}')
    case=dict(case_id='ood',initial='initial.h5',initial_check={'sha256':digest(tmp_path/'initial.h5')},
              pde_coefficients={'D_perp(1)':2e-5})
    plan=dict(cases=[case],template_sha256={'ood':digest(tmp_path/'prepared/ood/case.json')},control_sha256={},
              controls={'ood':{'coarse':'id_coarse','fine':'id_fine'}},coarse={},fine={},finer={})
    write(tmp_path/'plan.json',plan);specs=[];comparisons=[]
    def run(spec):specs.append(spec);return tmp_path/'runs'/spec['label']/'work'
    def compare(*args,**kwargs):
        comparisons.append(args)
        error=.02 if refine and len(comparisons)==1 else .001
        return dict(matched_frames=97,positive_sampled_rho_T=True,groups={'state':{'fields':{'Vpar':{'error_over_temporal_std':error}}}},energy_relative_errors={'kinetic':.0001})
    base=SimpleNamespace(environment=lambda *a:SimpleNamespace(run=run),export_case=lambda *a:{'frames':97})
    monkeypatch.setattr(b,'load',lambda name,path:base if 'base_builder' in name else SimpleNamespace(compare=compare))
    monkeypatch.setattr(b,'verify_runtime_coefficient',lambda *a:2e-5)
    b.run_case(tmp_path,0)
    assert all(s['template']=='ood' and s['restart']==str(tmp_path/'initial.h5') for s in specs)
    assert comparisons[0][2:]==('id_coarse','id_fine')
    if refine:assert comparisons[1][2:]==('id_fine','id_fine')
    report=json.loads((tmp_path/'cases/ood.json').read_text())
    assert report['status']=='accepted' and report['selected_reference_rate']==('finer' if refine else 'fine')


def test_gamma_sampling_and_runtime_readback(tmp_path):
    b=builder()
    parent={'cases':[dict(case_id='test', split='test', initial_check={'sha256':'x'})]}
    case=b.sample_cases(parent,gamma=2/3)[0]
    assert case['matched_id_case']=='test' and case['pde_coefficients']=={'D_perp(1)':1e-5,'gamma':2/3}
    for gamma in (0.,1.,5/3,2.1,float('nan'),[1.2,1.2]):
        with pytest.raises(ValueError):b.sample_cases(parent,gamma=gamma)
    parent['cases']=[dict(case_id=f'test{i}',split='test',initial_check={'sha256':str(i)}) for i in range(4)]
    cases=b.sample_cases(parent,gamma=[1.2,1.4,1.8,2.])
    assert [c['pde_coefficients']['gamma'] for c in cases]==[1.2,1.4,1.8,2.]
    assert len({c['matched_id_case'] for c in cases})==4
    work=tmp_path/'work';work.mkdir()
    (tmp_path/'stdout.log').write_text(' gamma = 6.6667E-01\n D_perp = 1.0000E-05\n')
    b.verify_runtime_coefficient(work,2/3,'gamma')
    with pytest.raises(ValueError):b.verify_runtime_coefficient(work,5/3,'gamma')


@pytest.mark.parametrize('family',['ar','node','opsplit'])
def test_gamma_rollout_and_explicit_signed_compression(tmp_path,family):
    from mastu_opsplit.coefficient_ood import GammaEulerModel
    torch.set_num_threads(1)
    rectangle(tmp_path/'mesh.h5');geometry,mean,_=read_restart(tmp_path/'mesh.h5')
    space=NativeSplineSpace(geometry);scale=np.ones(7)
    model=GammaEulerModel(family,space,configuration(),mean=mean,scale=scale,code_interval=.03125,
                         physical_linear=FrozenContinuity(geometry,mean,scale,.03125))
    state=torch.randn(2,len(mean),7,dtype=torch.float64,requires_grad=True)
    parameters=torch.tensor([[1.,5/3],[1.,2/3]])
    out=model(state,parameters);field_loss(out,state,space).backward()
    assert torch.isfinite(out).all() and torch.isfinite(state.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.testing.assert_close(model(state[:1],torch.ones(1,1)),model(state[:1],parameters[:1]))
    if family=='opsplit':
        # Separate advection from compression via gamma=1: only T changes.
        gammas=[torch.full((2,1),v,dtype=torch.float64) for v in (1.,5/3,2/3)]
        adv,baseline,ood=[model.linear_rhs(state,gamma=g) for g in gammas]
        torch.testing.assert_close(ood-adv,-.5*(baseline-adv),atol=1e-8,rtol=2e-4)
        torch.testing.assert_close(ood[...,[0,1,2,3,4,6]],baseline[...,[0,1,2,3,4,6]])
        assert (ood[...,5]-baseline[...,5]).abs().max()>0
