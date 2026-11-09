#!/usr/bin/env python3
"""Matched-test-IC density-diffusivity OOD simulations with native refinement."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

import h5py
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STUDY/'src'))
from mastu_opsplit.common import sha256_file as digest, write_json as write
from mastu_opsplit.case import namelist_overrides


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sample_cases(parent, seed=20261106, multipliers=None, gamma=None):
    if gamma is not None:
        values = np.atleast_1d(gamma).astype(float)
        if multipliers is not None or values.ndim != 1 or not len(values) or \
           not np.all(np.isfinite(values) & (values > 0) & (values <= 2) & (values != 1) & (values != 5/3)) or len(set(values)) != len(values):
            raise ValueError("Require distinct OOD gamma values in (0,2], excluding 1 and baseline 5/3")
        cases = sample_cases(parent, seed, multipliers=list(range(2,2+len(values))))
        for i,(case,value) in enumerate(zip(cases, values)):
            name = f"mastu_gamma_{i:03d}"
            case.update(case_id=name, initial=f"initials/{name}.h5", coefficient_multiplier=float(value)/(5/3),
                        pde_coefficients={"D_perp(1)":1e-5, "gamma":float(value)})
        return cases
    rng = np.random.default_rng(seed)
    held_out = [c for c in parent['cases'] if c['split']=='test']
    prefix = 'mastu_ood' if multipliers is None else 'mastu_ood_strong'
    if multipliers is None:
        # Preserve the original ten-case sampling and assignments exactly.
        positions = (np.arange(10)+rng.random(10))/10
        positions[0], positions[-1] = 0.,1.
        factors = np.exp(np.log(2)+positions*np.log(5)); factors[0],factors[-1]=2.,10.
    else:
        factors = np.asarray(multipliers,dtype=float)
    if factors.ndim != 1 or not 1 <= len(factors) <= len(held_out) or \
       not np.all(np.isfinite(factors) & (factors >= 2) & (factors <= 100)) or len(set(factors)) != len(factors):
        raise ValueError('Require distinct multipliers in [2,100] and sufficient held-out parent cases')
    cases=[]
    for i,(parent_index,multiplier) in enumerate(zip(rng.permutation(len(held_out))[:len(factors)],factors)):
        source = held_out[parent_index]
        case = {k:v for k,v in source.items() if k not in ('result','resolved_path')}
        case.update(case_id=f'{prefix}_{i:03d}',matched_id_case=source['case_id'],split='test',
                    coefficient_multiplier=float(multiplier),pde_coefficients={'D_perp(1)':1e-5*float(multiplier)},
                    initial=f'initials/{prefix}_{i:03d}.h5')
        cases.append(case)
    return cases


def prepare(root, parent_root, multipliers=None, gamma=None, binary=None):
    if gamma is not None and (binary is None or not binary.is_file()):
        raise ValueError("Gamma pilot requires the isolated gamma-enabled binary")
    if gamma is None and binary is not None:
        raise ValueError("Alternate binary only allowed for the gamma pilot")
    parent_root=parent_root.resolve()
    parent=json.loads((parent_root/'manifest.json').read_text())
    completion=json.loads((parent_root/'completion.json').read_text())
    if completion['status']!='dataset_ready' or digest(parent_root/'manifest.json')!=completion['manifest_sha256']:
        raise ValueError('Parent release has not passed verification')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists() or (root/'code').exists():
        raise FileExistsError('OOD already prepared or partially prepared')
    for folder in ('initials','cases','logs','prepared'):(root/folder).mkdir()
    code=root/'code'
    shutil.copytree(parent_root/'code',code,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.rmtree(code/'src')
    shutil.copytree(STUDY/'src',code/'src',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(__file__,code/'build_coefficient_ood.py')
    shutil.copyfile(STUDY/'scripts/render_coefficient_ood.py',code/'render_coefficient_ood.py')
    native=(code/'native_fields.py').read_text().replace(str(parent_root/'code'),str(code))
    (code/'native_fields.py').write_text(native)
    for name in ('geometry.h5','normalization.npz'):
        record=parent['geometry' if name=='geometry.h5' else 'normalization']
        if digest(parent_root/name)!=record['sha256']:raise ValueError('Parent artifact changed')
        shutil.copyfile(parent_root/name,root/name)
    shutil.copyfile(parent_root/'manifest.json',root/'parent_manifest.json')
    parent_plan=json.loads((parent_root/'plan.json').read_text())
    if digest(parent_root/'plan.json')!=parent['plan_sha256']:raise ValueError('Parent plan changed')
    cases=sample_cases(parent,multipliers=multipliers,gamma=gamma); controls={}; control_hashes={};template_hashes={}
    for case in cases:
        name=case['case_id'];source=next(c for c in parent['cases'] if c['case_id']==case['matched_id_case'])
        initial=parent_root/source['initial']
        if digest(initial)!=case['initial_check']['sha256']:raise ValueError('Parent initial changed')
        shutil.copyfile(initial,root/case['initial'])
        prepared=root/'prepared'/name
        shutil.copytree(parent_root/'prepared/baseline_fine',prepared)
        # The verified parent snapshot is read-only; only these new copies are edited.
        (prepared/'input').chmod(0o644)
        (prepared/'case.json').chmod(0o644)
        text=(prepared/'input').read_text()
        (prepared/'input').write_text(namelist_overrides(text,case['pde_coefficients']))
        manifest=json.loads((prepared/'case.json').read_text())
        manifest['case_id']=name;manifest['ood_coefficients']=case['pde_coefficients']
        if gamma is not None:
            for filename, origin in [('jorek.exe', binary), ('patched_models_model303_initialise_parameters.f90', binary.parent/'models/model303/initialise_parameters.f90')]:
                (prepared/filename).chmod(0o755 if filename=='jorek.exe' else 0o644)
                shutil.copyfile(origin, prepared/filename)
                manifest['snapshot_hashes'][filename]=digest(prepared/filename)
                manifest['files'][filename]={'source':str(origin), 'sha256':digest(origin)}
            manifest['executable_sha256']=digest(binary)
            manifest['gamma_build_provenance']=json.loads((binary.parent.parent/'provenance.json').read_text())
        manifest['snapshot_hashes']['input']=digest(prepared/'input')
        (prepared/'case.json').write_text(json.dumps(manifest,indent=2)+'\n')
        template_hashes[name]=digest(prepared/'case.json')
        controls[name]={}
        for rate in ('coarse','fine'):
            work=parent_root/'runs'/f"{case['matched_id_case']}_{rate}"/'work'
            result=json.loads((work.parent/'result.json').read_text())
            if result['status']!='completed':raise ValueError('Missing matched ID control')
            paths=[*sorted(work.glob('jorek??????.h5')),work/'jorek_restart.h5',work/'input']
            if len(paths)!=99:raise ValueError('Incomplete matched control')
            controls[name][rate]=str(work)
            control_hashes.update({str(p):digest(p) for p in paths})
        print(json.dumps(dict(prepared=name,matched_id=case['matched_id_case'],multiplier=case['coefficient_multiplier'])),flush=True)
    plan=dict(schema='mastu-density-diffusivity-ood-plan-v1',cases=cases,seed=20261106,
        coefficient_name='D_perp(1)',baseline_coefficient=1e-5,coefficient_multiplier_range=[min(c['coefficient_multiplier'] for c in cases),max(c['coefficient_multiplier'] for c in cases)],
        source_parent_root=str(parent_root),parent_manifest_sha256=digest(root/'parent_manifest.json'),
        geometry_sha256=digest(root/'geometry.h5'),normalization_sha256=digest(root/'normalization.npz'),
        binary_sha256=digest(binary) if binary is not None else parent_plan['binary_sha256'],controls=controls,control_sha256=control_hashes,
        template_sha256=template_hashes,split_counts={'train':0,'validation':0,'test':len(cases)},
        coarse=parent_plan['coarse'],fine=parent_plan['fine'],finer=parent_plan['finer'],
        expected_frames=97,dynamic_error_limit=.01,energy_error_limit=.01,
        physical_window_seconds=parent_plan['physical_window_seconds'],
        response_definition='coefficient-shift trajectory minus baseline-coefficient trajectory with identical held-out initial state',
        paper_protocol='2602.23113v2 Table 7 varies viscosity 10x plus initial conditions; this is an adapted density-diffusivity-only shift, not an exact reproduction.',
        normalization_policy='byte-identical parent training-only normalization; no fitting on OOD',
        model_class='mastu_opsplit.coefficient_ood.DensityDiffusionEulerModel',
        code_sha256={str(p.relative_to(root)):digest(p) for p in code.rglob('*') if p.is_file()})
    if gamma is not None:
        plan.update(schema='mastu-gamma-ood-plan-v1', coefficient_name='gamma', baseline_coefficient=5/3,
            model_class='mastu_opsplit.coefficient_ood.GammaEulerModel',
            paper_protocol='Gamma-only coefficient OOD at matched held-out initial states, baseline=5/3. Other input coefficients stay at baseline.',
            gamma=[c['pde_coefficients']['gamma'] for c in cases], required_build_regression=True,
            parameter_semantics='[D_perp(1)/1e-5, gamma]; baseline training=[1,5/3]',
            sheath_policy='Keep input gamma_sheath fixed; JOREK recomputes gamma_stangeby using the requested gamma.')
        # Compare the new interface at baseline gamma against the original executable.
        for name, source, baseline_gamma in [('regression_old', parent_root/'prepared/baseline_fine', False),
                                              ('regression_new', root/'prepared'/cases[0]['case_id'], True)]:
            destination=root/'prepared'/name;shutil.copytree(source,destination)
            (destination/'input').chmod(0o644);(destination/'case.json').chmod(0o644)
            if baseline_gamma:
                (destination/'input').write_text(namelist_overrides((destination/'input').read_text(), {'gamma':5/3}))
            record=json.loads((destination/'case.json').read_text())
            record['snapshot_hashes']['input']=digest(destination/'input')
            record['case_id']=name
            if baseline_gamma:record['ood_coefficients']={'D_perp(1)':1e-5,'gamma':5/3}
            (destination/'case.json').write_text(json.dumps(record,indent=2)+'\n')
            plan['template_sha256'][name]=digest(destination/'case.json')
    write(root/'plan.json',plan)


def verify_runtime_coefficient(work, value, name="D_perp"):
    if name not in ("D_perp", "gamma"):raise ValueError("Unknown runtime coefficient")
    text=(work.parent/'stdout.log').read_text(errors='replace')
    match=re.search(rf'^\s*{name}\s*=\s*([+\-\d.EeDd]+)',text,re.M)
    if not match or not np.isclose(float(match[1].replace('D','E').replace('d','e')),value,rtol=6e-5,atol=0):
        raise ValueError(f'Runtime {name} echo does not match requested PDE coefficient')
    return float(match[1].replace('D','E').replace('d','e'))


def regression(root):
    plan=json.loads((root/'plan.json').read_text())
    base=load('gamma_regression_builder',root/'code/build_euler_dataset.py')
    runner=base.environment(root,plan)
    initial=root/plan['cases'][0]['initial']
    works=[]
    for name in ('regression_old','regression_new'):
        if digest(root/'prepared'/name/'case.json') != plan['template_sha256'][name]:
            raise ValueError('Regression template changed')
        works.append(runner.run(dict(label=name,template=name,restart=str(initial),dt=.0078125,steps=2,nout=1)))
        verify_runtime_coefficient(works[-1],5/3,'gamma')
    from native_fields import NativeMesh
    mesh=NativeMesh(initial)
    with np.load(root/'normalization.npz') as f:scale=f['scale']
    maximum=0.
    for a,b in zip(sorted(works[0].glob('jorek??????.h5')), sorted(works[1].glob('jorek??????.h5'))):
        qa,ta=mesh.read(a);qb,tb=mesh.read(b)
        if ta!=tb:raise ValueError('Regression times differ')
        maximum=max(maximum,float(np.max(np.abs(mesh.evaluate(qa-qb))/scale)))
    if maximum>1e-8:raise ValueError(f'Baseline build regression failed: {maximum}')
    write(root/'build_regression.json',dict(status='passed',gamma=5/3,steps=2,
        maximum_normalized_point_difference=maximum,limit=1e-8,
        plan_sha256=digest(root/'plan.json'),binary_sha256=plan['binary_sha256']))


def run_case(root,index):
    plan=json.loads((root/'plan.json').read_text());case=plan['cases'][index];name=case['case_id']
    if (root/'cases'/f'{name}.json').exists():raise FileExistsError('Case already assessed')
    base=load('ood_base_builder',root/'code/build_euler_dataset.py')
    runner=base.environment(root,plan)
    if plan.get('required_build_regression'):
        check=json.loads((root/'build_regression.json').read_text())
        if check['status']!='passed' or check['binary_sha256']!=plan['binary_sha256'] or check['plan_sha256']!=digest(root/'plan.json'):
            raise ValueError('Gamma build regression must pass before OOD')
        base.BINARY_SHA=plan['binary_sha256']
    diagnostics=load('ood_diagnostics',root/'code/diagnostics_full.py')
    report=dict(case_id=name,status='failed',split='test',convergence_attempts=[],runtime_coefficients={})
    try:
        initial=root/case['initial']
        if digest(initial)!=case['initial_check']['sha256']:raise ValueError('Initial state changed')
        if digest(root/'prepared'/name/'case.json')!=plan['template_sha256'][name]:raise ValueError('Template changed')
        for p,sha in plan['control_sha256'].items():
            if any(Path(p).parent==Path(w) for w in plan['controls'][name].values()) and digest(p)!=sha:
                raise ValueError('Matched ID reference changed')
        for rate in ('coarse','fine','finer'):
            spec=dict(plan[rate],label=name+'_'+rate,restart=str(initial),template=name)
            work=runner.run(spec)
            report['runtime_coefficients'][rate]=verify_runtime_coefficient(work,case['pde_coefficients']['D_perp(1)'])
            if 'gamma' in case['pde_coefficients']:
                report.setdefault('runtime_gamma',{})[rate]=verify_runtime_coefficient(work,case['pde_coefficients']['gamma'],'gamma')
            if rate=='coarse':continue
            previous='coarse' if rate=='fine' else 'fine'
            c=diagnostics.compare(root/'runs'/f'{name}_{previous}'/'work',work,
                                  plan['controls'][name][previous],plan['controls'][name]['fine'],order=8)
            c['control_convention']='matched ID timesteps' if rate=='fine' else 'common fine ID trajectory'
            report['convergence_attempts'].append(dict(pair=[previous,rate],comparison=c))
            errors=[f['error_over_temporal_std'] for g in c['groups'].values() for f in g['fields'].values()]
            energy=list(c['energy_relative_errors'].values())
            if c['matched_frames']==97 and c['positive_sampled_rho_T'] and np.isfinite(errors+energy).all() and \
               max(errors)<=.01 and max(energy)<=.01:
                report.update(status='accepted',max_dynamic_error=max(errors),selected_reference_rate=rate,
                              data=base.export_case(root,case,work,c))
                break
        if report['status']!='accepted':raise ValueError('OOD failed convergence or positivity after refinement')
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        write(root/'cases'/f'{name}.json',report)


def finalize(root):
    plan=json.loads((root/'plan.json').read_text())
    cases=[]
    for case in plan['cases']:
        path=root/'cases'/f"{case['case_id']}.json"
        result=json.loads(path.read_text()) if path.exists() else {'status':'missing'}
        cases.append(dict(case,result=result))
    bad=[c['case_id'] for c in cases if c['result']['status']!='accepted']
    if bad:
        write(root/'incomplete.json',dict(ready=False,failed_or_missing=bad));raise ValueError('Incomplete OOD: '+str(bad))
    parent=json.loads((root/'parent_manifest.json').read_text())
    for name,sha in [('parent_manifest.json',plan['parent_manifest_sha256']),('geometry.h5',plan['geometry_sha256']),('normalization.npz',plan['normalization_sha256'])]:
        if digest(root/name)!=sha:raise ValueError('Frozen release artifact changed: '+name)
    manifest=dict(schema='mastu-density-diffusivity-ood-v1',ready_for_evaluation=True,synthetic=False,
        cases=cases,split_counts=plan['split_counts'],fields=parent['fields'],
        geometry=parent['geometry'],normalization=parent['normalization'],
        parent_manifest=dict(path='parent_manifest.json',sha256=plan['parent_manifest_sha256']),
        plan_sha256=digest(root/'plan.json'),observation_interval_seconds=parent['observation_interval_seconds'],
        coefficient_name=plan['coefficient_name'],baseline_coefficient=plan['baseline_coefficient'],
        coefficient_multiplier_range=plan['coefficient_multiplier_range'],
        parameter_semantics='D_perp(1) divided by baseline 1e-5; baseline training input is 1',
        paired_id_cases=[c['matched_id_case'] for c in cases],families=['opsplit','ar','node'],
        limits=['Test-only coefficient-shift extension; not additional training data.',
                'Independent held-out initial states reused for matched ID/OOD comparisons.',
                'One fixed mesh/equilibrium; altered coefficients start at the original warm-start time.',
                'Requires a model trained with the explicit coefficient split at baseline; old continuity-only checkpoints are not equivalent.',
                'Analytic spline derivatives approximate the physical fields; full native transport and boundary closure remain in the residual.'])
    if plan.get('gamma') is not None:
        manifest.update(schema='mastu-gamma-ood-v1', parameter_semantics=plan['parameter_semantics'],
            gamma=plan['gamma'], model_class=plan['model_class'],
            build_regression=json.loads((root/'build_regression.json').read_text()),
            interpretation=plan['paper_protocol'], sheath_policy=plan['sheath_policy'])
    # Verify before publishing the final manifest or success certificate.
    provisional=root/'manifest.pending.json';write(provisional,manifest)
    from mastu_opsplit.coefficient_ood import CoefficientOODDataset
    data=CoefficientOODDataset(provisional);windows=data.windows('test',4)
    for i in range(len(data.cases)):
        for j in (i*windows.starts,(i+1)*windows.starts-1):
            assert np.isfinite(windows[j]['state'].numpy()).all()
    provisional.replace(root/'manifest.json')
    write(root/'completion.json',dict(status='ood_dataset_ready',cases=len(cases),frames_per_case=97,
        total_frames=97*len(cases),four_interval_test_windows=len(windows),
        manifest_sha256=digest(root/'manifest.json'),normalization_refitted=False,model_training_performed=False))
    print((root/'completion.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','regression','run-case','finalize']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--index',type=int)
    p.add_argument('--multipliers',type=float,nargs='+',help='Optional explicit OOD factors in [2,100]')
    p.add_argument('--gamma',type=float,nargs='+');p.add_argument('--binary',type=Path)
    a=p.parse_args();root=a.root.resolve()
    if a.action=='prepare':
        if a.parent is None:p.error('prepare requires --parent')
        prepare(root,a.parent,a.multipliers,a.gamma,a.binary)
    elif a.action=='regression':regression(root)
    elif a.action=='run-case':
        if a.index is None or not 0<=a.index<len(json.loads((root/'plan.json').read_text())['cases']):
            p.error('run-case requires a valid index into the frozen plan')
        run_case(root,a.index)
    else:finalize(root)
