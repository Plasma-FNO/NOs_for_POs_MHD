"""Small chronological diagnostic pilot on converged native JOREK trajectories.

Shared-parent time windows cannot establish independent-trajectory generalization.
The pilot is explicitly separate from production training and physical-RHS labels.
"""
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from .euler import NativeSplineSpace, EulerSplineModel, field_loss
from .learning import parameter_counts
from .restart import read_restart


FIELDS = ('psi','u','zj','w','rho','T','Vpar')


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_pilot(config):
    report_path=Path(config['convergence_report'])
    report=json.loads(report_path.read_text())['reports']['native_reference_refinement']
    maximum=max(f['error_over_temporal_std'] for g in report['groups'].values() for f in g['fields'].values())
    if maximum>.01 or not report['positive_sampled_rho_T'] or report['matched_frames']!=97:
        raise ValueError('Reference convergence or positivity gate failed')
    trajectories=[]; provenance=[]; geometry=None; expected_times=None
    for directory in config['reference_directories']:
        run=Path(directory).parent
        native_provenance=json.loads((run/'provenance.json').read_text())
        if json.loads((run/'result.json').read_text())['status']!='completed':
            raise ValueError('Reference simulation did not complete')
        if native_provenance['binary_sha256']!='98d4348bc6e56d16168e1fc7ed82094036e13f33073ea5ddb18d1366026a4425':
            raise ValueError('Reference is not from the pinned unmodified JOREK binary')
        paths=sorted(Path(directory).glob('jorek??????.h5'))[::2]
        if len(paths)!=49:
            raise ValueError('Pilot requires 97 native frames sampled at stride two')
        states=[];times=[];files=[]
        for path in paths:
            with h5py.File(path) as f:
                fixed={key:f[key][()] for key in ('x','size','index','vertex','boundary','n_sons','constrained')}
                if geometry is None:
                    geometry,_,_=read_restart(path)
                    first_fixed=fixed
                    _,first,inverse=np.unique(fixed['index'].T.ravel(),return_index=True,return_inverse=True)
                if any(not np.array_equal(value,first_fixed[key]) for key,value in fixed.items()):
                    raise ValueError('Reference mesh changed')
                expanded=f['values'][()][:,:,0,:].transpose(2,1,0).reshape(-1,7)
                state=expanded[first]
                if not np.isfinite(state).all() or not np.allclose(expanded,state[inverse],rtol=1e-13,atol=1e-13):
                    raise ValueError('Invalid or inconsistent native coefficients')
                states.append(state)
                times.append(float(f['t_now'][0])*float(f['sqrt_mu0_rho0'][0]))
            files.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        times=np.asarray(times)
        if not np.allclose(np.diff(times),np.diff(times)[0],rtol=1e-10,atol=1e-18):
            raise ValueError('Nonuniform observation interval')
        if expected_times is not None and not np.array_equal(times,expected_times):
            raise ValueError('Reference times differ')
        if not np.isclose(times[0],report['start_time_seconds'],rtol=0,atol=1e-18) or not np.isclose(times[-1],report['end_time_seconds'],rtol=0,atol=1e-18):
            raise ValueError('Data window differs from validated reference window')
        expected_times=times;trajectories.append(np.stack(states));provenance.append(files)
    return geometry,np.stack(trajectories),expected_times,{'files':provenance,'reference_max_dynamic_error':maximum,
        'convergence_report':str(report_path),'convergence_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest()}


def normalize_training(data, space, stop):
    # Spatial reference and channel scales use training states only, including derivative DOFs.
    mean=data[:,:stop].mean(axis=(0,1))
    variance=np.zeros(data.shape[-1])
    for q in data[:,:stop].reshape(-1,*data.shape[2:]):
        values=space.basis@(q-mean)
        variance+=(values**2*space.weights[:,None]).sum(0)/space.weights.sum()
    scale=np.sqrt(variance/(len(data)*stop))
    if not np.all(np.isfinite(scale)&(scale>0)):
        raise ValueError('Training channels must have finite nonzero variation')
    return mean,scale


def physical_linear_from_reference(config, geometry, mean, scale, times, provenance):
    if config.get('linear_branch') != 'lspline_continuity':
        return None
    from .continuity import FrozenContinuity
    with h5py.File(provenance['files'][0][0]['path']) as f:
        time_unit=float(np.asarray(f['sqrt_mu0_rho0']).item())
    interval=float(times[1]-times[0])/time_unit
    provenance['physical_linear']={'background':'training frames [0,24), both parents',
        'code_interval':interval,'gamma':config['gamma'],
        'scope':'density and temperature advection/compression tangent only; other terms learned',
        'amplitude':'fixed native code-time interval; no learned multiplier',
        'gradient':'analytic L-SplineFNO perturbation gradients; native background gradients'}
    return FrozenContinuity(geometry,mean,scale,interval,config['gamma'])


@torch.no_grad()
def roll(model, sequence, parameters, *, substeps=1):
    state=sequence[:1]
    predictions=[state[0]]
    for _ in range(len(sequence)-1):
        state=model(state,parameters,substeps=substeps)
        if not torch.isfinite(state).all():
            raise FloatingPointError('Nonfinite autonomous rollout')
        predictions.append(state[0])
    return torch.stack(predictions)


def errors(pred, truth, space, mean, scale, q8):
    delta=space.apply(pred-truth,'sample')
    target=space.apply(truth,'sample')
    w=space.weights/space.weights.sum()
    mse=np.einsum('tpc,p->c',delta**2,w)/len(delta)
    variation=np.einsum('tpc,p->c',(target-target.mean(0))**2,w)/len(target)
    decoded=target*scale+space.basis@mean
    norm=np.einsum('tpc,p->c',decoded**2,w)/len(decoded)
    minima=np.full(2,np.inf)
    for state in pred:
        physical=state*scale+mean
        values=np.einsum('eql,elc->eqc',q8.basis,physical[q8.dof_map])
        minima=np.minimum(minima,values[...,4:6].min(axis=(0,1)))
    return {'normalized_volume_mse':float(mse.mean()),
            'dynamic_relative_rms':dict(zip(FIELDS,np.sqrt(mse/np.maximum(variation,1e-30)).tolist())),
            'decoded_relative_rms':dict(zip(FIELDS,np.sqrt(mse*scale**2/np.maximum(norm,1e-30)).tolist())),
            'q8_min_density_temperature':minima.tolist(),'q8_positive':bool((minima>0).all())}


def run_pilot(config, model_config, output):
    if config.get('purpose')!='chronological_diagnostic' or not 1<=config['steps']<=200:
        raise ValueError('This entry point is a bounded diagnostic, not independent-family training')
    if config.get('validate_every',0)<1 or not np.isfinite(config['learning_rate']) or config['learning_rate']<=0:
        raise ValueError('Require positive validation frequency and learning rate')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(int(config.get('threads',2)))
    started=time.monotonic()
    geometry,data,times,provenance=load_pilot(config)
    space=NativeSplineSpace(geometry)
    mean,scale=normalize_training(data,space,24)
    normalized=torch.from_numpy((data-mean)/scale)
    del data
    with torch.no_grad():
        probe=normalized[0,:1]
        projection_error=float(field_loss(space.project(space.sample(probe)),probe,space))
        if projection_error>1e-12:
            raise ValueError('Native projection roundtrip failed on the actual mesh')
    q8,_,_=read_restart(provenance['files'][0][0]['path'],quadrature_order=8)
    guard=None
    if config.get('physical_guard',False):
        from .physical_guard import PhysicalGuard
        guard=PhysicalGuard(geometry,q8,mean,scale,positivity_basis=config.get('positivity_basis','bernstein_subdivided'))
        # Native boundary derivative DOFs evolve; only nodal values are fixed.
        boundary=guard.fixed_boundary_values.numpy()
        trace=normalized[:,:,boundary,:2]
        drift=float((trace-trace[:,:1]).abs().max())
        if drift>1e-7:
            raise ValueError(f'Native potential boundary trace evolves: normalized drift {drift}')
        provenance['fixed_potential_boundary_value_max_normalized_drift']=drift
    architecture={**model_config,'architecture':{**model_config['architecture'],**config['architecture']}}
    linear_branch=config.get('linear_branch','lspline_analytic')
    linear_initialization=config.get('linear_initialization','positive')
    physical_linear=physical_linear_from_reference(config,geometry,mean,scale,times,provenance)
    np.savez(output/'normalization.npz',mean=mean,scale=scale,time_seconds=times)
    provenance.update(config=config,model_config=architecture,split={'train':[0,24],'validation':[24,32],'test':[32,49]},
        independent_trajectory_test=False,warning='Same parent simulations across disjoint time windows; diagnostic only',
        time_basis='one observation interval equals one pseudotime unit',observation_interval_seconds=float(times[1]-times[0]),
        linear_branch=linear_branch,
        linear_initialization=linear_initialization,
        linear_operator=('L-SplineFNO learned raw-field scatter -> analytic (dRR+dZZ) cubic synthesis -> native mass projection'
                         if linear_branch=='lspline_analytic' else
                         'frozen model303 continuity/temperature tangent with analytic L-SplineFNO gradients'
                         if linear_branch=='lspline_continuity' else '-M^-1 K from native bicubic gradients'),
        linear_stability_guaranteed=linear_branch=='native_bicubic' and linear_initialization=='positive',
        physical_constraints_enforced=guard is not None,
        constraint_scope='interior weak current/vorticity plus fixed potential boundary nodal values; derivative traces learned' if guard is not None else 'none',
        positivity_scope=('whole native bicubic cell via subdivided Bernstein convex hull'
                          if config.get('positivity_basis','bernstein_subdivided')=='bernstein_subdivided'
                          else 'Q8 samples only') if guard is not None else 'none',
        spectral_bound=space.spectral_bound,native_projection_mse=projection_error,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    write(output/'provenance.json',provenance)
    print(json.dumps({'phase':'data_ready','shape':list(normalized.shape),'spectral_bound':space.spectral_bound}),flush=True)
    params=torch.ones(1,1)
    schedule=np.random.default_rng(config['seed']).integers([0,0],[len(normalized),22],size=(config['steps'],2))
    results={}
    for family in ('ar','node','opsplit'):
        torch.manual_seed(config['seed'])
        model=EulerSplineModel(family,space,architecture,linear_branch=linear_branch,
                               physical_guard=guard,linear_initialization=linear_initialization,
                               physical_linear=physical_linear)
        if guard is not None:guard.reset_statistics()
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
        folder=output/family;folder.mkdir();history=[];best=float('inf');best_step=None
        wall=time.monotonic()
        for step,(branch,start) in enumerate(schedule,1):
            model.train();optimizer.zero_grad(set_to_none=True)
            frames=normalized[branch,start:start+3];state=frames[:1];loss=state.new_zeros(())
            for target in frames[1:]:
                state=model(state,params)
                loss=loss+field_loss(state,target[None],space)
                if guard is not None:
                    loss=loss+config.get('positivity_penalty',.001)*guard.last_penalty
            loss=loss/2
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True);optimizer.step()
            row={'step':step,'training_loss':float(loss.detach()),'elapsed_seconds':time.monotonic()-wall}
            if step%config['validate_every']==0 or step==config['steps']:
                model.eval();scores=[]
                for sequence in normalized[:,24:32]:
                    pred=roll(model,sequence,params)
                    scores.append(float(field_loss(pred[1:],sequence[1:],space)))
                score=float(np.mean(scores));row['validation_loss']=score
                if score<best:
                    best=score;best_step=step
                    torch.save({'model':model.state_dict(),'family':family,'step':step,'model_config':architecture,
                                'optimizer':optimizer.state_dict(),'normalization':'../normalization.npz',
                                'linear_branch':linear_branch,
                                'linear_initialization':linear_initialization,'physical_guard':guard is not None,
                                'formulation':'normalized_field_euler','independent_trajectory_test':False},folder/'best.pt')
            history.append(row);print(json.dumps({'family':family,**row}),flush=True)
        write(folder/'history.json',history)
        model.load_state_dict(torch.load(folder/'best.pt',weights_only=False,map_location='cpu')['model']);model.eval()
        metrics=[];refinement=[]
        for branch,sequence in enumerate(normalized[:,32:]):
            if guard is not None:guard.reset_statistics()
            prediction=roll(model,sequence,params)
            metrics.append(errors(prediction[1:].numpy(),sequence[1:].numpy(),space,mean,scale,q8))
            if guard is not None:
                metrics[-1]['guard_statistics']=dict(guard.statistics)
                bounds=np.full(2,np.inf)
                for q in prediction[1:].numpy()*scale+mean:
                    bounds=np.minimum(bounds,(guard.sampler.basis@q[:,4:6]).min(0))
                metrics[-1]['density_temperature_lower_bounds']=bounds.tolist()
                metrics[-1]['positive_in_limiter_basis']=bool((bounds>0).all())
                constraint_errors=[]
                for q in prediction[1:].numpy()*scale+mean:
                    defects=[]
                    for pair,pi,ai in ((guard.current,0,2),(guard.vorticity,1,3)):
                        a=pair.stiff@q[:,pi];b=pair.mass@q[:,ai]
                        defects.append(float(np.linalg.norm(a+b)/max(np.linalg.norm(a)+np.linalg.norm(b),1e-30)))
                    constraint_errors.append(defects)
                metrics[-1]['max_rollout_current_vorticity_defect']=np.max(constraint_errors,axis=0).tolist()
            if family!='ar':
                refined={n:roll(model,sequence[:5],params,substeps=n) for n in (1,2,4)}
                refinement.append({'one_vs_two_mse':float(field_loss(refined[1][1:],refined[2][1:],space)),
                                   'two_vs_four_mse':float(field_loss(refined[2][1:],refined[4][1:],space)),
                                   'test_loss_by_substeps':{str(n):float(field_loss(p[1:],sequence[1:5],space)) for n,p in refined.items()},
                                   'positivity_by_substeps':{str(n):errors(p[1:].numpy(),sequence[1:5].numpy(),space,mean,scale,q8)['q8_positive'] for n,p in refined.items()}})
            np.save(folder/f'last_prediction_{branch}.npy',prediction[-1].numpy()*scale+mean)
            if guard is not None:
                np.save(folder/f'prediction_{branch}.npy',prediction.numpy()*scale+mean)
        result={'best_step':best_step,'validation_loss':best,'parameters':parameter_counts(model),
                'test_by_branch':metrics,'fixed_weight_substep_checks':refinement,
                'elapsed_seconds':time.monotonic()-wall,'independent_trajectory_test':False}
        if family=='opsplit':
            if linear_branch!='lspline_continuity':
                result['effective_diffusion_per_frame']=model.diffusion().detach().tolist()
            result['linear_branch']=linear_branch
        write(folder/'metrics.json',result);results[family]=result
        print(json.dumps({'phase':'family_complete','family':family,'test_mse':[m['normalized_volume_mse'] for m in metrics]}),flush=True)
    write(output/'results.json',{'results':results,'elapsed_seconds':time.monotonic()-started,
                                'diagnostic_only':True,'independent_trajectory_test':False})
    return results
