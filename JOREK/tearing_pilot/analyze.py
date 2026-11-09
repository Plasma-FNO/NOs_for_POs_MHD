"""Native bicubic/Fourier diagnostics for the bounded, unrefined model199 pilot.

Reuses the source-audited JOREK bicubic basis. Integrates the n=0,cos(phi),sin(phi)
representation exactly in phi; no derivative DOFs are mistaken for field values.
This is a diagnostic reader, not an export of native equation residuals.
"""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mastu_opsplit/src'))
from mastu_opsplit.restart import cubic_basis

FIELDS = ('psi', 'u', 'j', 'omega', 'rho', 'T')
TOR_WEIGHTS = np.array([1., .5, .5])


def mesh(f, order=4):
    for key, expected in dict(jorek_model=199, n_tor=3, n_coord_tor=1,
                              n_order=3, n_degrees=4, n_var=6, n_period=1).items():
        if int(f[key][0]) != expected:
            raise ValueError(f'Unsupported {key}')
    if np.any(f['n_sons'][()] != 0) or np.any(f['constrained'][()] != b'F'):
        raise ValueError('Refinement/constraints require a native elimination map')
    vertex = f['vertex'][()].T-1
    sizes = f['size'][()].transpose(2, 1, 0)
    x = f['x'][()]
    xy = x[:, :, 0, :].transpose(2, 1, 0)
    points, weights = np.polynomial.legendre.leggauss(order)
    s, t = np.meshgrid((points+1)/2, (points+1)/2, indexing='ij')
    h, hs, ht = cubic_basis(s.ravel(), t.ravel())
    basis, bs, bt = [np.einsum('qvd,evd->eqvd', a, sizes).reshape(len(vertex), -1, 16)
                     for a in (h, hs, ht)]
    local_xy = xy[vertex].reshape(len(vertex), 16, 2)
    rz, rs, rt = [np.einsum('eql,elc->eqc', a, local_xy) for a in (basis, bs, bt)]
    jac = rs[..., 0]*rt[..., 1]-rs[..., 1]*rt[..., 0]
    if np.any(jac <= 0) or not np.isfinite(jac).all():
        raise ValueError('Nonpositive geometry Jacobian')
    vol = 2*np.pi*rz[..., 0]*jac*np.outer(weights/2, weights/2).ravel()
    return dict(vertex=vertex, sizes=sizes, x=x, basis=basis, rz=rz, volume=vol,
                minimum_jacobian=float(jac.min()))


def values(f, geo):
    for name, expected in [('vertex', geo['vertex'].T+1), ('size', geo['sizes'].transpose(2,1,0)), ('x',geo['x'])]:
        if not np.array_equal(f[name][()],expected):
            raise ValueError('Mesh changed across snapshots')
    v=f['values'][()]
    if v.shape != (6,4,3,geo['x'].shape[-1]) or not np.isfinite(v).all():
        raise ValueError('Invalid coefficients')
    local=v.transpose(0,2,3,1)[:,:,geo['vertex'],:].reshape(6,3,len(geo['vertex']),16)
    return np.einsum('eql,ftel->fteq',geo['basis'],local)


def norm(a, volume):
    """Per-field volume RMS, with optional leading time dimension."""
    return np.sqrt(np.sum(a*a*TOR_WEIGHTS[:,None,None]*volume,axis=(-3,-2,-1))/volume.sum())


def ratio(a,b):
    return np.divide(a,b,out=np.full_like(np.asarray(a,dtype=float),np.nan),where=np.asarray(b)>1e-30)


def clean(obj):
    if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
    if isinstance(obj,np.ndarray):return clean(obj.tolist())
    if isinstance(obj,(np.floating,float)):return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,np.integer):return int(obj)
    return obj


def load_case(folder,order=4):
    paths=sorted(folder.glob('jorek[0-9][0-9][0-9][0-9][0-9][0-9].h5'))
    if (folder/'initial.h5').exists():paths.insert(0,folder/'initial.h5')
    if not paths:raise ValueError(f'No frames in {folder}')
    with h5py.File(paths[0]) as f: geo=mesh(f,order)
    rows=[];times=[];indices=[]
    for p in paths:
        with h5py.File(p) as f:
            rows.append(values(f,geo));times.append(float(f['t_now'][0])*float(f['sqrt_mu0_rho0'][0]));indices.append(int(f['index_now'][0]))
    if np.any(np.diff(times)<=0):raise ValueError('Nonmonotonic times')
    return np.array(rows),np.array(times),indices,geo,paths


def case_metrics(a,t,geo):
    w=geo['volume']; start=norm(a[0],w); drift=norm(a[-1]-a[0],w)
    centered=a-a.mean(axis=0); std=np.sqrt(np.mean(norm(centered,w)**2,axis=0))
    nonaxis=a.copy();nonaxis[:,:,0]=0
    n1=norm(nonaxis,w)
    amplitude=np.sqrt(a[:,:,1]**2+a[:,:,2]**2)
    minima=(a[:,:,0]-amplitude).min(axis=(0,2,3))
    maxima=(a[:,:,0]+amplitude).max(axis=(0,2,3))
    persistence={}; slope={}
    for lag in (1,5,10):
        if len(a)>lag:
            err=norm(a[lag:]-a[:-lag],w)
            persistence[str(lag)]=ratio(np.sqrt(np.mean(err**2,axis=0)),std)
        if len(a)>lag+1:
            dt=(t[1:-lag]-t[:-lag-1]);future=t[lag+1:]-t[1:-lag]
            pred=a[1:-lag]+(a[1:-lag]-a[:-lag-1])*(future/dt)[:,None,None,None,None]
            slope[str(lag)]=ratio(np.sqrt(np.mean(norm(a[lag+1:]-pred,w)**2,axis=0)),std)
    fields={}
    for i,name in enumerate(FIELDS):
        weighted=(centered[:,i]*np.sqrt(TOR_WEIGHTS[:,None,None]*w/w.sum())).reshape(len(a),-1)
        eigen=np.maximum(np.linalg.eigvalsh(weighted@weighted.T)[::-1],0)
        rank=int(np.searchsorted(np.cumsum(eigen)/eigen.sum(),.99)+1) if eigen.sum()>0 else 0
        fields[name]=dict(initial_rms=start[i],final_minus_initial_rms=drift[i],
                         relative_endpoint_change=ratio(drift[i],start[i]),temporal_std_rms=std[i],
                         temporal_std_over_initial_rms=ratio(std[i],start[i]),
                         n1_initial_rms=n1[0,i],n1_final_rms=n1[-1,i],n1_growth=ratio(n1[-1,i],n1[0,i]),
                         minimum=minima[i],maximum=maxima[i],temporal_rank_99=rank,
                         persistence_error_over_temporal_std={k:v[i] for k,v in persistence.items()},
                         slope_error_over_temporal_std={k:v[i] for k,v in slope.items()})
    return dict(frames=len(a),time_start_seconds=t[0],time_end_seconds=t[-1],
                volume_m3=w.sum(),minimum_jacobian=geo['minimum_jacobian'],fields=fields),n1


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--refinement',type=Path);args=p.parse_args()
    run=json.loads((args.run/'run_report.json').read_text())
    report={'scope':'Pilot field activity and temporal sensitivity; no trained AR/NODE/OpsSplit comparison.',
            'metric':'Volume-weighted native bicubic field values; exact toroidal mean-square integration.',
            'cases':{},'comparisons':{},'quadrature_order':4}
    loaded={}; curves={}
    folders={name:args.run/name for name in run['cases']}
    if args.refinement:
        refined=json.loads((args.refinement/'run_report.json').read_text())
        run['cases'].update(refined['cases'])
        folders.update({name:args.refinement/name for name in refined['cases']})
    for name,result in run['cases'].items():
        if name=='regression' or not result.get('healthy_completion'):continue
        a,t,idx,g,paths=load_case(folders[name])
        report['cases'][name],n1=case_metrics(a,t,g)
        report['cases'][name]['indices']=idx
        # Compare forecasting controls at the same physical cadence, 3000 native units.
        stride=round(3000/result['dt_native'])
        if stride>1:
            sampled,_=case_metrics(a[::stride],t[::stride],g)
            report['cases'][name]['matched_cadence_fields']=sampled['fields']
        loaded[name]=(a,t,g);curves[name]=(t,n1)
    if 'baseline' in loaded:
        base,t,g=loaded['baseline'];activity=norm(base[-1]-base[0],g['volume'])
        for name in ('half_dt','eta_x2','quarter_dt'):
            if name not in loaded:continue
            other,to,go=loaded[name]
            if not np.array_equal(g['x'],go['x']) or abs(to[-1]-t[-1])>1e-12:raise ValueError('Comparison mismatch')
            error=norm(other[-1]-base[-1],g['volume'])
            report['comparisons'][name]={f:dict(absolute_endpoint_difference=error[i],
                relative_to_baseline_evolution=ratio(error[i],activity[i]),
                relative_to_baseline_field=ratio(error[i],norm(base[-1],g['volume'])[i])) for i,f in enumerate(FIELDS)}
        if 'quarter_dt' in loaded and 'half_dt' in loaded:
            coarse,tc,_=loaded['half_dt']; fine,tf,_=loaded['quarter_dt']
            if not np.allclose(tc,tf[::2],rtol=0,atol=1e-12):raise ValueError('Refinement timestamps differ')
            err=norm(fine[::2]-coarse,g['volume'])
            evolution=norm(fine[-1]-fine[0],g['volume'])
            std=np.sqrt(np.mean(norm(fine-fine.mean(axis=0),g['volume'])**2,axis=0))
            report['comparisons']['quarter_vs_half']={f:dict(endpoint_difference_over_evolution=ratio(err[-1,i],evolution[i]),
                trajectory_rms_error_over_temporal_std=ratio(np.sqrt(np.mean(err[:,i]**2)),std[i]),
                endpoint_difference_over_final_field=ratio(err[-1,i],norm(fine[-1],g['volume'])[i])) for i,f in enumerate(FIELDS)}
        # Quadrature convergence for the same native fields (not mesh convergence).
        last=args.run/'baseline'/f'jorek{run["cases"]["baseline"]["final_step"]:06d}.h5'
        with h5py.File(last) as f:
            g8=mesh(f,8);v8=values(f,g8)
        with h5py.File(last) as f:
            report['native_integral_crosscheck']=dict(volume_relative_error=abs(g8['volume'].sum()/f['volume_t'][-1]-1),
                density_integral_native=float(np.sum(g8['volume']*v8[4,0])),saved_density_total_SI=float(f['density_tot_t'][-1]),
                density_relative_error=abs(np.sum(g8['volume']*v8[4,0])*float(f['central_density'][0])*1e20/f['density_tot_t'][-1]-1))
        report['quadrature_q4_vs_q8_relative_rms_difference']=ratio(abs(norm(base[-1],g['volume'])-norm(v8,g8['volume'])),norm(v8,g8['volume']))
    (args.run/'analysis.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    for i,ax in enumerate(axes.flat):
        for name,(t,n1) in curves.items():ax.semilogy(t*1000,np.maximum(n1[:,i],1e-30),label=name)
        ax.set(title=FIELDS[i],xlabel='Time (ms)',ylabel='n=1 volume RMS (native units)');ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);fig.suptitle('Model199 tearing pilot: non-axisymmetric field evolution')
    fig.savefig(args.run/'field_evolution.png',dpi=160);plt.close(fig)
    if 'baseline' in loaded:
        a,t,g=loaded['baseline'];fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
        for i,ax in enumerate(axes.flat):
            change=(a[-1,i,0]+a[-1,i,1])-(a[0,i,0]+a[0,i,1]);limit=np.max(abs(change))
            im=ax.scatter(g['rz'][...,0].ravel(),g['rz'][...,1].ravel(),c=change.ravel(),s=2,cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.set(title=FIELDS[i],xlabel='R (m)',ylabel='Z (m)',aspect='equal');fig.colorbar(im,ax=ax,shrink=.75)
        fig.suptitle('Final minus initial field at toroidal angle phi=0 (native units)')
        fig.savefig(args.run/'field_changes.png',dpi=160);plt.close(fig)
    print(json.dumps(clean(report),indent=2))


if __name__=='__main__':main()
