"""Compare independently initialized growth trajectories at identical physical times."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from analyze import FIELDS, clean, mesh, norm, values
from check_reference import compare_fields, energies, match_physical_points

ROOT=Path(__file__).resolve().parent
WINDOW=(.015,.050)


def relative_rms(error,signal):
    numerator=np.sqrt(np.mean(error**2,axis=0))
    denominator=np.sqrt(np.mean(signal**2,axis=0))
    return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)


def compare(a,b,weights):
    centered=b-b.mean(axis=0)
    dynamic=relative_rms(norm(a-b,weights),norm(centered,weights))
    an=a.copy();bn=b.copy();an[:,:,0]=0;bn[:,:,0]=0
    nonaxis=relative_rms(norm(an-bn,weights),norm(bn,weights))
    full=relative_rms(norm(a-b,weights),norm(b,weights))
    return {f:dict(error_over_temporal_std=dynamic[i],n1_relative_l2_error=nonaxis[i],full_field_relative_l2_error=full[i]) for i,f in enumerate(FIELDS)}


def load_selected(folder,dt,code_times,reference_geo=None):
    samples=[]
    with h5py.File(folder/'jorek_restart.h5') as f:
        own=mesh(f,4)
        if reference_geo is None:reference_geo=own
        if not np.array_equal(own['vertex'],reference_geo['vertex']):raise ValueError('Different topology')
        geo,residual=match_physical_points(own,reference_geo['rz'])
        norm_time=float(f['sqrt_mu0_rho0'][0])
    for time in code_times:
        index=round(time/dt)
        if abs(index*dt-time)>1e-9:raise ValueError('Incommensurate times')
        with h5py.File(folder/f'jorek{index:06d}.h5') as f:
            if abs(float(f['t_now'][0])-time)>1e-9:raise ValueError('Snapshot time mismatch')
            samples.append(values(f,geo))
    return np.array(samples),reference_geo,norm_time,residual


def health_and_curves(folder):
    with h5py.File(folder/'jorek_restart.h5') as f:g=mesh(f,4)
    minimum=np.full(6,np.inf);maximum=np.full(6,-np.inf)
    ts=[];modes=[]
    for path in sorted(folder.glob('jorek[0-9][0-9][0-9][0-9][0-9][0-9].h5')):
        with h5py.File(path) as f:
            a=values(f,g);t=float(f['t_now'][0])*float(f['sqrt_mu0_rho0'][0])
        amplitude=np.sqrt(a[:,1]**2+a[:,2]**2)
        minimum=np.minimum(minimum,(a[:,0]-amplitude).min(axis=(1,2)))
        maximum=np.maximum(maximum,(a[:,0]+amplitude).max(axis=(1,2)))
        a[:,0]=0;ts.append(t);modes.append(norm(a,g['volume']))
    return dict(frames=len(ts),minima=minimum,maxima=maximum,positive_density_temperature=bool(np.all(minimum[4:]>0))),np.array(ts),np.array(modes)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('runs',type=Path,nargs='+');args=parser.parse_args()
    out=args.runs[0]/'convergence';out.mkdir(exist_ok=True)
    cases={}
    for run in args.runs:
        data=json.loads((run/'run_report.json').read_text())
        for name,result in data['cases'].items():
            if name!='regression' and result['healthy_completion']:
                cases[result['dt_native']]=(run/name,result)
    if len(cases)<2:raise ValueError('Need two healthy timestep levels')
    dts=sorted(cases,reverse=True)
    with h5py.File(cases[dts[0]][0]/'jorek_restart.h5') as f:time_norm=float(f['sqrt_mu0_rho0'][0])
    times=np.arange(np.ceil(WINDOW[0]/time_norm/3000),np.floor(WINDOW[1]/time_norm/3000)+1)*3000
    report=dict(window_seconds=WINDOW,comparison_times_seconds=times*time_norm,
                scope='Fixed mesh and n=0,1; fresh-start timestep convergence. No production dataset.',
                criteria=json.loads((ROOT/'growth_protocol.json').read_text())['acceptance'],cases={},pairs={})
    samples={};geo=None;curves={};energy={}
    for dt in dts:
        folder,result=cases[dt]
        a,geo,tn,res=load_selected(folder,dt,times,geo);samples[dt]=a
        h,ts,n1=health_and_curves(folder);curves[dt]=(ts,n1);energy[dt]=energies(folder/'jorek_restart.h5')
        report['cases'][str(dt)]={**result,**h,'physical_mapping_residual_m':res,'directory':str(folder)}
    for coarse,fine in zip(dts[:-1],dts[1:]):
        # Use every saved step of the coarser member, not only the sparsest grid.
        pair_times=np.arange(np.ceil(WINDOW[0]/time_norm/coarse),np.floor(WINDOW[1]/time_norm/coarse)+1)*coarse
        ca,_,_,_=load_selected(cases[coarse][0],coarse,pair_times,geo)
        fa,_,_,_=load_selected(cases[fine][0],fine,pair_times,geo)
        metrics=compare(ca,fa,geo['volume'])
        sparse_metrics=compare(samples[coarse],samples[fine],geo['volume'])
        ec=[];ef=[]
        for dt,selected in ((coarse,ec),(fine,ef)):
            te,e=energy[dt]
            for target in pair_times*time_norm:
                index=int(np.argmin(abs(te-target)))
                if abs(te[index]-target)>1e-12:raise ValueError('Energy timestamps mismatch')
                selected.append(e[index])
        ec=np.array(ec);ef=np.array(ef)
        energy_errors=np.linalg.norm(ec-ef,axis=0)/np.linalg.norm(ef,axis=0)
        passed=all(v['error_over_temporal_std']<=.01 and v['n1_relative_l2_error']<=.01 for v in metrics.values()) and bool(np.all(energy_errors<=.01))
        del ca,fa
        report['pairs'][f'{coarse:g}_vs_{fine:g}']=dict(fields=metrics,common_coarsest_cadence_fields=sparse_metrics,matched_samples=len(pair_times),
            matched_window_seconds=[pair_times[0]*time_norm,pair_times[-1]*time_norm],n1_magnetic_energy_relative_l2_error=energy_errors[0],
            n1_kinetic_energy_relative_l2_error=energy_errors[1],meets_pilot_temporal_target=passed)
    # Use one fixed energy threshold from the upstream endpoint for every dt.
    _,reference_energy=energies(ROOT/'snapshot/end.h5')
    report['growth_timing']={}
    for dt,(te,en) in energy.items():
        threshold=.5*reference_energy[-1,0]
        crossing=np.flatnonzero(en[:,0]>=threshold)
        crossing_time=None
        if len(crossing):
            k=int(crossing[0])
            crossing_time=float(te[k]) if k==0 else float(te[k-1]+(te[k]-te[k-1])*(threshold-en[k-1,0])/(en[k,0]-en[k-1,0]))
        mask=(en[:,0]>reference_energy[-1,0]*1e-6)&(en[:,0]<reference_energy[-1,0]*1e-2)
        slope=None;r2=None
        if mask.sum()>=3:
            x=te[mask];y=np.log(en[mask,0]);fit=np.polyfit(x,y,1)
            slope=float(fit[0]);r2=float(1-np.sum((y-np.polyval(fit,x))**2)/np.sum((y-y.mean())**2))
        report['growth_timing'][str(dt)]=dict(fixed_magnetic_energy_threshold=float(threshold),threshold_crossing_seconds=crossing_time,
            fitted_magnetic_energy_log_growth_per_second=slope,fit_r_squared=r2,
            fit_window_seconds=[float(te[mask][0]),float(te[mask][-1])] if mask.any() else None)
    report['finest_pair_passes']=list(report['pairs'].values())[-1]['meets_pilot_temporal_target']
    report['dataset_ready']=False
    report['remaining_checks']=['Mesh and higher-toroidal-mode convergence','Independent initial-condition families and benchmark split','Native model199 operator/constraint audit']
    coarse=cases[dts[0]][0]
    if (coarse/'jorek000031.h5').exists():
        report['reference_step31']=compare_fields(ROOT/'snapshot/end.h5',coarse/'jorek000031.h5')
        report['patch_vs_old_startup_step31']=compare_fields(ROOT/'run_2773515/startup/jorek000031.h5',coarse/'jorek000031.h5')
    # Initial-state equality is checked separately from evolving trajectories.
    initial={}
    for dt in dts:
        path=cases[dt][0]/'jorek000000.h5'
        if path.exists():
            with h5py.File(path) as f:initial[dt]={k:f[k][()] for k in ('x','size','vertex','values','deltas')}
    if len(initial)==len(dts):
        report['initial_state_max_abs_difference']={str(dt):{k:float(np.max(abs(initial[dt][k]-initial[dts[0]][k]))) for k in initial[dt]} for dt in dts}
    (out/'report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    for i,ax in enumerate(axes.flat):
        for dt,(ts,n1) in curves.items():ax.semilogy(ts*1000,np.maximum(n1[:,i],1e-30),label=f'dt={dt:g}')
        ax.set(title=FIELDS[i],xlabel='Time (ms)',ylabel='n=1 volume RMS',xlim=(15,50));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,title='Native timestep');fig.suptitle('Fresh-start tearing growth: timestep refinement')
    fig.savefig(out/'field_convergence.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for i,ax in enumerate(axes):
        for dt,(ts,e) in energy.items():ax.semilogy(ts*1000,np.maximum(e[:,i],1e-30),label=f'dt={dt:g}')
        ax.set(title=('n=1 magnetic energy','n=1 kinetic energy')[i],xlabel='Time (ms)',ylabel='Stored native energy',xlim=(15,50));ax.grid(alpha=.2)
    axes[0].legend(title='Native timestep');fig.savefig(out/'energy_convergence.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for i,ax in enumerate(axes):
        for dt,(ts,e) in energy.items():ax.plot(ts*1000,e[:,i],label=f'dt={dt:g}')
        ax.set(title=('n=1 magnetic energy','n=1 kinetic energy')[i],xlabel='Time (ms)',ylabel='Stored native energy',xlim=(40,52.5));ax.grid(alpha=.2)
    axes[0].legend(title='Native timestep',fontsize=8)
    fig.suptitle('Nonlinear transition: energy still evolves at the pilot endpoint')
    fig.savefig(out/'nonlinear_transition.png',dpi=160);plt.close(fig)
    print(json.dumps(clean(report),indent=2))


if __name__=='__main__':main()
