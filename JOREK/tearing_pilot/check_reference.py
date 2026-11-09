"""Compare the tearing pilot with the bundled upstream regression simulation.

Reference histories inherited by restarted runs are not independent validation.
Only the fresh-start pilot is compared against the reference's first 31 steps.
"""
import hashlib
import json
from pathlib import Path
import tarfile

import h5py
import numpy as np

from analyze import FIELDS, clean, cubic_basis, mesh, norm, values

ROOT = Path(__file__).resolve().parent
JR = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/src/jorek')


def match_physical_points(geo, target):
    """Invert each candidate element at the reference quadrature points."""
    order = int(np.sqrt(target.shape[1]))
    points = (np.polynomial.legendre.leggauss(order)[0]+1)/2
    s0,t0 = np.meshgrid(points,points,indexing='ij')
    s=np.broadcast_to(s0.ravel(),target.shape[:2]).copy()
    t=np.broadcast_to(t0.ravel(),target.shape[:2]).copy()
    local=geo['x'][:,:,0,:].transpose(2,1,0)[geo['vertex']].reshape(len(s),16,2)
    for iteration in range(8):
        h,hs,ht=cubic_basis(s,t)
        basis,bs,bt=[(a*geo['sizes'][:,None,:,:]).reshape(len(s),-1,16) for a in (h,hs,ht)]
        rz,rs,rt=[np.einsum('eql,elc->eqc',a,local) for a in (basis,bs,bt)]
        residual=rz-target
        error=float(np.max(np.linalg.norm(residual,axis=-1)))
        if error<1e-12:
            return {**geo,'basis':basis,'rz':rz},error
        jac=rs[...,0]*rt[...,1]-rs[...,1]*rt[...,0]
        if np.any(jac<=0):raise ValueError('Invalid candidate mapping')
        s-=(residual[...,0]*rt[...,1]-residual[...,1]*rt[...,0])/jac
        t-=(rs[...,0]*residual[...,1]-rs[...,1]*residual[...,0])/jac
    raise ValueError('Physical point matching did not converge')


def compare_fields(reference, candidate):
    with h5py.File(reference) as f, h5py.File(candidate) as g:
        if float(f['t_now'][0]) != float(g['t_now'][0]):
            raise ValueError('Comparison times differ')
        gf, gg = mesh(f, 8), mesh(g, 8)
        if not np.array_equal(gf['vertex'], gg['vertex']):
            raise ValueError('Different mesh connectivity')
        shift = np.max(np.linalg.norm(gf['rz']-gg['rz'], axis=-1))
        gg,match_error=match_physical_points(gg,gf['rz'])
        vf, vg = values(f, gf), values(g, gg)
        delta, scale = norm(vg-vf, gf['volume']), norm(vf, gf['volume'])
        raw = np.max(np.abs(g['values'][()]-f['values'][()]))
        result = dict(reference=str(reference), candidate=str(candidate),
                      time_native=float(f['t_now'][0]),
                      exact_same_geometry=all(np.array_equal(f[k][()], g[k][()]) for k in ('x','size','vertex')),
                      max_query_shift_before_mapping_m=shift, max_query_residual_after_mapping_m=match_error, coefficient_max_abs_error=raw,
                      upstream_values_tolerance_pass=bool(raw <= 1e-12),
                      fields={name:dict(relative_volume_l2_error=delta[i]/scale[i],absolute_volume_rms_error=delta[i]) for i,name in enumerate(FIELDS)})
    return result


def energies(path):
    with h5py.File(path) as f:
        t=f['xtime'][()]*float(f['sqrt_mu0_rho0'][0])
        a=f['energies'][()]
    # Fortran energies(n_tor, magnetic_or_kinetic, time); HDF5 reverses axes.
    # Native jorek2_diagno sums the cosine and sine energies for each n>0.
    return t, np.stack((a[:,0,1]+a[:,0,2],a[:,1,1]+a[:,1,2]),axis=1)


def main():
    out=ROOT/'reference_check';out.mkdir(exist_ok=True)
    version=(JR/'src/reg_tests/testcases/tear_circ_199/.version').read_text().strip()
    archive=JR/'downloads/regdata'/f'tear_circ_199{version}.tgz'
    expected=Path(str(archive)+'.sha256').read_text().split()[0]
    actual=hashlib.sha256(archive.read_bytes()).hexdigest()
    if expected != actual:raise ValueError('Archive checksum differs')
    refs=ROOT/'snapshot'; pilot=ROOT/'run_2773515'
    with tarfile.open(archive) as tar:
        info=tar.extractfile('.info').read().decode()
        members={}
        for name in ('begin.h5','end.h5'):
            digest=hashlib.sha256(tar.extractfile(name).read()).hexdigest()
            if digest != hashlib.sha256((refs/name).read_bytes()).hexdigest():
                raise ValueError('Pilot reference differs from archive')
            members[name]=digest
    comparisons={
        'restart_regression':compare_fields(refs/'end.h5',pilot/'regression/jorek_restart.h5'),
        'continuation_first_step':compare_fields(refs/'end.h5',pilot/'baseline/jorek000031.h5'),
        'independent_startup_step30':compare_fields(refs/'begin.h5',pilot/'startup/jorek000030.h5'),
        'independent_startup_step31':compare_fields(refs/'end.h5',pilot/'startup/jorek000031.h5')}
    tr,er=energies(refs/'end.h5');ts,es=energies(pilot/'startup/jorek_restart.h5')
    if not np.array_equal(tr,ts[:len(tr)]):raise ValueError('History timestamps differ')
    history={}
    for i,name in enumerate(('n1_magnetic','n1_kinetic')):
        ref=er[:,i];candidate=es[:len(tr),i]
        valid=abs(ref)>abs(ref).max()*1e-12
        history[name]=dict(relative_l2_error=np.linalg.norm(candidate-ref)/np.linalg.norm(ref),
                           max_abs_error_over_reference_peak=np.max(abs(candidate-ref))/np.max(abs(ref)),
                           max_pointwise_relative_error_above_floor=np.max(abs((candidate[valid]-ref[valid])/ref[valid])),
                           reference_first=ref[0],reference_last=ref[-1],reference_peak=ref.max(),
                           reference_peak_time_seconds=tr[ref.argmax()])
    with h5py.File(refs/'end.h5') as f,h5py.File(pilot/'startup/jorek_restart.h5') as g:
        versions={'reference':f['RCS_version'][0].decode(),'pilot':g['RCS_version'][0].decode()}
        scalars={}
        for key in ('volume_t','density_tot_t','thermal_tot_t','current_t'):
            ref=f[key][()];cand=g[key][()][:len(ref)]
            scalars[key]=dict(relative_l2_error=np.linalg.norm(cand-ref)/np.linalg.norm(ref))
    report=dict(archive=str(archive),archive_sha256=actual,archive_checksum_verified=True,
                reference_members_sha256=members,reference_info=info,versions=versions,
                reference_scope='Full fields at steps 30/31 only; scalar and mode-energy histories through step31. No independent reference beyond t=93000.',
                comparisons=comparisons,independent_startup_energy_history=history,
                independent_startup_scalar_history=scalars,
                caveats=['Fresh-start pilot emitted HDF5 equilibrium-export errors and remains excluded from accepted data.',
                         'Restart history before step31 was inherited; it is not counted as independent evidence.',
                         'The fresh-start comparison is supplementary: upstream tolerance applies to the one-step restart test.',
                         'Fine-timestep runs are convergence probes, not independent upstream references.',
                         'Agreement at upstream dt=3000 does not establish temporal or mesh convergence.'])
    (out/'comparison.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    tb,eb=energies(pilot/'baseline/jorek_restart.h5')
    for i,ax in enumerate(axes):
        ax.semilogy(tr*1000,er[:,i],color='black',marker='o',ms=3,label='Upstream reference (steps 1–31)')
        ax.semilogy(ts*1000,es[:,i],color='tab:orange',ls='--',label='Fresh startup (HDF5 warnings)')
        new=tb>90000*6.482912854348901e-7
        ax.semilogy(tb[new]*1000,eb[new,i],color='tab:blue',label='Pilot continuation (new steps only)')
        ax.axvline(tr[-1]*1000,color='gray',ls=':',label='End of upstream reference')
        ax.set(title=('n=1 magnetic energy','n=1 kinetic energy')[i],xlabel='Time (ms)',ylabel='Stored native energy');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Tearing pilot versus bundled upstream reference simulation')
    fig.savefig(out/'reference_energy_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(clean(report),indent=2))


if __name__=='__main__':main()
