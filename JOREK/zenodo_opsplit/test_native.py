import math
import pytest
import torch

from native import (N0, MU0, EV, TAU, F0, decode_release, encode_release,
                    read_input, common_coefficients, MomentumSolve, NativeReference, wall_trace)


def grid(n):
    r = torch.linspace(9.5, 10.5, n+2, dtype=torch.float64)[1:-1]
    z = torch.linspace(-.5, .5, n+2, dtype=torch.float64)[1:-1]
    zz, rr = torch.meshgrid(z, r, indexing='ij')
    return r, z, rr, zz


def coefficients(case='es'):
    return common_coefficients(dict(d_perp=1e-6, zk_perp=1e-6, visco=1e-5,
                                    tstep=.5, rho_0=.05, t_0=5e-6), case)


@pytest.mark.parametrize('case', ['es', 'em'])
@pytest.mark.parametrize('residual', [False, True])
def test_rejected_neural_trials_fall_back_to_exact_analytic_step_and_adjoint(case, residual):
    from native_rollout import NativeRollout

    class RejectCorrection(torch.nn.Module):
        is_residual = residual

        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor(0.))

        def forward(self, state):
            raise FloatingPointError('Manufactured rejected neural trial')

    r, z, rr, zz = grid(15)
    c = coefficients(case)
    c.update(viscosity='old', conservative=False)
    shape = torch.sin(math.pi*(rr-9.5))*torch.sin(math.pi*(zz+.5))
    rho = c['rho_wall']+.01*shape
    fields = [rho, rho*c['T_wall'], .1*shape*(1+.3*torch.cos(2*math.pi*zz))]
    if case == 'em':
        fields.append(.001*shape)
    state = torch.stack(fields)[None]
    reference = NativeReference(r, z, wall_trace(state, r, z, c), c)
    analytic = NativeRollout(reference, state)
    learned = NativeRollout(reference, state, RejectCorrection())
    initial = state.clone().requires_grad_()
    fallback_initial = state.clone().requires_grad_()
    expected = analytic.step(initial, .02, substeps=1)
    actual = learned.step(fallback_initial, .02, substeps=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.linspace(.1, 1., state.numel(), dtype=state.dtype).reshape_as(state)
    expected_gradient, = torch.autograd.grad((expected*weights).sum(), initial)
    actual_gradient, = torch.autograd.grad((actual*weights).sum(), fallback_initial)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    assert learned.limiter_stats['analytic_fallbacks'] == analytic.limiter_stats['substeps'] > 0
    assert learned.limiter_stats['energy_limited'] == 0


def test_native_conversion_round_trip():
    q = torch.randn(2, 4, 9, 9, dtype=torch.float64)
    q[:, :2] = q[:, :2].abs()+1
    w, j = q[:, 0]*.01, q[:, 0]*.02
    raw = encode_release(q, w, j)
    torch.testing.assert_close(decode_release(raw), q)
    torch.testing.assert_close(raw[:, 2], -q[:, 2]*F0/TAU)
    torch.testing.assert_close(raw[:, 3], w)


def test_input_rescaling(tmp_path):
    p=tmp_path/'input'
    p.write_text('restart=.t.\ntstep=0.005 ! note\nd_perp=1.d-4\nzk_perp=1.d-4\nvisco=1.d-3\nvisco_num=1.d-6\nrho_0=0.05\nt_0=0.05\n')
    c=common_coefficients(read_input(p),'es')
    assert c['nominal_frame_dt']==5 and c['d']==1e-6 and c['mu_hyper']==1e-8
    assert c['T_wall']==5e-6


def test_momentum_continuum_refinement():
    errors=[]
    for n in (15,31):
        r,z,rr,zz=grid(n)
        exact=torch.sin(math.pi*(rr-9.5))*torch.sin(math.pi*(zz+.5))
        solver=MomentumSolve(r,z)
        x=solver.solve(torch.ones_like(rr),2*math.pi**2*exact,tolerance=1e-10)
        errors.append(torch.linalg.vector_norm(x-exact)/torch.linalg.vector_norm(exact))
    assert errors[1] < errors[0]/3.8


def test_variable_mass_solve_and_autograd():
    r,z,rr,zz=grid(9)
    solver=MomentumSolve(r,z)
    a=(rr**3*(.1+torch.cos(zz))).requires_grad_()
    exact=torch.sin(math.pi*(rr-9.5))*torch.sin(math.pi*(zz+.5))
    cr,cz=solver.edges(a)
    rhs=(solver.apply(exact,cr,cz)/solver.volume).detach().requires_grad_()
    value=solver.solve(a,rhs,tolerance=1e-10)
    torch.testing.assert_close(value,exact,rtol=1e-8,atol=1e-8)
    value.square().sum().backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(rhs.grad).all()
    with pytest.raises(ValueError,match='Positive'):
        solver.solve(a*0,rhs)


@pytest.mark.parametrize('case', ['es','em'])
def test_native_equilibrium_and_boundary_does_not_clamp_samples(case):
    r,z,rr,zz=grid(13)
    c=coefficients('em'); c['conservative']=case=='em'
    q=torch.zeros(1,4 if case=='em' else 3,13,13,dtype=torch.float64)
    q[:,0]=c['rho_wall']; q[:,1]=c['rho_wall']*c['T_wall']
    if case=='em':q[:,3]=.1
    wall=wall_trace(q,r,z,c)
    reference=NativeReference(r,z,wall,c)
    terms=reference.terms(q)
    for value in terms.values():
        assert value.abs().max()<1e-10
    perturbed=q.clone();perturbed[...,0,0]+=1e-6
    torch.testing.assert_close(reference.extend(perturbed)[...,1:-1,1:-1],perturbed)
    assert torch.count_nonzero(wall[...,1:-1,1:-1])==0


def test_pressure_load_sign_and_temperature_coupling():
    r,z,rr,zz=grid(13)
    c=coefficients('em');c['viscosity']='old';c['conservative']=False
    q=torch.stack((rr*0+1,zz*.01+1,rr*0))[None]
    ref=NativeReference(r,z,wall_trace(q,r,z,c),c)
    terms=ref.terms(q)
    torch.testing.assert_close(terms['fluid_momentum'][...,2:-2,2:-2],-.02*rr[2:-2,2:-2][None])
    # Evolving rho at fixed p automatically lowers T; no independent T Laplacian.
    updated=q.clone();updated[:,0]*=2
    torch.testing.assert_close(encode_release(updated,rr[None]*0)[:,1],encode_release(q,rr[None]*0)[:,1]/2)


def test_separate_native_modules_backward_and_transfer():
    from native_terms import NativeOperators
    norms=dict(input_mean=torch.zeros(4),input_scale=torch.ones(4),target_scale=torch.ones(5))
    model=NativeOperators(**norms,width=4,modes=4,layers=(1,1,1,1))
    x=torch.randn(2,4,12,12)
    model(x).square().mean().backward()
    assert all(any(p.grad is not None for p in block.parameters()) for block in model.blocks.values())
    target=NativeOperators(**norms,width=4,modes=4,layers=(1,1,1,1))
    target.transfer(model,['scalar_advection','particle_diffusion'],freeze=True)
    assert all(not p.requires_grad for p in target.blocks['scalar_advection'].parameters())
    assert all(p.requires_grad for p in target.blocks['fluid_momentum'].parameters())
    clone=NativeOperators(**norms,width=4,modes=4,layers=(1,1,1,1))
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone(x),model(x))


def test_imex_keeps_equilibrium_and_dissipates_temperature():
    from native_imex import IMEXReference
    r,z,rr,zz=grid(13)
    c=coefficients('em');c.update(viscosity='old',conservative=False)
    q=torch.zeros(1,3,13,13,dtype=torch.float64)
    q[:,0]=c['rho_wall'];q[:,1]=c['rho_wall']*c['T_wall']
    ref=NativeReference(r,z,wall_trace(q,r,z,c),c)
    solver=IMEXReference(ref,q)
    out=solver.step(q,5,2)
    torch.testing.assert_close(out,q,atol=1e-15,rtol=1e-8)
    q[:,1]+=1e-9*torch.sin(math.pi*(rr-9.5))*torch.sin(math.pi*(zz+.5))
    # Suppress nonlinear force for the isolated heat-conduction manufactured check.
    original=ref.terms
    def diffusion_only(state):
        terms=original(state)
        for key in ('scalar_advection','fluid_momentum','conservative','lorentz'):
            terms[key]=terms[key]*0
        return terms
    ref.terms=diffusion_only
    out=solver.step(q,5,4)
    assert (out[:,1]-c['rho_wall']*c['T_wall']).norm() < (q[:,1]-c['rho_wall']*c['T_wall']).norm()
    torch.testing.assert_close(out[:,0],q[:,0])


def test_sparse_implicit_adjoint_matches_finite_differences():
    from native_rollout import SparseSolve
    values=torch.tensor([3.,-.4,.2,2.],dtype=torch.float64,requires_grad=True)
    rhs=torch.tensor([.1,.7],dtype=torch.float64,requires_grad=True)
    indices=torch.tensor([[0,0,1,1],[0,1,0,1]])
    assert torch.autograd.gradcheck(lambda a,b:SparseSolve.apply(a,b,indices),(values,rhs))


@pytest.mark.parametrize('case',['es','em'])
def test_differentiable_imex_matches_reference_and_state_derivative(case):
    from native_imex import IMEXReference
    from native_rollout import NativeRollout
    r,z,rr,zz=grid(11)
    c=coefficients('em');c.update(viscosity='old',conservative=case=='em',eta=1e-5,eta_hyper=1e-8)
    shape=torch.sin(math.pi*(rr-9.5))*torch.sin(math.pi*(zz+.5))
    q=torch.stack((c['rho_wall']+shape*.001,c['rho_wall']*c['T_wall']+shape*1e-9,shape*1e-8))[None]
    if case=='em':q=torch.cat((q,(.1+shape*1e-5)[None,None]),1)
    ref=NativeReference(r,z,wall_trace(q,r,z,c),c)
    solver=NativeRollout(ref,q)
    expected=IMEXReference(ref,q).step(q,1.,2)
    actual=solver.step(q,1.,2)
    torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-14)
    # Derivative of pressure amplification includes the density-dependent solves.
    alpha=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    direction=q*0;direction[:,1]=shape*1e-9
    value=solver.step(q+alpha*direction,1.,2)[:,1].sum()
    derivative=torch.autograd.grad(value,alpha)[0]
    eps=1e-4
    numeric=(solver.step(q+(1+eps)*direction,1.,2)[:,1].sum()-solver.step(q+(1-eps)*direction,1.,2)[:,1].sum())/(2*eps)
    torch.testing.assert_close(derivative,numeric,rtol=1e-5,atol=1e-13)


def test_rollout_backpropagates_to_each_learned_operator():
    from native_terms import NativeOperators
    from native_rollout import NativeRollout
    r,z,rr,zz=grid(11);c=coefficients('em');c.update(viscosity='old',conservative=False)
    shape=torch.sin(math.pi*(rr-9.5))*torch.sin(math.pi*(zz+.5))
    # Nonzero physical forcing: equilibrium cannot supply neural kinetic energy.
    q=torch.stack((c['rho_wall']+shape*.001,c['rho_wall']*c['T_wall']+shape*1e-8,shape*1e-7))[None]
    ref=NativeReference(r,z,wall_trace(q,r,z,c),c)
    model=NativeOperators(torch.zeros(4),torch.ones(4),torch.ones(5)*1e-9,width=4,modes=4,layers=(1,1,1,1))
    solver=NativeRollout(ref,q,model)
    result=solver.step(q,1.,1)
    (result-q).square().sum().backward()
    for block in model.blocks.values():
        grads=[p.grad for p in block.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0


def test_native_positivity_guard_limits_correction_without_clipping_state():
    from native_rollout import NativeRollout
    r,z,rr,zz=grid(11);c=coefficients('em');c.update(viscosity='old',conservative=False)
    q=torch.stack((rr*0+c['rho_wall'],rr*0+c['rho_wall']*c['T_wall'],rr*0))[None]
    ref=NativeReference(r,z,wall_trace(q,r,z,c),c)
    class BadPressure(torch.nn.Module):
        def __init__(self):
            super().__init__();self.rate=torch.nn.Parameter(torch.tensor(-1e-5));self.target_scale=torch.ones(5)
        def forward(self,x):
            value=x[:,0:1]*0
            return torch.cat((value,value+self.rate,value,value,value),1)
    model=BadPressure();solver=NativeRollout(ref,q,model)
    result=solver.step(q,1.,1)
    assert torch.all(result[:,1]>0)
    assert solver.limiter_stats['temperature_limited']==1
    assert 0<solver.limiter_stats['minimum_scalar_fraction']<1
    result[:,1].sum().backward()
    assert torch.isfinite(model.rate.grad) and model.rate.grad>0
    torch.testing.assert_close(result[:,0],q[:,0])


def test_zero_density_windows_are_reported_and_excluded():
    from train_native_rollout import admissible_windows
    q=torch.ones(2,8,3,11,11)
    q[0,3,0,2,2]=0
    accepted,excluded=admissible_windows(q,5)
    assert accepted==[(1,0),(1,1),(1,2)]
    assert excluded==[(0,0),(0,1),(0,2)]


def test_native_launcher_defaults_to_smoke_after_terms(tmp_path,monkeypatch):
    import json
    import orchestrate_native
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    monkeypatch.setattr('sys.argv',['orchestrate_native.py','--manifest',str(manifest),'--output',str(tmp_path/'run'),'--case','em'])
    orchestrate_native.main()
    jobs=json.loads((tmp_path/'run/jobs.json').read_text())
    assert [j['stage'] for j in jobs]==['targets','term-training','rollout']
    assert all(j['job_id'] is None for j in jobs)
    assert '--smoke' in (tmp_path/'run/rollout.sbatch').read_text()
    assert '--resume' in (tmp_path/'run/rollout.sbatch').read_text()
    assert '--case em' in (tmp_path/'run/targets.sbatch').read_text()


def test_low_temperature_enhanced_conduction_is_implicit():
    from native_rollout import NativeRollout
    from native_imex import IMEXReference
    r,z,rr,zz=grid(21);c=coefficients('em')
    c.update(viscosity='old',conservative=False,T_threshold=5e-7,k_negative=1e-4)
    q=torch.stack((rr*0+c['rho_wall'],rr*0+c['rho_wall']*1e-8,rr*0))[None]
    ref=NativeReference(r,z,wall_trace(q,r,z,c),c)
    out=NativeRollout(ref,q).step(q,5.,1)
    expected=IMEXReference(ref,q).step(q,5.,1)
    torch.testing.assert_close(out,expected,rtol=1e-8,atol=1e-14)
    assert torch.all(out[:,1]>0)
    assert out[:,1].min()>q[:,1].min()


def test_full_native_launcher_uses_gpu_and_checkpointed_requeue(tmp_path,monkeypatch):
    import json
    import orchestrate_native
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    pretrained=tmp_path/'terms.pt';pretrained.write_text('fixture')
    monkeypatch.setattr('sys.argv',['orchestrate_native.py','--manifest',str(manifest),'--pretrained',str(pretrained),
                                   '--output',str(tmp_path/'run'),'--case','es','--full-rollout'])
    orchestrate_native.main()
    jobs=json.loads((tmp_path/'run/jobs.json').read_text())
    assert [j['stage'] for j in jobs]==['validate-windows','rollout']
    assert '--requeue' in jobs[1]['submission']
    assert '--validation-report' in (tmp_path/'run/rollout.sbatch').read_text()
    assert '--dependency=afterok:<validate-windows-job-id>' in jobs[1]['submission']
    body=(tmp_path/'run/rollout.sbatch').read_text()
    assert '--elliptic-device cuda' in body and '--max-seconds 79200' in body
    assert '--smoke' not in body and 'scontrol requeue "$SLURM_JOB_ID"' in body
    assert 'training_status=$?' in body and 'exit "$training_status"' in body


def test_upwind_transport_is_conservative_and_positive_implicit():
    from native_rollout import NativeRollout,solve
    r,z,rr,zz=grid(11);c=coefficients('em');c.update(viscosity='old',conservative=False)
    q=torch.stack((rr*0+c['rho_wall'],rr*0+c['rho_wall']*c['T_wall'],
                   torch.sin(math.pi*(rr-9.5))*torch.sin(math.pi*(zz+.5))*.01))[None]
    ref=NativeReference(r,z,wall_trace(q,r,z,c),c);solver=NativeRollout(ref,q)
    transport=solver.upwind_transport(ref.extend(q))
    dense=transport.to_dense()
    torch.testing.assert_close(dense.sum(0),torch.zeros(solver.n,dtype=torch.float64),atol=1e-14,rtol=0)
    assert torch.all((dense-torch.diag(dense.diag()))<=0)
    initial=q[0,0].flatten()
    result=solve(solver.W+10*transport,solver.V*solver.R*initial)
    assert torch.all(result>0)
    torch.testing.assert_close((solver.V*solver.R*result).sum(),(solver.V*solver.R*initial).sum())


def test_scalar_limiter_global_fraction_and_gradient():
    from native_rollout import NativeRollout
    solver=object.__new__(NativeRollout)
    solver.limiter_stats=dict(density_limited=0,minimum_scalar_fraction=1.)
    low=torch.tensor([1.,2.,3.],dtype=torch.float64,requires_grad=True)
    high=torch.tensor([-1.,4.,2.],dtype=torch.float64,requires_grad=True)
    result=solver.positive_scalar(high,low,'density')
    torch.testing.assert_close(result,low+.495*(high-low))
    assert torch.all(result>0)
    assert torch.autograd.gradcheck(lambda a,b:solver.positive_scalar(a,b,'density'),(high,low))


def test_full_training_rejects_stale_or_partial_window_audit(tmp_path):
    import json
    from train_native_rollout import require_preflight
    provenance=dict(case='es',manifest_sha256='manifest',source_sha256='weights',code_sha256={'solver':'version'},
                    train_ids=['train'],val_ids=['val'],protocol=dict(substeps=8,blend=1.))
    report=dict(passed=True,samples=0,case='es',manifest_sha256='manifest',checkpoint_sha256='weights',
                code_sha256={'solver':'version'},train_ids=['train'],val_ids=['val'],substeps=8)
    path=tmp_path/'audit.json';path.write_text(json.dumps(report));require_preflight(path,provenance)
    for key,value in [('passed',False),('samples',1),('code_sha256',{}),('substeps',16),('checkpoint_sha256','other')]:
        path.write_text(json.dumps(dict(report,**{key:value})))
        with pytest.raises(ValueError):require_preflight(path,provenance)


def test_nonpositive_pressure_source_windows_are_reported():
    from train_native_rollout import admissible_windows
    q=torch.ones(1,8,3,5,5);q[0,3,1,2,2]=-1
    accepted,excluded=admissible_windows(q,5)
    assert not accepted and excluded==[(0,0),(0,1),(0,2)]


def test_upwind_transport_converges_to_native_density_advection():
    from native_rollout import NativeRollout,mv
    errors=[]
    for n in (23,47):
        r,z,rr,zz=grid(n);c=coefficients('em');c.update(viscosity='old',conservative=False)
        shape=torch.sin(math.pi*(rr-9.5))*torch.sin(math.pi*(zz+.5))
        rho=c['rho_wall']+.02*shape
        q=torch.stack((rho,rho*c['T_wall'],shape*.001))[None]
        ref=NativeReference(r,z,wall_trace(q,r,z,c),c);solver=NativeRollout(ref,q)
        flux=solver.upwind_transport(ref.extend(q))
        actual=(-mv(flux,rho.flatten())/(solver.V*solver.R)).reshape_as(rho)
        expected=ref.terms(q)['scalar_advection'][0,0]
        errors.append((actual[2:-2,2:-2]-expected[2:-2,2:-2]).square().mean().sqrt())
    assert errors[1]<errors[0]/1.7


@pytest.mark.parametrize('case',['es','em'])
def test_native_explicit_cfl_tracks_fluid_and_magnetic_speeds(case):
    from native_rollout import NativeRollout
    r,z,rr,zz=grid(15);c=coefficients('em');c.update(viscosity='old',conservative=case=='em')
    shape=torch.sin(math.pi*(rr-9.5))*torch.sin(math.pi*(zz+.5))
    q=torch.stack((rr*0+c['rho_wall'],rr*0+c['rho_wall']*c['T_wall'],shape*.1 if case=='es' else rr*0))[None]
    if case=='em':q=torch.cat((q,(rr*.1)[None,None]),1)
    ref=NativeReference(r,z,wall_trace(q,r,z,c),c);solver=NativeRollout(ref,q)
    first=solver.stable_timestep(q,10.)
    doubled=q.clone();doubled[:,2 if case=='es' else 3]*=2
    ref2=NativeReference(r,z,wall_trace(doubled,r,z,c),c);solver2=NativeRollout(ref2,doubled)
    second=solver2.stable_timestep(doubled,10.)
    assert 0<first<10
    assert second==pytest.approx(first/2)


def test_adaptive_step_advances_the_full_requested_interval():
    from native_rollout import NativeRollout
    solver=object.__new__(NativeRollout);solver.operators=None;solver.positivity_guard=False;solver.checkpoint_steps=False;solver.reset_limiter_stats()
    solver.stable_timestep=lambda q,h:min(.3,h)
    solver._trial_step=lambda q,h,correction_scale:q+h
    result=solver.step(torch.tensor(0.),1.,1)
    torch.testing.assert_close(result,torch.tensor(1.))
    assert solver.limiter_stats['substeps']==4
    solver.stable_timestep=lambda q,h:1e-14
    with pytest.raises(FloatingPointError,match='CFL'):solver.step(torch.tensor(0.),1.,1)





def test_positive_density_correction_respects_transport_bound():
    from native_rollout import NativeRollout
    solver=object.__new__(NativeRollout);solver.reset_limiter_stats()
    # Merely positive high-order density can still make momentum nearly singular.
    high=torch.tensor([1e-10,.3],dtype=torch.float64,requires_grad=True)
    low=torch.tensor([.1,.2],dtype=torch.float64)
    result=solver.positive_scalar(high,low,'density',lower=.05)
    assert torch.all(result>=.05) and torch.all(result>0)
    fraction=(result[1]-low[1])/(high[1]-low[1])
    torch.testing.assert_close(result,low+fraction*(high-low))
    assert 0<fraction<1
    result.sum().backward();assert torch.isfinite(high.grad).all()


def test_kinetic_envelope_limits_growth_and_preserves_forcing_and_gradient():
    from native_rollout import NativeRollout,diagonal,mv
    solver=object.__new__(NativeRollout);solver.reset_limiter_stats()
    old=torch.tensor([.2,-.1],dtype=torch.float64)
    mass=diagonal(torch.tensor([2.,3.],dtype=torch.float64))
    new_mass=diagonal(torch.tensor([3.,2.],dtype=torch.float64))
    forced=old*1.2
    high=torch.tensor([1.,-.7],dtype=torch.float64,requires_grad=True)
    def limited(value):return solver.limit_kinetic_energy(old,value,forced,mass,new_mass)
    result=limited(high)
    ceiling=torch.maximum(.5*(old*mv(mass,old)).sum(),.5*(forced*mv(new_mass,forced)).sum())
    torch.testing.assert_close(.5*(result*mv(new_mass,result)).sum(),ceiling)
    torch.testing.assert_close(result[0]/high[0],result[1]/high[1])
    assert torch.autograd.gradcheck(limited,(high,))
    torch.testing.assert_close(limited(forced),forced)
    torch.testing.assert_close(limited(high*.01),high*.01)
    # Without external work, artificial momentum cannot create kinetic energy.
    zero=torch.zeros_like(old)
    torch.testing.assert_close(solver.limit_kinetic_energy(zero,high,zero,mass,mass),zero)


def test_conservative_momentum_includes_learned_density_advection_defect():
    from native_terms import labels
    from native_rollout import NativeRollout,solve
    r,z,rr,zz=grid(15);c=coefficients('em');c.update(viscosity='old',conservative=False)
    shape=torch.sin(math.pi*(rr-9.5))*torch.sin(math.pi*(zz+.5))
    q=torch.stack((rr*0+c['rho_wall'],rr*0+c['rho_wall']*c['T_wall'],shape*1e-6))[None]
    ref=NativeReference(r,z,wall_trace(q,r,z,c),c)
    class DensityDefect(torch.nn.Module):
        def __init__(self):
            super().__init__();self.rate=torch.nn.Parameter(torch.tensor(.001,dtype=torch.float64))
            self.register_buffer('base',labels(ref,q).detach())
            self.register_buffer('target_scale',torch.ones(5,dtype=torch.float64))
        def forward(self,x):
            change=torch.zeros_like(self.base);change[:,0]=self.rate
            return self.base+change
    model=DensityDefect();outputs=[]
    for conservative in (False,True):
        cc=dict(c,conservative=conservative)
        reference=NativeReference(r,z,wall_trace(q,r,z,cc),cc)
        solver=NativeRollout(reference,q,model)
        # Isolate the source-derived load identity before energy relaxation.
        solver.limit_kinetic_energy=lambda old,high,forced,old_mass,new_mass:high
        outputs.append(solver._trial_step(q,.1))
    ur,uz=ref.op.grad(ref.extend(q)[:,2])
    delta=torch.nn.functional.pad(solver.mask*model.rate,(1,1,1,1))
    load=-ref.div(ref.r**3*delta*ur,ref.r**3*delta*uz)[:,1:-1,1:-1]
    mass=solver.stiffness_tensor((q[0,0].flatten()*solver.R**3).reshape(solver.shape))
    expected=solve(mass+.1*c['mu']*solver.H+.1*c['mu_hyper']*solver.Hu,-.1*solver.V*load.flatten())
    torch.testing.assert_close((outputs[1][:,2]-outputs[0][:,2]).flatten(),expected,rtol=1e-8,atol=1e-16)
    torch.testing.assert_close(outputs[1][:,:2],outputs[0][:,:2])
    assert expected.norm()>0
    (outputs[1][:,2]-outputs[0][:,2]).square().sum().backward()
    assert torch.isfinite(model.rate.grad) and model.rate.grad.abs()>0


def test_corner_flux_has_no_wall_divergence_amplification():
    from native_rollout import NativeRollout,mv
    r,z,rr,zz=grid(31);c=coefficients('em');c.update(viscosity='old',conservative=False)
    shape=torch.sin(math.pi*(rr-9.5))*torch.sin(math.pi*(zz+.5))
    q=torch.stack((rr*0+c['rho_wall'],rr*0+c['rho_wall']*c['T_wall'],shape*.001))[None]
    ref=NativeReference(r,z,wall_trace(q,r,z,c),c);solver=NativeRollout(ref,q)
    div=mv(solver.upwind_transport(ref.extend(q)),torch.ones(solver.n,dtype=torch.float64))/(solver.V*solver.R)
    # The zero-corner dual-wall closure has at most a 3/2 edge derivative
    # amplification on this grid: bounded O(u_Z), not O(R*u/(wall gap)).
    uz=ref.op.grad(ref.extend(q)[:,2])[1][:,1:-1,1:-1]
    assert div.abs().max()<3.2*uz.abs().max()


@pytest.mark.parametrize('case',['es','em'])
def test_imex_terms_preserve_all_explicit_loads(case):
    r,z,rr,zz=grid(15);c=coefficients('em');c.update(viscosity='old',conservative=case=='em')
    shape=torch.sin(math.pi*(rr-9.5))*torch.sin(math.pi*(zz+.5))
    q=torch.stack((c['rho_wall']+.01*shape,c['rho_wall']*c['T_wall']+shape*1e-8,shape*1e-7))[None]
    if case=='em':q=torch.cat((q,(rr*.01+shape*1e-5)[None,None]),1)
    ref=NativeReference(r,z,wall_trace(q,r,z,c),c)
    full,imex=ref.terms(q),ref.terms(q,imex=True)
    for key in ('scalar_advection','scalar_diffusion','isotropic_diffusion','fluid_momentum','conservative','lorentz','induction'):
        torch.testing.assert_close(full[key],imex[key],rtol=1e-12,atol=1e-18)


def test_substep_checkpoint_preserves_value_gradient_and_limiter_counts():
    from native_rollout import NativeRollout
    r,z,rr,zz=grid(15);c=coefficients('em');c.update(viscosity='old',conservative=False)
    shape=torch.sin(math.pi*(rr-9.5))*torch.sin(math.pi*(zz+.5))
    q=torch.stack((c['rho_wall']+shape*.001,c['rho_wall']*c['T_wall']+shape*1e-8,shape*1e-8))[None]
    ref=NativeReference(r,z,wall_trace(q,r,z,c),c);answers=[]
    for enabled in (False,True):
        state=q.clone().requires_grad_();solver=NativeRollout(ref,q,checkpoint_steps=enabled)
        out=solver.step(state,.2,2);stats=dict(solver.limiter_stats)
        out.square().sum().backward()
        assert solver.limiter_stats==stats
        answers.append((out.detach(),state.grad))
    for a,b in zip(*answers):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-16)
