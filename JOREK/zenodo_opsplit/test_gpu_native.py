"""GPU-only checks; skipped explicitly without a CUDA allocation."""
import pytest
import torch
from test_native import grid,coefficients
from native import NativeReference,wall_trace
from native_rollout import NativeRollout

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA allocation required')


@pytest.mark.parametrize('case',['es','em'])
def test_gpu_zero_residual_recovers_analytic(case):
    from native_terms import NativeOperators
    from gpu_sparse import clear_cache
    r,z,rr,zz=grid(15);c=coefficients('em');c.update(viscosity='old',conservative=case=='em')
    shape=torch.sin(torch.pi*(rr-9.5))*torch.sin(torch.pi*(zz+.5))
    q=torch.stack((c['rho_wall']+shape*.001,c['rho_wall']*c['T_wall']+shape*1e-8,shape*1e-7))[None]
    if case=='em':q=torch.cat((q,(.1+shape*1e-5)[None,None]),1)
    ref=NativeReference(r,z,wall_trace(q,r,z,c),c)
    model=NativeOperators(torch.zeros(4),torch.ones(4),torch.ones(5)*1e-8,
                          width=4,modes=4,layers=(1,1,1,1),correction_fraction=.1).cuda()
    learned=NativeRollout(ref,q,model,device='cuda')
    expected=NativeRollout(ref,q,device='cuda').step(q.cuda(),1.,2)
    actual=learned.step(q.cuda(),1.,2)
    torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-9)
    actual[:,1].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    clear_cache()


@pytest.mark.parametrize('columns',[1,2])
def test_gpu_sparse_nonsymmetric_adjoint_and_cache(columns):
    from gpu_sparse import GPUSparseSolve,clear_cache
    indices=torch.tensor([[0,0,1,1],[0,1,0,1]],device='cuda')
    values=torch.tensor([3.,-.4,.2,2.],device='cuda',dtype=torch.float64,requires_grad=True)
    rhs=torch.tensor([.1,.7],device='cuda',dtype=torch.float64,requires_grad=True)
    if columns==2:rhs=torch.stack((rhs,rhs*2),1).detach().requires_grad_()
    first=GPUSparseSolve.apply(values,rhs,indices)
    torch.testing.assert_close(first,torch.linalg.solve(values.reshape(2,2),rhs))
    assert torch.autograd.gradcheck(lambda a,b:GPUSparseSolve.apply(a,b,indices),(values,rhs))
    second=GPUSparseSolve.apply(values*2,rhs*3,indices)
    torch.testing.assert_close(second,first*1.5)
    torch.testing.assert_close(GPUSparseSolve.apply(values,rhs*0,indices),rhs*0)
    clear_cache()


@pytest.mark.parametrize('case',['es','em'])
def test_gpu_native_state_and_derivative_match_cpu(case):
    from gpu_sparse import clear_cache
    r,z,rr,zz=grid(15);c=coefficients('em')
    c.update(viscosity='old',conservative=case=='em',eta=1e-5,eta_hyper=1e-8,mu_hyper=1e-8,
             d_hyper=1e-10,k_hyper=1e-10,T_threshold=5e-7,k_negative=1e-4)
    shape=torch.sin(torch.pi*(rr-9.5))*torch.sin(torch.pi*(zz+.5))
    q=torch.stack((c['rho_wall']+shape*.001,c['rho_wall']*c['T_wall']+shape*1e-9,shape*1e-8))[None]
    if case=='em':q=torch.cat((q,(.1+shape*1e-5)[None,None]),1)
    reference=NativeReference(r,z,wall_trace(q,r,z,c),c)
    outputs=[];derivatives=[]
    for device,compiled in (('cpu',False),('cuda',False),('cuda',True)):
        solver=NativeRollout(reference,q,device=device,compiled_terms=compiled)
        alpha=torch.tensor(1.,dtype=torch.float64,device=device,requires_grad=True)
        direction=q*0;direction[:,1]=shape*1e-9
        result=solver.step(q.to(device)+alpha*direction.to(device),1.,2)
        derivatives.append(torch.autograd.grad(result[:,1].sum(),alpha)[0].cpu())
        outputs.append(result.detach().cpu())
    for i in (1,2):
        torch.testing.assert_close(outputs[i],outputs[0],rtol=1e-7,atol=1e-14)
        torch.testing.assert_close(derivatives[i],derivatives[0],rtol=1e-7,atol=1e-14)
    clear_cache()


def test_gpu_solver_rejects_singular_matrix():
    from gpu_sparse import GPUSparseSolve,clear_cache
    indices=torch.tensor([[0,1],[0,1]],device='cuda')
    values=torch.zeros(2,device='cuda',dtype=torch.float64)
    rhs=torch.ones(2,device='cuda',dtype=torch.float64)
    with pytest.raises(RuntimeError):GPUSparseSolve.apply(values,rhs,indices)
    clear_cache()


def test_gpu_scalar_fallback_matches_cpu_and_gradient():
    r,z,rr,zz=grid(15);c=coefficients('em');c.update(viscosity='old',conservative=False)
    q=torch.stack((rr*0+c['rho_wall'],rr*0+c['rho_wall']*c['T_wall'],rr*0))[None]
    class NegativeSource(torch.nn.Module):
        def __init__(self):
            super().__init__();self.rate=torch.nn.Parameter(torch.tensor(-1.,dtype=torch.float64))
            self.register_buffer('target_scale',torch.tensor([1.,1e-5,1.,1.,1.]))
        def forward(self,x):
            zero=x[:,:1].double()*0
            forcing=zero+self.rate+torch.linspace(0,.3,x.shape[-1],device=x.device,dtype=torch.float64)
            return torch.cat((forcing,forcing,zero,zero,zero),1)
    ref=NativeReference(r,z,wall_trace(q,r,z,c),c)
    results=[];gradients=[]
    for device in ['cpu','cuda']:
        model=NegativeSource().to(device);solver=NativeRollout(ref,q,model,device=device)
        out=solver.step(q.to(device),1.,1)
        assert solver.limiter_stats['density_limited']==1
        assert solver.limiter_stats['temperature_limited']==1
        assert torch.all(out[:,:2]>0)
        (out[:,0].sum()+out[:,1].sum()*1e6).backward()
        results.append(out.detach().cpu());gradients.append(model.rate.grad.cpu())
    torch.testing.assert_close(results[0],results[1],rtol=1e-7,atol=1e-14)
    torch.testing.assert_close(gradients[0],gradients[1],rtol=1e-5,atol=1e-7)


def test_gpu_kinetic_envelope_matches_cpu_and_active_gradient():
    from native_rollout import diagonal
    old=torch.tensor([.2,-.1],dtype=torch.float64)
    high=torch.tensor([1.,-.7],dtype=torch.float64)
    results=[];gradients=[]
    for device in ('cpu','cuda'):
        solver=object.__new__(NativeRollout);solver.reset_limiter_stats()
        value=high.to(device).detach().clone().requires_grad_()
        mass=diagonal(torch.tensor([2.,3.],dtype=torch.float64,device=device))
        new_mass=diagonal(torch.tensor([3.,2.],dtype=torch.float64,device=device))
        result=solver.limit_kinetic_energy(old.to(device),value,old.to(device)*1.2,mass,new_mass)
        result.sum().backward()
        assert solver.limiter_stats['energy_limited']==1 and torch.isfinite(value.grad).all()
        results.append(result.detach().cpu());gradients.append(value.grad.cpu())
    torch.testing.assert_close(results[0],results[1],rtol=1e-10,atol=1e-14)
    torch.testing.assert_close(gradients[0],gradients[1],rtol=1e-10,atol=1e-14)
