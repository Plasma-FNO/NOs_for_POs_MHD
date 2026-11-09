import pytest
import numpy as np
import torch
from mastu_opsplit.continuity import continuity_tangent, FrozenContinuity
from mastu_opsplit.euler import EulerSplineModel, NativeSplineSpace, field_loss
from mastu_opsplit.restart import read_restart
from test_restart import rectangle
from test_learning import configuration


def native_advection(q,g,r,gamma):
    # Independent translation of native rho and pressure source rows; then
    # T_t=(p_t-T*rho_t)/rho. No sources, heating or diffusive weak fluxes.
    rho_u=g[...,4,0]*g[...,1,1]-g[...,4,1]*g[...,1,0]
    t_u=g[...,5,0]*g[...,1,1]-g[...,5,1]*g[...,1,0]
    rho_psi=g[...,4,0]*g[...,0,1]-g[...,4,1]*g[...,0,0]
    t_psi=g[...,5,0]*g[...,0,1]-g[...,5,1]*g[...,0,0]
    v_psi=g[...,6,0]*g[...,0,1]-g[...,6,1]*g[...,0,0]
    dr=r*rho_u+2*q[...,4]*g[...,1,1]-(q[...,6]*rho_psi+q[...,4]*v_psi)/r
    dp=(r*(q[...,4]*t_u+q[...,5]*rho_u)+2*gamma*q[...,4]*q[...,5]*g[...,1,1]
        -q[...,6]*(q[...,4]*t_psi+q[...,5]*rho_psi)/r-gamma*q[...,4]*q[...,5]*v_psi/r)
    out=torch.zeros_like(q);out[...,4]=dr;out[...,5]=(dp-q[...,5]*dr)/q[...,4]
    return out


@pytest.mark.parametrize('gamma',[5/3,2/3])
def test_physical_tangent_matches_native_pressure_chain_rule(gamma):
    torch.manual_seed(4)
    q=torch.randn(13,7,dtype=torch.float64);q[:,4]=2.
    g=torch.randn(13,7,2,dtype=torch.float64)
    dq=torch.randn_like(q);dg=torch.randn_like(g);r=torch.linspace(.3,2.,13)
    eps=1e-5
    numerical=(native_advection(q+eps*dq,g+eps*dg,r,gamma)-native_advection(q-eps*dq,g-eps*dg,r,gamma))/(2*eps)
    actual=continuity_tangent(q,g,dq,dg,r,gamma)
    torch.testing.assert_close(actual,numerical,rtol=1e-8,atol=1e-8)
    dq.requires_grad_();dg.requires_grad_()
    assert torch.autograd.gradcheck(lambda v,d:continuity_tangent(q,g,v,d,r,gamma),(dq,dg))


@pytest.mark.parametrize('branch', ['lspline_continuity', 'native_continuity'])
def test_physical_branch_linear_scaled_differentiable_and_parameter_matched(tmp_path, branch):
    torch.set_num_threads(1)
    rectangle(tmp_path/'mesh.h5');geometry,mean,_=read_restart(tmp_path/'mesh.h5')
    space=NativeSplineSpace(geometry)
    physical=FrozenContinuity(geometry,mean,np.arange(1,8),.0625)
    torch.manual_seed(17)
    node=EulerSplineModel('node',space,configuration())
    torch.manual_seed(17)
    model=EulerSplineModel('opsplit',space,configuration(),linear_branch=branch,physical_linear=physical)
    assert sum(p.numel() for p in node.parameters())==sum(p.numel() for p in model.parameters())
    for a,b in zip(node.operator.parameters(),model.operator.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    x=torch.randn(1,16,7,dtype=torch.float64,requires_grad=True)
    y=torch.randn_like(x)
    torch.testing.assert_close(model.linear_rhs(x+y),model.linear_rhs(x)+model.linear_rhs(y),rtol=1e-4,atol=1e-7)
    zero=model.linear_rhs(torch.zeros_like(x));assert torch.count_nonzero(zero)==0
    output=model(model(x,torch.ones(1,1)),torch.ones(1,1))
    field_loss(output,x,space).backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_native_continuity_uses_exact_gradients_and_state_adjoint(tmp_path):
    torch.manual_seed(31)
    rectangle(tmp_path/'mesh.h5')
    geometry, mean, _ = read_restart(tmp_path/'mesh.h5')
    space = NativeSplineSpace(geometry)
    physical = FrozenContinuity(geometry, mean, np.arange(1, 8), .0625)
    model = EulerSplineModel('opsplit', space, configuration(), linear_branch='native_continuity', physical_linear=physical)
    state = torch.randn(1, 16, 7, dtype=torch.float64, requires_grad=True)
    gradient = np.einsum('eqlr,belc->beqcr', geometry.basis_grad,
                         state.detach().numpy()[:, geometry.dof_map]).reshape(1, -1, 7, 2)
    expected = space.project(physical(space.sample(state.detach()), torch.from_numpy(gradient)))
    torch.testing.assert_close(model.linear_rhs(state), expected, rtol=1e-12, atol=1e-12)
    weights = torch.randn_like(state)
    direction = torch.randn_like(state)
    derivative, = torch.autograd.grad((model.linear_rhs(state)*weights).sum(), state)
    # The branch is linear, so a larger displacement avoids solve cancellation without truncation error.
    eps = 1e-2
    finite_difference = ((model.linear_rhs(state+eps*direction)-model.linear_rhs(state-eps*direction))*weights).sum()/(2*eps)
    torch.testing.assert_close((derivative*direction).sum(), finite_difference, rtol=1e-8, atol=1e-10)
    # The physical branch must not depend on trainable scatter kernels.
    with torch.no_grad():
        for parameter in model.operator.parameters():
            parameter.add_(.01*torch.randn_like(parameter))
    torch.testing.assert_close(model.linear_rhs(state), expected, rtol=1e-12, atol=1e-12)
