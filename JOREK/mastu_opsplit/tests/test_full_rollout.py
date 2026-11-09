"""Component tests are not trajectory or timestep-convergence evidence."""
import struct
import numpy as np
import pytest
import torch
from mastu_opsplit.full_rollout import (native_jets,linear_terms,assemble_terms,
                                      FullOpsSplitOperators,read_linear,write_prediction)
from mastu_opsplit.numerical import evaluate_native,evaluate_native_grad,assemble_strong,assemble_flux
from test_numerical import geometry_fixture
from test_learning import configuration


def test_native_jets_assembly_and_derivative_dofs():
    g=geometry_fixture();g.basis[:,:,1]*=-1
    q=torch.randn(2,3,7,dtype=torch.float64,requires_grad=True)
    value,gradient=native_jets(q,g)
    for b in range(2):
        np.testing.assert_allclose(value[b].detach(),evaluate_native(q[b].detach().numpy(),g))
        np.testing.assert_allclose(gradient[b].detach(),evaluate_native_grad(q[b].detach().numpy(),g))
    c=torch.randn(2,2,2,5,3,dtype=torch.float64,requires_grad=True)
    assembled=assemble_terms(c,g)
    for b in range(2):
        a=c[b].detach().numpy()
        expected=assemble_strong(a[...,0],g)-assemble_flux(a[...,1:],g)
        np.testing.assert_allclose(assembled[b].detach(),expected,atol=1e-14)
    assert torch.autograd.gradcheck(lambda x:assemble_terms(x,g),(c,))
    assert torch.autograd.gradcheck(lambda x:native_jets(x,g),(q,))


def test_fixed_operator_is_linear_and_torch_matches_numpy():
    rng=np.random.default_rng(4)
    q=rng.normal(size=(2,4,7));dq=rng.normal(size=q.shape)
    grad=rng.normal(size=(*q.shape,2));dgrad=rng.normal(size=grad.shape)
    r=rng.uniform(.4,1.5,size=(2,4));a=rng.normal(size=(2,4,10))
    expected=2*linear_terms(q,grad,r,a)-3*linear_terms(dq,dgrad,r,a)
    np.testing.assert_allclose(linear_terms(2*q-3*dq,2*grad-3*dgrad,r,a),expected,atol=1e-13)
    args=[torch.tensor(x,requires_grad=i<2) for i,x in enumerate((q,grad,r,a))]
    actual=linear_terms(*args)
    np.testing.assert_allclose(actual.detach(),linear_terms(q,grad,r,a))
    assert torch.autograd.gradcheck(lambda q,g:linear_terms(q,g,args[2],args[3]),tuple(args[:2]))


def test_native_signs_and_channel_mapping():
    q=np.zeros((1,7));g=np.zeros((1,7,2));r=np.array([2.]);a=np.arange(1,11.)[None]
    q[0,2]=8;q[0,6]=3;g[0,4]=[2,3];g[0,5]=[5,7];g[0,3]=[11,13];g[0,6]=[17,19]
    actual=linear_terms(q,g,r,a)[0]
    np.testing.assert_allclose(actual,[[2,0,0],[0,-22,-26],[0,-18,-23],[0,-79,-91],[0,-123,-171]])


def test_full_no_all_five_blocks_and_gradient_paths():
    torch.set_num_threads(1);torch.manual_seed(42)
    model=FullOpsSplitOperators([[1,2],[-1,1]],configuration())
    coords=torch.tensor([[1.1,-.8],[1.3,.4],[1.6,-.1],[1.9,.7]])
    fields=torch.randn(1,4,7,requires_grad=True)
    out=model(fields,coords,torch.tensor([[1.]]))
    assert out.shape==(1,4,5,3)
    out.square().sum().backward()
    assert fields.grad.abs().sum()>0 and torch.isfinite(fields.grad).all()
    for block in model.blocks.values():
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters() if p.requires_grad)
        assert sum(p.grad.abs().sum() for p in block.parameters() if p.requires_grad)>0


def test_binary_geometry_stamp_finiteness_and_truncation(tmp_path):
    path=tmp_path/'linear.bin';vertices=np.array([[1,2,3,4]],dtype='<i4')
    with path.open('wb') as f:
        f.write(struct.pack('<3i',303046,1,16));vertices.tofile(f);np.ones((1,16,46),dtype='<f8').tofile(f)
    v,a=read_linear(path);np.testing.assert_array_equal(v,vertices)
    pred=np.arange(240.).reshape(1,16,5,3)
    dest=tmp_path/'prediction.bin';write_prediction(dest,145,pred,vertices,path)
    with dest.open('rb') as f:
        assert struct.unpack('<4i',f.read(16))==(303015,145,1,16)
        np.testing.assert_array_equal(np.fromfile(f,dtype='<f8').reshape(pred.shape),pred)
    with pytest.raises(ValueError,match='geometry'):write_prediction(dest,146,pred,vertices+1,path)
    pred[0,0,0,0]=np.nan
    with pytest.raises(ValueError,match='prediction'):write_prediction(dest,146,pred,vertices,path)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError,match='size'):read_linear(path)


def test_centered_forcing_has_exact_baseline_and_correct_increment():
    from mastu_opsplit.full_rollout import centered_linear_forcing
    rng=np.random.default_rng(32)
    frozen=rng.normal(size=(3,46));q=frozen[:,10:17].copy()
    grad=np.stack((frozen[:,17:24],frozen[:,24:31]),axis=-1)
    r=np.ones(3);base=frozen[:,31:46].reshape(3,5,3)
    np.testing.assert_array_equal(centered_linear_forcing(q,grad,r,frozen),base)
    dq=rng.normal(size=q.shape);dg=rng.normal(size=grad.shape)
    np.testing.assert_allclose(centered_linear_forcing(q+dq,grad+dg,r,frozen)-base,
                               linear_terms(dq,dg,r,frozen[:,:10]),atol=2e-15)


def test_cnab_midpoint_mass_and_nonlinear_history_are_second_order():
    # Manufactured scalar DAE-volume analogue only, not a MAST-U validation gate.
    # q*q'=-q has q(t)=2-t. Split L=-0.4 with centered remainder -0.6*(q-2).
    errors=[]
    for steps in (10,20,40):
        h=.5/steps;q=2.;previous=q;old_n=0.
        for k in range(steps):
            n=-.6*(q-2.)
            weight=.5 if k else 1.
            ratio=.5 if k else 0.
            mass=q+ratio*(q-previous)
            rhs=-2.-.4*(q-2.)+n+ratio*(n-old_n)
            new=q+h*rhs/(mass+weight*h*.4)
            previous,q,old_n=q,new,n
        errors.append(abs(q-1.5))
    assert errors[0]/errors[1]>3.8 and errors[1]/errors[2]>3.8


def test_sbdf2_variable_mass_and_history_are_second_order():
    errors=[]
    for steps in (10,20,40):
        h=.5/steps;q=2.;previous=q;old_n=0.
        for k in range(steps):
            n=-.6*(q-2.)
            ratio=1. if k else 0.
            mass=q+ratio*(q-previous)
            a0=1.5 if k else 1.;history=.5 if k else 0.
            rhs=-2.-.4*(q-2.)+n+ratio*(n-old_n)
            new=q+(h*rhs+history*mass*(q-previous))/(a0*mass+h*.4)
            previous,q,old_n=q,new,n
        errors.append(abs(q-1.5))
    assert errors[0]/errors[1]>3.7 and errors[1]/errors[2]>3.7


def test_full_tangent_native_column_layout_and_autograd():
    from mastu_opsplit.full_rollout import centered_linear_forcing
    frozen=np.zeros((1,361));jac=frozen[:,46:].reshape(1,21,15)
    jac[0,2,0]=2.;jac[0,14,4]=3.;jac[0,12,11]=5.;jac[0,6,12]=7.
    q=np.zeros((1,7));grad=np.zeros((1,7,2))
    q[0,2]=2.;grad[0,0,1]=3.;grad[0,5,0]=5.;q[0,6]=7.
    expected=np.zeros((1,5,3));expected[0,0,0]=4.;expected[0,1,1]=9.;expected[0,3,2]=25.;expected[0,4,0]=49.
    np.testing.assert_array_equal(centered_linear_forcing(q,grad,np.ones(1),frozen),expected)
    qt=torch.tensor(q,requires_grad=True);gt=torch.tensor(grad,requires_grad=True)
    function=lambda q,g:centered_linear_forcing(q,g,torch.ones(1,dtype=torch.float64),torch.tensor(frozen))
    np.testing.assert_array_equal(function(qt,gt).detach(),expected)
    assert torch.autograd.gradcheck(function,(qt,gt))


def test_normalization_preserves_small_evolution_on_large_background():
    from mastu_opsplit.full_rollout import normalize_current_fields
    mean=torch.full((7,),1.e6,dtype=torch.float64)
    increments=torch.tensor([1.,-2.,3.,-4.,5.,-6.,7.],dtype=torch.float64)*1.e-3
    actual=normalize_current_fields((mean+increments)[None,None],mean,torch.full((7,),1.e-3,dtype=torch.float64))
    assert actual.dtype==torch.float32
    torch.testing.assert_close(actual[0,0],(increments/1.e-3).float(),rtol=1.e-6,atol=1.e-6)
