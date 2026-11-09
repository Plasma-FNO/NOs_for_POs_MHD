"""Trajectory-supervised Euler models in the native bicubic field space.

The linear branch is an effective, learnable scalar diffusion prior per channel,
not the model303 temporal/constraint solver. The default differentiates the
L-SplineFNO synthesis of raw-field scatter coefficients. Physical boundary and
constraint errors require evaluation. Physical continuity can instead use exact
native bicubic gradients, preserving the same frozen tangent and neural residual.
"""
import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import torch
from torch import nn

from .learning import FieldSplineOperator
from .full_rollout import native_jets
from .spline_derivatives import SplineDerivatives


class _FixedNative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, space, operation):
        ctx.space, ctx.operation = space, operation
        result = space.apply(values.detach().cpu().double().numpy(), operation)
        return torch.as_tensor(result, device=values.device, dtype=values.dtype)

    @staticmethod
    def backward(ctx, gradient):
        result = ctx.space.apply(gradient.detach().cpu().double().numpy(), ctx.operation, transpose=True)
        return torch.as_tensor(result, device=gradient.device, dtype=gradient.dtype), None, None


class NativeSplineSpace:
    """Consistent volume projection and stiffness from analytic physical gradients.

    Fixed sparse solves have exact transpose actions for first-order autograd.
    shortcut: solves run on CPU; optimize transfers before large GPU training.
    """
    def __init__(self, geometry):
        geometry.validate()
        self.geometry = geometry
        e, q, local = geometry.basis.shape
        ndof = len(geometry.dof_ids)
        rows = np.broadcast_to(np.arange(e*q).reshape(e,q,1), (e,q,local)).ravel()
        cols = np.broadcast_to(geometry.dof_map[:,None,:], (e,q,local)).ravel()
        self.basis = sp.coo_matrix((geometry.basis.ravel(), (rows,cols)), shape=(e*q,ndof)).tocsr()
        self.weights = geometry.volume_weights.ravel()
        b, g, w = geometry.basis, geometry.basis_grad, geometry.volume_weights
        local_mass = np.einsum('eql,eqm,eq->elm', b,b,w)
        local_stiff = np.einsum('eqlr,eqmr,eq->elm', g,g,w)
        ij = np.broadcast_arrays(geometry.dof_map[:,:,None], geometry.dof_map[:,None,:])
        def assemble(a):
            return sp.coo_matrix((a.ravel(), (ij[0].ravel(),ij[1].ravel())), shape=(ndof,ndof)).tocsc()
        self.mass, self.stiffness = assemble(local_mass), assemble(local_stiff)
        self.diagonal = np.sqrt(self.mass.diagonal())
        if not np.all(self.diagonal > 0):
            raise ValueError('Native mass matrix has unsupported degrees of freedom')
        scale = sp.diags(1/self.diagonal)
        self.factor = splu((scale @ self.mass @ scale).tocsc())
        # Local generalized eigenvalue bounds also bound the assembled Rayleigh quotient.
        self.spectral_bound = 1.000001*max(float(scipy.linalg.eigh(k,m,subset_by_index=[local-1,local-1],
                                          eigvals_only=True)[0]) for k,m in zip(local_stiff,local_mass))
        if not np.isfinite(self.spectral_bound) or self.spectral_bound <= 0:
            raise ValueError('Invalid native diffusion stability bound')

    def solve(self, x, transpose=False):
        return self.factor.solve(x/self.diagonal[:,None], trans='T' if transpose else 'N')/self.diagonal[:,None]

    def apply(self, x, operation, transpose=False):
        if x.ndim != 3 or not np.isfinite(x).all():
            raise ValueError('Expected finite [batch,point_or_dof,channel]')
        batch, _, channels = x.shape
        a = x.transpose(1,0,2).reshape(x.shape[1],-1)
        if operation == 'sample':
            result = self.basis.T @ a if transpose else self.basis @ a
        elif operation == 'project':
            result = self.weights[:,None]*(self.basis @ self.solve(a,True)) if transpose else self.solve(self.basis.T @ (self.weights[:,None]*a))
        elif operation == 'laplace':
            result = -(self.stiffness @ self.solve(a,True)) if transpose else -self.solve(self.stiffness @ a)
        else:
            raise ValueError('Unknown native operation')
        return result.reshape(-1,batch,channels).transpose(1,0,2).copy()

    def sample(self, x):
        return _FixedNative.apply(x,self,'sample')

    def project(self, x):
        return _FixedNative.apply(x,self,'project')

    def laplace(self, x):
        return _FixedNative.apply(x,self,'laplace')


class EulerSplineModel(nn.Module):
    """AR, Euler NODE, and additive Euler OpsSplit with identical neural backbones."""
    def __init__(self, family, space, model_config, channels=7, *, linear_branch='lspline_analytic',
                 physical_guard=None, linear_initialization='positive', physical_linear=None):
        super().__init__()
        if family not in ('ar','node','opsplit'):
            raise ValueError('Expected ar, node, or opsplit')
        self.family, self.space = family, space
        if linear_branch not in ('lspline_analytic', 'native_bicubic', 'lspline_continuity', 'native_continuity'):
            raise ValueError('Unknown linear derivative formulation')
        self.linear_branch = linear_branch
        self.physical_guard = physical_guard
        if linear_initialization not in ('positive','zero'):
            raise ValueError('Unknown linear initialization')
        self.linear_initialization = linear_initialization
        self.operator = FieldSplineOperator(channels,channels,space.geometry.bounds,model_config,
                                            condition_transport=True)
        self.register_buffer('coordinates',torch.tensor(space.geometry.quadrature_rz.reshape(-1,2),dtype=torch.float64))
        if family == 'opsplit':
            if linear_branch in ('lspline_continuity', 'native_continuity'):
                if physical_linear is None:
                    raise ValueError('Continuity branch needs training-only physical background')
                self.physical_linear = physical_linear
            else:
                self.log_diffusion = nn.Parameter(torch.full((channels,),0. if linear_initialization=='zero' else -2.))
            if linear_branch in ('lspline_analytic','lspline_continuity'):
                self.derivatives = SplineDerivatives(self.coordinates, self.operator.physical_bounds,
                                                     self.operator.backbone._get_grid_axes())
                spacing = [(float(axis[1]-axis[0])*float(hi-lo)) for axis, (lo,hi) in
                           zip(self.operator.backbone._get_grid_axes(), self.operator.physical_bounds)]
                # Grid-scale coefficient normalization, NOT a stability proof for
                # learned scatter + spline differentiation + native projection.
                self.diffusion_scale = 1.8/(4*sum(h**-2 for h in spacing))
            else:
                self.diffusion_scale = 1.8/space.spectral_bound

    def diffusion(self):
        # A zero-initialized signed curvature correction contains NODE exactly
        # at initialization; its sign/magnitude are learned on training data.
        gate = self.log_diffusion.tanh() if self.linear_initialization=='zero' else self.log_diffusion.sigmoid()
        return self.diffusion_scale*gate

    def linear_rhs(self, state, values=None, *, gamma=None):
        if self.linear_branch == 'native_continuity':
            values = self.space.sample(state) if values is None else values
            gradients = native_jets(state, self.space.geometry)[1].flatten(1, 2)
            return self.space.project(self.physical_linear(values, gradients, gamma=gamma))
        if self.linear_branch == 'native_bicubic':
            laplace = self.space.laplace(state)
        else:
            values = self.space.sample(state) if values is None else values
            source = self.operator.normalized_coordinates(self.coordinates, state.shape[0]).to(values.float())
            # Share learned scatter kernels with N, but keep this path linear
            # in the current normalized fields for fixed network parameters.
            coefficients = self.operator.backbone._encode(values.float().transpose(1,2), source)
            if self.linear_branch == 'lspline_continuity':
                gradients = torch.stack((self.derivatives(coefficients,(1,0)),
                                         self.derivatives(coefficients,(0,1))),dim=-1).to(state.dtype)
                return self.space.project(self.physical_linear(values,gradients, gamma=gamma))
            laplace = self.space.project(self.derivatives.laplace(coefficients).to(state.dtype))
        return laplace*self.diffusion().to(state.dtype)

    def rhs(self, state, parameters):
        values = self.space.sample(state)
        neural = self.operator(values.float(),self.coordinates,parameters.float()).to(state.dtype)
        result = self.space.project(neural)
        if self.family == 'opsplit':
            result = result + self.linear_rhs(state, values)
        return result

    def forward(self, state, parameters, *, substeps=1, interval=1.):
        if type(substeps) is not int or substeps < 1 or not 0 < interval <= 1:
            raise ValueError('Require positive substeps and 0 < frame interval <= 1')
        if self.family == 'ar':
            if substeps != 1 or interval != 1:
                raise ValueError('AR has no substep refinement')
            proposal = self.rhs(state,parameters)
            return self.physical_guard(state,proposal) if self.physical_guard is not None else proposal
        for _ in range(substeps):
            proposal = state + interval/substeps*self.rhs(state,parameters)
            state = self.physical_guard(state,proposal) if self.physical_guard is not None else proposal
        return state


def field_loss(prediction, target, space):
    error = space.sample(prediction-target)
    weights = torch.as_tensor(space.weights/space.weights.sum(),device=error.device,dtype=error.dtype)
    return (error.square()*weights[None,:,None]).sum(1).mean()
