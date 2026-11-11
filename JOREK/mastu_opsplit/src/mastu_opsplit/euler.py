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


def _on_device(tensor, space, operation):
    """Use the space's device kernels (gpu_native) for non-CPU tensors when it has them."""
    return tensor.device.type != 'cpu' and operation in getattr(space, 'torch_operations', ())


class _FixedNative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, space, operation):
        ctx.space, ctx.operation = space, operation
        if _on_device(values, space, operation):
            return space.apply_torch(values.detach(), operation)
        result = space.apply(values.detach().cpu().double().numpy(), operation)
        return torch.as_tensor(result, device=values.device, dtype=values.dtype)

    @staticmethod
    def backward(ctx, gradient):
        if _on_device(gradient, ctx.space, ctx.operation):
            return ctx.space.apply_torch(gradient.detach(), ctx.operation, transpose=True), None, None
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

    torch_operations = ('sample', 'project')

    def _device_solve(self, rhs):
        """Scaled-mass solve on rhs.device (cuDSS); the scaled mass is symmetric, so it is its own transpose."""
        from .gpu_native import device_tensor, direct_solver
        cache = self.__dict__.setdefault('_device_solvers', {})
        key = str(rhs.device)
        if key not in cache:
            scaled = sp.diags(1/self.diagonal) @ self.mass @ sp.diags(1/self.diagonal)
            if (abs(scaled - scaled.T) > 1e-14*abs(scaled).max()).nnz:
                raise ValueError('Device projection assumes a symmetric native mass matrix')
            cache[key] = direct_solver(scaled, rhs.device)
        diagonal = device_tensor(self, 'diagonal', lambda: self.diagonal, rhs.device, torch.float64)[:, None]
        return cache[key].solve(rhs/diagonal)/diagonal

    def apply_torch(self, x, operation, transpose=False):
        """Device versions of 'sample' and 'project' (same linear maps and exact transposes); laplace stays on the CPU."""
        from .gpu_native import apply_batched, device_matrix, device_tensor
        if operation not in self.torch_operations or x.ndim != 3 or not torch.isfinite(x).all():
            raise ValueError('Expected finite [batch,point_or_dof,channel]')
        if operation == 'sample':
            return apply_batched(device_matrix(self, 'basis', lambda: self.basis, x.device, transpose), x).to(x.dtype)
        batch, n, channels = x.shape
        flat = x.to(torch.float64).transpose(0, 1).reshape(n, batch*channels)
        weights = device_tensor(self, 'weights', lambda: self.weights, x.device, torch.float64)[:, None]
        if transpose:   # (B^T W)^T M^-T = W B M^-1
            result = weights*(device_matrix(self, 'basis', lambda: self.basis, x.device) @ self._device_solve(flat))
        else:
            result = self._device_solve(device_matrix(self, 'basis', lambda: self.basis, x.device, True) @ (weights*flat))
        return result.reshape(-1, batch, channels).transpose(0, 1).to(x.dtype)

    def sample(self, x):
        return _FixedNative.apply(x,self,'sample')

    def project(self, x):
        return _FixedNative.apply(x,self,'project')

    def laplace(self, x):
        return _FixedNative.apply(x,self,'laplace')


class EulerSplineModel(nn.Module):
    """AR, Euler NODE, additive Euler OpsSplit and the one-step time-conditional model, sharing one backbone.

    'conditional' is the Euler NODE update with the L-SplineFNO's FiLM input widened from [D/D0] to
    [D/D0, t/T]: t is the physical time of the state being advanced, measured from the trajectory start,
    and T the conditioning time scale (the 96-interval trajectory). It takes one state to the next exactly
    like the other families, so it sees the same windows, targets and rollout feedback.
    """
    def __init__(self, family, space, model_config, channels=7, *, linear_branch='lspline_analytic',
                 physical_guard=None, linear_initialization='positive', physical_linear=None, time_conditioning=None):
        super().__init__()
        if family not in ('ar','node','opsplit','conditional'):
            raise ValueError('Expected ar, node, opsplit, or conditional')
        self.family, self.space = family, space
        self.direct_query = False
        if (family == 'conditional') != (time_conditioning is not None):
            raise ValueError('The conditional family, and only it, needs time conditioning')
        if time_conditioning is not None:
            values = [float(time_conditioning[name]) for name in ('origin_seconds', 'scale_seconds', 'interval_seconds')]
            if not np.isfinite(values).all() or values[1] <= 0 or values[2] <= 0:
                raise ValueError('Time conditioning needs a finite origin and positive scale and interval')
            self.time_origin_seconds, self.time_scale_seconds, self.interval_seconds = values
        if linear_branch not in ('lspline_analytic', 'native_bicubic', 'lspline_continuity', 'native_continuity'):
            raise ValueError('Unknown linear derivative formulation')
        self.linear_branch = linear_branch
        self.physical_guard = physical_guard
        if linear_initialization not in ('positive','zero'):
            raise ValueError('Unknown linear initialization')
        self.linear_initialization = linear_initialization
        self.operator = FieldSplineOperator(channels,channels,space.geometry.bounds,model_config,
                                            condition_transport=True, condition_time=family == 'conditional')
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

    def normalized_time(self, time_seconds, state):
        """[B] values of (t - t0)/T for the state being advanced."""
        time = torch.as_tensor(time_seconds, device=state.device, dtype=torch.float64)
        time = time.expand(state.shape[0]) if time.ndim == 0 else time
        tau = (time - self.time_origin_seconds)/self.time_scale_seconds
        if tau.shape != (state.shape[0],) or not torch.isfinite(tau).all() or (tau < -1e-12).any():
            raise ValueError('State time must be finite, not before the trajectory start, scalar or [batch]')
        return tau.clamp_min(0)

    def rhs(self, state, parameters, tau=None):
        values = self.space.sample(state)
        if self.family == 'conditional':
            parameters = torch.cat((parameters.to(tau), tau[:, None]), dim=1)
        neural = self.operator(values.float(),self.coordinates,parameters.float()).to(state.dtype)
        result = self.space.project(neural)
        if self.family == 'opsplit':
            result = result + self.linear_rhs(state, values)
        return result

    def forward(self, state, parameters, *, substeps=1, interval=1., time_seconds=None):
        if type(substeps) is not int or substeps < 1 or not 0 < interval <= 1:
            raise ValueError('Require positive substeps and 0 < frame interval <= 1')
        if (self.family == 'conditional') != (time_seconds is not None):
            raise ValueError('Pass the state time to the conditional family, and only to it')
        if self.family == 'ar':
            if substeps != 1 or interval != 1:
                raise ValueError('AR has no substep refinement')
            proposal = self.rhs(state,parameters)
            return self.physical_guard(state,proposal) if self.physical_guard is not None else proposal
        tau = self.normalized_time(time_seconds, state) if self.family == 'conditional' else None
        for substep in range(substeps):
            if tau is None:
                rate = self.rhs(state,parameters)
            else:   # Euler substeps advance the conditioning time with the state.
                rate = self.rhs(state,parameters, tau + substep*interval/substeps*self.interval_seconds/self.time_scale_seconds)
            proposal = state + interval/substeps*rate
            state = self.physical_guard(state,proposal) if self.physical_guard is not None else proposal
        return state


def field_loss(prediction, target, space):
    error = space.sample(prediction-target)
    weights = torch.as_tensor(space.weights/space.weights.sum(),device=error.device,dtype=error.dtype)
    return (error.square()*weights[None,:,None]).sum(1).mean()
