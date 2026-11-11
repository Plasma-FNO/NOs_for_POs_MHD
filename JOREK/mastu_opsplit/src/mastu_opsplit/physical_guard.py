"""Shared native elliptic closure and sampled positivity for all pilot families.

Predict j,w,rho,T,Vpar; recover psi,u from their weak interior definitions,
retaining fixed boundary nodal values but learning boundary derivative DOFs.
The positivity limiter blends toward the training mean (a positive fixed anchor),
not the previous near-floor state or individual samples. Subdivided Bernstein
coefficients bound the entire native bicubic cell, including edges and vertices.
Neither procedure asserts the remaining MHD equations.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import torch
from torch import nn

from .euler import _FixedNative


class EllipticPair:
    def __init__(self, geometry, current):
        self.interior = np.flatnonzero(geometry.boundary_labels == 0)
        self.boundary = np.flatnonzero(geometry.boundary_labels != 0)
        if not len(self.interior) or not len(self.boundary):
            raise ValueError('Elliptic closure requires interior and fixed boundary DOFs')
        w = geometry.volume_weights
        if current:
            w = w/geometry.quadrature_rz[..., 0]**2
        b, g = geometry.basis, geometry.basis_grad
        ij = np.broadcast_arrays(geometry.dof_map[:, :, None], geometry.dof_map[:, None, :])
        def assemble(local):
            return sp.coo_matrix((local.ravel(), (ij[0].ravel(), ij[1].ravel())),
                                 shape=(len(geometry.dof_ids),)*2).tocsr()
        mass = assemble(np.einsum('eql,eqm,eq->elm', b, b, w))
        stiff = assemble(np.einsum('eqlr,eqmr,eq->elm', g, g, w))
        self.mass = mass[self.interior]
        self.cross = stiff[self.interior][:, self.boundary]
        self.stiff = stiff[self.interior]
        block = stiff[self.interior][:, self.interior].tocsc()
        self.diagonal = np.sqrt(block.diagonal())
        d = sp.diags(1/self.diagonal)
        self.scaled = (d @ block @ d).tocsc()
        self.factor = splu(self.scaled)

    def solve(self, rhs, transpose=False):
        return self.factor.solve(rhs/self.diagonal[:, None], trans='T' if transpose else 'N')/self.diagonal[:, None]

    def apply(self, x, operation, transpose=False):
        if operation != 'closure' or x.ndim != 3 or x.shape[-1] != 2 or not np.isfinite(x).all():
            raise ValueError('Expected finite native potential/auxiliary pair')
        p, a = x[..., 0].T, x[..., 1].T
        if transpose:
            z = self.solve(p[self.interior], True)
            potential = np.zeros_like(p)
            potential[self.boundary] = p[self.boundary] - self.cross.T @ z
            auxiliary = a - self.mass.T @ z
        else:
            potential = p.copy()
            potential[self.interior] = self.solve(-self.mass @ a - self.cross @ p[self.boundary])
            auxiliary = a
        return np.stack((potential.T, auxiliary.T), axis=-1)

    torch_operations = ('closure',)

    def _device_solve(self, rhs, transpose):
        """K^-1 rhs (or K^-T) on rhs.device: cuDSS on CUDA, this pair's SuperLU on the CPU."""
        from .gpu_native import device_tensor, direct_solver
        if rhs.device.type == 'cpu':
            return torch.as_tensor(self.solve(rhs.numpy(), transpose), dtype=rhs.dtype)
        cache = self.__dict__.setdefault('_device_solvers', {})
        if transpose and 'symmetric' not in cache:
            cache['symmetric'] = (self.scaled != self.scaled.T).nnz == 0
        key = (str(rhs.device), transpose and not cache['symmetric'])
        if key not in cache:
            cache[key] = direct_solver(self.scaled.T if key[1] else self.scaled, rhs.device)
        diagonal = device_tensor(self, 'diagonal', lambda: self.diagonal, rhs.device, torch.float64)[:, None]
        return cache[key].solve(rhs/diagonal)/diagonal

    def apply_torch(self, x, operation, transpose=False):
        """Device version of apply (same linear map and exact transpose)."""
        from .gpu_native import device_matrix, device_tensor
        if operation != 'closure' or x.ndim != 3 or x.shape[-1] != 2 or not torch.isfinite(x).all():
            raise ValueError('Expected finite native potential/auxiliary pair')
        device, xd = x.device, x.to(torch.float64)
        interior = device_tensor(self, 'interior', lambda: self.interior, device, torch.long)
        boundary = device_tensor(self, 'boundary', lambda: self.boundary, device, torch.long)
        mass = device_matrix(self, 'mass', lambda: self.mass, device, transpose)
        cross = device_matrix(self, 'cross', lambda: self.cross, device, transpose)
        p, a = xd[..., 0].T, xd[..., 1].T   # [ndof, batch]
        if transpose:
            z = self._device_solve(p[interior], True)
            potential = torch.zeros_like(p)
            potential[boundary] = p[boundary] - cross @ z
            auxiliary = a.clone()
            auxiliary -= mass @ z
        else:
            potential = p.clone()
            potential[interior] = self._device_solve(-(mass @ a) - cross @ p[boundary], False)
            auxiliary = a
        return torch.stack((potential.T, auxiliary.T), dim=-1).to(x.dtype)

    def __call__(self, pair):
        return _FixedNative.apply(pair, self, 'closure')


def bernstein_basis(geometry):
    """Exact bicubic coefficient map on four half-size reference-cell patches.

    The Bernstein convex hull gives a sufficient whole-cell positivity condition.
    One de Casteljau subdivision is less restrictive than the unsplit hull.
    """
    e,q,local=geometry.basis.shape
    order=int(np.sqrt(q))
    if order**2!=q or order<4:
        raise ValueError('Require tensor Gauss quadrature of order >=4')
    t=(np.polynomial.legendre.leggauss(order)[0]+1)/2
    b=np.stack(((1-t)**3,3*t*(1-t)**2,3*t**2*(1-t),t**3),axis=1)
    inverse=np.linalg.inv(b) if order==4 else np.linalg.pinv(b)
    control=np.einsum('ia,jb,eabl->eijl',inverse,inverse,geometry.basis.reshape(e,order,order,local))
    left=np.array([[1,0,0,0],[.5,.5,0,0],[.25,.5,.25,0],[.125,.375,.375,.125]])
    right=left[::-1,::-1]
    return np.concatenate([np.einsum('ia,jb,eabl->eijl',a,b,control).reshape(e,16,local)
                           for a in (left,right) for b in (left,right)],axis=1)


class QuadratureSampler:
    def __init__(self, geometry, basis=None):
        basis=geometry.basis if basis is None else basis
        e, q, local = basis.shape
        rows = np.broadcast_to(np.arange(e*q).reshape(e,q,1), (e,q,local)).ravel()
        cols = np.broadcast_to(geometry.dof_map[:,None,:], (e,q,local)).ravel()
        self.basis = sp.coo_matrix((basis.ravel(), (rows,cols)),
                                   shape=(e*q, len(geometry.dof_ids))).tocsr()

    def apply(self, x, operation, transpose=False):
        batch, _, channels = x.shape
        a = x.transpose(1,0,2).reshape(x.shape[1],-1)
        result = self.basis.T @ a if transpose else self.basis @ a
        return result.reshape(-1,batch,channels).transpose(1,0,2).copy()

    torch_operations = ('sample',)

    def apply_torch(self, x, operation, transpose=False):
        from .gpu_native import apply_batched, device_matrix
        if operation != 'sample':
            raise ValueError('Unknown sampler operation')
        return apply_batched(device_matrix(self, 'basis', lambda: self.basis, x.device, transpose), x).to(x.dtype)

    def __call__(self, x):
        return _FixedNative.apply(x, self, 'sample')


class PhysicalGuard(nn.Module):
    def __init__(self, geometry, q8, mean, scale, *, positivity_basis='bernstein_subdivided'):
        super().__init__()
        self.current = EllipticPair(geometry, True)
        self.vorticity = EllipticPair(geometry, False)
        if positivity_basis not in ('bernstein_subdivided','q8'):
            raise ValueError('Unknown positivity basis')
        self.sampler = (QuadratureSampler(geometry,bernstein_basis(geometry))
                        if positivity_basis=='bernstein_subdivided' else QuadratureSampler(q8))
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float64))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float64))
        fixed = np.zeros(len(geometry.dof_ids),dtype=bool)
        fixed[np.unique(geometry.dof_map[:,::4])] = True
        fixed &= geometry.boundary_labels != 0
        self.register_buffer('fixed_boundary_values',torch.from_numpy(fixed))
        anchor = self.sampler.basis @ np.asarray(mean)[:,4:6]
        minimum = anchor.min(0)
        if not (minimum > 0).all():
            raise ValueError('Training mean must have positive coefficients in the limiter basis')
        self.register_buffer('floor', torch.from_numpy(minimum*1e-6))
        self.register_buffer('anchor_values',torch.from_numpy(anchor),persistent=False)
        self.last_penalty = None
        self.reset_statistics()

    def reset_statistics(self):
        self.statistics = {'steps':0, 'limited_field_steps':0, 'minimum_alpha':1.,
                           'worst_proposed_density_bound':None, 'worst_proposed_temperature_bound':None}

    def forward(self, previous, proposal):
        old = previous*self.scale+self.mean
        new = proposal*self.scale+self.mean
        new_values = self.sampler(new[...,4:6])
        if not torch.isfinite(new_values).all():
            raise ValueError('Positivity limiter needs finite proposals')
        ratios = torch.where(new_values < self.floor,
                             (self.anchor_values-self.floor)/(self.anchor_values-new_values).clamp_min(1e-30),
                             torch.ones_like(new_values)).amin(dim=1).clamp(0,1)
        alpha = torch.where(ratios < 1, .99*ratios, torch.ones_like(ratios))
        positive = self.mean[...,4:6]+alpha[:,None,:]*(new[...,4:6]-self.mean[...,4:6])
        violation = ((self.floor-new_values).clamp_min(0)/self.scale[4:6]).square()
        self.last_penalty = violation.amax(dim=1).mean()
        potential = torch.where(self.fixed_boundary_values[None,:,None],old[...,:2],new[...,:2])
        current = self.current(torch.stack((potential[...,0],new[...,2]),dim=-1))
        vort = self.vorticity(torch.stack((potential[...,1],new[...,3]),dim=-1))
        physical = torch.cat((current[...,:1],vort[...,:1],new[...,2:4],positive,new[...,6:]),dim=-1)
        with torch.no_grad():
            self.statistics['steps'] += len(previous)
            self.statistics['limited_field_steps'] += int((alpha < 1).sum())
            self.statistics['minimum_alpha'] = min(self.statistics['minimum_alpha'],float(alpha.min()))
            for i,key in enumerate(('worst_proposed_density_bound','worst_proposed_temperature_bound')):
                value = float(new_values[...,i].min())
                old_min = self.statistics[key]
                self.statistics[key] = value if old_min is None else min(value,old_min)
        return (physical-self.mean)/self.scale
