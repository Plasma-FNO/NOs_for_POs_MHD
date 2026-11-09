"""Frozen tangent of model303 density/temperature advection and compression.

From native/full_split_procedures.f90: rho_t = a.grad(rho) + c*rho,
p_t = a.grad(p) + gamma*c*p. With p=rho*T this gives
T_t = a.grad(T) + (gamma-1)*c*T. Sources, diffusion and all other
equations remain in N. This is a partial physical spatial operator, not
the full JOREK temporal Jacobian or a conservative discretization of it.
"""
import numpy as np
import torch
from torch import nn


def continuity_tangent(q, g, dq, dg, radius, gamma):
    """Exact directional derivative at fixed physical background jets q,g."""
    r = radius
    a = torch.stack((r*g[...,1,1]-q[...,6]*g[...,0,1]/r,
                     -r*g[...,1,0]+q[...,6]*g[...,0,0]/r), -1)
    da = torch.stack((r*dg[...,1,1]-(dq[...,6]*g[...,0,1]+q[...,6]*dg[...,0,1])/r,
                      -r*dg[...,1,0]+(dq[...,6]*g[...,0,0]+q[...,6]*dg[...,0,0])/r), -1)
    c = 2*g[...,1,1]-(g[...,6,0]*g[...,0,1]-g[...,6,1]*g[...,0,0])/r
    dc = 2*dg[...,1,1]-(dg[...,6,0]*g[...,0,1]+g[...,6,0]*dg[...,0,1]
                           -dg[...,6,1]*g[...,0,0]-g[...,6,1]*dg[...,0,0])/r
    result = torch.zeros_like(dq)
    for field, factor in ((4,1.), (5,gamma-1.)):
        result[...,field] = ((a*dg[...,field,:]+da*g[...,field,:]).sum(-1)
                             + factor*(c*dq[...,field]+dc*q[...,field]))
    return result


class FrozenContinuity(nn.Module):
    def __init__(self, geometry, mean, scale, code_interval, gamma=5/3):
        super().__init__()
        if not np.isfinite(code_interval) or code_interval <= 0 or not 0 < gamma <= 2:
            raise ValueError('Require positive native code-time interval and positive gamma')
        local = np.asarray(mean)[geometry.dof_map]
        q = np.einsum('eql,elc->eqc',geometry.basis,local).reshape(-1,7)
        g = np.einsum('eqlr,elc->eqcr',geometry.basis_grad,local).reshape(-1,7,2)
        for name, value in (('background',q),('gradient',g),('scale',scale),
                            ('radius',geometry.quadrature_rz[...,0].ravel())):
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float64))
        self.code_interval, self.gamma = float(code_interval), float(gamma)

    def forward(self, values, gradients, gamma=None):
        gamma = self.gamma if gamma is None else gamma
        if torch.is_tensor(gamma):
            if gamma.shape != (values.shape[0], 1) or not torch.all(torch.isfinite(gamma) & (gamma > 0) & (gamma <= 2)):
                raise ValueError("Gamma must be [B,1] with 0 < gamma <= 2")
        dq = values*self.scale
        dg = gradients*self.scale[:,None]
        return self.code_interval*continuity_tangent(self.background,self.gradient,dq,dg,
                                                    self.radius,gamma)/self.scale
