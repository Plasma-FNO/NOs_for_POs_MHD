"""Analytic physical derivatives of the pinned L-SplineFNO synthesis basis.

Differentiate the normalized, boundary-truncated cubic basis, including the
quotient rule. This operates on spline coefficients, before any output MLP.
The upstream learned scatter can therefore supply a field-linear encoding;
passing through its FNO, lift, FiLM or output MLP would not be a linear branch.
"""
import importlib

import torch
from torch import nn


class SplineDerivatives(nn.Module):
    def __init__(self, coordinates, bounds, grid_axes):
        super().__init__()
        self.grid_shape = tuple(len(axis) for axis in grid_axes)
        for d, axis in enumerate(grid_axes):
            # Use the actual upstream knots, including their float32 spacing.
            axis = axis.to(coordinates)
            extent = bounds[d, 1] - bounds[d, 0]
            h = axis[1] - axis[0]
            t = ((coordinates[:, d] - bounds[d, 0])/extent - axis[0])/h
            k = torch.floor(t).long()[:, None] + torch.arange(-1, 3, device=t.device)
            z = t[:, None] - k
            a = z.abs()
            valid = (k >= 0) & (k < len(axis))
            b = torch.where(a < 1, 2/3 - a**2 + a**3/2,
                            torch.where(a < 2, (2-a)**3/6, 0.))*valid
            db = torch.where(a < 1, -2*a + 1.5*a**2,
                             torch.where(a < 2, -(2-a)**2/2, 0.))*z.sign()*valid
            ddb = torch.where(a < 1, -2 + 3*a,
                              torch.where(a < 2, 2-a, 0.))*valid
            s, ds, dds = (v.sum(-1, keepdim=True) for v in (b, db, ddb))
            if not (s > 1e-12).all():
                raise ValueError('Derivative queries require supported spline coordinates')
            values = (b/s, (db/s - b*ds/s**2)/(h*extent),
                      (ddb/s - 2*db*ds/s**2 - b*dds/s**2 + 2*b*ds**2/s**3)/(h*extent)**2)
            self.register_buffer(f'indices_{d}', k.clamp(0, len(axis)-1)[None], persistent=False)
            for order, value in enumerate(values):
                self.register_buffer(f'axis_{d}_{order}', value[None], persistent=False)

    def forward(self, coefficients, orders=(0, 0)):
        """[B,C,G] coefficients -> [B,N,C] differentiated synthesis."""
        if len(orders) != 2 or any(o not in (0, 1, 2) for o in orders):
            raise ValueError('Require two derivative orders in {0,1,2}')
        upstream = importlib.import_module('Models.learnable_spline_fno')
        stencil = upstream._FactoredStencil(
            [getattr(self, f'indices_{d}') for d in range(2)],
            [getattr(self, f'axis_{d}_{orders[d]}').to(coefficients) for d in range(2)],
            self.grid_shape)
        return upstream._decode_gather(stencil, coefficients).transpose(1, 2)

    def laplace(self, coefficients):
        return self(coefficients, (2, 0)) + self(coefficients, (0, 2))
