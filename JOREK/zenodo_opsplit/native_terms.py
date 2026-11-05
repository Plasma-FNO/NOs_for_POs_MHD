"""Independent native FNO terms: whole operators or bounded additive corrections."""
import torch
from torch import nn
from model import fno

NAMES = ('scalar_advection', 'fluid_momentum', 'particle_diffusion', 'thermal_diffusion')
SLICES = (slice(0, 2), slice(2, 3), slice(3, 4), slice(4, 5))


def features(q):
    rho, p, u = q[:, 0:1], q[:, 1:2], q[:, 2:3]
    if torch.any(rho <= 0):
        raise ValueError('Native term features require positive density')
    return torch.cat((rho, p, u, p/rho), 1)


def labels(reference, q):
    """Coefficient-free D targets; explicit target coefficients applied at composition.

    Labels are FD approximations to source-derived equation terms, not native FEM
    residual exports. Supervision is restricted to a four-cell interior crop.
    """
    terms = reference.terms(q)
    extended = reference.extend(q)
    rho, p = extended[:, 0], extended[:, 1]
    return torch.cat((terms['scalar_advection'], terms['fluid_momentum'][:, None],
                      reference.laplacian(rho)[:, None, 1:-1, 1:-1],
                      reference.laplacian(p/rho)[:, None, 1:-1, 1:-1]), 1)


class NativeOperators(nn.Module):
    def __init__(self, input_mean, input_scale, target_scale, width=64, modes=32, layers=(2, 2, 1, 1),
                 correction_fraction=None):
        super().__init__()
        self.register_buffer('input_mean', torch.as_tensor(input_mean, dtype=torch.float32))
        self.register_buffer('input_scale', torch.as_tensor(input_scale, dtype=torch.float32))
        self.register_buffer('target_scale', torch.as_tensor(target_scale, dtype=torch.float32))
        if torch.any(self.input_scale <= 0) or torch.any(self.target_scale <= 0):
            raise ValueError('Positive normalizer scales required')
        self.config = dict(width=width, modes=modes, layers=list(layers))
        self.correction_fraction = correction_fraction
        self.is_residual = correction_fraction is not None
        if self.is_residual:
            if not 0 < correction_fraction <= 1:
                raise ValueError('Correction fraction must be in (0,1]')
            self.config['correction_fraction'] = correction_fraction
        self.blocks = nn.ModuleDict({name: fno(3 if i<2 else 1, 2 if i==0 else 1, width, modes, layers[i], True)
                                    for i, name in enumerate(NAMES)})
        if self.is_residual:
            # Zero outputs preserve analytic dynamics; the final layer still receives gradients.
            for block in self.blocks.values():
                nn.init.zeros_(block.projection.fcs[-1].weight)
                nn.init.zeros_(block.projection.fcs[-1].bias)

    def forward(self, x):
        normalized=(x-self.input_mean[None,:,None,None])/self.input_scale[None,:,None,None]
        inputs=(normalized[:,:3], normalized[:,:3], normalized[:,0:1], normalized[:,3:4])
        output = torch.cat([self.blocks[name](value) for name,value in zip(NAMES, inputs)], 1)
        return self.correction_fraction*output.tanh() if self.is_residual else output

    def physical_terms(self, q):
        """Physical operator values, or additive corrections when is_residual=True."""
        result=self(features(q).float())*self.target_scale[None,:,None,None]
        return {name:result[:, part] for name,part in zip(NAMES,SLICES)}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict=state_dict.copy();state_dict.pop('_metadata',None)
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    def transfer(self, source, names, freeze=False):
        # Transfer weights only under the identical interface normalization.
        if self.correction_fraction != source.correction_fraction:
            raise ValueError('Transfer requires identical operator/correction semantics')
        for key in ('input_mean','input_scale','target_scale'):
            if not torch.equal(getattr(self,key),getattr(source,key)):
                raise ValueError('Transfer requires shared source interface normalizers')
        for name in names:
            self.blocks[name].load_state_dict(source.blocks[name].state_dict())
            if freeze:self.blocks[name].requires_grad_(False)
