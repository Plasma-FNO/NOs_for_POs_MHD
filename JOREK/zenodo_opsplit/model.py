"""Additive effective RHS split in saved-frame time, using the installed neuralop FNO."""
import torch
from torch import nn
import torch.nn.functional as F
from neuralop.models import FNO


def fno(inputs, outputs, width, modes, layers, paper_architecture=False):
    return FNO(n_modes=(modes, modes), in_channels=inputs, out_channels=outputs,
               hidden_channels=width, n_layers=layers, domain_padding=0.2,
               **(dict(lifting_channel_ratio=1, projection_channel_ratio=4) if paper_architecture else {}))


def laplacian(x, r, z):
    """Axisymmetric scalar Laplacian on interior points; no periodic wraparound."""
    dr, dz = (r[-1] - r[0]) / (len(r)-1), (z[-1] - z[0]) / (len(z)-1)
    center = x[..., 1:-1, 1:-1]
    rr = (x[..., 1:-1, 2:] - 2*center + x[..., 1:-1, :-2]) / dr**2
    radial = (x[..., 1:-1, 2:] - x[..., 1:-1, :-2]) / (2*dr*r[1:-1])
    zz = (x[..., 2:, 1:-1] - 2*center + x[..., :-2, 1:-1]) / dz**2
    return F.pad(rr + radial + zz, (1, 1, 1, 1))


class SplitFNO(nn.Module):
    def __init__(self, r, z, channels=4, width=16, modes=12, layers=3, monolithic=False, autoregressive=False, paper_architecture=False):
        super().__init__()
        if autoregressive and not monolithic:
            raise ValueError('AR requires a whole-state operator')
        self.autoregressive = autoregressive
        if channels not in (4, 6):
            raise ValueError('Expected four ES or six EM channels')
        r, z = torch.as_tensor(r, dtype=torch.float32), torch.as_tensor(z, dtype=torch.float32)
        for coordinate in (r, z):
            if len(coordinate) < 3 or not torch.allclose(torch.diff(coordinate), (coordinate[1]-coordinate[0]).expand(len(coordinate)-1), rtol=1e-3, atol=1e-7):
                raise ValueError('Uniform grid required')
        self.register_buffer('r', r)
        self.register_buffer('z', z)
        self.channels, self.monolithic = channels, monolithic
        self.config = dict(channels=channels, width=width, modes=modes, layers=layers, monolithic=monolithic,
                           autoregressive=autoregressive, paper_architecture=paper_architecture)
        if monolithic:
            self.whole = fno(channels, channels, width, modes, layers, paper_architecture)
        else:
            self.fluid = fno(4, 4, width, modes, layers, paper_architecture)
            # Effective positive diffusivities per saved frame; not claimed as JOREK coefficients.
            self.log_diffusion = nn.Parameter(torch.full((4,), -4.0))
            if channels == 6:
                self.magnetic = fno(6, 6, width, modes, layers, paper_architecture)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # neuralop 1.0.2 inserts constructor metadata at the parent dictionary root.
        state_dict = state_dict.copy()
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def fluid_rhs(self, x):
        # Bound explicit diffusion below the 2D Euler stability limit on this grid.
        cap = .2 / ((self.r[1]-self.r[0])**-2 + (self.z[1]-self.z[0])**-2)
        diffusion = cap * self.log_diffusion.sigmoid()[None, :, None, None]
        return self.fluid(x) + diffusion * laplacian(x, self.r, self.z)

    def rhs(self, x, zero_shot=False):
        if self.monolithic:
            if zero_shot:
                raise ValueError('Zero-shot requires the shared fluid split')
            return self.whole(x)
        fluid = self.fluid_rhs(x[:, :4])
        if self.channels == 4:
            return fluid
        result = torch.cat((fluid, torch.zeros_like(x[:, 4:])), dim=1)
        return result if zero_shot else result + self.magnetic(x)

    def forward(self, x, zero_shot=False):
        # The saved grid lies 1 mm inside the wall. Its edge samples are NOT exact Dirichlet nodes.
        update = self.rhs(x, zero_shot)
        return update if self.autoregressive else x + update

    def transfer(self, source, freeze=False):
        if source.monolithic or self.monolithic or source.channels != 4 or self.channels != 6:
            raise ValueError('Transfer must be split ES -> split EM')
        if not (torch.equal(self.r, source.r) and torch.equal(self.z, source.z)):
            raise ValueError('Transfer grids differ')
        self.fluid.load_state_dict(source.fluid.state_dict())
        self.log_diffusion.data.copy_(source.log_diffusion.data)
        if freeze:
            self.fluid.requires_grad_(False)
            self.log_diffusion.requires_grad_(False)
