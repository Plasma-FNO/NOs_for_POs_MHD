"""Composite fluid/EM dynamics with learned auxiliary defects and linear IMEX steps.

Neural rates use saved-frame time. Fixed diffusion uses nominal common-native
frame_dt=5. Auxiliary defects include sampling and boundary reconstruction error;
these are effective dynamics, not independently identified JOREK equation terms.
"""
import torch
from torch import nn
import torch.nn.functional as F
from model import fno
from native import N0, MU0, EV, TAU, F0
from data import SCALES


def positive_increment(value, rate, reference):
    """Softplus coordinates: logarithmic near zero, additive for large fields.

    An exponential update amplifies a network rate proportional to the state into
    super-exponential feedback. This map has derivative at most reference in rate.
    """
    scaled=value/reference
    coordinate=scaled+torch.log(-torch.expm1(-scaled))
    return reference*torch.logaddexp(torch.zeros_like(coordinate),coordinate+rate)


class InteriorDiffusion(nn.Module):
    """Backward Euler on the sampled interior; learned edge values supply Dirichlet data.

    The outer strip is part of the learned state, not the physical wall. Constant
    cylindrical coefficients admit a separable Sylvester solve with cached bases.
    """
    def __init__(self, r, z, coefficient, dt, sign=1, strip=4):
        super().__init__()
        r, z = torch.as_tensor(r, dtype=torch.float64), torch.as_tensor(z, dtype=torch.float64)
        if strip < 2 or min(len(r), len(z)) <= 2*strip+2:
            raise ValueError('Interior grid is too small')
        for x in (r, z):
            if not torch.all(torch.diff(x)>0) or not torch.allclose(torch.diff(x), (x[1]-x[0]).expand(len(x)-1), rtol=1e-3, atol=1e-10):
                raise ValueError('Uniform increasing sampled grid required')
        if coefficient < 0 or dt <= 0 or sign not in (-1,1):
            raise ValueError('Invalid linear diffusion parameters')
        self.strip, self.step = strip, float(coefficient*dt)
        ri = r[strip:-strip]; dr, dz = r[1]-r[0], z[1]-z[0]
        lo, hi = dr**-2-sign/(2*dr*ri), dr**-2+sign/(2*dr*ri)
        if torch.any(lo <= 0) or torch.any(hi <= 0): raise ValueError('Nonmonotone radial stencil')
        ar = torch.diag(torch.full_like(ri,-2/dr**2))
        off = torch.sqrt(lo[1:]*hi[:-1]); ar += torch.diag(off,1)+torch.diag(off,-1)
        er, qr = torch.linalg.eigh(ar)
        sr = torch.cat((torch.ones(1,dtype=r.dtype),torch.cumprod(torch.sqrt(lo[1:]/hi[:-1]),0)))
        nz = len(z)-2*strip
        az = torch.diag(torch.full((nz,),-2/dz**2,dtype=r.dtype))
        az += torch.diag(torch.full((nz-1,),1/dz**2,dtype=r.dtype),1)+torch.diag(torch.full((nz-1,),1/dz**2,dtype=r.dtype),-1)
        ez, qz = torch.linalg.eigh(az)
        for name,value in dict(qr=qr,qz=qz,sr=sr,denominator=1-self.step*(ez[:,None]+er[None,:]),
                               lo=lo,hi=hi,zscale=dz**-2).items(): self.register_buffer(name,value)

    def forward(self, x):
        if self.step == 0: return x
        s=self.strip
        rhs=x[...,s:-s,s:-s].clone()
        rhs[...,0] += self.step*self.lo[0]*x[...,s:-s,s-1]
        rhs[...,-1] += self.step*self.hi[-1]*x[...,s:-s,-s]
        rhs[...,0,:] += self.step*self.zscale*x[...,s-1,s:-s]
        rhs[...,-1,:] += self.step*self.zscale*x[...,-s,s:-s]
        modes=self.qz.T @ (rhs/self.sr) @ self.qr
        interior=(self.qz @ (modes/self.denominator) @ self.qr.T)*self.sr
        # Keep the learned boundary strip unchanged; no sampled edge is clamped to a wall value.
        result=x.clone();result[...,s:-s,s:-s]=interior
        return result


class CompositeSplit(nn.Module):
    def __init__(self, r, z, mean, scale, channels=4, width=64, modes=32, layers=6,
                 em_layers=2, diffusion=1e-6, resistivity=1e-5, frame_dt=5., substeps=1):
        super().__init__()
        if channels not in (4,6) or substeps < 1: raise ValueError('Invalid channels or substeps')
        self.channels, self.substeps = channels, substeps
        self.config=dict(channels=channels,width=width,modes=modes,layers=layers,em_layers=em_layers,
                         diffusion=diffusion,resistivity=resistivity,frame_dt=frame_dt,substeps=substeps,
                         formulation='composite-imex-auxiliary-defect',boundary_strip=4,positive_update='softplus-rho-T',
                         state_representation='exported-float64')
        for name,value in dict(r=r,z=z,mean=mean,scale=scale).items():
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float64).detach().clone())
        if self.mean.shape != (channels,) or self.scale.shape != (channels,) or torch.any(self.scale<=0):
            raise ValueError('Expected positive fieldwise normalization scales')
        factor=torch.tensor(SCALES[:channels],dtype=torch.float64)
        factor[:3] *= torch.tensor([1/N0,N0*MU0*EV,-TAU/F0],dtype=torch.float64)
        self.register_buffer('native_factor',factor)
        self.register_buffer('fluid_mean',self.mean[:4].clone())
        self.register_buffer('fluid_scale',self.scale[:4].clone())
        self.fluid=fno(4,4,width,modes,layers,True)
        if channels==6: self.magnetic=fno(6,6,width,modes,em_layers,True)
        for block in [self.fluid]+([self.magnetic] if channels==6 else []):
            nn.init.zeros_(block.projection.fcs[-1].weight)
            if block.projection.fcs[-1].bias is not None: nn.init.zeros_(block.projection.fcs[-1].bias)
        self.particle=InteriorDiffusion(r,z,diffusion,frame_dt/substeps)
        if channels==6:self.resistive=InteriorDiffusion(r,z,resistivity,frame_dt/substeps,sign=-1)

    def constraints(self, q):
        """Fourth-order interior reconstruction; the outer four cells are learned defects."""
        def lap(x, sign):
            dr,dz=self.r[1]-self.r[0],self.z[1]-self.z[0];c=x[...,2:-2,2:-2]
            rr=(-x[...,2:-2,4:]+16*x[...,2:-2,3:-1]-30*c+16*x[...,2:-2,1:-3]-x[...,2:-2,:-4])/(12*dr**2)
            zz=(-x[...,4:,2:-2]+16*x[...,3:-1,2:-2]-30*c+16*x[...,1:-3,2:-2]-x[...,:-4,2:-2])/(12*dz**2)
            first=(-x[...,2:-2,4:]+8*x[...,2:-2,3:-1]-8*x[...,2:-2,1:-3]+x[...,2:-2,:-4])/(12*dr)
            interior=rr+zz+sign*first/self.r[2:-2]
            return F.pad(interior[...,2:-2,2:-2],(4,4,4,4))
        return [lap(q[:,2],1)]+([lap(q[:,4],-1)] if self.channels==6 else [])

    def forward(self, x):
        # Centered coordinates erase tiny positive fields on decoding, even in float64.
        exported=x.double()
        for _ in range(self.substeps):
            q=exported*self.native_factor[None,:,None,None]
            self.check_state(q,'input')
            shared=((exported[:,:4]-self.fluid_mean[None,:,None,None])/self.fluid_scale[None,:,None,None]).float()
            fluid_units=torch.cat((torch.ones_like(self.fluid_scale[:2]),self.fluid_scale[2:]))
            rate=self.fluid(shared).double()*fluid_units[None,:,None,None]
            if self.channels==6:
                rate=torch.cat((rate,torch.zeros_like(exported[:,4:])),1)
                em_units=torch.cat((torch.ones_like(self.scale[:2]),self.scale[2:]))
                normalized=(exported-self.mean[None,:,None,None])/self.scale[None,:,None,None]
                rate=rate+self.magnetic(normalized.float()).double()*em_units[None,:,None,None]
            previous=self.constraints(q)
            increment=rate*self.native_factor[None,:,None,None]/self.substeps
            # Shared train-only scales keep the positive-coordinate interface transferable.
            positive_reference=self.fluid_scale[:2]*self.native_factor[:2]
            candidate=torch.stack([positive_increment(q[:,i],rate[:,i]/self.substeps,positive_reference[i]) if i<2 else q[:,i]+increment[:,i]
                                   for i in range(self.channels)],1)
            self.check_state(candidate,'neural update')
            candidate=torch.stack([self.particle(candidate[:,i]) if i==0 else
                                   self.resistive(candidate[:,i]) if i==4 and self.channels==6 else candidate[:,i]
                                   for i in range(self.channels)],1)
            self.check_state(candidate,'implicit diffusion')
            updated=self.constraints(candidate)
            # Network auxiliary outputs are increments to omega-Delta(u) and j-Delta*(psi).
            correction={3:updated[0]-previous[0]}
            if self.channels==6:correction[5]=updated[1]-previous[1]
            candidate=torch.stack([candidate[:,i]+correction.get(i,0) for i in range(self.channels)],1)
            self.check_state(candidate,'auxiliary reconstruction')
            exported=candidate/self.native_factor[None,:,None,None]
        return exported

    def check_state(self, value, stage):
        if torch.isfinite(value).all() and torch.all(value[:,:2]>0):return
        names=['rho','T','potential','omega','psi','j'][:self.channels]
        details=[]
        for index,name in enumerate(names):
            field=value[:,index]
            bad=~torch.isfinite(field)
            if index<2:bad=bad | (field<=0)
            if bad.any():
                location=bad.nonzero()[0].tolist()
                details.append(f'{name}: min={float(field.min()):.6g}, max={float(field.max()):.6g}, first_bad_batch_Z_R={location}')
        raise FloatingPointError(f'Composite {stage}: '+ '; '.join(details))

    def transfer(self, source, freeze=False):
        if source.channels!=4 or self.channels!=6:raise ValueError('Transfer requires ES source and EM target')
        if not torch.equal(self.r,source.r) or not torch.equal(self.z,source.z):raise ValueError('Transfer grids differ')
        self.fluid.load_state_dict(source.fluid.state_dict())
        self.fluid_mean.copy_(source.fluid_mean);self.fluid_scale.copy_(source.fluid_scale)
        self.fluid.requires_grad_(not freeze)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict=state_dict.copy();state_dict.pop('_metadata',None)
        return super().load_state_dict(state_dict,strict=strict,assign=assign)
