"""Differentiable native IMEX composition with separate learned equation terms.

Sparse LU and its implicit adjoint use SciPy on CPU or cuDSS on GPU.
The CPU backend remains the independent reference for GPU validation.
The four-cell label-excluded strip retains analytic terms. No field clipping.
"""
from contextlib import contextmanager, nullcontext

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from native_imex import IMEXReference
from native_terms import features,NAMES


class SparseSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, rhs, indices):
        ij=indices.detach().cpu().numpy()
        matrix=sp.coo_matrix((values.detach().cpu().double().numpy(),ij),shape=(len(rhs),len(rhs))).tocsc()
        factor=splu(matrix)
        result=factor.solve(rhs.detach().cpu().double().numpy())
        residual=matrix@result-rhs.detach().cpu().double().numpy()
        if not np.isfinite(result).all() or np.any(np.linalg.norm(residual,axis=0)>1e-8*np.maximum(np.linalg.norm(rhs.detach().cpu().numpy(),axis=0),1e-30)):
            raise RuntimeError('Sparse native solve failed residual check')
        output=torch.as_tensor(result,dtype=rhs.dtype,device=rhs.device)
        ctx.factor=factor;ctx.save_for_backward(output,indices)
        return output

    @staticmethod
    def backward(ctx, grad):
        solution,indices=ctx.saved_tensors
        adjoint=torch.as_tensor(ctx.factor.solve(grad.detach().cpu().double().numpy(),trans='T'),device=grad.device,dtype=grad.dtype)
        values_grad=-adjoint[indices[0]]*solution[indices[1]]
        return values_grad.sum(-1) if solution.ndim==2 else values_grad,adjoint,None


def sparse(matrix):
    coo=matrix.tocoo()
    return torch.sparse_coo_tensor(np.stack((coo.row,coo.col)),coo.data,coo.shape,dtype=torch.float64).coalesce()


def diagonal(values):
    i=torch.arange(len(values),device=values.device)
    return torch.sparse_coo_tensor(torch.stack((i,i)),values,(len(values),len(values)),is_coalesced=True)


def mv(matrix,vector):
    return torch.sparse.mm(matrix,vector[:,None])[:,0]


def solve(matrix,rhs):
    matrix=matrix.coalesce()
    if rhs.is_cuda:
        from gpu_sparse import GPUSparseSolve
        return GPUSparseSolve.apply(matrix.values(),rhs,matrix.indices())
    return SparseSolve.apply(matrix.values(),rhs,matrix.indices())


class NativeRollout(IMEXReference):
    def __init__(self,reference,initial,operators=None,blend=1.,modules=NAMES,positivity_guard=True,device=None,checkpoint_steps=False,compiled_terms=False):
        super().__init__(reference,initial.detach())
        if not 0<=blend<=1:raise ValueError('Blend must be in [0,1]')
        self.operators,self.blend=operators,blend
        if not set(modules)<=set(NAMES):raise ValueError('Unknown native operator')
        self.modules=set(modules)
        self.positivity_guard=positivity_guard
        self.checkpoint_steps=checkpoint_steps
        self.reset_limiter_stats()
        for name in ('W','Ws','Kr','Ks','K1','H','Hu'):
            setattr(self,name,sparse(getattr(self,name)))
        for name in ('V','R','br','bs'):
            setattr(self,name,torch.as_tensor(getattr(self,name),dtype=torch.float64))
        # Keep products sparse; avoid any n-by-n dense allocation.
        self.Kmag=torch.sparse.mm(torch.sparse.mm(self.K1,diagonal(self.R/self.V)),self.Ks).coalesce()
        if initial.shape[1]==4:self.bj=torch.as_tensor(self.bj)
        self.mask=F.pad(torch.ones_like(initial[:,0,4:-4,4:-4]),(4,4,4,4))*blend
        ny,nx=self.shape;index=torch.arange(ny*nx).reshape(ny,nx)
        self.index=index
        rows=[index.flatten()];cols=[index.flatten()]
        for left,right in ((index[:,:-1],index[:,1:]),(index[:-1],index[1:])):
            rows.extend((left.flatten(),right.flatten()));cols.extend((right.flatten(),left.flatten()))
        ij=torch.stack((torch.cat(rows),torch.cat(cols)))
        self.stencil_order=torch.argsort(ij[0]*self.n+ij[1])
        self.stencil_indices=ij[:,self.stencil_order]
        self.device=torch.device(device or 'cpu')
        if self.device.type=='cuda':
            from native import NativeReference
            self.ref=NativeReference(reference.mass.r.to(self.device),reference.mass.z.to(self.device),
                                     reference.wall.to(self.device),reference.c)
            for name in ('W','Ws','Kr','Ks','K1','H','Hu','V','R','br','bs','Kmag','mask','index','stencil_order','stencil_indices'):
                setattr(self,name,getattr(self,name).to(self.device))
            if initial.shape[1]==4:self.bj=self.bj.to(self.device)
            if compiled_terms:self.ref._terms=torch.compile(self.ref._terms,fullgraph=True)

    def reset_limiter_stats(self):
        self.limiter_stats=dict(substeps=0,limited_substeps=0,analytic_fallbacks=0,minimum_fraction=1.,
                                density_limited=0,temperature_limited=0,minimum_scalar_fraction=1.,
                                cfl_limited_substeps=0,minimum_dt_fraction=1.,
                                energy_limited=0,minimum_energy_fraction=1.)

    @contextmanager
    def _recompute_context(self):
        # Replayed substeps must not double-count limiter activity in reports.
        saved=dict(self.limiter_stats)
        try:yield
        finally:self.limiter_stats=saved

    def stiffness_tensor(self,a):
        cr,cz=self.ref.mass.edges(a)
        vals=torch.cat(((cr[:,1:]+cr[:,:-1]+cz[1:]+cz[:-1]).flatten(),
                        -cr[:,1:-1].flatten(),-cr[:,1:-1].flatten(),
                        -cz[1:-1].flatten(),-cz[1:-1].flatten()))
        return self.stencil_matrix(vals)

    def stencil_matrix(self,values):
        return torch.sparse_coo_tensor(self.stencil_indices,values[self.stencil_order],
                                      (self.n,self.n),is_coalesced=True)

    def upwind_transport(self,extended):
        """Integrated div(R v scalar), v=(-R u_Z, R u_R), impermeable wall.

        Interior face fluxes cancel pairwise. The off-diagonals are nonpositive,
        making W+h*(transport+diffusion) a nonsingular M-matrix for h>0.
        """
        # Shared corner potentials make the mixed derivatives cancel. Averaging
        # cell-centred derivatives independently leaves an O(R**2/h) divergence
        # defect next to the imposed impermeable wall, draining density there.
        u=extended[0,2,1:-1,1:-1]
        corners=F.pad((u[1:,1:]+u[1:,:-1]+u[:-1,1:]+u[:-1,:-1])/4,(1,1,1,1))
        r=self.ref.mass.r
        rf=(r[1:]+r[:-1])/2
        fr=-rf.square()[None,:]*(corners[1:,1:-1]-corners[:-1,1:-1])
        fz=r.square()[None,:]*(corners[1:-1,1:]-corners[1:-1,:-1])
        rp,rn=fr.clamp_min(0),fr.clamp_max(0)
        zp,zn=fz.clamp_min(0),fz.clamp_max(0)
        diag=F.pad(rp,(0,1))-F.pad(rn,(1,0))+F.pad(zp,(0,0,0,1))-F.pad(zn,(0,0,1,0))
        return self.stencil_matrix(torch.cat((diag.flatten(),rn.flatten(),-rp.flatten(),zn.flatten(),-zp.flatten())))

    def positive_scalar(self,high,low,field,lower=0.):
        """Convex correction of the entire scalar update, never pointwise clipping.

        The low-order implicit upwind/diffusion update is positive. Retain the
        full high-order update whenever admissible; otherwise retain 99% of the
        largest globally admissible correction. Differentiate the limiter too.
        """
        if not torch.isfinite(high).all() or not torch.isfinite(low).all() or torch.any(low<=0):
            raise FloatingPointError('Invalid low/high-order scalar update')
        # Keep the analytic bound below the low-order solution by roundoff only.
        lower=torch.as_tensor(lower,device=low.device,dtype=low.dtype)
        if torch.any(low<lower*(1-1e-8)):raise FloatingPointError('Low-order density violates transport bound')
        lower=torch.minimum(lower,low.min()*(1-1e-12))
        negative=high<=lower
        denominator=torch.where(negative,low-high,torch.ones_like(low))
        fraction=torch.where(negative,.99*(low-lower)/denominator,torch.ones_like(low)).min()
        result=low+fraction*(high-low)
        self.limiter_stats[field+'_limited']+=1
        self.limiter_stats['minimum_scalar_fraction']=min(self.limiter_stats['minimum_scalar_fraction'],float(fraction.detach()))
        if not torch.isfinite(result).all() or torch.any(result<=0) or torch.any(result<lower):
            raise FloatingPointError('Scalar convex correction failed positivity')
        return result

    def limit_kinetic_energy(self, old, high, forced, old_mass, new_mass):
        """Relax momentum to a force-only kinetic-energy envelope.

        Independent learned scalar/momentum terms do not cancel transport work.
        Retain the old kinetic energy or the pressure/Lorentz/diffusion-driven
        reference energy, whichever is larger. This is a numerical safeguard,
        not an exact discrete total-energy law for JOREK. Preserve direction and
        the homogeneous wall condition; differentiate the scalar relaxation.
        """
        ceiling=torch.maximum(.5*(old*mv(old_mass,old)).sum(),
                              .5*(forced*mv(new_mass,forced)).sum())
        actual=.5*(high*mv(new_mass,high)).sum()
        if not torch.isfinite(actual) or not torch.isfinite(ceiling) or ceiling<0:
            raise FloatingPointError('Invalid kinetic-energy envelope')
        if actual>ceiling:
            fraction=(ceiling/actual).sqrt() if ceiling>0 else actual*0
            self.limiter_stats['energy_limited']+=1
            self.limiter_stats['minimum_energy_fraction']=min(self.limiter_stats['minimum_energy_fraction'],float(fraction.detach()))
            return high*fraction
        return high

    @torch.no_grad()
    def stable_timestep(self,q,maximum):
        """CFL bound for explicit fluid transport and poloidal Alfven coupling.

        Scalar diffusion is implicit. Freeze controller choices in the adjoint,
        as for the discrete backtracking choices. No state is clipped.
        """
        extended=self.ref.extend(q)
        ur,uz=self.ref.op.grad(extended[:,2])
        ur,uz=ur[:,1:-1,1:-1],uz[:,1:-1,1:-1]
        r=self.ref.mass.r
        rate=(r*uz).abs()/self.ref.mass.wr+(r*ur).abs()/self.ref.mass.wz[:,None]
        rate=rate+2*self.ref.c['gamma']*uz.abs()
        if q.shape[1]==4:
            pr,pz=self.ref.op.grad(extended[:,3])
            br,bz=-pz[:,1:-1,1:-1]/r,pr[:,1:-1,1:-1]/r
            rate=rate+(br.abs()/self.ref.mass.wr+bz.abs()/self.ref.mass.wz[:,None])/q[:,0].sqrt()
        fastest=float(rate.max())
        if not np.isfinite(fastest):raise FloatingPointError('Nonfinite explicit wave speed')
        return min(maximum,.4/fastest) if fastest else maximum

    def step(self,q,dt,substeps=16):
        """Backtrack the neural correction, never clip the state.

        Each accepted trial re-solves all implicit equations with one common
        correction fraction. Zero fraction is the complete analytic IMEX step.
        Discrete line-search choices are held fixed in the implicit adjoint.
        Report limiter_stats alongside every learned rollout result.
        """
        if substeps<1 or dt<=0:raise ValueError('Positive dt and substep count required')
        x=q
        fractions=[2.**(-k) for k in range(9)]+[0.] if self.positivity_guard and self.operators is not None else [1.]
        elapsed=0.;accepted=0;maximum=dt/substeps
        while elapsed<dt*(1-1e-14):
            h=self.stable_timestep(x,min(maximum,dt-elapsed))
            if accepted>=4096 or h<dt*1e-12:
                raise FloatingPointError('Explicit CFL requires excessive substeps; refusing unstable advance')
            for fraction in fractions:
                try:
                    if self.checkpoint_steps and torch.is_grad_enabled():
                        candidate=checkpoint(self._trial_step,x,h,correction_scale=fraction,
                                             checkpoint_model=False,use_reentrant=False,
                                             context_fn=lambda:(nullcontext(),self._recompute_context()))
                    else:
                        candidate=self._trial_step(x,h,correction_scale=fraction)
                except FloatingPointError:
                    if fraction==fractions[-1]:raise
                    continue
                x=candidate;elapsed+=h;accepted+=1
                self.limiter_stats['substeps']+=1
                self.limiter_stats['limited_substeps']+=int(fraction<1)
                self.limiter_stats['analytic_fallbacks']+=int(fraction==0)
                self.limiter_stats['minimum_fraction']=min(self.limiter_stats['minimum_fraction'],fraction)
                self.limiter_stats['cfl_limited_substeps']+=int(h<maximum*(1-1e-12))
                self.limiter_stats['minimum_dt_fraction']=min(self.limiter_stats['minimum_dt_fraction'],h/maximum)
                break
        return x

    def _trial_step(self,q,dt,substeps=1,correction_scale=1.,checkpoint_model=True):
        if q.shape[0]!=1 or q.device.type!=self.device.type:raise ValueError('Use one trajectory on the configured elliptic device')
        if substeps<1 or dt<=0:raise ValueError('Positive dt and substep count required')
        c=self.ref.c
        if c['viscosity']!='old':raise ValueError('Validated release profile uses old viscosity')
        x=q.double();h=dt/substeps;mask=self.mask*correction_scale
        for _ in range(substeps):
            terms=self.ref.terms(x,imex=True)
            rho,p,u=x[0,:3].reshape(3,-1)
            extended=self.ref.extend(x)
            laprho=self.ref.laplacian(extended[:,0])[:,1:-1,1:-1]
            lapT=self.ref.laplacian(extended[:,1]/extended[:,0])[:,1:-1,1:-1]
            A=terms['scalar_advection']
            correction=terms['scalar_diffusion']-terms['isotropic_diffusion']
            load=terms['fluid_momentum']+terms['lorentz']+terms['conservative']
            if self.operators is not None and self.blend and correction_scale:
                device=next(self.operators.parameters()).device
                xx=features(x).to(device=device,dtype=torch.float32)
                output=checkpoint(self.operators,xx,use_reentrant=False) if checkpoint_model and torch.is_grad_enabled() else self.operators(xx)
                output=(output*self.operators.target_scale[None,:,None,None]).to(dtype=x.dtype,device=x.device)
                if getattr(self.operators,'is_residual',False):
                    residual=output
                else:
                    residual=output-torch.cat((A,terms['fluid_momentum'][:,None],laprho[:,None],lapT[:,None]),1)
                if 'scalar_advection' in self.modules:A=A+mask[:,None]*residual[:,:2]
                if 'fluid_momentum' in self.modules:load=load+mask*residual[:,2]
                delta=mask[:,None]*torch.stack((c['d']*residual[:,3] if 'particle_diffusion' in self.modules else laprho*0,
                                               c['k']*residual[:,4] if 'thermal_diffusion' in self.modules else lapT*0),1)
                correction=correction+delta
                if c['conservative']:
                    ur,uz=self.ref.op.grad(extended[:,2])
                    # Conservative momentum uses rho_t+div(rho*v), including
                    # the learned continuity defect, not only learned diffusion.
                    density_delta=F.pad(delta[:,0]+A[:,0]-terms['scalar_advection'][:,0],(1,1,1,1))
                    extra=-self.ref.div(self.ref.r**3*density_delta*ur,self.ref.r**3*density_delta*uz)
                    load=load+extra[:,1:-1,1:-1]
            A=A[0].reshape(2,-1);correction=correction[0].reshape(2,-1)
            d=torch.where(rho<c['rho_threshold'],rho.new_tensor(c['d_negative']),rho.new_tensor(c['d']))
            k=torch.where(p/rho<c['T_threshold'],rho.new_tensor(c['k_negative']),rho.new_tensor(c['k']))
            Kd=self.stiffness_tensor((d*self.R).reshape(self.shape))
            Kk=self.stiffness_tensor((k*self.R).reshape(self.shape))
            matrix=self.W+h*Kd+h*c['d_hyper']*self.H
            rhs=self.V*self.R*(rho+h*(A[0]+correction[0]))+h*mv(Kd,torch.full_like(rho,c['rho_wall']))+h*c['d_hyper']*mv(self.H,torch.full_like(rho,c['rho_wall']))
            rhon=solve(matrix,rhs)
            transport=self.upwind_transport(extended)
            # Discrete continuity maximum principle: A*1 = W*1+h*div(v)*W*1
            # plus nonnegative wall diffusion. This lower bound scales with the
            # current state and compression; it is not a fixed density floor.
            divergence=mv(transport,torch.ones_like(rho))/(self.V*self.R)
            lower=torch.minimum(rho.min(),rho.new_tensor(c['rho_wall']))/(1+h*divergence.clamp_min(0).max())
            if torch.any(rhon<lower):
                low=solve(self.W+h*(Kd+transport),self.V*self.R*rho+
                          h*mv(Kd,torch.full_like(rho,c['rho_wall'])))
                rhon=self.positive_scalar(rhon,low,'density',lower)
            matrix=diagonal(self.V*self.R*rhon)+h*Kk+h*c['k_hyper']*self.H
            rhs=self.V*self.R*(p+h*(A[1]+correction[1]))+h*mv(Kk,torch.full_like(rho,c['T_wall']))+h*c['k_hyper']*mv(self.H,torch.full_like(rho,c['T_wall']))
            Tn=solve(matrix,rhs)
            if torch.any(Tn<=0):
                if transport is None:transport=self.upwind_transport(extended)
                # p_t=-div(p v) -(gamma-1)*p*div(v); div(v)=-2*u_Z.
                uz=self.ref.op.grad(extended[:,2])[1][0,1:-1,1:-1].flatten()
                reaction=2*(c['gamma']-1)*uz
                operator=(transport+diagonal(self.V*self.R*(-reaction).clamp_min(0))).coalesce()
                # Right multiplication by diag(rhon) changes p to T unknowns.
                operator=torch.sparse_coo_tensor(operator.indices(),
                    operator.values()*rhon[operator.indices()[1]],operator.shape,device=self.device).coalesce()
                low=solve(diagonal(self.V*self.R*rhon)+h*(Kk+operator),
                    self.V*self.R*p*(1+h*reaction.clamp_min(0))+
                    h*mv(Kk,torch.full_like(rho,c['T_wall'])))
                Tn=self.positive_scalar(Tn,low,'temperature')
            Ka=self.stiffness_tensor((rho*self.R**3).reshape(self.shape))
            momentum_matrix=Ka+h*c['mu']*self.H+h*c['mu_hyper']*self.Hu
            momentum_rhs=mv(Ka,u)-h*self.V*load.flatten()
            if self.operators is not None and self.blend:
                pressure=(-2*self.ref.r*self.ref.op.grad(extended[:,1])[1])[:,1:-1,1:-1]
                force=pressure+terms['lorentz']+terms['conservative']
                if getattr(self.operators,'is_residual',False):
                    # Bound correction energy against the complete analytic candidate.
                    # The historical force-only envelope would alter zero-correction dynamics.
                    force=terms['fluid_momentum']+terms['lorentz']+terms['conservative']
                un,forced=solve(momentum_matrix,torch.stack((momentum_rhs,mv(Ka,u)-h*self.V*force.flatten()),1)).unbind(1)
                Kn=self.stiffness_tensor((rhon*self.R**3).reshape(self.shape))
                un=self.limit_kinetic_energy(u,un,forced,Ka,Kn)
            else:un=solve(momentum_matrix,momentum_rhs)
            result=[rhon,rhon*Tn,un]
            if x.shape[1]==4:
                matrix=self.Ws+h*c['eta']*self.Ks+h*c['eta_hyper']*self.Kmag
                rhs=self.V/self.R*(x[0,3].flatten()+h*terms['induction'].flatten())+h*c['eta']*self.bs+h*c['eta_hyper']*(mv(self.K1,self.R/self.V*self.bs)-self.bj)
                result.append(solve(matrix,rhs))
            x=torch.stack(result).reshape_as(x)
            if not torch.isfinite(x).all() or torch.any(x[:,1]<=0):raise FloatingPointError('Invalid native thermodynamic state')
        return x
