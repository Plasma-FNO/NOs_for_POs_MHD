"""CPU sparse IMEX reference for generating/validating native split targets.

Implicit scalar diffusion, pressure conduction, viscosity and magnetic diffusion
remove the explicit stiffness of the 1 mm wall strip. Not differentiable; use
native.MomentumSolve for differentiable elliptic tests, and term-supervised FNOs
for learning. This is a coarse FV/FD reference, not a replica of JOREK FEM.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import torch


class IMEXReference:
    def __init__(self, reference, initial):
        self.ref=reference
        self.shape=tuple(initial.shape[-2:]);self.n=int(np.prod(self.shape))
        self.V=reference.mass.volume.numpy().ravel()
        self.R=np.broadcast_to(reference.mass.r.numpy()[None,:],self.shape).ravel()
        self.W=sp.diags(self.V*self.R);self.Wi=sp.diags(1/(self.V*self.R))
        self.Ws=sp.diags(self.V/self.R);self.Wsi=sp.diags(self.R/self.V)
        self.Kr,self.br=self.stiffness(torch.as_tensor(self.R.reshape(self.shape)),np.ones((self.shape[0]+2,self.shape[1]+2)))
        self.Ks,self.bs=self.stiffness(torch.as_tensor((1/self.R).reshape(self.shape)),reference.wall[0,3].numpy() if initial.shape[1]==4 else np.zeros((self.shape[0]+2,self.shape[1]+2)))
        self.K1,_=self.stiffness(torch.ones(self.shape,dtype=torch.float64),np.zeros((self.shape[0]+2,self.shape[1]+2)))
        self.H=self.Kr@self.Wi@self.Kr
        L=-self.Wi@self.Kr
        self.Hu=-(L.T@sp.diags(self.V)@L@L)
        if initial.shape[1]==4:
            j0=(self.bs-self.Ks@initial[0,3].numpy().ravel())/(self.V/self.R)
            jb=np.pad(j0.reshape(self.shape),1,mode='edge')
            _,self.bj=self.stiffness(torch.ones(self.shape,dtype=torch.float64),jb)

    def stiffness(self,a,wall):
        cr,cz=(x.numpy() for x in self.ref.mass.edges(a))
        ny,nx=self.shape;index=np.arange(ny*nx).reshape(ny,nx)
        diag=(cr[:,1:]+cr[:,:-1]+cz[1:]+cz[:-1]).ravel()
        rows=[index.ravel()];cols=[index.ravel()];vals=[diag]
        for aa,bb,cc in ((index[:,:-1],index[:,1:],cr[:,1:-1]),(index[:-1],index[1:],cz[1:-1])):
            rows.extend((aa.ravel(),bb.ravel()));cols.extend((bb.ravel(),aa.ravel()));vals.extend((-cc.ravel(),-cc.ravel()))
        matrix=sp.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(self.n,self.n)).tocsr()
        b=np.zeros(self.shape);b[:,0]+=cr[:,0]*wall[1:-1,0];b[:,-1]+=cr[:,-1]*wall[1:-1,-1]
        b[0]+=cz[0]*wall[0,1:-1];b[-1]+=cz[-1]*wall[-1,1:-1]
        return matrix,b.ravel()

    def step(self,q,dt,substeps=16):
        if q.shape[0]!=1 or q.device.type!='cpu':raise ValueError('Reference accepts one CPU trajectory')
        c=self.ref.c
        if c['viscosity']!='old':raise ValueError('IMEX reference currently implements old viscosity only')
        x=q.detach().double().clone();h=dt/substeps
        for _ in range(substeps):
            terms=self.ref.terms(x)
            fields=x[0].numpy().reshape(x.shape[1],-1)
            rho,p,u=fields[:3]
            A=terms['scalar_advection'][0].numpy().reshape(2,-1)
            # Threshold transport may be 100x larger: retain it implicitly.
            D=terms['scalar_diffusion'][0].numpy().reshape(2,-1)
            corr=D-terms['isotropic_diffusion'][0].numpy().reshape(2,-1)
            d=np.where(rho<c['rho_threshold'],c['d_negative'],c['d'])
            k=np.where(p/rho<c['T_threshold'],c['k_negative'],c['k'])
            Kd,bd=self.stiffness(torch.from_numpy((d*self.R).reshape(self.shape)),np.full((self.shape[0]+2,self.shape[1]+2),c['rho_wall']))
            Kk,bk=self.stiffness(torch.from_numpy((k*self.R).reshape(self.shape)),np.full((self.shape[0]+2,self.shape[1]+2),c['T_wall']))
            mat=self.W+h*Kd+h*c['d_hyper']*self.H
            b=self.V*self.R*(rho+h*(A[0]+corr[0]))+h*bd+h*c['d_hyper']*(self.H@np.full(self.n,c['rho_wall']))
            rhon=spsolve(mat.tocsc(),b)
            if np.any(rhon<=0):raise FloatingPointError('Nonpositive density after IMEX step')
            mat=sp.diags(self.V*self.R*rhon)+h*Kk+h*c['k_hyper']*self.H
            b=self.V*self.R*(p+h*(A[1]+corr[1]))+h*bk+h*c['k_hyper']*(self.H@np.full(self.n,c['T_wall']))
            Tn=spsolve(mat.tocsc(),b)
            Ka,_=self.stiffness(torch.from_numpy((rho*self.R**3).reshape(self.shape)),np.zeros((self.shape[0]+2,self.shape[1]+2)))
            load=sum(terms[key][0].numpy().ravel() for key in ('fluid_momentum','lorentz','conservative'))
            un=spsolve((Ka+h*c['mu']*self.H+h*c['mu_hyper']*self.Hu).tocsc(),Ka@u-h*self.V*load)
            result=[rhon,rhon*Tn,un]
            if len(fields)==4:
                matrix=self.Ws+h*c['eta']*self.Ks+h*c['eta_hyper']*self.K1@self.Wsi@self.Ks
                rhs=self.V/self.R*(fields[3]+h*terms['induction'][0].numpy().ravel())+h*c['eta']*self.bs+h*c['eta_hyper']*(self.K1@(self.R/self.V*self.bs)-self.bj)
                result.append(spsolve(matrix.tocsc(),rhs))
            x=torch.from_numpy(np.stack(result).reshape(1,len(fields),*self.shape))
            if not torch.isfinite(x).all() or torch.any(x[:,1]<=0):raise FloatingPointError('Invalid thermodynamic state after IMEX step')
        return x
