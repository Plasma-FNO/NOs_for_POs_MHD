"""Native-unit, axisymmetric JOREK term diagnostics; NOT a time integrator.

Inputs use native stream function u, not the release's exported potential.
See README's equation audit before applying these kernels to any HDF5 data.
All results are on the two-cell-cropped interior. No wall condition is invented.
These kernels omit momentum inversion, hyperdiffusion, floors and stabilization.
"""
import torch


class JorekTerms:
    """Differentiable reference terms on [..., Z, R] rectilinear samples."""

    def __init__(self, r, z):
        self.r, self.z = r, z
        for x in (r, z):
            if (x.ndim != 1 or len(x) < 5 or not torch.isfinite(x).all()
                    or not torch.all(torch.diff(x) > 0)):
                raise ValueError('At least five finite, increasing coordinates required')
        if not torch.all(r > 0):
            raise ValueError('Cylindrical radius must be positive')

    @staticmethod
    def interior(x):
        return x[..., 2:-2, 2:-2]

    def grad(self, x):
        if x.shape[-2:] != (len(self.z), len(self.r)):
            raise ValueError('Expected trailing dimensions [Z, R]')
        dz, dr = torch.gradient(x, spacing=(self.z, self.r), dim=(-2, -1), edge_order=2)
        return dr, dz

    def bracket(self, a, b):
        """[a,b] = a_R b_Z - a_Z b_R; full-grid intermediate only."""
        ar, az = self.grad(a)
        br, bz = self.grad(b)
        return ar*bz - az*br

    def divergence(self, fr, fz):
        """Axisymmetric divergence, R^-1 d_R(R f_R) + d_Z(f_Z)."""
        return self.grad(self.r*fr)[0]/self.r + self.grad(fz)[1]

    def constraints(self, u, psi):
        ur, uz = self.grad(u)
        pr, pz = self.grad(psi)
        return dict(omega=self.interior(self.divergence(ur, uz)),
                    j=self.interior(self.grad(pr)[0] - pr/self.r + self.grad(pz)[1]))

    def fluid_advection(self, rho, p, u, gamma):
        """Continuity and PRESSURE transport, including toroidal compression."""
        uz = self.grad(u)[1]
        return dict(rho=self.interior(self.r*self.bracket(rho, u) + 2*rho*uz),
                    p=self.interior(self.r*self.bracket(p, u) + 2*gamma*p*uz),
                    # For the projected momentum weak form with measure dR dZ.
                    pressure_force=self.interior(-2*self.r*self.grad(p)[1]))

    def scalar_diffusion(self, rho, temperature, d_perp, k_perp):
        """Coefficient fields allowed; k is JOREK ZK, already gamma-normalized."""
        rr, rz = self.grad(rho)
        tr, tz = self.grad(temperature)
        return dict(rho=self.interior(self.divergence(d_perp*rr, d_perp*rz)),
                    p=self.interior(self.divergence(k_perp*tr, k_perp*tz)))

    def magnetic(self, u, psi, j, eta):
        """Ideal induction, resistivity, and projected Lorentz-force density.

        lorentz_force is a momentum LOAD, not du/dt or domega/dt.
        Current must be compatible with psi. No current source assumed here.
        """
        return dict(induction=self.interior(self.r*self.bracket(psi, u)),
                    resistivity=self.interior(eta*j),
                    lorentz_force=self.interior(self.bracket(psi, j)))

    def anisotropic_correction(self, scalar, psi, f0, parallel, perpendicular):
        """div((k_parallel-k_perp) b b.grad(scalar)), not total diffusion.

        This correction survives k_parallel=0 when poloidal B is nonzero.
        Add to scalar_diffusion; use scalar=T for the pressure equation.
        """
        pr, pz = self.grad(psi)
        sr, sz = self.grad(scalar)
        br, bz, bt = -pz/self.r, pr/self.r, f0/self.r
        b2 = br**2 + bz**2 + bt**2
        if torch.any(b2 <= 0):
            raise ValueError('Magnetic direction requires nonzero field')
        flux = (parallel-perpendicular)*(br*sr + bz*sz)/b2
        return self.interior(self.divergence(flux*br, flux*bz))
