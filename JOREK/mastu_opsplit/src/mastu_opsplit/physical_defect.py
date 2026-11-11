"""Exact native volume physics plus a learned defect anchored at the native q*.

The temporal operator remains the frozen, coupled native M*. Algebraic current,
vorticity and boundary rows are solved with every implicit stage; M* is never
inverted on its own. ``be`` is first-order IMEX Euler (backward Euler for L*),
not a fully nonlinear backward-Euler solve. ``ars222`` is the stiffly accurate
ARS(2,2,2) IMEX method. Its second solve reuses the gamma-stage factorization.
"""
import math

import numpy as np
import torch

from .implicit_opsplit import ImplicitOpsSplitModel, _ImplicitStep


class PhysicalDefectModel(ImplicitOpsSplitModel):
    """Keep F(q) explicitly known and learn only G(q)-G(q*).

    ``reference_state`` is the physical native coefficient array [ndof,7], not
    the training mean. ``physical_rhs`` supplies ``pointwise(state, parameters)``
    in the 13 active weak channels and ``linear_forcing(state)`` for
    F(q*) + L*(q-q*). Both take normalized states. The defect vanishes at q*;
    the physical F(q*) is retained and need not vanish.
    """
    variant = 'implicit_native_physical_defect'
    gamma = 1. - 1./math.sqrt(2.)
    delta = 1. - 1./(2.*gamma)

    def __init__(self, space, system, model_config, remainder_scale, *, physical_guard,
                 physical_rhs, reference_state, scheme='be', physics_only=False,
                 zero_output=True, boundary_tolerance=1e-10, boundary_max_iterations=12,
                 held_dof_tolerance=None, record_raw_prediction=True, internal_substeps=1):
        if scheme not in ('be', 'ars222'):
            raise ValueError("scheme must be 'be' or 'ars222'")
        if system.theta != 1.:
            raise ValueError('Physical defect stages require system.theta=1')
        if type(internal_substeps) is not int or internal_substeps < 1:
            raise ValueError('internal_substeps must be a positive integer')
        reference = np.asarray(reference_state, dtype=np.float64)
        if reference.shape != (system.ndof, 7) or not np.isfinite(reference).all():
            raise ValueError('reference_state must be finite physical [ndof,7] native coefficients')
        if not np.isfinite(boundary_tolerance) or boundary_tolerance <= 0 or \
                type(boundary_max_iterations) is not int or boundary_max_iterations < 1:
            raise ValueError('Require positive boundary tolerance and iteration limit')
        if held_dof_tolerance is not None and (not np.isfinite(held_dof_tolerance) or held_dof_tolerance <= 0):
            raise ValueError('held_dof_tolerance must be positive or None (diagnostic only)')
        super().__init__(space, system, model_config, remainder_scale, physical_guard=physical_guard,
                         physics_only=physics_only, zero_output=zero_output)
        self.physical_rhs, self.scheme = physical_rhs, scheme
        self.internal_substeps = internal_substeps
        self.boundary_tolerance = float(boundary_tolerance)
        self.boundary_max_iterations = boundary_max_iterations
        # Both tolerances use normalized native coefficient units, not SI units.
        self.held_dof_tolerance = held_dof_tolerance
        self.record_raw_prediction = bool(record_raw_prediction)
        self.last_raw_prediction = None
        self.register_buffer('reference_state', torch.as_tensor(reference))
        self.register_buffer('reference_normalized', torch.as_tensor((reference-system.mean)/system.scale))
        self.reset_stage_statistics()

    def reset_stage_statistics(self):
        self.stage_statistics = dict(stages=0, boundary_iterations=0, maximum_boundary_iterations=0,
                                     maximum_boundary_residual=0., guarded_stages=0,
                                     maximum_guard_change=0., maximum_held_dof_change=0.,
                                     physical_rhs_evaluations=0, linear_rhs_evaluations=0,
                                     learned_network_evaluations=0)

    def learned_defect(self, state, parameters):
        """Pointwise learned correction; exactly zero at q* for any network weights."""
        if self.physics_only:
            return state.new_zeros(state.shape[0], len(self.coordinates), len(self.remainder_scale))
        self.stage_statistics['learned_network_evaluations'] += 2
        reference = self.reference_normalized.to(state)[None].expand(state.shape[0], -1, -1)
        return (super().remainder(state, parameters)
                - super().remainder(reference, parameters))

    def remainder(self, state, parameters):
        """Known nonlinear volume remainder plus anchored learned defect."""
        self.stage_statistics['physical_rhs_evaluations'] += 1
        known = self.physical_rhs.pointwise(state, parameters)
        return known if self.physics_only else known + self.learned_defect(state, parameters)

    def forcing(self, state, parameters):
        # Physics-only still includes the exact nonlinear physical remainder.
        return self.assembler(self.remainder(state, parameters))

    def _boundary_residual(self, state):
        r = self.system.relations
        if not len(r.rows):
            return state.new_zeros(())
        physical = state*torch.as_tensor(self.system.scale, device=state.device) + \
            torch.as_tensor(self.system.mean, device=state.device)
        expected = r.evaluate(physical[..., 5])
        rows = torch.as_tensor(r.rows, device=state.device)
        return ((physical[:, rows, 6]-expected)/self.system.scale[6]).abs().amax()

    def _stage(self, state, forcing, fraction):
        """Solve one coupled DAE stage, including the nonlinear Mach wall relation."""
        system, current = self.system, state
        if forcing.shape != (state.shape[0], system.ndof, 5) or not torch.isfinite(forcing).all():
            raise ValueError('Expected finite assembled [batch,ndof,5] forcing')
        for iteration in range(1, self.boundary_max_iterations+1):
            proposal = _ImplicitStep.apply(state, forcing, system.relation_explicit(current), system, fraction)
            with torch.no_grad():
                if not torch.isfinite(proposal).all():
                    raise FloatingPointError('Native stage produced a non-finite state')
                residual = float(self._boundary_residual(proposal))
            if not math.isfinite(residual):
                raise FloatingPointError('Non-finite native stage boundary residual')
            if residual <= self.boundary_tolerance:
                break
            current = proposal
        else:
            raise FloatingPointError(f'Native stage Mach relation did not converge: {residual:.3g}')
        stats = self.stage_statistics
        stats['stages'] += 1
        stats['boundary_iterations'] += iteration
        stats['maximum_boundary_iterations'] = max(stats['maximum_boundary_iterations'], iteration)
        stats['maximum_boundary_residual'] = max(stats['maximum_boundary_residual'], residual)
        # Final-stage proposal from the actual guarded stage history; this is
        # not an independently unguarded trajectory. Detach to avoid retaining
        # training graphs solely for evaluation diagnostics.
        if self.record_raw_prediction:
            self.last_raw_prediction = proposal.detach()
        if self.physical_guard is not None:
            guarded = self.physical_guard(state, proposal)
            # The existing limiter can change T. Its Mach boundary must follow T.
            r = system.relations
            if len(r.rows):
                temperature = guarded[..., 5]*system.scale[5] + torch.as_tensor(
                    system.mean[:, 5], device=state.device, dtype=state.dtype)
                rows = torch.as_tensor(r.rows, device=state.device)
                guarded = guarded.clone()
                guarded[:, rows, 6] = (r.evaluate(temperature) - torch.as_tensor(
                    system.mean[r.rows, 6], device=state.device))/system.scale[6]
            fixed = torch.as_tensor(system.fixed, device=state.device)
            with torch.no_grad():
                change = float((guarded-proposal).abs().amax())
                boundary_change = float((guarded-state)[:, fixed].abs().amax()) if fixed.any() else 0.
            stats['maximum_held_dof_change'] = max(stats['maximum_held_dof_change'], boundary_change)
            if self.held_dof_tolerance is not None and boundary_change > self.held_dof_tolerance:
                raise FloatingPointError(f'Physical guard changed a held boundary DOF: {boundary_change:.3g}')
            stats['guarded_stages'] += 1
            stats['maximum_guard_change'] = max(stats['maximum_guard_change'], change)
            proposal = guarded
        return proposal

    def forward(self, state, parameters, *, substeps=1, interval=1.):
        if type(substeps) is not int or substeps < 1 or not 0 < interval <= 1:
            raise ValueError('Require positive substeps and 0 < frame interval <= 1')
        if state.ndim != 3 or state.shape[1:] != (self.system.ndof, 7):
            raise ValueError('Expected normalized [batch,ndof,7] state')
        self.last_raw_prediction = None
        substeps *= self.internal_substeps
        fraction = interval/substeps
        for _ in range(substeps):
            first = self.forcing(state, parameters)
            if self.scheme == 'be':
                state = self._stage(state, first, fraction)
            else:
                stage = self._stage(state, first, self.gamma*fraction)
                second = self.forcing(stage, parameters)
                self.stage_statistics['linear_rhs_evaluations'] += 1
                implicit = self.physical_rhs.linear_forcing(stage)
                combined = (self.delta*first + (1.-self.delta)*second
                            + (1.-self.gamma)*implicit)/self.gamma
                state = self._stage(state, combined, self.gamma*fraction)
        return state
