"""Anchoring, DAE-stage and time-order checks; no training runs."""
from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch import nn

from mastu_opsplit.euler import NativeSplineSpace
from mastu_opsplit.implicit_opsplit import ACTIVE_CHANNELS, BLOCK_ROWS, ImplicitNativeSystem, WeakAssembler
from mastu_opsplit.physical_defect import PhysicalDefectModel
from mastu_opsplit.physical_guard import PhysicalGuard, QuadratureSampler
from test_implicit_opsplit import dirichlet, frozen_fixture, jets, relation_fixture
from test_learning import configuration
from test_physical_guard import fixture


class FixtureRemainder(nn.Module):
    """Nonlinear pointwise fixture, with the exact frozen affine part separately."""
    def __init__(self, geometry, frozen, mean, scale):
        super().__init__()
        self.samplers = [QuadratureSampler(geometry, basis) for basis in
                         (geometry.basis, geometry.basis_grad[..., 0], geometry.basis_grad[..., 1])]
        for name, value in dict(mean=mean, scale=scale, reference=frozen.jets.reshape(-1, 21),
                                tangent=frozen.tangent.reshape(-1, 21, 15)[..., ACTIVE_CHANNELS],
                                constant=frozen.forcing.reshape(-1, 15)[..., ACTIVE_CHANNELS]).items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64))
        self.assembler = WeakAssembler(geometry)

    def jet(self, state):
        physical = state*self.scale + self.mean
        return torch.cat([sample(physical) for sample in self.samplers], -1)

    def pointwise(self, state, parameters=None):
        delta = self.jet(state)-self.reference
        return .01*delta[..., :13].square()

    def linear_forcing(self, state):
        return self.assembler(self.constant + torch.einsum('pjo,bpj->bpo', self.tangent,
                                                          self.jet(state)-self.reference))


def make_model(tmp_path, *, scheme='be', physics_only=False, zero_output=True,
               guard=False, relations=False, device='cpu'):
    geometry, q8, reference = fixture(tmp_path)
    # Deliberately distinct from native q*: anchoring at the training mean is wrong.
    mean = reference.copy()
    mean[:, 0] += .02
    scale = np.linspace(.8, 1.2, 7)
    frozen = frozen_fixture(geometry)
    values, gradients, _ = jets(reference, geometry)
    frozen = replace(frozen, values=values, gradients=gradients,
                     tangent=frozen.tangent*.01, forcing=frozen.forcing*.001)
    relation, fixed = relation_fixture(geometry) if relations else (None, dirichlet(geometry))
    system = ImplicitNativeSystem(geometry, frozen, mean, scale, fixed,
                                  code_interval=1e-5, relations=relation,
                                  backend='torch' if device != 'cpu' else 'scipy', device=device)
    physical = FixtureRemainder(geometry, frozen, mean, scale)
    limiter = PhysicalGuard(geometry, q8, mean, scale) if guard else None
    model = PhysicalDefectModel(NativeSplineSpace(geometry), system, configuration(), np.ones(13),
                                physical_guard=limiter, physical_rhs=physical, reference_state=reference,
                                scheme=scheme, physics_only=physics_only, zero_output=zero_output).to(device)
    return model


def test_defect_anchors_at_native_reference_not_training_mean_and_retains_forcing(tmp_path):
    model = make_model(tmp_path, zero_output=False)
    reference = model.reference_normalized[None]
    parameters = torch.ones(1, 1)
    torch.testing.assert_close(model.learned_defect(reference, parameters),
                               torch.zeros(1, len(model.coordinates), 13, dtype=torch.float64), rtol=0, atol=0)
    assert model.learned_defect(torch.zeros_like(reference), parameters).abs().max() > 0
    physical = model.physical_rhs.linear_forcing(reference)
    assert physical.abs().max() > 0
    torch.testing.assert_close(model.forcing(reference, parameters), torch.zeros_like(physical), atol=1e-28, rtol=0)
    # An anchored defect must not artificially make an evolving native baseline stationary.
    assert (model(reference, parameters)-reference).abs().max() > 1e-8


@pytest.mark.parametrize('scheme', ['be', 'ars222'])
def test_zero_initialized_defect_matches_full_physics_and_has_finite_gradients(tmp_path, scheme):
    model = make_model(tmp_path, scheme=scheme)
    state = model.reference_normalized[None] + torch.randn(1, model.system.ndof, 7, dtype=torch.float64)*1e-4
    state.requires_grad_()
    parameters = torch.ones(1, 1)
    learned = model(state, parameters, substeps=2)
    assert model.last_raw_prediction is not None and not model.last_raw_prediction.requires_grad
    model.physics_only = True
    physics = model(state, parameters, substeps=2)
    torch.testing.assert_close(learned, physics, atol=0, rtol=0)
    model.physics_only = False
    learned.square().mean().backward()
    gradient = model.operator.backbone.project.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert torch.isfinite(state.grad).all()
    assert model.stage_statistics['stages'] == (4 if scheme == 'be' else 8)
    assert len(model.system._factors) == 1


@pytest.mark.parametrize('scheme', ['be', 'ars222'])
def test_each_stage_preserves_native_algebraic_and_held_rows(tmp_path, scheme):
    model = make_model(tmp_path, scheme=scheme, physics_only=True)
    state = model.reference_normalized[None]
    stages = []
    original = model._stage
    def record(*args):
        out = original(*args)
        stages.append(out.detach().numpy()[0])
        return out
    model._stage = record
    model(state, torch.ones(1, 1))
    system = model.system
    assert len(stages) == (1 if scheme == 'be' else 2)
    for normalized in stages:
        physical = normalized*system.scale + system.mean
        for name, potential, auxiliary in [('current', 0, 2), ('vorticity', 1, 3)]:
            stiff, mass = system.constraints[name]
            free = ~system.fixed[:, auxiliary]
            residual = stiff@physical[:, potential] + mass@physical[:, auxiliary]
            np.testing.assert_allclose(residual[free], 0, atol=1e-10)
        np.testing.assert_array_equal(normalized[system.fixed], state.numpy()[0][system.fixed])


def test_stage_converges_actual_mach_relation_and_differentiates_it(tmp_path):
    model = make_model(tmp_path, relations=True, physics_only=True)
    state = model.reference_normalized[None].clone().requires_grad_()
    forcing = model.forcing(state, torch.ones(1, 1))
    out = model._stage(state, forcing, 1.)
    assert float(model._boundary_residual(out)) <= model.boundary_tolerance
    weights = torch.randn_like(out)
    direction = torch.randn_like(state)*.01
    gradient, = torch.autograd.grad((out*weights).sum(), state)
    epsilon = 1e-5
    def evaluate(x):
        return (model._stage(x, model.forcing(x, torch.ones(1, 1)), 1.)*weights).sum()
    with torch.no_grad():
        difference = (evaluate(state+epsilon*direction)-evaluate(state-epsilon*direction))/(2*epsilon)
    assert float((gradient*direction).sum()) == pytest.approx(float(difference), rel=1e-5, abs=1e-7)
    model.boundary_max_iterations = 1
    model.boundary_tolerance = 1e-30
    with pytest.raises(FloatingPointError, match='did not converge'):
        model._stage(state, forcing, 1.)


class LinearDAERemainder(nn.Module):
    """Synthetic coupled-mass DAE: M qdot=(lambda+mu) M q, a=2p."""
    def __init__(self, system, explicit_rate=-.3, implicit_rate=-2.):
        super().__init__()
        self.system = system
        self.explicit_rate, self.implicit_rate = explicit_rate, implicit_rate

    def mass(self, state):
        return torch.stack((state[..., 0], state[..., 1], state[..., 4],
                            .3*state[..., 4]+state[..., 5], .2*state[..., 0]+state[..., 6]), -1)

    def forward(self, state, parameters=None):
        return self.explicit_rate*self.mass(state)

    def linear_forcing(self, state):
        return self.implicit_rate*self.mass(state)


def linear_dae_model(tmp_path, scheme):
    model = make_model(tmp_path, scheme=scheme, physics_only=True)
    system, n = model.system, model.system.ndof
    identity = sp.identity(n, format='csr')
    system.mean[:] = 0.
    system.scale[:] = 1.
    system.fixed[:] = False
    system.forcing_index = np.concatenate([b*n+np.arange(n) for b in range(5)])
    system.forcing_global_rows = np.concatenate([row*n+np.arange(n) for row in BLOCK_ROWS])
    system.M = {(0, 0): identity, (1, 1): identity, (2, 4): identity,
                (3, 4): .3*identity, (3, 5): identity, (4, 0): .2*identity, (4, 6): identity}
    system.L = {k: -2.*matrix for k, matrix in system.M.items()}
    system.reference_forcing[:] = 0.
    system.constraints = {name: (-2.*identity, identity) for name in ('current', 'vorticity')}
    system.code_interval = .4
    model.physical_rhs = LinearDAERemainder(system)
    model.forcing = model.physical_rhs.forward
    return model


@pytest.mark.parametrize('scheme,minimum_ratio', [('be', 1.8), ('ars222', 3.8)])
def test_time_order_on_coupled_mass_singular_dae(tmp_path, scheme, minimum_ratio):
    model = linear_dae_model(tmp_path, scheme)
    state = torch.ones(1, model.system.ndof, 7, dtype=torch.float64)
    state[..., 2:4] = 2.
    exact = state*np.exp(-2.3*.4)
    errors = []
    for count in (8, 16, 32):
        result = model(state, torch.ones(1, 1), substeps=count)
        errors.append(float((result-exact).abs().amax()))
        torch.testing.assert_close(result[..., 2:4], 2*result[..., :2], atol=1e-13, rtol=0)
    assert min(errors[i]/errors[i+1] for i in range(2)) > minimum_ratio


def test_guard_runs_at_each_ars_stage_and_reports_normalized_changes(tmp_path):
    model = make_model(tmp_path, scheme='ars222', guard=True, physics_only=True)
    model(model.reference_normalized[None], torch.ones(1, 1), substeps=2)
    assert model.physical_guard.statistics['steps'] == 4
    assert model.stage_statistics['guarded_stages'] == 4
    assert np.isfinite(model.stage_statistics['maximum_held_dof_change'])
    assert model.last_raw_prediction is not None
    model.record_raw_prediction = False
    model(model.reference_normalized[None], torch.ones(1, 1))
    assert model.last_raw_prediction is None


def test_held_dof_failure_threshold_is_configurable_in_normalized_units(tmp_path):
    model = make_model(tmp_path, physics_only=True)
    class ChangedHeldDof(nn.Module):
        def forward(self, previous, proposal):
            return proposal + torch.as_tensor(model.system.fixed)*1e-4
    model.physical_guard = ChangedHeldDof()
    model(model.reference_normalized[None], torch.ones(1, 1))
    assert model.stage_statistics['maximum_held_dof_change'] == pytest.approx(1e-4)
    model.held_dof_tolerance = 1e-5
    with pytest.raises(FloatingPointError, match='held boundary DOF'):
        model(model.reference_normalized[None], torch.ones(1, 1))


def test_internal_substeps_match_explicit_requested_substeps_and_count_actual_work(tmp_path):
    model = make_model(tmp_path, scheme='ars222', physics_only=True)
    state, parameters = model.reference_normalized[None], torch.ones(1, 1)
    requested = model(state, parameters, substeps=4)
    model.internal_substeps = 2
    model.reset_stage_statistics()
    internal = model(state, parameters, substeps=2)
    torch.testing.assert_close(internal, requested, rtol=0, atol=0)
    assert model.stage_statistics['stages'] == 8
    assert model.stage_statistics['physical_rhs_evaluations'] == 8
    assert model.stage_statistics['linear_rhs_evaluations'] == 4
    assert model.stage_statistics['learned_network_evaluations'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_native_stage_forward_and_adjoint(tmp_path):
    model = make_model(tmp_path, scheme='ars222', device='cuda')
    state = (model.reference_normalized[None] + 1e-4*torch.randn_like(model.reference_normalized[None])).requires_grad_()
    result = model(state, torch.ones(1, 1, device='cuda'))
    result.square().mean().backward()
    assert result.device.type == 'cuda' and state.grad.device.type == 'cuda'
    assert torch.isfinite(state.grad).all()
    assert model.operator.backbone.project.weight.grad.abs().sum() > 0
    assert model.system.report()['backend'] == 'torch'
