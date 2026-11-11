"""Independent affine forcing check for ARS on a coupled singular mass system."""
import numpy as np
import pytest
import torch

from test_physical_defect import LinearDAERemainder, linear_dae_model


@pytest.mark.parametrize('scheme,order_ratio', [('be', 1.8), ('ars222', 3.8)])
def test_nonzero_affine_forcing_retained_once_per_step(tmp_path, scheme, order_ratio):
    model = linear_dae_model(tmp_path, scheme)
    equilibrium = torch.full((1, model.system.ndof, 7), .7, dtype=torch.float64)
    equilibrium[..., 2:4] *= 2

    class Affine(LinearDAERemainder):
        def forward(self, state, parameters=None):
            return self.explicit_rate*self.mass(state-equilibrium)

        def linear_forcing(self, state):
            return self.implicit_rate*self.mass(state-equilibrium)

    rhs = Affine(model.system)
    model.physical_rhs = rhs
    model.forcing = rhs.forward
    model.system.reference_forcing[:] = (-rhs.implicit_rate*rhs.mass(equilibrium)[0]).numpy()
    state = torch.ones_like(equilibrium)
    state[..., 2:4] = 2
    exact = equilibrium+(state-equilibrium)*np.exp(-2.3*.4)
    errors = []
    for count in (8, 16, 32):
        actual = model(state, torch.ones(1, 1), substeps=count)
        errors.append(float((actual-exact).abs().max()))
        torch.testing.assert_close(actual[..., 2:4], 2*actual[..., :2], rtol=0, atol=1e-13)
    assert min(errors[i]/errors[i+1] for i in range(2)) > order_ratio
    # Affine constants cancel their corresponding linear terms without equilibrium drift.
    torch.testing.assert_close(model(equilibrium, torch.ones(1, 1)), equilibrium, rtol=0, atol=1e-13)
