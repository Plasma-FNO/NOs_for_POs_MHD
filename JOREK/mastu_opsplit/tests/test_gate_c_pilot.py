import itertools

import numpy as np
import pytest

from mastu_opsplit.common import load_config
from mastu_opsplit.sampling import training_envelope_pilot, transport_profile_overrides, validate_family_splits
from pathlib import Path


def test_pilot_covers_every_training_corner_and_matching_transport_baselines():
    config = load_config(Path(__file__).resolve().parents[1] / 'configs/sampling.yaml')
    rows = training_envelope_pilot(config)
    assert len(rows) == 10
    corners = [r for r in rows if r['A_rho']]
    assert {(r['A_rho'], r['A_T'], r['s_perp']) for r in corners} == set(itertools.product(config['train_amplitude'], config['train_amplitude'], config['train_transport']))
    baselines = {r['case_id']: r for r in rows if not r['A_rho']}
    assert all(baselines[r['baseline_id']]['s_perp'] == r['s_perp'] for r in corners)
    assert all(r['split'] == 'development' for r in rows)
    refinement = {**corners[0], 'case_id': corners[0]['case_id']+'-refined'}
    validate_family_splits(rows+[refinement])
    with pytest.raises(ValueError, match='leaks'):
        validate_family_splits(rows+[{**refinement, 'split': 'id_test'}])


def test_native_profile_scaling_preserves_tanh_shape_and_second_amplitude():
    # Pinned diffusivities.get_dperp1/get_zkperp1: entries 2..5 are shape controls.
    profile = np.array([1e-5, .3, .04, .2, .8, 2e-5, 0., 0., 0., 0.])
    scale = 1.4
    overrides = transport_profile_overrides({'D_perp': profile, 'ZK_perp': profile}, scale)
    assert set(overrides) == {'D_perp(1)', 'D_perp(6)', 'ZK_perp(1)', 'ZK_perp(6)'}
    updated = profile.copy()
    updated[[0, 5]] = [overrides['D_perp(1)'], overrides['D_perp(6)']]
    def native(p, psi):
        return p[0]*((1-p[1])+p[1]*(.5-.5*np.tanh((psi-p[4])/p[3]))) + p[5]*p[1]*(.5-.5*np.tanh((-psi+p[4]+p[2])/p[3]))
    psi = np.linspace(-.2, 1.5, 50)
    np.testing.assert_allclose(native(updated, psi), scale*native(profile, psi), rtol=2e-15)
    with pytest.raises(ValueError, match='positive'):
        transport_profile_overrides({'D_perp': profile, 'ZK_perp': profile}, -1)
