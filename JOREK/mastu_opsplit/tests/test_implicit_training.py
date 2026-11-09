from types import SimpleNamespace
import json

import h5py
import numpy as np
import pytest
import torch

from mastu_opsplit.common import write_json
from mastu_opsplit.ensemble_training import evaluate_trajectory, train_family
from mastu_opsplit.euler import NativeSplineSpace
from mastu_opsplit.implicit_labels import RECORD, label_scale, read_export, remainder_from_export, sha256
from mastu_opsplit.implicit_opsplit import ACTIVE_CHANNELS, ImplicitNativeSystem, ImplicitOpsSplitModel
from mastu_opsplit.implicit_training import LabelStore, pretrain
from mastu_opsplit.physical_guard import PhysicalGuard, QuadratureSampler
from test_ensemble_training import Windows
from test_implicit_opsplit import dirichlet, frozen_fixture, jets
from test_learning import configuration
from test_physical_guard import fixture


def native_export(geometry, frozen, state, physical):
    values, gradients, _ = jets(state, geometry)
    e, q = geometry.basis.shape[:2]
    data = np.zeros((e, q, 364))
    data[..., :2] = geometry.quadrature_rz
    data[..., 3:10] = values
    data[..., 10:17] = gradients[..., 0]
    data[..., 17:24] = gradients[..., 1]
    data[..., 34:49] = physical
    data[..., 49:] = frozen.tangent.reshape(e, q, 315)
    return data


def test_remainder_label_is_exact_complement_of_frozen_linearization(tmp_path):
    geometry, _, _ = fixture(tmp_path)
    frozen = frozen_fixture(geometry)
    rng = np.random.default_rng(4)
    state = rng.normal(size=(len(geometry.dof_ids), 7))
    _, _, jet = jets(state, geometry)
    linear = frozen.forcing.reshape(*frozen.forcing.shape[:2], 15) + np.einsum('eqjo,eqj->eqo', frozen.tangent, jet-frozen.jets)
    nonlinear = rng.normal(size=linear.shape)
    nonlinear[..., [1, 2]] = 0
    label, metrics = remainder_from_export(native_export(geometry, frozen, state, linear+nonlinear), frozen, geometry, state)
    np.testing.assert_allclose(label, nonlinear[..., list(ACTIVE_CHANNELS)].reshape(-1, 13), atol=1e-12)
    assert metrics['value_relative_error'] < 1e-14
    with pytest.raises(ValueError, match='differs from dataset frame'):
        remainder_from_export(native_export(geometry, frozen, state, linear), frozen, geometry, state+1e-3)


def test_export_records_are_reassembled_in_restart_element_order(tmp_path):
    geometry, _, _ = fixture(tmp_path)
    vertices = np.array([[1, 2, 3, 4]], dtype='<i4')
    data = np.random.default_rng(0).normal(size=(1, 16, 364))
    records = np.zeros(16, dtype=RECORD)
    records['vertices'] = vertices[0]
    records['query'] = np.arange(16)[::-1]+1
    records['data'] = data[0, ::-1]
    records[:8].tofile(tmp_path/'opsplit_export_0.bin')
    records[8:].tofile(tmp_path/'opsplit_export_1.bin')
    np.testing.assert_array_equal(read_export(sorted(tmp_path.glob('opsplit_export_*.bin')), vertices), data)
    records[:8].tofile(tmp_path/'opsplit_export_0.bin')
    with pytest.raises(ValueError, match='Missing'):
        read_export([tmp_path/'opsplit_export_0.bin'], vertices)


def label_store(tmp_path, geometry, states):
    root = tmp_path/'labels'
    (root/'states').mkdir(parents=True)
    entries = []
    rng = np.random.default_rng(9)
    with h5py.File(tmp_path/'case.h5', 'w') as stream:
        stream['native_state'] = states
    for frame in range(len(states)):
        np.save(root/'states'/f'case_{frame:03d}.npy', rng.normal(size=(geometry.basis.shape[0]*16, 13)).astype(np.float32))
        entries.append(dict(case_id='case', frame=frame, data=str(tmp_path/'case.h5')))
    write_json(root/'plan.json', dict(tangent_sha256='t', active_channels=list(ACTIVE_CHANNELS), cases=['case'],
                                      entries=entries, frames=list(range(len(states)))))
    scale = label_scale(sorted((root/'states').glob('*.npy')), geometry.volume_weights)
    write_json(root/'summary.json', dict(plan_sha256=sha256(root/'plan.json'), label_scale=scale.tolist()))
    return root


def test_label_store_rejects_foreign_tangent_test_cases_and_edits(tmp_path):
    geometry, _, _ = fixture(tmp_path)
    root = label_store(tmp_path, geometry, np.zeros((3, 16, 7)))
    store = LabelStore(root, tangent_sha256='t', training_case_ids=['case', 'other'])
    assert len(store.entries) == 3 and store.scale.shape == (13,)
    with pytest.raises(ValueError, match='different frozen tangent'):
        LabelStore(root, tangent_sha256='u', training_case_ids=['case'])
    with pytest.raises(ValueError, match='non-training'):
        LabelStore(root, tangent_sha256='t', training_case_ids=['other'])
    plan = json.loads((root/'plan.json').read_text())
    plan['frames'] = [0]
    (root/'plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='changed after finalization'):
        LabelStore(root, tangent_sha256='t', training_case_ids=['case'])


def test_pretraining_then_rollout_resume_exactly_and_evaluate(tmp_path):
    torch.set_num_threads(1)
    geometry, q8, mean = fixture(tmp_path)
    space = NativeSplineSpace(geometry)
    frozen = frozen_fixture(geometry)
    n = len(geometry.dof_ids)
    rng = np.random.default_rng(3)
    states = mean[None] + .01*rng.normal(size=(4, n, 7))
    store = LabelStore(label_store(tmp_path, geometry, states), tangent_sha256='t', training_case_ids=['case'])
    guard = PhysicalGuard(geometry, q8, mean, np.ones(7))
    system = ImplicitNativeSystem(geometry, frozen, mean, np.ones(7), dirichlet(geometry), code_interval=.03125)
    data = SimpleNamespace(mean=mean, scale=np.ones(7), windows=lambda split, horizon, stride=1: Windows(mean, split, horizon))
    config = dict(seed=17, steps=2, pretrain_steps=3, horizon=2, stride=1, learning_rate=.001,
                  pretrain_learning_rate=.001, positivity_penalty=.001)

    def model():
        torch.manual_seed(17)
        return ImplicitOpsSplitModel(space, system, configuration(), store.scale, physical_guard=guard)
    whole, partial = model(), model()
    pretrain(whole, data, store, space, config, tmp_path/'whole', 'p')
    assert pretrain(partial, data, store, space, config, tmp_path/'partial', 'p', stop_after=1) is None
    resumed = model()
    pretrain(resumed, data, store, space, config, tmp_path/'partial', 'p', resume=True)
    for name, value in whole.operator.state_dict().items():
        torch.testing.assert_close(value, resumed.operator.state_dict()[name], rtol=0, atol=0)
    history = json.loads((tmp_path/'whole'/'history.json').read_text())
    assert len(history) == 3 and all(np.isfinite(row['remainder_loss']) for row in history)
    assert whole.operator.backbone.project.weight.abs().sum() > 0
    with pytest.raises(ValueError, match='provenance'):
        pretrain(model(), data, store, space, config, tmp_path/'partial', 'q', resume=True)
    (tmp_path/'rollout').mkdir()
    train_family(whole, data, space, config, tmp_path/'rollout', 'p')
    records = evaluate_trajectory(whole, data.windows('test', 2)[0], space, guard, QuadratureSampler(q8), [1, 2], 2)
    assert all(record['audit_passed'] for record in records)
    assert records[-1]['guard_statistics']['steps'] == 4
