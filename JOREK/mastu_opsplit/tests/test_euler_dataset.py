from dataclasses import fields
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from mastu_opsplit.common import sha256_file as digest, write_json as write
from mastu_opsplit.euler_dataset import EulerDataset
from mastu_opsplit.numerical import NativeGeometry
from mastu_opsplit.restart import read_restart
from test_restart import rectangle


def builder():
    path = Path(__file__).resolve().parents[1] / 'scripts/build_euler_dataset.py'
    spec = importlib.util.spec_from_file_location('ensemble_builder_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fixture_release(root):
    rectangle(root / 'mesh.h5')
    geometry, _, _ = read_restart(root / 'mesh.h5')
    with h5py.File(root / 'geometry.h5', 'x') as f:
        for v in fields(NativeGeometry):
            f[v.name] = getattr(geometry, v.name)
    (root / 'cases').mkdir()
    cases = []; train = []; rng = np.random.default_rng(7)
    for i, split in enumerate(['train', 'train', 'validation', 'test']):
        case = dict(case_id=f'case{i}', parent_family_id=f'parent{i}', split=split,
                    initial_check={'sha256': f'unique-initial-{i}'})
        states = rng.normal(size=(3, 16, 7)) + (1e9 if split != 'train' else 0)
        if split == 'train':
            train.append(states)
        path = root / 'cases' / f'case{i}.h5'
        with h5py.File(path, 'x') as f:
            f['native_state'] = states; f['time_seconds'] = np.arange(3)*1e-8
            f.create_dataset('metadata_json', data=json.dumps(case), dtype=h5py.string_dtype('utf-8'))
        write(root / 'cases' / f'case{i}.json', dict(status='accepted',
              data=dict(path=str(path.relative_to(root)), sha256=digest(path), frames=3,
                        time_seconds=(np.arange(3)*1e-8).tolist())))
        cases.append(case)
    write(root / 'plan.json', dict(cases=cases, split_counts={'train': 2, 'validation': 1, 'test': 1},
                                  geometry_sha256=digest(root / 'geometry.h5'), limits=['Synthetic unit fixture only']))
    builder().finalize(root)
    return np.concatenate(train), geometry


def test_sampling_is_reproducible_and_split_before_simulation():
    sample = builder().sample_cases
    cases = sample()
    assert cases == sample() and cases != sample(91)
    assert {s: sum(c['split'] == s for c in cases) for s in ('train', 'validation', 'test')} == {
        'train': 60, 'validation': 20, 'test': 20}
    assert len({c['parent_family_id'] for c in cases}) == 100
    assert len({tuple(c['center'] + c['width'] + c['amplitude']) for c in cases}) == 100
    assert all(.01 <= abs(a) <= .05 for c in cases for a in c['amplitude'])


def test_release_normalization_uses_only_independent_training_cases(tmp_path):
    train, geometry = fixture_release(tmp_path)
    data = EulerDataset(tmp_path / 'manifest.json')
    np.testing.assert_allclose(data.mean, train.mean(0), atol=1e-14)
    values = np.einsum('eql,telc->teqc', geometry.basis, (train-train.mean(0))[:, geometry.dof_map])
    expected = np.sqrt(np.einsum('teqc,eq->c', values**2, geometry.volume_weights) /
                       (len(train)*geometry.volume_weights.sum()))
    np.testing.assert_allclose(data.scale, expected, atol=1e-14)
    windows = data.windows('train', horizon=2)
    assert len(windows) == 2 and {windows[i]['case_id'] for i in range(2)} == {'case0', 'case1'}
    assert windows[0]['state'].shape == (3, 16, 7)
    np.testing.assert_allclose(windows[0]['state'], (train[:3]-data.mean)/data.scale)
    assert len(data.windows('test', horizon=1, stride=2)) == 1
    with pytest.raises(ValueError, match='horizon'):
        data.windows('test', horizon=3)


def test_duplicate_parent_and_tampered_data_are_rejected(tmp_path):
    fixture_release(tmp_path)
    path = tmp_path / 'manifest.json'; original = path.read_text(); m = json.loads(original)
    m['cases'][-1]['parent_family_id'] = m['cases'][0]['parent_family_id']
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='parent'):
        EulerDataset(path)
    path.write_text(original)
    with h5py.File(tmp_path / 'cases/case3.h5', 'r+') as f:
        f['native_state'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='hash'):
        EulerDataset(path)


def test_incomplete_ensemble_cannot_release(tmp_path):
    write(tmp_path / 'plan.json', {'cases': [{'case_id': 'missing', 'split': 'test'}]})
    with pytest.raises(ValueError, match='incomplete'):
        builder().finalize(tmp_path)
    assert not (tmp_path / 'manifest.json').exists()
    assert json.loads((tmp_path / 'incomplete.json').read_text())['ready'] is False


@pytest.mark.parametrize('needs_refinement', [False, True])
def test_refinement_preserves_case_and_split(tmp_path, monkeypatch, needs_refinement):
    from types import SimpleNamespace
    module = builder(); calls = []; comparisons = []
    (tmp_path / 'cases').mkdir(); initial = tmp_path / 'initial.h5'; initial.write_bytes(b'fixed state')
    case = dict(case_id='case', split='test', initial='initial.h5', initial_check={'sha256': digest(initial)})
    plan = dict(cases=[case], coarse=dict(dt=1/64), fine=dict(dt=1/128), finer=dict(dt=1/256),
                controls=dict(coarse='control64', fine='control128'), dynamic_error_limit=.01, energy_error_limit=.01)
    write(tmp_path / 'plan.json', plan)
    monkeypatch.setattr(module, 'environment', lambda *args: SimpleNamespace(run=lambda spec: calls.append(spec)))
    def compare(*args, **kwargs):
        comparisons.append(args)
        error = .02 if needs_refinement and len(comparisons) == 1 else .001
        return dict(groups={'state': {'fields': {'rho': {'error_over_temporal_std': error}}}},
                    matched_frames=97, positive_sampled_rho_T=True, energy_relative_errors={'magnetic': 0., 'kinetic': 0.})
    monkeypatch.setattr(module, 'load_module', lambda *args: SimpleNamespace(compare=compare))
    monkeypatch.setattr(module, 'export_case', lambda *args: {'fixture': True})
    report = module.run_case(tmp_path, 0)
    assert len(calls) == (3 if needs_refinement else 2)
    assert all(c['restart'] == str(initial) for c in calls)
    assert report['status'] == 'accepted' and report['split'] == 'test'
    if needs_refinement:
        assert comparisons[-1][2:] == ('control128', 'control128')
        assert report['selected_reference_rate'] == 'finer'
