import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pytest
import torch

from mastu_opsplit.common import sha256_file
from mastu_opsplit.euler import NativeSplineSpace
from mastu_opsplit.euler_dataset import EulerDataset
from mastu_opsplit import ensemble_training, physical_data_cache as cache
from test_euler_dataset import fixture_release
from test_restart import rectangle
from mastu_opsplit.restart import read_restart


@pytest.fixture
def release():
    # No preparation/training artifacts in the repository or home-directory /tmp.
    with tempfile.TemporaryDirectory(prefix='mastu-prepared-test-', dir='/scratch_local') as directory:
        root = Path(directory)
        fixture_release(root)
        with h5py.File(root/'mesh.h5', 'r+') as stream:
            stream['boundary'][...] = 1
            stream['boundary'][0] = 0
        geometry, reference, _ = read_restart(root/'mesh.h5')
        manifest = json.loads((root/'manifest.json').read_text())
        with h5py.File(root/manifest['geometry']['path'], 'r+') as stream:
            stream['boundary_labels'][...] = geometry.boundary_labels
        manifest['geometry']['sha256'] = sha256_file(root/manifest['geometry']['path'])
        train = []
        for index, case in enumerate(manifest['cases']):
            initial = root/f'initial{index}.h5'
            rectangle(initial)
            with h5py.File(initial, 'r+') as stream:
                stream['boundary'][...] = 1
                stream['boundary'][0] = 0
                stream['values'][0, 0, 0, 0] += (index+1)*1e-5
            case['initial'] = initial.name
            case['initial_check']['sha256'] = sha256_file(initial)
            states = np.stack([reference + (index+frame+1)*1e-5*np.ones_like(reference) for frame in range(3)])
            states[:, ::4, :2] = reference[::4, :2]
            if case['split'] == 'test':
                states[..., 2:4] += 1e5  # Must never enter pooled training normalization.
            if case['split'] == 'train':
                train.extend(states)
            path = root/case['result']['data']['path']
            with h5py.File(path, 'r+') as stream:
                stream['native_state'][...] = states
            case['result']['data']['sha256'] = sha256_file(path)
        train = np.stack(train)
        mean = train.mean(0)
        space = NativeSplineSpace(geometry)
        scale = np.sqrt(sum(np.einsum('pc,p->c', (space.basis@(state-mean))**2,
                                      space.weights/space.weights.sum()) for state in train)/len(train))
        normalization = root/manifest['normalization']['path']
        np.savez(normalization, mean=mean, scale=scale, time_seconds=np.arange(3)*1e-8,
                 training_case_ids=[c['case_id'] for c in manifest['cases'] if c['split'] == 'train'])
        manifest['normalization']['sha256'] = sha256_file(normalization)
        path = root/'manifest.json'
        path.write_text(json.dumps(manifest))
        (root/'completion.json').write_text(json.dumps(dict(manifest_sha256=sha256_file(path), status='dataset_ready')))
        config = dict(manifest=str(path), manifest_sha256=sha256_file(path),
                      source_split_counts=dict(train=2, validation=1, test=1),
                      split_counts=dict(train=3, validation=0, test=1), horizon=1, stride=1,
                      evaluation_horizons=[1, 2], gamma=5/3, prepared_data_cache=str(root/'cache'))
        yield root, config


def uncached(config):
    assert 'prepared_data_cache' not in config
    return ensemble_training.prepare_data(config)


def test_cache_roundtrip_rebuilds_factors_and_preserves_windows_training_normalization_and_guards(release, monkeypatch):
    root, config = release
    first = cache.prepare_cached_data(config, uncached)
    assert first[-1]['verification_reused'] is False
    real_hash = cache.sha256_file
    forbidden = {str(root/c['result']['data']['path']) for c in first[0].cases}
    forbidden |= {str(root/c['initial']) for c in first[0].cases}
    def no_dataset_rehash(path):
        assert str(path) not in forbidden
        return real_hash(path)
    monkeypatch.setattr(cache, 'sha256_file', no_dataset_rehash)
    monkeypatch.setattr(EulerDataset, '__init__', lambda *args: pytest.fail('Cache reran dataset initialization'))
    second = cache.prepare_cached_data({**config, 'learning_rate': .002, 'seed': 92}, uncached)
    data, space, guard, physical, dense, report = second
    assert report['verification_reused'] is True
    assert report['cached_setup']['artifact_fingerprints_checked'] == 12  # 2 release + 2 shared + 4 case + 4 initial.
    assert report['cached_setup']['solver_factors_serialized'] is False
    assert first[1].factor is not space.factor
    assert set(report['normalization']['training_case_ids']) == {'case0', 'case1', 'case2'}
    assert np.abs(data.mean[:, 2:4]).max() < 100
    np.testing.assert_array_equal(data.mean, first[0].mean)
    np.testing.assert_array_equal(data.scale, first[0].scale)
    sample = data.windows('train', 1)[0]['state']
    torch.testing.assert_close(sample, first[0].windows('train', 1)[0]['state'], atol=0, rtol=0)
    torch.testing.assert_close(space.sample(sample), first[1].sample(sample), atol=0, rtol=0)
    np.testing.assert_array_equal(dense.basis.toarray(), first[4].basis.toarray())
    for key, value in physical.state_dict().items():
        torch.testing.assert_close(value, first[3].state_dict()[key], atol=0, rtol=0)
    torch.testing.assert_close(guard.mean, first[2].mean, atol=0, rtol=0)
    # Successful reuse leaves no partial temporary payloads.
    assert not list((root/'cache').glob('*.tmp'))


@pytest.mark.parametrize('kind', ['geometry', 'normalization', 'case', 'initial', 'completion'])
def test_every_verified_artifact_kind_invalidates_cache_even_when_content_is_unchanged(release, kind):
    root, config = release
    result = cache.prepare_cached_data(config, uncached)
    manifest = result[0].manifest
    paths = dict(geometry=root/manifest['geometry']['path'], normalization=root/manifest['normalization']['path'],
                 case=root/manifest['cases'][-1]['result']['data']['path'],
                 initial=root/manifest['cases'][-1]['initial'], completion=root/'completion.json')
    path = paths[kind]
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000))
    with pytest.raises(cache.PreparedDataCacheError, match='fingerprint changed'):
        cache.prepare_cached_data(config, uncached)


def test_preparation_config_and_source_identity_changes_fail_explicitly(release, monkeypatch):
    root, config = release
    cache.prepare_cached_data(config, uncached)
    with pytest.raises(cache.PreparedDataCacheError, match='configuration or preparation source changed'):
        cache.prepare_cached_data({**config, 'gamma': 1.4}, uncached)
    identity = cache._identity
    def changed_source(*args):
        value = identity(*args)
        value['source_sha256']['changed.py'] = '0'*64
        return value
    monkeypatch.setattr(cache, '_identity', changed_source)
    with pytest.raises(cache.PreparedDataCacheError, match='preparation source changed'):
        cache.prepare_cached_data(config, uncached)


def test_payload_corruption_is_detected_before_deserialization(release):
    root, config = release
    cache.prepare_cached_data(config, uncached)
    path = root/'cache/prepared_data.pkl'
    with path.open('r+b') as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(cache.PreparedDataCacheError, match='payload hash mismatch'):
        cache.prepare_cached_data(config, uncached)


def test_artifact_change_during_original_verification_prevents_cache_publication(release):
    root, config = release
    def changed(config):
        result = uncached(config)
        path = root/result[0].manifest['cases'][0]['initial']
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000))
        return result
    with pytest.raises(cache.PreparedDataCacheError, match='artifact changed'):
        cache.prepare_cached_data(config, changed)
    assert not (root/'cache/prepared_data.pkl').exists()


def test_disabled_cache_preserves_original_preparation_api():
    config = {'arbitrary_original_setting': 3}
    calls = []
    assert cache.prepare_cached_data(config, lambda value: calls.append(value) or ('original',)) == ('original',)
    assert calls == [config]
