"""Opt-in, trusted-local reuse of a fully verified immutable native dataset.

The first call runs the original preparation and records file stat fingerprints
around that complete verification. Later calls recheck preparation identity,
source hashes and every fingerprint; they do not independently re-hash the
trajectory dataset. Solver factors are rebuilt, never serialized. Pickle is
accepted only as a trusted local cache produced within the same research session.
A stale cache fails explicitly: choose a fresh directory to run verification anew.
"""
import copy
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import platform
import tempfile
import time

import numpy as np
import scipy
import torch

from .common import config_hash, sha256_file
from .continuity import FrozenContinuity
from .euler import NativeSplineSpace
from .euler_dataset import EulerDataset
from .physical_guard import PhysicalGuard, QuadratureSampler

SCHEMA = 'mastu-prepared-native-data-v1'
MAGIC = b'MOPSC1\n'
PREPARATION_KEYS = ('manifest', 'manifest_sha256', 'source_split_counts', 'split_counts',
                    'horizon', 'stride', 'evaluation_horizons', 'gamma')
PREPARATION_SOURCES = ('common.py', 'numerical.py', 'restart.py', 'euler_dataset.py', 'euler.py',
                       'physical_guard.py', 'continuity.py', 'ensemble_training.py', 'physical_data_cache.py')
STAT_FIELDS = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')


class PreparedDataCacheError(ValueError):
    """An existing cache cannot be reused; it is never silently refreshed."""


def _identity(config, uncached_prepare):
    selected = {key: config[key] for key in PREPARATION_KEYS}
    selected['manifest'] = str(Path(selected['manifest']).resolve())
    files = {Path(__file__).with_name(name) for name in PREPARATION_SOURCES}
    implementation = inspect.getsourcefile(uncached_prepare)
    if implementation is None:
        raise ValueError('Cached preparation requires an inspectable local implementation')
    files.add(Path(implementation).resolve())
    return dict(preparation=selected,
                source_sha256={str(path.resolve()): sha256_file(path) for path in sorted(files)},
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              scipy=scipy.__version__, torch=torch.__version__))


def _fingerprint(path):
    path = Path(path)
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f'Expected native artifact file: {path}')
    return dict(resolved_path=str(path.resolve()), **{key: getattr(stat, key) for key in STAT_FIELDS})


def _artifacts(config):
    manifest = Path(config['manifest']).resolve()
    if sha256_file(manifest) != config['manifest_sha256']:
        raise ValueError('Manifest hash differs from the pinned release')
    record = json.loads(manifest.read_text())
    paths = {str(manifest): config['manifest_sha256'], str(manifest.parent/'completion.json'): None}
    artifacts = [record['geometry'], record['normalization']]
    for case in record['cases']:
        artifacts += [case['result']['data'], dict(path=case['initial'], sha256=case['initial_check']['sha256'])]
    for artifact in artifacts:
        path = manifest.parent / artifact['path']
        if not path.resolve().is_relative_to(manifest.parent):
            raise ValueError(f'Dataset artifact escapes its release directory: {path}')
        path = str(path.absolute())
        if path in paths and paths[path] != artifact['sha256']:
            raise ValueError(f'Conflicting native artifact hashes: {path}')
        paths[path] = artifact['sha256']
    return {path: dict(verified_sha256=digest, stat=_fingerprint(path)) for path, digest in sorted(paths.items())}


def _check_artifacts(expected):
    for path, record in expected.items():
        try:
            current = _fingerprint(path)
        except OSError as error:
            raise PreparedDataCacheError(f'Prepared-data cache artifact unavailable: {path}') from error
        if current != record['stat']:
            raise PreparedDataCacheError(f'Prepared-data cache artifact changed: {path}; '
                                         'use a fresh cache directory for complete verification')


def _check_training_normalization(data, report):
    expected = {case['case_id'] for case in data.cases if case['split'] == 'train'}
    test = {case['case_id'] for case in data.manifest['cases'] if case['split'] == 'test'}
    normalization = report['normalization']
    if expected & test or set(normalization['training_case_ids']) != expected:
        raise PreparedDataCacheError('Cached normalization membership includes test data or differs from training')
    for name in ('mean', 'scale'):
        if hashlib.sha256(getattr(data, name).tobytes()).hexdigest() != normalization[f'{name}_sha256']:
            raise PreparedDataCacheError(f'Cached training normalization {name} hash differs')


class _HashWriter:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def write(self, block):
        self.digest.update(block)
        return self.stream.write(block)


def _write(path, header, payload):
    # Atomic replacement requires a temporary file on the cache filesystem.
    # Unit fixtures and independent scratch work use /scratch_local.
    descriptor, temporary = tempfile.mkstemp(prefix='.prepared-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w+b') as stream:
            header = {**header, 'payload_sha256': '0'*64}
            encoded = json.dumps(header, sort_keys=True, allow_nan=False).encode()
            stream.write(MAGIC + len(encoded).to_bytes(8, 'big') + encoded)
            writer = _HashWriter(stream)
            pickle.dump(payload, writer, protocol=pickle.HIGHEST_PROTOCOL)
            header['payload_sha256'] = writer.digest.hexdigest()
            finalized = json.dumps(header, sort_keys=True, allow_nan=False).encode()
            if len(finalized) != len(encoded):
                raise RuntimeError('Cache header size changed')
            stream.seek(len(MAGIC)+8)
            stream.write(finalized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path, identity, artifacts):
    with path.open('rb') as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise PreparedDataCacheError('Unknown prepared-data cache format')
        size = int.from_bytes(stream.read(8), 'big')
        if not 0 < size <= 16*1024*1024:
            raise PreparedDataCacheError('Invalid prepared-data cache header length')
        header = json.loads(stream.read(size))
        if header.get('schema') != SCHEMA or header.get('identity') != identity:
            raise PreparedDataCacheError('Prepared-data cache configuration or preparation source changed; '
                                         'use a fresh cache directory for complete verification')
        if header.get('artifacts') != artifacts:
            raise PreparedDataCacheError('Prepared-data cache artifact fingerprint changed; '
                                         'use a fresh cache directory for complete verification')
        start = stream.tell()
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
        if digest.hexdigest() != header.get('payload_sha256'):
            raise PreparedDataCacheError('Prepared-data cache payload hash mismatch')
        stream.seek(start)
        payload = pickle.load(stream)  # Trusted local artifact only; never load a downloaded cache.
    return header, payload


def prepare_cached_data(config, uncached_prepare):
    """Same six-item return as prepare_data, with optional local verification reuse."""
    directory = config.get('prepared_data_cache')
    clean = {key: value for key, value in config.items() if key != 'prepared_data_cache'}
    if directory is None:
        return uncached_prepare(clean)
    if not isinstance(directory, (str, Path)) or not str(directory).strip():
        raise ValueError('prepared_data_cache must be a nonempty trusted local directory path')
    started, cpu_started = time.monotonic(), time.process_time()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'prepared_data.pkl'
    # Serialize first builds and reads so parallel workers cannot observe an incomplete cache.
    with (directory/'prepare.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identity = _identity(clean, uncached_prepare)
        artifacts = _artifacts(clean)
        reused = path.exists()
        if reused:
            header, payload = _read(path, identity, artifacts)
            data, verification = payload['data'], payload['verification']
            if type(data) is not EulerDataset:
                raise PreparedDataCacheError('Cached payload is not a native EulerDataset')
            _check_training_normalization(data, verification)
            geometry = data.geometry
            space = NativeSplineSpace(geometry)
            guard = PhysicalGuard(geometry, None, data.mean, data.scale, positivity_basis='bernstein_subdivided')
            physical = FrozenContinuity(geometry, data.mean, data.scale, verification['code_interval'], clean['gamma'])
            dense = QuadratureSampler.__new__(QuadratureSampler)
            dense.basis = payload['dense_basis']
            result = data, space, guard, physical, dense, verification
        else:
            result = uncached_prepare(clean)
            data, space, guard, physical, dense, verification = result
            if type(data) is not EulerDataset or verification.get('all_trajectory_and_initial_hashes_verified') is not True:
                raise ValueError('Prepared-data cache requires complete native dataset hash verification')
            _check_training_normalization(data, verification)
            _check_artifacts(artifacts)
            if _identity(clean, uncached_prepare) != identity:
                raise PreparedDataCacheError('Preparation source changed during complete verification')
            header = dict(schema=SCHEMA, identity=identity, artifacts=artifacts)
            _write(path, header, dict(data=data, dense_basis=dense.basis, verification=verification))
        _check_artifacts(artifacts)
    report = copy.deepcopy(result[-1])
    report['verification_reused'] = reused
    report['cached_setup'] = dict(cache_path=str(path), reused=reused,
        preparation_identity_sha256=config_hash(identity), artifact_fingerprints_checked=len(artifacts),
        seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started,
        solver_factors_serialized=False, original_verification_seconds=verification['verification_seconds'],
        verification_scope='Within-session reuse of the originally hash-verified immutable dataset; '
                           'manifest hash, preparation source hashes and artifact stat fingerprints rechecked. '
                           'Reuse is not a new independent trajectory hash verification.')
    return (*result[:-1], report)
