"""Versioned HDF5 exchange contract for native model-303 trajectories."""
from dataclasses import fields as dataclass_fields
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from .numerical import NativeGeometry, assemble_strong, evaluate_native, residual_identity

SCHEMA_VERSION = 'mastu-native-v1'
STATE_CHANNELS = ('psi', 'u', 'zj', 'w', 'rho', 'T', 'Vpar')
TERM_CHANNELS = ('density_advection_compression', 'pressure_advection_compression')
ARRAY_NAMES = ('time_seconds', 'time_code', 'native_state', 'state_samples', 'pointwise_terms',
               'selected_residuals', 'retained_residual', 'total_residual')
REQUIRED_METADATA = ('case_id', 'parent_family_id', 'split', 'seed', 'parameters', 'config_hash',
                     'source_commit', 'executable_hash', 'equilibrium_hash', 'baseline_id',
                     'field_units', 'normalization', 'residual_units', 'solver_status',
                     'required_final_time_seconds', 'synthetic', 'mapping_status')


def geometry_hash(geometry):
    geometry.validate()
    digest = hashlib.sha256()
    for field in dataclass_fields(geometry):
        value = np.ascontiguousarray(getattr(geometry, field.name))
        digest.update(field.name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def validate_trajectory(geometry, arrays, metadata):
    geometry.validate()
    missing = set(REQUIRED_METADATA) - set(metadata)
    if missing:
        raise ValueError(f'missing trajectory metadata: {sorted(missing)}')
    if metadata['mapping_status'] not in ('candidate', 'provisional', 'verified_native'):
        raise ValueError('unknown native mapping status')
    if not metadata['synthetic'] and metadata.get('acceptance_status') != 'provisional_exchange':
        raise ValueError('real exchange remains provisional until independent native mapping and rollout gates pass')
    if metadata['solver_status'] != 'success':
        raise ValueError('unsuccessful runs belong in the failure manifest, not successful exchange HDF5')
    if tuple(metadata.get('state_channels', ())) != STATE_CHANNELS:
        raise ValueError('state_channels must preserve native model303 order')
    if tuple(metadata.get('term_channels', ())) != TERM_CHANNELS:
        raise ValueError('term_channels must declare density and pressure residual blocks')
    if metadata.get('pointwise_term_kind') != 'strong' or metadata.get('spatial_residual_convention') != 'axisymmetric_volume_weak':
        raise ValueError('labels require declared strong form and common volume weak residual units')
    if len(metadata['field_units']) != 7 or not metadata['normalization'] or not metadata['residual_units']:
        raise ValueError('units and exact normalization must be specified')
    if not isinstance(metadata['synthetic'], bool):
        raise ValueError('synthetic must be an explicit boolean')
    missing = set(ARRAY_NAMES) - set(arrays)
    if missing:
        raise ValueError(f'missing trajectory arrays: {sorted(missing)}')
    time = np.asarray(arrays['time_seconds'])
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or (np.diff(time) <= 0).any():
        raise ValueError('physical timestamps must be finite and strictly increasing')
    horizon = float(metadata['required_final_time_seconds'])
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError('required physical horizon must be finite and positive')
    if time[-1] < horizon:
        raise ValueError('trajectory does not reach required physical horizon')
    t, d = len(time), len(geometry.dof_ids)
    e, q = geometry.basis.shape[:2]
    shapes = {'time_code': (t,), 'native_state': (t, d, 7), 'state_samples': (t, e*q, 7),
              'pointwise_terms': (t, e, q, 2), 'selected_residuals': (t, d, 7, 2),
              'retained_residual': (t, d, 7), 'total_residual': (t, d, 7)}
    for name, shape in shapes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'invalid {name}: expected finite {shape}')
    if (np.diff(arrays['time_code']) <= 0).any():
        raise ValueError('original code timestamps must be strictly increasing')
    evaluated = evaluate_native(arrays['native_state'], geometry).reshape(t, e*q, 7)
    if not np.allclose(evaluated, arrays['state_samples'], rtol=1e-10, atol=1e-12):
        raise ValueError('native FE state evaluation differs from stored source samples')
    if (np.asarray(arrays['state_samples'])[..., 4:6] <= 0).any():
        raise ValueError('density and temperature must remain positive')
    assembled = assemble_strong(arrays['pointwise_terms'], geometry)
    expected = np.zeros((t, d, 7, 2), dtype=np.float64)
    expected[..., 4, 0] = assembled[..., 0]
    expected[..., 5, 1] = assembled[..., 1]
    if not np.allclose(expected, arrays['selected_residuals'], rtol=1e-8, atol=1e-12):
        raise ValueError('pointwise labels do not reproduce selected weak residual blocks')
    return residual_identity(arrays['total_residual'], arrays['retained_residual'], arrays['selected_residuals'])


def write_geometry(path, geometry):
    """Write fixed geometry once; exclusive creation preserves accepted datasets."""
    geometry.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'x') as handle:
        handle.attrs['schema_version'] = SCHEMA_VERSION
        handle.attrs['geometry_hash'] = geometry_hash(geometry)
        for field in dataclass_fields(geometry):
            handle.create_dataset(field.name, data=getattr(geometry, field.name))
    return geometry_hash(geometry)


def read_geometry(path):
    with h5py.File(path, 'r') as handle:
        if handle.attrs['schema_version'] != SCHEMA_VERSION:
            raise ValueError('unsupported geometry schema')
        geometry = NativeGeometry(**{field.name: handle[field.name][...] for field in dataclass_fields(NativeGeometry)})
        if geometry_hash(geometry) != handle.attrs['geometry_hash']:
            raise ValueError('geometry hash mismatch')
    return geometry


def write_trajectory(path, geometry, geometry_path, arrays, metadata):
    checks = validate_trajectory(geometry, arrays, metadata)
    if geometry_hash(read_geometry(geometry_path)) != geometry_hash(geometry):
        raise ValueError('trajectory references a different geometry')
    metadata_json = json.dumps(metadata, sort_keys=True, allow_nan=False)
    checks_json = json.dumps(checks, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'x') as handle:
        handle.attrs['schema_version'] = SCHEMA_VERSION
        handle.attrs['geometry_hash'] = geometry_hash(geometry)
        handle.attrs['geometry_path'] = str(Path(geometry_path).resolve())
        handle.attrs['metadata_json'] = metadata_json
        handle.attrs['checks_json'] = checks_json
        for name in ARRAY_NAMES:
            handle.create_dataset(name, data=arrays[name], compression='gzip', shuffle=True)
    return checks


def read_trajectory(path, geometry_path=None):
    with h5py.File(path, 'r') as handle:
        if handle.attrs['schema_version'] != SCHEMA_VERSION:
            raise ValueError('unsupported trajectory schema')
        geometry = read_geometry(geometry_path or handle.attrs['geometry_path'])
        if geometry_hash(geometry) != handle.attrs['geometry_hash']:
            raise ValueError('trajectory geometry hash mismatch')
        arrays = {name: handle[name][...] for name in ARRAY_NAMES}
        metadata = json.loads(handle.attrs['metadata_json'])
    validate_trajectory(geometry, arrays, metadata)
    return geometry, arrays, metadata
