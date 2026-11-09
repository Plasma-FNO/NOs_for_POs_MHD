import numpy as np
import pytest
from mastu_opsplit.data import (STATE_CHANNELS, TERM_CHANNELS, geometry_hash,
                               read_trajectory, validate_trajectory, write_geometry, write_trajectory)
from mastu_opsplit.numerical import assemble_strong, evaluate_native
from test_numerical import geometry_fixture


def trajectory_fixture():
    geometry = geometry_fixture()
    coefficients = np.ones((2, 3, 7))
    labels = np.ones((2, 2, 2, 2))
    assembled = assemble_strong(labels, geometry)
    selected = np.zeros((2, 3, 7, 2))
    selected[..., 4, 0] = assembled[..., 0]
    selected[..., 5, 1] = assembled[..., 1]
    arrays = {'time_seconds': np.array([0., 1.]), 'time_code': np.array([0., 10.]),
              'native_state': coefficients, 'state_samples': evaluate_native(coefficients, geometry).reshape(2, 4, 7),
              'pointwise_terms': labels, 'selected_residuals': selected,
              'retained_residual': np.ones((2, 3, 7)), 'total_residual': 1+selected.sum(axis=-1)}
    metadata = dict(case_id='synthetic', parent_family_id='manufactured-1', split='train', seed=1,
                    parameters={'s_perp': 1.}, config_hash='fixture', source_commit='fixture',
                    executable_hash='fixture', equilibrium_hash='fixture', baseline_id='fixture',
                    field_units=['fixture']*7, normalization={'kind': 'explicit manufactured units'},
                    residual_units='fixture', solver_status='success', required_final_time_seconds=1.,
                    synthetic=True, mapping_status='provisional', state_channels=list(STATE_CHANNELS), term_channels=list(TERM_CHANNELS),
                    pointwise_term_kind='strong', spatial_residual_convention='axisymmetric_volume_weak')
    return geometry, arrays, metadata


def test_roundtrip_preserves_native_coefficients_and_geometry(tmp_path):
    geometry, arrays, metadata = trajectory_fixture()
    gp, tp = tmp_path/'geometry.h5', tmp_path/'trajectory.h5'
    write_geometry(gp, geometry)
    write_trajectory(tp, geometry, gp, arrays, metadata)
    restored, values, info = read_trajectory(tp)
    assert geometry_hash(restored) == geometry_hash(geometry)
    assert info == metadata
    for key, expected in arrays.items():
        np.testing.assert_array_equal(values[key], expected)
    with pytest.raises(OSError):
        write_trajectory(tp, geometry, gp, arrays, metadata)


def test_reject_label_mapping_incomplete_runs_and_nonphysical_fields():
    geometry, arrays, metadata = trajectory_fixture()
    arrays['selected_residuals'][0, 0, 4, 0] += 1
    with pytest.raises(ValueError, match='weak residual'):
        validate_trajectory(geometry, arrays, metadata)
    geometry, arrays, metadata = trajectory_fixture()
    metadata['required_final_time_seconds'] = 2
    with pytest.raises(ValueError, match='horizon'):
        validate_trajectory(geometry, arrays, metadata)
    geometry, arrays, metadata = trajectory_fixture()
    arrays['state_samples'][0, 0, 4] = -1
    with pytest.raises(ValueError, match='FE state evaluation'):
        validate_trajectory(geometry, arrays, metadata)


def test_no_state_or_schema_leakage():
    geometry, arrays, metadata = trajectory_fixture()
    arrays['native_state'][0, 0, 0] += .1
    with pytest.raises(ValueError, match='FE state evaluation'):
        validate_trajectory(geometry, arrays, metadata)
    geometry, arrays, metadata = trajectory_fixture()
    del metadata['normalization']
    with pytest.raises(ValueError, match='metadata'):
        validate_trajectory(geometry, arrays, metadata)


def test_real_exchange_cannot_claim_release_acceptance():
    geometry, arrays, metadata = trajectory_fixture()
    metadata['synthetic'] = False
    with pytest.raises(ValueError, match='provisional'):
        validate_trajectory(geometry, arrays, metadata)
    metadata['acceptance_status'] = 'provisional_exchange'
    assert validate_trajectory(geometry, arrays, metadata)['relative_error'] == 0
    metadata['required_final_time_seconds'] = float('nan')
    with pytest.raises(ValueError, match='finite and positive'):
        validate_trajectory(geometry, arrays, metadata)


def test_export_command_dry_run_and_exclusive_output(tmp_path):
    from dataclasses import fields
    import json
    from pathlib import Path
    import subprocess
    import sys
    import yaml
    geometry, arrays, metadata = trajectory_fixture()
    geometry_npz = tmp_path/'geometry.npz'
    trajectory_npz = tmp_path/'trajectory.npz'
    metadata_json = tmp_path/'metadata.json'
    config = tmp_path/'export.yaml'
    output = tmp_path/'trajectory.h5'
    geometry_h5 = tmp_path/'geometry.h5'
    np.savez(geometry_npz, **{field.name: getattr(geometry, field.name) for field in fields(geometry)})
    np.savez(trajectory_npz, **arrays)
    metadata_json.write_text(json.dumps(metadata))
    config.write_text(yaml.safe_dump(dict(geometry_npz=str(geometry_npz), trajectory_npz=str(trajectory_npz),
                                         metadata_json=str(metadata_json), geometry_h5=str(geometry_h5))))
    command = [sys.executable, str(Path(__file__).parents[1]/'scripts/export_dataset.py'),
               '--config', str(config), '--output', str(output)]
    result = subprocess.run(command+['--dry-run'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not output.exists() and not geometry_h5.exists()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert read_trajectory(output)[2]['synthetic']
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_invalid_metadata_leaves_no_partial_trajectory(tmp_path):
    geometry, arrays, metadata = trajectory_fixture()
    geometry_path = tmp_path/'geometry.h5'
    output = tmp_path/'trajectory.h5'
    write_geometry(geometry_path, geometry)
    metadata['parameters']['s_perp'] = float('nan')
    with pytest.raises(ValueError):
        write_trajectory(output, geometry, geometry_path, arrays, metadata)
    assert not output.exists()
