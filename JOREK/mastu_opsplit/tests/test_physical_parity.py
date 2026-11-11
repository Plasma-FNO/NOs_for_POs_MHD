"""Regression checks for independently weighted native label parity evidence."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('verify_physical_defect',
    Path(__file__).resolve().parents[1]/'scripts/verify_physical_defect.py')
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def test_physical_parity_uses_native_volume_weights_and_fixed_training_scale():
    native = np.zeros((2, 13))
    prediction = np.zeros_like(native)
    prediction[:, 0] = [1., 3.]
    scale = np.full(13, 2.)
    row = script.compare_labels(prediction, native, np.array([3., 1.]), scale)
    assert row['channel_rms_error'][0] == pytest.approx(np.sqrt(3.))
    assert row['channel_training_scaled_rms'][0] == pytest.approx(np.sqrt(3.)/2)
    assert row['channel_training_scaled_max'][0] == 1.5
    assert np.isfinite(row['channel_relative_rms_with_floor']).all()
    assert row['channel_relative_rms_with_floor'][0] == pytest.approx(np.sqrt(3.)/2e-6)


def test_float32_native_labels_have_small_reported_error():
    rng = np.random.default_rng(31)
    exact = rng.normal(size=(300, 13))*np.geomspace(1e-11, 1e-3, 13)
    scale = np.sqrt((exact**2).mean(0))
    row = script.compare_labels(exact, exact.astype(np.float32).astype(float), np.ones(300), scale)
    assert 1e-8 < max(row['channel_training_scaled_rms']) < 1e-7


def test_nonfinite_or_nonpositive_metrics_are_rejected():
    a = np.ones((2, 13))
    with pytest.raises(ValueError):
        script.compare_labels(a, a, np.zeros(2), np.ones(13))
    with pytest.raises(ValueError):
        script.compare_labels(a, a*np.nan, np.ones(2), np.ones(13))
    with pytest.raises(ValueError):
        script.compare_labels(a, a, np.ones(2), np.zeros(13))
