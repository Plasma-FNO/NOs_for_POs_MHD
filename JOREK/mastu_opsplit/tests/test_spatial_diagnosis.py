import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def test_weighted_localization_and_zero_response(monkeypatch):
    reports = Path(__file__).resolve().parents[1]/'reports'
    monkeypatch.syspath_prepend(str(reports))
    spec = importlib.util.spec_from_file_location('diagnosis', reports/'cpu_spatial_diagnosis_20261103.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    error = np.zeros(100)
    error[37] = 2.
    weights = np.ones(100)
    points = np.column_stack((np.arange(100), np.zeros(100)))
    region = np.arange(100) > 30
    result = module.concentration(error, np.ones(100), weights, points, {'selected': region})
    assert result['error_energy'] == pytest.approx(4.)
    assert result['reference_energy'] == pytest.approx(100.)
    assert result['regional_error_energy_fractions']['selected'] == 1.
    assert result['error_fraction_in_largest_error_one_percent_volume'] == 1.
    assert result['bbox_containing_largest_contributors_90percent_energy_m'] == [[37., 0.], [37., 0.]]
    result = module.concentration(error*0, error*0, weights, points, {'selected': region})
    assert result['error_energy'] == result['reference_energy'] == 0.
    assert result['regional_error_energy_fractions']['selected'] == 0.
