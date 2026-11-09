import json

import numpy as np
import pytest

from mastu_opsplit.common import config_hash, require_gates, write_json
from mastu_opsplit.evaluation import (fit_training_scales, paired_family_interval,
                                     resource_totals, summarize, weighted_errors)
from mastu_opsplit.sampling import candidate_parameters, validate_family_splits


def test_volume_weighted_errors_and_decaying_perturbations():
    truth = np.array([[[1.], [3.]], [[0.], [0.]]])
    prediction = truth + 1
    metrics = weighted_errors(truth, prediction, np.array([1., 3.]), np.array([2.]))
    np.testing.assert_allclose(metrics["absolute_l2"], 1)
    np.testing.assert_allclose(metrics["relative_l2"][0], 1/np.sqrt(7))
    np.testing.assert_allclose(metrics["scaled_l2"], .5)
    assert np.isfinite(metrics["relative_l2"]).all()
    with pytest.raises(ValueError):
        weighted_errors(truth, prediction, [-1., 2.], [2.])


def test_scales_reject_test_data():
    record = {"split": "train", "perturbation": np.ones((2, 3, 1))*2, "weights": np.ones(3)}
    np.testing.assert_allclose(fit_training_scales([record]), [2])
    with pytest.raises(ValueError, match="training"):
        fit_training_scales([{**record, "split": "id_test"}])


def test_failed_rollouts_are_counted_and_all_failed_is_defined():
    success = dict(model="opsplit", seed=0, regime="id", case_id="a", parent_family_id="f",
                   completed=True, perturbation_error=.03)
    failure = {**success, "case_id": "b", "completed": False, "failure_reason": "nonfinite", "perturbation_error": None}
    result = summarize([success, failure])[0]
    assert result["completion_fraction"] == .5
    assert result["failed_case_ids"] == ["b"]
    assert summarize([failure])[0]["median_error_completed_only"] is None
    with pytest.raises(ValueError, match="duplicate"):
        summarize([success, success])


def test_bootstrap_clusters_correlated_cases_and_requires_pairs():
    left = {"a": ("family1", 2), "b": ("family1", 4), "c": ("family2", 8)}
    right = {"a": ("family1", 1), "b": ("family1", 3), "c": ("family2", 7)}
    result = paired_family_interval(left, right, samples=50)
    assert result["parent_families"] == 2
    assert result["interval"] == [1., 1.]
    with pytest.raises(ValueError, match="paired"):
        paired_family_interval(left, {"a": right["a"]})


def test_resource_accounting_includes_failed_attempts():
    record = dict(attempt_id="a", elapsed_seconds=3600, cpu_cores=4, gpus=0,
                  raw_bytes=10, export_bytes=5, peak_memory_bytes=100, outcome="failed")
    totals = resource_totals([record, {**record, "attempt_id": "b", "elapsed_seconds": 1800}])
    assert totals["cpu_core_hours"] == 6
    assert totals["raw_bytes"] == 20
    assert totals["peak_memory_bytes"] == 100
    with pytest.raises(ValueError, match="duplicate"):
        resource_totals([record, record])


def test_gates_require_passed_status_and_existing_evidence(tmp_path):
    gate_path = tmp_path / "gates.json"
    gate_path.write_text(json.dumps({"A": {"status": "passed", "evidence": []}}))
    with pytest.raises(ValueError):
        require_gates(gate_path, ("A",))
    (tmp_path / "evidence.json").write_text("{}")
    gate_path.write_text(json.dumps({"A": {"status": "passed", "evidence": ["evidence.json"]}}))
    require_gates(gate_path, ("A",))


def test_exclusive_report_and_stable_hash(tmp_path):
    output = tmp_path / "report.json"
    write_json(output, {"ok": True})
    with pytest.raises(FileExistsError):
        write_json(output, {})
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        write_json(tmp_path / "invalid.json", {"value": float("nan")})
    assert not (tmp_path / "invalid.json").exists()


def test_candidate_sampling_and_family_leakage():
    config = dict(seed=4, splits={"train": 8, "combined_test": 8}, train_amplitude=[.02,.15],
                  strong_amplitude=[.2,.3], train_transport=[.7,1.4], low_transport=[.35,.5], high_transport=[2,2.8])
    rows = candidate_parameters(config)
    assert rows == candidate_parameters(config)
    assert all(.02 <= row["A_rho"] <= .15 for row in rows[:8])
    assert all(row["A_T"] >= .2 for row in rows[8:])
    assert all(row["center_m"] is None for row in rows)
    with pytest.raises(ValueError, match="leaks"):
        validate_family_splits([rows[0], {**rows[0], "split": "id_test", "case_id": "new"}])
    with pytest.raises(ValueError, match="overlap"):
        candidate_parameters({**config, "strong_amplitude": [.1, .3]})
    with pytest.raises(ValueError, match="unknown"):
        candidate_parameters({**config, "splits": {"typo_test": 8}})
