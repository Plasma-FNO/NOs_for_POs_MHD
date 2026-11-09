import json

import pytest

from mastu_opsplit.manifest import release_manifest


def passed_gates(tmp_path):
    (tmp_path / "evidence.json").write_text("{}")
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps({key: {"status": "passed", "evidence": ["evidence.json"]}
                                for key in ("A", "B", "C")}))
    return gates


def test_release_cannot_run_before_gates(tmp_path):
    gates = tmp_path / "gates.json"
    gates.write_text('{"A":{"status":"pending"}}')
    with pytest.raises(ValueError, match="Gate A"):
        release_manifest([], tmp_path / "release.json", gates_file=gates, frozen_protocol_hash="frozen")
    assert not (tmp_path / "release.json").exists()


def test_release_rejects_synthetic_and_retains_failure_records(tmp_path, monkeypatch):
    from test_data import trajectory_fixture
    gates = passed_gates(tmp_path)
    geometry, arrays, metadata = trajectory_fixture()
    monkeypatch.setattr("mastu_opsplit.manifest.read_trajectory", lambda _: (geometry, arrays, metadata))
    trajectory = tmp_path / "reference.h5"
    trajectory.write_bytes(b"manufactured-test-file")
    row = dict(case_id=metadata["case_id"], parent_family_id=metadata["parent_family_id"],
               split="train", status="accepted", path=str(trajectory))
    with pytest.raises(ValueError, match="synthetic"):
        release_manifest([row], tmp_path / "release.json", gates_file=gates, frozen_protocol_hash="frozen")
    # Mock verified native metadata only to exercise bookkeeping, never gate evidence.
    metadata.update(synthetic=False, mapping_status="verified_native")
    with pytest.raises(ValueError, match="frozen"):
        release_manifest([row], tmp_path / "release.json", gates_file=gates, frozen_protocol_hash="frozen")
    metadata["frozen_protocol_hash"] = "frozen"
    attempt = tmp_path / "attempt.json"
    attempt.write_text('{"status":"timeout"}')
    failure = dict(case_id="failed-case", parent_family_id="failed-family", split="id_test",
                   status="failed", failure_reason="timeout", attempts=[str(attempt)])
    report = release_manifest([row, failure], tmp_path / "release.json", gates_file=gates, frozen_protocol_hash="frozen")
    assert report["accepted"] == report["failed"] == 1
    assert report["trajectories"][1]["attempt_records"][0]["sha256"]
