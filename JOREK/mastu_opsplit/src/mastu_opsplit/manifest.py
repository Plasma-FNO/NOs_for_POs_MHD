"""Release manifests preserve rejected runs and verify accepted trajectory identity."""
from pathlib import Path

from .common import require_gates, sha256_file, write_json
from .data import read_trajectory
from .sampling import validate_family_splits


def release_manifest(records, output, *, gates_file, frozen_protocol_hash):
    """Publish a local manifest only after gates; no data is sent externally."""
    require_gates(gates_file, ("A", "B", "C"))
    if not frozen_protocol_hash or not records:
        raise ValueError("frozen configuration hash and attempted cases required")
    validate_family_splits(records)
    verified, geometry_hashes = [], set()
    for row in records:
        if row["status"] == "failed":
            if not row.get("failure_reason") or not row.get("attempts"):
                raise ValueError("failed cases require reason and preserved attempts")
            attempts = []
            for path in row["attempts"]:
                attempts.append({"path": str(Path(path).resolve()), "sha256": sha256_file(path)})
            verified.append({**row, "attempt_records": attempts})
            continue
        if row["status"] != "accepted":
            raise ValueError("case status must be accepted or failed")
        geometry, arrays, metadata = read_trajectory(row["path"])
        if metadata["synthetic"]:
            raise ValueError("synthetic fixtures cannot enter a scientific release")
        if metadata.get("frozen_protocol_hash") != frozen_protocol_hash:
            raise ValueError("trajectory belongs to a different frozen sampling/evaluation protocol")
        for key in ("case_id", "parent_family_id", "split"):
            if metadata[key] != row[key]:
                raise ValueError(f"trajectory manifest identity mismatch: {key}")
        if metadata.get("mapping_status") != "verified_native":
            raise ValueError("native weak-form mapping must be verified before release")
        from .data import geometry_hash
        geometry_hashes.add(geometry_hash(geometry))
        verified.append({**row, "path": str(Path(row["path"]).resolve()),
                         "sha256": sha256_file(row["path"]),
                         "geometry_hash": geometry_hash(geometry),
                         "final_time_seconds": float(arrays["time_seconds"][-1])})
    if len(geometry_hashes) != 1:
        raise ValueError("release requires accepted trajectories with one fixed physical geometry")
    report = {"schema_version": "mastu-release-v1", "frozen_protocol_hash": frozen_protocol_hash,
              "geometry_hash": next(iter(geometry_hashes)), "trajectories": verified,
              "accepted": sum(row["status"] == "accepted" for row in records),
              "failed": sum(row["status"] == "failed" for row in records)}
    write_json(output, report)
    return report
