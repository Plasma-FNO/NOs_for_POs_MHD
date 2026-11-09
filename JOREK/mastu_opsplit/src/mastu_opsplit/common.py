"""Shared configuration, provenance, and evidence helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def load_config(path) -> dict:
    with Path(path).open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("configuration must contain a mapping")
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize before opening so invalid/nonfinite reports leave no partial file.
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(payload)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(mapping) -> str:
    return hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def require_gates(path, required: tuple[str, ...]):
    path = Path(path)
    gates = json.loads(path.read_text())
    for name in required:
        gate = gates.get(name, {})
        evidence = gate.get("evidence", [])
        if gate.get("status") != "passed" or not evidence:
            raise ValueError(f"Gate {name} has not passed with evidence")
        for item in evidence:
            evidence_path = Path(item)
            if not evidence_path.is_absolute():
                evidence_path = path.parent / evidence_path
            if not evidence_path.is_file():
                raise ValueError(f"Gate {name} evidence missing: {evidence_path}")

