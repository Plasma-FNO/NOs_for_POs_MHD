"""Snapshot the pinned JOREK case without changing the production build."""
from __future__ import annotations

import re
import math
import shutil
import subprocess
from pathlib import Path

from .common import config_hash, sha256_file, write_json


def namelist_overrides(text: str, values: dict) -> str:
    """Replace simple scalar/array assignments; reject duplicate existing assignments."""
    for name, value in values.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\(\d+(?::\d+)?(?:,\d+(?::\d+)?)*\))?", name):
            raise ValueError(f"invalid namelist key: {name}")
        if isinstance(value, bool):
            rendered = ".true." if value else ".false."
        elif isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("nonfinite namelist value")
            rendered = str(value)
        elif isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9 _.,+\-]+", value):
            if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?", value):
                raise ValueError(f"numeric-looking string for {name}; use a numeric YAML value with an exponent sign")
            rendered = "'" + value + "'"
        else:
            raise ValueError(f"unsupported namelist value for {name}")
        key = re.escape(name).replace(r"\(", r"\s*\(\s*").replace(r"\)", r"\s*\)")
        key = key.replace(":", r"\s*:\s*").replace(",", r"\s*,\s*")
        pattern = re.compile(r"^\s*" + key + r"\s*=.*$", re.I | re.M)
        matches = list(pattern.finditer(text))
        if len(matches) > 1:
            raise ValueError(f"duplicate namelist assignment: {name}")
        assignment = f" {name} = {rendered}"
        if matches:
            text = pattern.sub(assignment, text)
        else:
            end = list(re.finditer(r"^\s*/\s*(?:!.*)?$", text, re.M))
            if len(end) != 1:
                raise ValueError("expected one terminating namelist slash")
            text = text[:end[0].start()] + assignment + "\n" + text[end[0].start():]
    return text


def case_plan(config: dict, output) -> dict:
    mode = config["mode"]
    if mode not in ("equilibrium_regression", "unperturbed_pilot"):
        raise ValueError("perturbed evolution requires the reviewed, built, validated initializer patch")
    evolution = config["evolution"]
    if type(evolution["nstep_n"]) is not int or evolution["nstep_n"] < 0:
        raise ValueError("nstep_n must be a nonnegative integer")
    if mode == "equilibrium_regression" and evolution["nstep_n"] != 0:
        raise ValueError("equilibrium regression must retain nstep_n=0")
    if mode == "unperturbed_pilot" and evolution["nstep_n"] < 1:
        raise ValueError("unperturbed pilot requires evolution")
    if not isinstance(evolution["tstep_n"], (int, float)) or not math.isfinite(evolution["tstep_n"]) or evolution["tstep_n"] <= 0:
        raise ValueError("positive finite timestep required")
    if type(evolution["nout"]) is not int or evolution["nout"] < 1:
        raise ValueError("output cadence must be a positive integer")
    if config.get("perturbation", {}).get("enabled", False):
        raise ValueError("initializer patch is not yet build/regression validated")
    jorek = config["jorek"]
    source = Path(jorek["source"]).resolve()
    executable = Path(jorek["executable"]).resolve()
    inputs = Path(jorek["input_directory"]).resolve()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != jorek["pinned_commit"]:
        raise ValueError("JOREK source commit differs from configured pin")
    executable_hash = sha256_file(executable)
    if executable_hash != jorek["executable_sha256"]:
        raise ValueError("JOREK executable differs from configured pin")
    overrides = {**config.get("input_defaults", {}), **evolution, "restart": False, "regrid": False}
    namelist = namelist_overrides((inputs / "input").read_text(), overrides)
    return {
        "case_id": config["case_id"], "parent_family_id": config["parent_family_id"],
        "mode": mode, "output": str(Path(output).resolve()), "config_hash": config_hash(config),
        "source_commit": commit, "executable_sha256": executable_hash,
        "files": {name: {"source": str(path), "sha256": sha256_file(path)} for name, path in {
            "jorek.exe": executable, "input.original": inputs / "input", "eqdsk.dat": inputs / "eqdsk.dat",
            "jorek-env.sh": Path(jorek["runtime_environment"]),
            "equilibrium_reference.h5": inputs / "ref/end.h5",
            "build_config.txt": executable.parent / "build_config.txt",
            "model_settings.f90": executable.parent / "models/model303/mod_model_settings.f90",
            "mesh_settings.f90": executable.parent / "models/mod_settings.f90",
            "source_initial_conditions.f90": source / "models/model303/initial_conditions.f90",
            "source_residual.f90": source / "models/model303/mod_elt_matrix_fft.f90",
            "source_defaults.f90": source / "models/preset_parameters.f90",
        }.items()},
        "namelist": namelist, "evolution": evolution,
        "gate_A_passed": False, "physical_acceptance": "not assessed",
    }


def prepare_case(config: dict, output, *, dry_run=False) -> dict:
    plan = case_plan(config, output)
    if dry_run:
        return plan
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name, record in plan["files"].items():
        target = destination / name
        shutil.copyfile(record["source"], target)
        if sha256_file(target) != record["sha256"]:
            raise ValueError(f"input changed while snapshotting: {name}")
        target.chmod(0o555 if name == "jorek.exe" else 0o444)
    (destination / "input").write_text(plan.pop("namelist"))
    (destination / "input").chmod(0o444)
    plan["snapshot_hashes"] = {p.name: sha256_file(p) for p in destination.iterdir() if p.is_file()}
    write_json(destination / "case.json", plan)
    return plan
