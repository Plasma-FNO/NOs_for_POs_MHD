"""Bounded study runner adapted from tkit.datasets.jorek_run (MIT, V. Gopakumar).

Source revision 14fcecb664cbb084b86051370bc415d8446e1637. Retains its process-group
termination and RTree-only retry policy. Importing tkit enables JAX globally;
this narrow standard-library adaptation keeps the PyTorch study independent.
"""
from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .common import config_hash, sha256_file, write_json

SOLVER_ABORT = re.compile(r"NO\s+CONVERGENCE\s+AFTER\s+\d+\s+ITERATIONS[.\s]*ABORTING", re.I)


def _resource_caps(config):
    authorization = config.get("resource_authorization", {})
    profile = authorization.get("resource_profile")
    if profile is None:
        return {"mpi_ranks": 4, "attempts": 2, "seconds": 1800, "memory_gib": 16}
    if profile != "cpu_refinement_20261102" or authorization.get("submission_allowed") is not True:
        raise ValueError("unknown or unapproved resource profile")
    path = Path(authorization["budget_record"])
    if sha256_file(path) != authorization.get("budget_record_sha256"):
        raise ValueError("CPU budget record identity differs")
    budget = json.loads(path.read_text())
    if budget["resource_profile"] != profile or budget["slurm_account"] != "fupa2_fauna":
        raise ValueError("CPU budget record does not authorize this profile/account")
    phase = budget["phase"]
    credit = min(budget["remaining_local_resource_hours"], budget["monthly_remaining_local_resource_hours"])
    if credit < phase["conservative_local_resource_hour_envelope"]:
        raise ValueError("insufficient verified CPU budget for the refinement envelope")
    resources = config["slurm"]
    if resources["account"] != "fupa2_fauna" or resources["partition"] != "dcgp_fua_prod":
        raise ValueError("extended CPU caps require the verified production CPU account/partition")
    return {"mpi_ranks": 16, "attempts": 1, "seconds": 7200, "memory_gib": 128}


def classify(returncode, timed_out, text):
    if timed_out:
        return "timeout"
    if "tools/RTree.h" in text and re.search(r"assertion", text, re.I):
        return "rtree_assertion"
    if SOLVER_ABORT.search(text):
        return "solver_aborted"
    if "HDF5-DIAG: Error detected" in text:
        return "hdf5_error"
    if returncode != 0:
        return "exit_nonzero"
    return "completed"


def restart_health(path, *, expected_steps: int, final_time_s: float) -> dict:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as restart:
        count = int(restart["index_now"][()])
        n_tor = int(restart["n_tor"][()])
        unit = float(restart["sqrt_mu0_rho0"][()])
        times = np.asarray(restart["xtime"][()] if "xtime" in restart else [], dtype=float)
        energies = np.asarray(restart["energies"][()] if "energies" in restart else np.zeros((0, 2, n_tor)))
    if n_tor != 1 or count < expected_steps or times.shape != (count,) or energies.shape != (count, 2, n_tor):
        raise ValueError("restart harmonic count, step count, or diagnostics layout is incomplete")
    if not math.isfinite(unit) or unit <= 0 or not np.isfinite(times).all() or not np.isfinite(energies).all():
        raise ValueError("nonfinite restart diagnostics or invalid normalization")
    if times.size > 1 and not (np.diff(times) > 0).all():
        raise ValueError("restart time must increase strictly")
    last = float(times[-1] * unit) if times.size else 0.0
    tolerance = 1e-10 * max(abs(final_time_s), 1e-12)
    if last + tolerance < final_time_s:
        raise ValueError(f"incomplete final time: {last} < {final_time_s} seconds")
    return {"steps": count, "final_time_s": last, "restart_sha256": sha256_file(path)}


def equilibrium_regression(result, reference, *, dataset="values", absolute_tolerance=1e-4):
    import h5py
    import numpy as np
    if not math.isfinite(absolute_tolerance) or absolute_tolerance <= 0:
        raise ValueError("positive finite regression tolerance required")
    with h5py.File(result, "r") as result_h5, h5py.File(reference, "r") as reference_h5:
        actual, expected = np.asarray(result_h5[dataset]), np.asarray(reference_h5[dataset])
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("equilibrium regression shape/finite check failed")
    discrepancy = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    if discrepancy > absolute_tolerance:
        raise ValueError(f"equilibrium regression discrepancy {discrepancy} > {absolute_tolerance}")
    return {"dataset": dataset, "maximum_absolute_difference": discrepancy,
            "absolute_tolerance": absolute_tolerance, "reference_sha256": sha256_file(reference)}


def run_plan(config: dict, prepared, output) -> dict:
    prepared = Path(prepared).resolve()
    manifest = json.loads((prepared / "case.json").read_text())
    if manifest["config_hash"] != config_hash(config):
        raise ValueError("prepared case configuration identity differs")
    for name, expected in manifest["snapshot_hashes"].items():
        if sha256_file(prepared / name) != expected:
            raise ValueError(f"prepared input changed: {name}")
    execution = config["execution"]
    for name in ("mpi_ranks", "omp_threads", "max_attempts"):
        if type(execution[name]) is not int or execution[name] < 1:
            raise ValueError(f"positive integer {name} required")
    # These study ceilings match configs/resources.yaml; production remains disabled.
    caps = _resource_caps(config)
    if execution["max_attempts"] > caps["attempts"] or execution["mpi_ranks"] > caps["mpi_ranks"] or execution["omp_threads"] > 1:
        raise ValueError(f"study caps permit at most {caps['mpi_ranks']} MPI ranks, 1 thread, and {caps['attempts']} attempts")
    if not math.isfinite(execution["timeout_s"]) or not 0 < execution["timeout_s"] <= caps["seconds"]:
        raise ValueError(f"positive finite timeout at most {caps['seconds']} seconds required")
    acceptance = config["acceptance"]
    if not acceptance["required_files"] or any(Path(name).name != name for name in acceptance["required_files"]):
        raise ValueError("explicit plain required output filenames required")
    final_time = acceptance["final_time_s"]
    if not math.isfinite(final_time) or final_time < 0:
        raise ValueError("finite nonnegative final physical time required")
    if config["evolution"]["nstep_n"] > 0 and final_time <= 0:
        raise ValueError("evolution requires a pilot-selected positive final physical time")
    launcher = execution.get("launcher", ["mpirun", "--bind-to", "none", "-n"])
    if not isinstance(launcher, list) or any(not isinstance(x, str) or not x for x in launcher):
        raise ValueError("launcher must be an explicit argument list")
    command = [*launcher, str(execution["mpi_ranks"]), str(prepared / "jorek.exe")] if launcher else [str(prepared / "jorek.exe")]
    return {"prepared": str(prepared), "output": str(Path(output).resolve()), "command": command,
            "execution": execution, "acceptance": acceptance, "config_hash": config_hash(config)}


def render_slurm(config, prepared, output, config_path) -> str:
    plan = run_plan(config, prepared, output)
    resources = config["slurm"]
    for key in ("account", "partition", "walltime", "memory"):
        if not isinstance(resources.get(key), str) or not re.fullmatch(r"[A-Za-z0-9_:.-]+", resources[key]):
            raise ValueError(f"explicit current Slurm {key} required")
    if type(resources["nodes"]) is not int or resources["nodes"] != 1:
        raise ValueError("initial bounded request requires exactly one node")
    execution = config["execution"]
    caps = _resource_caps(config)
    parts = resources["walltime"].split(":")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError("walltime must use HH:MM:SS")
    hours, minutes, seconds = map(int, parts)
    wall_seconds = 3600 * hours + 60 * minutes + seconds
    if minutes >= 60 or seconds >= 60 or not 0 < wall_seconds <= caps["seconds"]:
        raise ValueError(f"study walltime cap is {caps['seconds']//60} minutes")
    if wall_seconds < execution["timeout_s"] * execution["max_attempts"] + 60:
        raise ValueError("walltime must cover all attempts plus 60 seconds for setup/cleanup")
    memory = re.fullmatch(r"([1-9][0-9]*)([MG])", resources["memory"], re.I)
    if not memory or int(memory[1]) / (1024 if memory[2].upper() == "M" else 1) > caps["memory_gib"]:
        raise ValueError(f"study memory cap is {caps['memory_gib']}G; specify positive M or G units")
    command = [resources.get("python", "python"), str(Path(__file__).resolve().parents[2] / "scripts/run_case.py"),
               "--config", str(Path(config_path).resolve()), "--prepared", plan["prepared"], "--output", plan["output"]]
    return "\n".join([
        "#!/bin/bash", "#SBATCH --job-name=mastu-opsplit",
        f"#SBATCH --account={resources['account']}", f"#SBATCH --partition={resources['partition']}",
        "#SBATCH --nodes=1", f"#SBATCH --ntasks={execution['mpi_ranks']}",
        f"#SBATCH --cpus-per-task={execution['omp_threads']}",
        f"#SBATCH --time={resources['walltime']}", f"#SBATCH --mem={resources['memory']}",
        "set -euo pipefail", "source " + shlex.quote(str(Path(prepared).resolve() / "jorek-env.sh")),
        shlex.join(command), "",
    ])


def run_case(config: dict, prepared, output, *, dry_run=False) -> dict:
    plan = run_plan(config, prepared, output)
    if dry_run:
        return plan
    if config.get("resource_authorization", {}).get("submission_allowed") is False:
        raise RuntimeError("this concrete resource request is proposed, not authorized")
    if config["execution"].get("require_slurm", True) and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("simulation execution requires a Slurm allocation; use --dry-run on login nodes")
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "plan.json", plan)
    execution = config["execution"]
    prepared = Path(prepared).resolve()
    attempts = []
    for index in range(1, execution["max_attempts"] + 1):
        attempt = destination / f"attempt_{index}"
        work = attempt / "work"
        work.mkdir(parents=True)
        for name in ("input", "eqdsk.dat"):
            shutil.copyfile(prepared / name, work / name)
        env = {**os.environ, "OMP_NUM_THREADS": str(execution["omp_threads"]),
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        timed_out, returncode, error = False, None, None
        start = time.perf_counter()
        process = None
        with (work / "input").open("rb") as stdin, (attempt / "stdout.log").open("wb") as stdout, (attempt / "stderr.log").open("wb") as stderr:
            try:
                process = subprocess.Popen(plan["command"], cwd=work, env=env, stdin=stdin,
                                           stdout=stdout, stderr=stderr, start_new_session=True)
                try:
                    returncode = process.wait(timeout=execution["timeout_s"])
                except subprocess.TimeoutExpired:
                    timed_out = True
            except OSError as exception:
                error = str(exception)
            finally:
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    if returncode is None:
                        returncode = process.wait()
        text = (attempt / "stdout.log").read_text(errors="replace") + (attempt / "stderr.log").read_text(errors="replace")
        status = "launch_error" if error else classify(returncode, timed_out, text)
        completed_steps = len(re.findall(r"W_mag,_kin\s*=", text))
        health = None
        if status == "completed":
            try:
                for filename in config["acceptance"]["required_files"]:
                    if not (work / filename).is_file() or (work / filename).stat().st_size == 0:
                        raise ValueError(f"missing/empty required output: {filename}")
                health = restart_health(work / "jorek_restart.h5", expected_steps=config["evolution"]["nstep_n"],
                                        final_time_s=config["acceptance"]["final_time_s"])
                regression = config["acceptance"].get("equilibrium_regression")
                if regression:
                    health["equilibrium_regression"] = equilibrium_regression(
                        work / "jorek_restart.h5", prepared / "equilibrium_reference.h5", **regression)
            except (OSError, KeyError, TypeError, ValueError) as exception:
                status, error = "incomplete_output", str(exception)
        record = {"attempt": index, "status": status, "returncode": returncode, "error": error,
                  "wall_s": time.perf_counter() - start, "steps_logged": completed_steps, "health": health}
        write_json(attempt / "attempt.json", record)
        attempts.append(record)
        if status != "rtree_assertion" or completed_steps:
            break
    wall = sum(item["wall_s"] for item in attempts)
    result = {"case_id": config["case_id"], "parent_family_id": config["parent_family_id"],
              "config_hash": plan["config_hash"], "status": attempts[-1]["status"], "attempts": attempts,
              "physical_acceptance": False, "gate_A_passed": False, "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
              "modules": os.environ.get("LOADEDMODULES"), "wall_s": wall,
              "runner_sha256": sha256_file(__file__),
              "allocated_cpus": os.environ.get("SLURM_JOB_CPUS_PER_NODE"),
              "process_core_hours_estimate": wall * execution["mpi_ranks"] * execution["omp_threads"] / 3600,
              "billable_usage": "pending scheduler accounting; process estimates exclude node allocation overhead",
              "cpu_core_hours": wall * execution["mpi_ranks"] * execution["omp_threads"] / 3600,
              "raw_bytes": sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())}
    write_json(destination / "result.json", result)
    return result
