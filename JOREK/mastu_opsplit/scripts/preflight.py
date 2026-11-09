#!/usr/bin/env python3
"""Audit local case/dependency provenance without launching scientific work."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mastu_opsplit.common import config_hash, load_config, sha256_file, write_json


def git(directory, *arguments):
    result = subprocess.run(["git", "-C", str(directory), *arguments], text=True,
                            capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError(result.stderr.strip())
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Case YAML")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    case, model = load_config(args.config), load_config(args.model_config)
    if args.dry_run:
        print(json.dumps({"action": "inspect_paths_hashes_and_versions", "output": args.output,
                          "simulation": False, "training": False, "writes": False}))
        return
    jorek = case["jorek"]
    root = Path(model["dependency"]["path"])
    assets = {"executable": jorek["executable"], "runtime_environment": jorek["runtime_environment"],
              "input": str(Path(jorek["input_directory"]) / "input"),
              "equilibrium": str(Path(jorek["input_directory"]) / "eqdsk.dat")}
    files = {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in assets.items()}
    source = jorek["source"]
    jorek_commit, cfo_commit = git(source, "rev-parse", "HEAD"), git(root, "rev-parse", "HEAD")
    if jorek_commit != jorek["pinned_commit"] or cfo_commit != model["dependency"]["commit"]:
        raise ValueError("source revision differs from pinned case/model configuration")
    dependency_files = ("Models/learnable_spline_fno.py", "Models/factory.py",
                        "Models/conditioning/film.py", "Models/region_routing.py")
    packages = {}
    for name in ("numpy", "torch", "h5py", "PyYAML", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"case_config_hash": config_hash(case), "model_config_hash": config_hash(model),
              "assets": files, "jorek_commit": jorek_commit, "jorek_dirty": git(source, "status", "--short"),
              "dependency": {"path": str(root), "commit": cfo_commit,
                             "status": git(root, "status", "--short"),
                             "tracked_patch": git(root, "diff", "--binary", "HEAD"),
                             "source_sha256": {name: sha256_file(root/name) for name in dependency_files}},
              "python_executable": sys.executable, "python_version": platform.python_version(),
              "packages": packages, "scheduler": {name: shutil.which(name) for name in ("sbatch", "squeue", "sacct")},
              "disk": {"home_free_bytes": shutil.disk_usage(Path.home()).free,
                       "workspace_free_bytes": shutil.disk_usage(Path(__file__).resolve().parent).free},
              "gates": {"A": "pending", "B": "pending", "C": "pending"},
              "executed_scientific_jobs": 0}
    write_json(args.output, report)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error))
