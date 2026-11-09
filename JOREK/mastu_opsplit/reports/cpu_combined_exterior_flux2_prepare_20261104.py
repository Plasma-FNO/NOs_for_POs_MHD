"""Probe exterior flux-surface refinement to cover all spatial directions with frozen physics."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import sha256_file

B = Path("/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102")
parent = B / "combined_refined_solver_20261103"
probe = parent / "control3"
base = B / "combined_exterior_flux2_20261104"
record_path = R / "cpu_combined_exterior_flux2_20261104.json"
if base.exists() or record_path.exists():
    raise ValueError("preserve existing attempts")
evidence = json.loads((probe / "result.json").read_text())
binary = parent / "source/jorek_model303"
if evidence["status"] != "completed" or sha256_file(binary) != evidence["binary_sha256"]:
    raise ValueError("healthy same-binary original control proof required")
for moment, name in [("initial", "jorek000000.h5"), ("final", "jorek_restart.h5")]:
    if sha256_file(probe / name) != evidence[moment]["restart_sha256"]:
        raise ValueError("qualified original control output changed")
original = json.loads((parent / "manifest.json").read_text())
for name, digest in original["input_hashes"].items():
    if sha256_file(name) != digest:
        raise ValueError("qualified source/input changed: " + name)
row = next(line for line in subprocess.check_output(["saldo", "-b", "--dcgp", "-n"], text=True).splitlines()
           if line.split()[0] == "FUPA2_FAUNA")
cols = row.split()
if min(int(cols[3])-int(cols[4]), int(cols[7])-int(cols[8])) < 768:
    raise ValueError("insufficient current credit for the capped probe")
(base / "source").mkdir(parents=True)
shutil.copyfile(binary, base / "source/jorek_model303")
(base / "source/jorek_model303").chmod(0o555)
for name in ["worker.py", "jorek-env.sh"]:
    shutil.copyfile(parent / name, base / name)
work = base / "probe3"
work.mkdir()
(work / "input").write_text(namelist_overrides((probe / "input").read_text(), {"n_open": 8, "n_outer": 64, "n_inner": 16, "n_private": 16, "n_up_priv": 16}))
for name in ["eqdsk.dat", "jorek_equil_rz.h5"]:
    shutil.copyfile(probe / name, work / name)
budget = {"authorization": "User requested continued solver convergence and science-plan progress.",
          "scope": "One three-step exterior_flux2 probe with the unchanged combined Newton/MUMPS executable; no physical or acceptance changes.",
          "mpi_ranks": 4, "memory_GiB": 256, "wall_hours_max": 3, "rank_CPU_hours_max": 12, "saldo_row": row}
files = [base / "source/jorek_model303", base / "worker.py", base / "jorek-env.sh",
         probe / "result.json", *work.iterdir()]
manifest = {key: original[key] for key in ["final_time_s", "tstep_code_units", "field_relative_target",
                                           "integral_relative_target", "equil_accuracy"]}
manifest.update(stages={"probe3": {"steps": 3, "amplitudes": [0., 0.], "timeout_seconds": 10000}},
                validated_unchanged_binary_sha256=evidence["binary_sha256"], budget=budget,
                input_hashes={str(p): sha256_file(p) for p in files},
                limitation="Probe qualification only. Only exterior/open/private radial flux counts are doubled; closed-core, poloidal, leg-length and wall counts retained. A healthy matched full pair is required for directional spatial acceptance.")
(base / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
batch = R / "cpu_combined_exterior_flux2_20261104.sbatch"
batch.write_text(f"""#!/bin/bash
#SBATCH --job-name=mastu-combined-exterior_flux2
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=03:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {base}/worker.py {base} {ROOT} probe3
""")
subprocess.run(["bash", "-n", str(batch)], check=True)
command = ["sbatch", "--parsable", "--chdir", str(base), "--output", str(base / "batch-%j.out"),
           "--error", str(base / "batch-%j.err"), str(batch)]
preflight = subprocess.run(["sbatch", "--test-only", *command[1:]], capture_output=True, text=True, check=True)
record = {"status": "prepared", "kind": "simulation", "bulk_output": str(base), "budget": budget,
          "binary_sha256": evidence["binary_sha256"], "manifest_sha256": sha256_file(base / "manifest.json"),
          "script_sha256": sha256_file(batch), "scheduler_preflight": preflight.stderr, "scientific_gates_passed": []}
record_path.write_text(json.dumps(record, indent=2) + "\n")
record.update(status="submitted", job_id=subprocess.check_output(command, text=True).strip().split(";")[0])
record_path.write_text(json.dumps(record, indent=2) + "\n")
print(record["job_id"])
