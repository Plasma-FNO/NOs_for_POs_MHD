"""Plot qualified native GS histories; spatial acceptance remains separate."""
from pathlib import Path
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
B = Path("/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102")
qualification = json.loads((R / "cpu_combined_refined_qualification_20261103.json").read_text())
if qualification["status"] != "combined_numerical_probes_verified":
    raise ValueError("completed combined qualification required")


def history(folder):
    samples = re.findall(r"Iteration\s+(\d+): diff=\s*(\S+)", (folder / "stdout.log").read_text())
    return list(zip(*[(int(i), float(v)) for i, v in samples]))


fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
reference = history(B / "tight_exact_picard_reference_20261103/picard1")
axes[0].semilogy(*reference, color="0.5", label="Tight Picard reference", linewidth=1.8)
for stage, label, color in [("control1", "Original mesh, tight Newton", "tab:blue"),
                            ("radial160", "Radial160", "tab:orange"),
                            ("wall80", "Wall80", "tab:green"),
                            ("core_poloidal440", "Core-poloidal440", "tab:purple")]:
    values = history(B / "combined_refined_solver_20261103" / stage)
    for ax in axes:
        ax.semilogy(*values, color=color, label=f"{label}: {values[0][-1]} iterations", linewidth=1.8)
    if qualification["stages"][stage]["cycle_fallback_activated"]:
        axes[1].plot(values[0][-1], values[1][-1], "o", color=color, markersize=5)

for ax in axes:
    ax.axhline(1e-10, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Equilibrium iteration", ylabel="Native mean absolute raw flux update")
    ax.grid(alpha=.2)
axes[0].set_title("Independent reference and combined solver")
axes[1].set_title("Newton detail; dashed line = 1e-10 requirement")
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8)
fig.suptitle("Equilibrium convergence verified; full spatial acceptance remains separate", fontsize=11)
for suffix in ["png", "pdf"]:
    fig.savefig(R / f"cpu_combined_solver_convergence_20261103.{suffix}", dpi=160)
