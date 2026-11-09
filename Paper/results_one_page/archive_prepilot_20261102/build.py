#!/usr/bin/env python3
"""Build the results companion; publish only a verified single-page PDF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_evidence():
    config = json.loads((HERE / "results.json").read_text())
    if config["scope"] != "current_jorek_experiment" or config["scientific_comparison_metrics"] is not None:
        raise ValueError("This pre-pilot page requires current JOREK evidence, with scientific metrics unmeasured")
    study = ROOT / config["study_root"]
    paths = {key: study / config[key] for key in ("validation", "initializer_build", "regression_status", "gates")}
    evidence = {key: json.loads(path.read_text()) for key, path in paths.items()}
    snapshot = study / config["reference_snapshot"]
    manifest = json.loads((snapshot / "case.json").read_text())
    reference = snapshot / "equilibrium_reference.h5"
    if sha256(reference) != manifest["snapshot_hashes"]["equilibrium_reference.h5"]:
        raise ValueError("Pinned equilibrium reference changed")
    evidence["reference"] = reference
    evidence["source_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in paths.values()}
    evidence["source_sha256"][str(reference.relative_to(ROOT))] = sha256(reference)
    audit = study / config["equation_audit"]
    equation_source = snapshot / "source_residual.f90"
    if sha256(equation_source) != manifest["snapshot_hashes"]["source_residual.f90"]:
        raise ValueError("Pinned equation source changed")
    for path in (audit, equation_source):
        evidence["source_sha256"][str(path.relative_to(ROOT))] = sha256(path)
    return evidence


def write_values(output, evidence, nodes):
    escape = lambda value: "".join({"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#",
                                    "$": r"\$", "{": r"\{", "}": r"\}", "\\": r"\textbackslash{}"}.get(c, c)
                                   for c in str(value))
    status = evidence["regression_status"]
    values = {
        "ReferenceNodes": nodes,
        "StudyChecks": evidence["validation"]["integrated_tests"]["passed"],
        "SplineChecks": evidence["validation"]["upstream_selected_tests"]["passed"],
        "BuildStatus": "Compiled/linked" if evidence["initializer_build"]["status"] == "compiled_and_linked" else "Not verified",
        "BuildSeconds": f'{evidence["initializer_build"]["elapsed_seconds"]:.2f}',
        "RegressionJob": status["job_id"], "RegressionState": status["state"],
        "RegressionReason": status.get("queue_reason") or "None recorded",
        "RegressionOutcome": "Equilibrium comparison passed; physical evolution remains unverified." if status["equilibrium_passed"] else "No verified reference comparison is available yet.",
        **{f"Gate{key}": evidence["gates"][key]["status"] for key in ("A", "B", "C")},
    }
    output.write_text("\n".join("\\newcommand{\\" + name + "}{" + escape(value) + "}" for name, value in values.items()) + "\n")


def make_figure(output, evidence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    from matplotlib.patches import FancyBboxPatch

    with h5py.File(evidence["reference"], "r") as handle:
        if int(handle["jorek_model"][()].item()) != 303 or int(handle["n_tor"][()].item()) != 1:
            raise ValueError("Expected the pinned axisymmetric model303 equilibrium")
        coordinates = handle["x"][:, 0, 0, :]
        fields = handle["values"][:, 0, 0, :]
        nodes = int(handle["n_nodes"][()].item())
    if coordinates.shape != (2, nodes) or fields.shape != (7, nodes) or not np.isfinite(coordinates).all() or not np.isfinite(fields).all():
        raise ValueError("Invalid native equilibrium node layout")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"],
                         "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
                         "font.size": 9, "axes.linewidth": .5, "pdf.fonttype": 42, "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.12), gridspec_kw={"width_ratios": [1, 1.5]})
    axis = axes[0]
    points = axis.scatter(*coordinates, c=fields[4]*fields[5], s=1.8, cmap="viridis", linewidths=0)
    axis.set_aspect("equal")
    axis.set_xlabel("R (m)", fontsize=9, labelpad=1)
    axis.set_ylabel("Z (m)", fontsize=9, labelpad=1)
    axis.set_title("(a) Pinned equilibrium nodes", fontsize=10, pad=5)
    axis.tick_params(width=.5, length=2, labelsize=8)
    colorbar = fig.colorbar(points, ax=axis, pad=.03, fraction=.05)
    colorbar.set_label(r"$\rho T$ (code units)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7, length=2, width=.5)
    axis = axes[1]
    axis.set_axis_off()
    axis.set_title("(b) Proposed hybrid update", fontsize=10, pad=5)
    labels = ["Current native FE state q\nSeven coupled fields", "Two L-SplineFNO modules\nDensity / pressure terms",
              "Quadrature weak assembly\n+ retained numerical residuals", "Temporal / constraint solve\nNext native state"]
    centers = [.85, .62, .39, .16]
    for index, (label, y, color) in enumerate(zip(labels, centers, ["#51829B", "#78ABA8", "#DA6C6C", "#51829B"])):
        axis.add_patch(FancyBboxPatch((.04, y-.095), .92, .19, boxstyle="round,pad=0.008",
                       edgecolor=color, facecolor=color+"20", linewidth=.8, linestyle="--", transform=axis.transAxes))
        axis.text(.5, y, label, ha="center", va="center", fontsize=8, transform=axis.transAxes)
        if index < 3:
            axis.annotate("", xy=(.5, centers[index+1]+.1), xytext=(.5, y-.1),
                          arrowprops={"arrowstyle": "->", "lw": .7, "mutation_scale": 5, "shrinkA": 0, "shrinkB": 0}, xycoords="axes fraction")
    fig.subplots_adjust(left=.08, right=.99, bottom=.19, top=.85, wspace=.28)
    fig.savefig(output, format="pdf")
    plt.close(fig)
    return nodes


def page_count(path):
    result = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True,
                            check=True, timeout=30)
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.M)
    if match is None:
        raise RuntimeError("Could not verify PDF page count")
    return int(match[1])


def build(source, output):
    source, output = Path(source), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = load_evidence()
    # Compile beside the destination, so replacing the verified PDF is atomic.
    with tempfile.TemporaryDirectory(prefix=".onepage-", dir=output.parent) as directory:
        work = Path(directory)
        nodes = make_figure(work / "results_figure.pdf", evidence)
        write_values(work / "experiment_values.tex", evidence, nodes)
        (work / "companion.tex").write_text(source.read_text())
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                                 "-no-shell-escape", "companion.tex"], cwd=work,
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError("LaTeX failed; existing PDF preserved.\n" + result.stdout[-2500:])
        log = (work / "companion.log").read_text(errors="replace")
        if re.search(r"Overfull \\[hv]box", log):
            raise RuntimeError("Layout overflow; shorten the content. Existing PDF preserved.")
        pdf = work / "companion.pdf"
        pages = page_count(pdf)
        if pages != 1:
            raise RuntimeError(f"ONE PAGE LIMIT: generated {pages} pages; existing PDF preserved")
        os.replace(pdf, output)
    return {"pages": 1, "pdf_sha256": sha256(output), "tex_sha256": sha256(source),
            "results_sha256": sha256(HERE / "results.json"), "scope": "current_jorek_experiment",
            "evidence_sha256": evidence["source_sha256"], "reference_nodes": nodes,
            "scientific_comparison_metrics": None, "regression_state": evidence["regression_status"]["state"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results_one_page.pdf")
    args = parser.parse_args()
    original = HERE.parents[1] / "2602.23113v2.pdf"
    if args.output.suffix.lower() != ".pdf" or args.output.resolve() == original.resolve():
        parser.error("Output must be a separate .pdf file; the original paper is protected")
    original_hash = sha256(original)
    report = build(HERE / "results_one_page.tex", args.output)
    report.update(style_reference_sha256=original_hash, source="Current JOREK study reports and pinned upstream equilibrium; original paper is style/method reference only")
    args.output.with_suffix(".build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Built {args.output}: exactly one page")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error))
