# One-page JOREK experiment record

`results_one_page.pdf` now covers the current **MAST-U/model303 OpsSplit experiment**. The original paper supplies typography and method context only. Its Navier–Stokes benchmark scores have been removed.

The page contains the pinned upstream equilibrium nodes, the reduced-MHD balances, explicit density/pressure learning targets, the proposed neural replacement and native assembly equations, plus compact implementation and pending-comparison status. The equilibrium figure is not a new run, and fixture tests/build success are not physical-accuracy measurements. Gate A/B/C status stays explicit.

The editable page is `results_one_page.tex`. `results.json` names the study evidence files and the pinned reference snapshot. The build reads those reports, verifies the reference and pinned model303 equation-source hashes against the immutable manifest, generates the geometry figure and fills the evidence/status values. The local equation audit is also fingerprinted. It never submits jobs or runs JOREK. Scheduler status is the recorded snapshot, not a live query; refresh the study report before rebuilding when status changes.

Build from any directory:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/opsplit-paper-matplotlib \
  /pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python \
  /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/Paper/results_one_page/build.py
```

Requires the existing Matplotlib/h5py environment, `pdflatex`, `pdfinfo`, the study reports and local reference snapshot named in `results.json`. Keep the original paper at the project root as a style-source fingerprint. The `.build.json` report records PDF/source/config/evidence hashes and the geometry source. Missing or changed reference evidence fails instead of substituting an illustrative mesh.

**One page is a hard limit.** LaTeX rejects a second shipped page. The build independently checks actual PDF page count and rejects overfull layout boxes. It replaces the delivered PDF only after verification. Overflow preserves the last valid PDF; shorten or replace content instead of shrinking the body font.

When scientific results become available, replace the pending-comparison status with accepted JOREK metrics and update the pre-pilot prose/evidence contract together. `scientific_comparison_metrics: null` currently documents that those results do not exist. The current template rejects a non-null entry until that explicit update is made, so newly supplied metrics cannot be silently ignored beneath a pending table.

Run the cap and content checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/opsplit-paper-matplotlib \
  /pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python -m pytest -q \
  /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/Paper/results_one_page/test_build.py
```

Checks exercise both overflow guards, preserve previous output, protect the original paper/source, and reject a stale published-benchmark artifact in place of the JOREK page. The original paper remains a separate source document.

## MHD-to-neural mapping

Equation (1) uses the model303 field ansatz and selected density, pressure, induction, current and vorticity equations for axisymmetry and the configured zero-diamagnetic setting. Equation (2) gives the exact candidate density/pressure advection-compression blocks, retaining cylindrical and poloidal parallel-flow terms. Equation (3) replaces only these two scalar blocks with independent L-SplineFNOs. Equation (4) injects their native weak assemblies into the coupled temporal/constraint solve. Pressure is p=rho*T; its RHS is not a standalone T tendency. Other MHD equations and nonlinear transport remain numerical, including timestep-dependent stabilization in the native assembly.

The source audit is `JOREK/mastu_opsplit/reports/equations.md`; the immutable `source_residual.f90` copy is included in build provenance. An independent 128-state algebra check verifies both displayed targets against the source-derived implementation; this is not physical-rollout validation. The MHD formulas and the replacement remain a proposed partial experiment until the numerical gates pass.
