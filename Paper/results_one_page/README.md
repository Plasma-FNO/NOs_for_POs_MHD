# MAST-U OpsSplit methods and interim results (two pages)

Open `results_one_page.pdf`. The file name is kept from the original one-page record so existing links still work. It is the MAST-U companion to the blob (ES-to-EM) report in `../jorek_transfer_one_page/`, with the same letter-paper, two-column layout.

The report covers the 80/0/20 perturbation ensemble on one MAST-U equilibrium:

- **Page 1:** headline, figure (native mesh; held-out and D⊥-OOD error vs rollout length), the case and split, the shared protocol, and the start of Section 3.
- **Section 3, how each method is conducted:**
  - AR
  - Euler NODE
  - explicit physical OpsSplit (frozen ρ/T advection–compression tangent), with its native-gradient and density-diffusion variants
  - implicit native-tangent OpsSplit and its three arms (labels + rollout, rollout only, physics only)
  - coefficient physics for the OOD set
- **Page 2:** Table 1 with every method's test and OOD scores, interim results and caveats.

The previous 2 November pre-pilot page (L-SplineFNO density/pressure replacement plan, gates A/B/C) is archived unchanged in `archive_prepilot_20261102/`, with its old builder.

## Evidence

`results.json` is a frozen snapshot. `build.py --refresh` creates it as follows:

1. Run the study's own `JOREK/mastu_opsplit/scripts/summarize_implicit.py` over the live campaign folders. Incomplete or failed-audit horizons are never averaged, and the snapshot keeps that rule.
2. Add the explicit-OpsSplit variants: the native-gradient study `comparison.json` and the seed-17 density-diffusion run.
3. Add the native coefficient check, label-pretraining losses, dataset facts and parameter counts.
4. Record the SHA-256 of every evaluation file read (about 1,070), the summarizer, the dataset manifests, geometry and checkpoints.

Evaluation files are written once per trajectory, so the refresh reruns the summarizer until no file appears or changes while it runs. A plain build re-verifies every hash and renders only the snapshot. It never submits jobs or queries Slurm.

Roots are fixed in `build.py`:
- implicit campaign: `/pitagora_scratch/userexternal/vgopakum/mastu_implicit_opsplit_20261108/{campaign_v2,ood,coefficient}`
- ensemble: `/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_ensemble80_comparison_20261106`
- gradient study: `/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_gradient_retraining_20261106`
- density-diffusion run: `/pitagora_scratch/userexternal/vgopakum/mastu_density_diffusion_seed17_20261106T163039Z`

The snapshot of 9 November 2026, 11:54 UTC, includes the completed physics-only test arm. It was taken while these jobs were still running:
- implicit rollout-only seeds 18/19: training finished, test evaluation in progress
- the fixed-conditioning OOD runs for rollout-only seeds 18/19 and physics-only
- the coefficient-physics OOD runs for rollout-only seeds 18/19

Their cells read `run.`/`pend.` and fill in on the next refresh. AR and NODE have only seed 17 (seeds 18/19 of array 2790285 were cancelled). As in `summarize_implicit.py`, explicit OpsSplit seeds 18/19 come from the gradient study's learned-gradient arm.

## Training times

`Train (h)` is the summed wall time of the optimizer updates in each run's `history.json` (`training_seconds`, or `seconds` for label pretraining), averaged over seeds whose training finished all 2,000 updates per stage. It is taken from the same runs as the score columns. It excludes data verification, operator setup (LU factorization) and evaluation. Every history file used is hashed in the snapshot.

The explicit families ran on 4 CPUs with 2 Torch threads. The implicit arms ran on 8 CPUs with 4 threads, so the times are not directly comparable. Labels + rollout also depends on a one-off export of 2,000 native labels, about 8.8 h summed (`labels/summary.json`), run in parallel and shared by all three seeds. Physics-only trains nothing. The D⊥ rows reuse the trained checkpoints.

## Prose guards

The headline and Section 4 make claims that later results could overturn. The build refuses to publish when:

- any paired implicit-vs-NODE test comparison at 96 intervals does not improve (the headline says all do);
- a completed physics-only result (test, or coefficient-physics OOD) differs from the trained implicit arms by more than 2× at any horizon (the headline says physics-only matches them).

If either fires after a refresh, revise the text rather than the check. Numbers in the text, such as gains, shares, substep and gradient effects and parameter counts, come from the snapshot.

## Build and test

```bash
cd /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD
PY=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/opsplit-paper-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY Paper/results_one_page/build.py --refresh   # freeze new evidence, then build
$PY Paper/results_one_page/build.py             # rebuild from the frozen snapshot
cd Paper/results_one_page && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest -q test_build.py
```

Two pages is a hard limit (`PAGE_LIMIT`). LaTeX and an independent `pdfinfo` check enforce it, and overfull boxes reject the build. The previous PDF is replaced only after these checks pass. Shorten the text rather than shrinking it. Table 1 is defined on page 1 so it floats to the top of page 2; defining it later pushes it onto a third page. The build refuses to write over the original paper, the blob report or the source.

`sha256` and `page_count` in `build.py` are also used by `../jorek_transfer_one_page/build.py`.
