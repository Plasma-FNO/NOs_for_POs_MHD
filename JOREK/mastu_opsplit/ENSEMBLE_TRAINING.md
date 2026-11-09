# MAST-U 80/20 ensemble training

The 6 November 2026 user revision supersedes the handoff's 60/20/20 training
protocol: the original 60 training and 20 validation trajectories form 80 training
trajectories. The original 20 test trajectories remain held out. The accepted
dataset and its manifest are unchanged. There is no validation set, early stopping,
hyperparameter search, or test-based checkpoint selection. The selected checkpoint
is the final update of the prescribed budget, even if an earlier checkpoint would
score better on test trajectories.

`scripts/train_ensemble.py` is a separate entry point. The chronological pilot and
the older physical-RHS experiment retain their existing entry points and behavior.

## Data and physics

Every process runs the full `EulerDataset` hash/identity/geometry/time verification.
It also hashes every initial restart, verifies common native geometry and time
normalization, checks finite coefficients and fixed potential boundary nodal values
in all 9,700 frames, and verifies native consistent-mass projection.

Normalization combines the verified original 60-trajectory population moments
with moments fitted to the former 20 validation trajectories, including the
between-population mean correction to variance. Only these 7,760 training frames
contribute. The resulting native coefficient mean and volume-weighted field scales
are saved in `normalization.npz`, with training IDs and content hashes in provenance.
Centering stays float64; only neural evaluation converts to float32. Four-step
stride-one windows number 7,440 training and 1,860 test windows. Sampling never
crosses trajectories or changes test membership.

The physical background and positivity anchor both use the new 80-trajectory mean.
The continuity multiplier is the effective observation interval divided by verified
native `sqrt_mu0_rho0`, approximately 0.03125 for stride one and 0.0625 for stride
two. Euler substeps divide pseudotime only; the physical multiplier is not applied
twice. All models share native elliptic closure, fixed potential boundary nodal
values, subdivided Bernstein positivity, and the same negative-proposal penalty.

## Matching, checkpoints, and audits

Seeds 17, 18, and 19 each initialize identical neural weights across AR, NODE, and
Physical OpsSplit. All use the same 16x32 grid, four modes, width eight, two layers,
Adam at 0.001, gradient clipping at two, four-step autonomous training loss, batch
one, and update budget. Architecture and initialization hashes, sampled case/start
pairs, optimizer settings, source hashes, split membership, and physical interval
are recorded. Perturbation metadata is not an additional conditioning input.

`last.pt` atomically stores neural weights, Adam state, all RNG states, immutable
sampling identity, completed update, and history after every update. Resume checks
provenance and continues at the next window. `selected.pt` contains the final
fixed-budget neural checkpoint. Reconstruct fixed native operators and guards from
the verified dataset and saved normalization; they have no additional trained
parameters. Slurm scripts lock each output to prevent simultaneous resume writers.

Evaluation streams each original test trajectory from frame zero. The 16-, 32-,
and 96-step horizons cover approximately 0.234319, 0.468638, and 1.405913 microseconds
at stride one. NODE and OpsSplit also run fixed-weight two- and four-substep
versions over exactly these durations. Each completed trajectory/substep report is
atomic and reused on resume. Change stride only with compatible horizons, a fresh
configuration, and a fresh output snapshot.

Reports include aggregate and per-field normalized volume MSE, decoded absolute
and relative RMS, dynamic-relative RMS, physical duration, whole-cell Bernstein
bounds, independent Q12 and vertex minima, weak interior current/vorticity defects,
boundary drift, raw proposed minima, limiter counts, and retained fractions.
Dynamic denominators use each spatial point's temporal variance of future truth
frames; decoded denominators use future physical truth squared. Both denominator
floors are 1e-30. Positivity and constraint failures produce a failed audit, and
numerical exceptions are recorded per trajectory. The CLI exits nonzero if audits
fail. This does not assert full MHD conservation or independent equilibria.

`summarize_ensemble.py` verifies matching within each seed and reports seed means,
seed standard deviations, and within-seed trajectory variation. Overlapping windows
are not treated as independent uncertainty samples. Failed outcomes remain visible.

Each horizon/substep aggregate records `expected_trajectories`, `audit_failures`,
and `complete`. Primary means and standard deviations are withheld unless every
expected trajectory reaches that horizon with finite scores and passing audits.
A later numerical failure retains earlier completed horizons, including on resume;
the overall evaluation still fails. Empty evaluations never pass. Seed summaries
require a complete horizon in all three seeds. Legacy reports lacking this flag
must pass their global audit before contributing a score.

`scripts/diagnose_ensemble_linear.py --run RUN --output FRESH_DIRECTORY` compares
learned spline derivatives against exact native bicubic derivatives at fixed seed-17
weights. It defaults to the final `selected.pt`; `--checkpoint last` is available
for explicitly labeled intermediate diagnostics. It verifies frozen training source
hashes, checkpoint provenance, and normalization, then samples sixteen training
windows from eight trajectories. Autonomous rollouts compare learned gradients,
native gradients, and no linear branch, with separate guard statistics. The report
contains per-field derivative errors and future-frame volume MSE. This is a fixed
weight diagnostic; it does not establish the benefit of retraining with native
gradients or performance on held-out trajectories.

## Matched native versus learned gradient retraining

`slurm/gradients.sbatch` uses the same ensemble configuration for six OpsSplit
runs: seeds 17, 18, and 19, each with `lspline_continuity` (learned scatter/spline
gradients) and `native_continuity` (exact native bicubic gradients). Only the
derivative calculation changes. Neural initialization, architecture, deterministic
sampled windows, 2,000-update budget, optimizer, four-step loss, normalization,
physical tangent, and guards stay matched. Both variants train from scratch and
select the final update. The native branch remains differentiable through the
current state; it adds no trainable parameters.

Freeze a fresh snapshot with `scripts/freeze_ensemble.py`, then use
`study/slurm/gradients.sbatch` instead of the generic ensemble script. Submit a
two-task smoke array `0-1%2` with `SMOKE=1` on the debug partition first. After its
matching and audits pass, submit `0-5%3` with `SMOKE=0` on the production partition.
Task pairs 0/1, 2/3, and 4/5 are seeds 17, 18, and 19; each pair is ordered learned,
native. Outputs are `seedSEED_VARIANT/opsplit`. Resume uses the same frozen source,
configuration, schedule, optimizer, RNG state, and cached trajectory reports.

Both variants evaluate the original 20 held-out trajectories from frame zero at
16, 32, and 96 steps, with 1/2/4 substeps and all existing physical audits.
`scripts/summarize_gradients.py --root RUN --output RUN/comparison.json` checks
matching provenance (allowing only the derivative branch to differ), initial
weights, parameter counts, normalization contents, and every sampled window.
It pairs held-out cases within each seed, reports per-field MSE, mean differences,
within-seed paired trajectory variation, and variation across the three seed
means. Incomplete or invalid horizons cannot produce primary comparison scores.
Schedule this command after the six training/evaluation tasks with Slurm
`afterany` so failures also produce an explicit incomplete or failed-audit report.

## Reproducible Slurm use

Use the existing Python interpreter:

```bash
PYTHON=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
STUDY=/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit
"$PYTHON" "$STUDY/scripts/freeze_ensemble.py" --output /absolute/fresh/run_directory
```

The command creates read-only source/config files and prints two submission
commands. The first runs a bounded, three-update matched smoke test. The second
prepares array tasks 0–8, ordered seed then AR/NODE/OpsSplit, with at most three
concurrent jobs. Do not submit that longer array before reviewing smoke results,
runtime estimates, and the prescribed budget. The default is 2,000 updates per
model and seed; this is a controlled comparison budget, not a convergence claim.

Resources: account `fupa2_fauna`, four allocated CPUs, two Torch/OMP threads,
one OpenBLAS thread, 24 GB RAM. The debug partition has a 30-minute limit; production
has a 24-hour limit, checked on 6 November 2026. The scripts disable bytecode writes
and verify snapshot hashes before execution. Dependency commit/source pins remain
in `configs/lsplinefno.yaml` and are verified by the model adapter.

To resume a timed-out or failed task, resubmit the same snapshot/export command with
`--array=TASK_ID`. The script always passes `--resume`. Completed optimizer updates
and evaluation reports are retained. Check `sacct` state and exit code; submission
alone is not success. A repeatable numerical failure requires a newly reviewed
experiment, not silently changing the immutable run. Shared scratch data still
requires separate archival retention.

After all nine tasks finish:

```bash
"$PYTHON" /absolute/fresh/run_directory/study/scripts/summarize_ensemble.py \
  --root /absolute/fresh/run_directory \
  --output /absolute/fresh/run_directory/comparison.json
```

The smoke configuration deliberately restricts evaluation to two test trajectories
and four steps. It validates integration and measures throughput; it does not
establish long-rollout stability, convergence, or model superiority. Test results
must not be used to tune architecture or training budget in this no-validation
protocol.

## Coefficient-aware seed-17 run

Freeze a fresh experiment as above, but do not use the generic array commands
printed by the freezer. Set `OUTPUT_ROOT` to that fresh absolute directory and
`STUDY_ROOT=$OUTPUT_ROOT/study`. Use the dedicated launcher:

```bash
sbatch --partition=dcgp_fua_dbg --time=00:30:00 \
  --export=ALL,STUDY_ROOT=$STUDY_ROOT,OUTPUT_ROOT=$OUTPUT_ROOT,SMOKE=1 \
  --output=$OUTPUT_ROOT/smoke-%j.log $STUDY_ROOT/slurm/density_diffusion.sbatch
sbatch --export=ALL,STUDY_ROOT=$STUDY_ROOT,OUTPUT_ROOT=$OUTPUT_ROOT,SMOKE=0 \
  --output=$OUTPUT_ROOT/train-%j.log $STUDY_ROOT/slurm/density_diffusion.sbatch
```

Review the completed smoke before submitting production. Production uses
`ensemble_density_diffusion.yaml`, `lsplinefno.yaml`, seed 17, and only OpsSplit.
It starts fresh in `seed17_opsplit`; `--resume` only continues that directory.
Never copy a continuity-only checkpoint into it. The physical density-diffusion
split uses analytic spline derivatives, the training physical mean and scale,
and the native observation interval. Baseline coefficient input is 1. Training
still pools the original 60 training and 20 validation trajectories into 80
training cases, with 20 untouched test cases and no validation selection.
The budget remains 2,000 Adam updates on four-interval windows; final-checkpoint
evaluation uses 16/32/96 intervals and 1/2/4 Euler substeps with the same guards.

OOD evaluation is separate and resumable. Set `OOD_ROOT` to the ten-case
`mastu_coefficient_ood10_20261106` campaign and `TRAIN_JOB_ID` to the new production
job. The campaign packaging job is 2794493; its parent simulation array is 2794492.

```bash
sbatch --dependency=afterok:$TRAIN_JOB_ID:2794493 --kill-on-invalid-dep=yes \
  --export=ALL,STUDY_ROOT=$STUDY_ROOT,OUTPUT_ROOT=$OUTPUT_ROOT,OOD_ROOT=$OOD_ROOT \
  --output=$OUTPUT_ROOT/ood-%j.log $STUDY_ROOT/slurm/density_diffusion_ood.sbatch
```

The evaluator requires `ood_dataset_ready`, verifies the manifest hash and all
ten test-only cases, and loads `CoefficientOODDataset` with the run's saved
`normalization.npz` and SHA-256. It also checks that normalization against the
training provenance and reverified 80-case statistics. No OOD normalization is
fitted and no OOD cases train the model. Evaluation uses the same final checkpoint,
observation times, 16/32/96 horizons, 1/2/4 substeps, positivity and constraint
audits, writing resumable reports into `seed17_opsplit_ood`. Resubmit the same
frozen scripts/exports to resume; a failed dependency needs review before retry.
