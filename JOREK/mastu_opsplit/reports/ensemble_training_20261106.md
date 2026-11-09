# Ensemble trainer verification and launch report — 6 November 2026

Implemented AR, Euler NODE, and Physical OpsSplit ensemble training with the
user-requested revised split: **80 training / 0 validation / 20 test trajectories**.
The original validation trajectories join training; the original test trajectories
and accepted dataset files are unchanged. Select the final fixed-budget update.
No test score selects checkpoints, architecture, or the training budget.

## Frozen runnable experiment

```text
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_ensemble80_comparison_20261106
```

The `study/` snapshot contains read-only source, tests, configuration, documentation,
Slurm scripts, and `source.sha256`. The smoke run is under `smoke/`; test evidence
is in `integration_tests.log`. Implementation details are in
`JOREK/mastu_opsplit/ENSEMBLE_TRAINING.md`.

## Verification results

- All **49 focused integration and regression tests passed** on the exact frozen
  snapshot in 43.80 seconds. Coverage includes dataset split isolation, pooled
  training-only normalization, observation stride/time compatibility, exact
  interrupted/resumed training for all three families, elliptic adjoints,
  Bernstein positivity, differentiability, the continuity tangent, and matched
  multi-seed aggregation. Resume comparisons are bitwise on native rectangle
  fixtures; a physical Slurm preemption was not injected.
- Full loader verification rehashed all 100 trajectory files, geometry and original
  normalization. Additional checks rehashed all 100 initial restarts and verified
  common native mesh/time metadata and finite coefficients in all 9,700 frames.
- Effective split: 80/0/20. Training normalization uses **7,760 frames**, using
  stable pooled population moments. Four-interval windows: **7,440 / 0 / 1,860**.
- Native geometry: 46,203 DOFs; 97 frames per trajectory. Observation interval:
  `1.4644928241958522e-08 s`. Verified native time unit:
  `4.686377037426928e-07 s`. Physical multiplier:
  `0.031249999999998657`, approximately **0.03125**.
- New training-mean Bernstein lower bounds: density `3.012052819581261e-4`,
  temperature `8.929267925419277e-5`. Both anchors remain strictly positive.
- Maximum normalized potential boundary nodal drift: `2.706715025493984e-12`.
  Native mass-projection roundtrip MSE: `5.1458113506374054e-30`.
- Final snapshot smoke job **2789157 completed, exit 0:0, in 4m35s**. Data
  verification took 148.95 seconds. Earlier smoke job 2788897 also completed,
  exit 0:0, in 5m12s. Neural weights after three updates match bitwise between
  these runs for every family.
- All three families have **9,673 trainable parameters**, identical initial neural
  hashes and identical sampled case/start pairs. Final selected update: three.
- All corrected rollouts passed whole-cell Bernstein, independent Q12, vertex,
  fixed potential boundary, and current/vorticity audits. NODE and OpsSplit also
  passed two- and four-substep checks over the same four observation intervals.
  **Zero numerical failures.** Maximum relative constraint defect across all
  models/substeps: `7.305811599368147e-13`.
- No limiter interventions occurred in this tiny smoke. Negative-proposal limiter
  activation and gradients are exercised by the integration tests. Raw proposal
  minima and limiter statistics remain recorded in training and rollout reports.
- Final process peak RSS was approximately **3.86 GiB**; the earlier run peaked
  near **9.9 GiB**. Keep the prepared 24 GiB request.

## Bounded smoke scores

Three updates, four-step autonomous training, two test trajectories, start frame
zero, duration approximately 58.58 ns. These numbers check the implementation;
they do not establish a converged ranking or independent-seed superiority.

| Family | One-substep normalized MSE | Two-substep MSE | Four-substep MSE |
|---|---:|---:|---:|
| AR | 1.54424284 | not applicable | not applicable |
| NODE | 4.53928157 | 4.75150715 | 4.87670375 |
| Physical OpsSplit | 4.53929680 | 4.75152605 | 4.87672574 |

Per-field metrics, decoded/dynamic errors, denominator definitions, trajectory
variation and all audit minima are in each family's `metrics.json` and the run's
`provenance.json`. No persistence model is included.

## Longer-run budget and runtime estimate

Prepared budget: **2,000 updates per model per seed**, seeds **17, 18, 19**.
All use four-step training, batch one, Adam 0.001, clipping at two, and the same
16x32 / modes-four / width-eight / two-layer architecture. Budget selection uses
measured throughput, not test scores. This budget does not claim convergence.

Evaluation covers every original test trajectory from frame zero, with reported
16-, 32-, and 96-step prefixes. The physical durations are approximately 0.234319,
0.468638 and 1.405913 microseconds. NODE/OpsSplit also use fixed-weight 2/4-substep
rollouts over these same durations. The stream accumulates all three horizon
reports from the longest rollout; it does not rerun overlapping prefixes.

| Family | Seconds/update | Estimated hours/seed, including full evaluation | With 30% headroom |
|---|---:|---:|---:|
| AR | 4.337 | 2.80 | 3.64 |
| NODE | 4.159 | 4.29 | 5.58 |
| Physical OpsSplit | 6.300 | 6.15 | 8.00 |

Estimate = `2,000 * mean_smoke_update_seconds + 240 * smoke_evaluation_seconds +
verification_seconds`. Evaluation scaling is `(20 trajectories * 96 intervals) /
(2 trajectories * 4 intervals)`; each family's smoke already includes its required
substep settings. Approximate total: **39.7 task-hours / 159 allocated CPU-hours**,
or 51.7 task-hours / 207 CPU-hours with headroom. Queue delays are excluded.
The array caps concurrency at three; actual wall time depends on scheduling.
Long-rollout numerical stability and throughput after substantial training remain
unmeasured, so these are planning estimates.

Resources: account `fupa2_fauna`, production partition `dcgp_fua_prod`, four CPUs,
two Torch/OMP threads, one OpenBLAS thread, 24 GiB, 24-hour per-task limit. Current
scheduler limits were checked. `sbatch --test-only` accepted the exact nine-task
array request. **No longer comparison jobs have been submitted.**

## Launch and resume commands

Submit only after reviewing this report:

```bash
RUN=/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_ensemble80_comparison_20261106
sbatch --array=0-8%3 \
  --export=ALL,STUDY_ROOT="$RUN/study",OUTPUT_ROOT="$RUN",SMOKE=0 \
  --output="$RUN/ensemble-%A_%a.log" "$RUN/study/slurm/ensemble.sbatch"
```

Task IDs: 0/1/2 = seed 17 AR/NODE/OpsSplit; 3/4/5 = seed 18; 6/7/8 = seed 19.
After a failed or timed-out task has stopped, use the same command with
`--array=TASK_ID` to resume that output. The script locks the output, rechecks
source/data/provenance, restores Adam and RNG state, skips completed updates,
and reuses completed trajectory audits. Changing source/config/budget requires
a fresh frozen experiment. Check job state and exit code with:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

After all nine tasks complete:

```bash
PYTHON=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
"$PYTHON" "$RUN/study/scripts/summarize_ensemble.py" \
  --root "$RUN" --output "$RUN/comparison.json"
```

The aggregator rejects mismatched initialization, sampling, parameter counts,
provenance, and budgets within a seed. It reports variation across trajectories
and seeds without treating overlapping windows as independent samples.

Not yet checked: trained 16/32/96-step performance, long-run convergence, actual
Slurm-preemption recovery, or cross-equilibrium generalization. The original
dataset, completed pilot, and unrelated repository changes were preserved.
