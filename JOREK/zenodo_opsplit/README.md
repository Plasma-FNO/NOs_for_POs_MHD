# JOREK gridded ES → EM OpsSplit transfer

A separate, runnable experiment for the published blob datasets while MAST-U native-mesh work continues. This reuses the repository's additive RHS approach and its installed `neuralop` FNO backend. The old `Expts/operator_splitting.py::JOREK_ES_OS_RHS` is incomplete (no potential evolution, undefined `kappa`, periodic stencils); it is not used by this workflow. No MAST-U code or running solver job is changed.

Current comparison: see [Matched four-arm comparison](#matched-four-arm-comparison-2026-11-04). The older anchored native workflow and effective-split sections are historical experiments.

## Full bounded-correction transfer sweep (submitted 2026-11-05)

Following the explicit request to proceed after the pilot, training array
**2780720** contains 36 trials: seeds **0/1/2**, nested EM trajectory budgets
**8/32/64**, and **scratch / finetune / advection-transfer / diffusion-transfer**.
Controls array **2780719** evaluates the analytic solver and frozen ES source
once each; they have no EM training budget or target seed. Summary job
**2780721** depends on both arrays terminating, including failures.

Campaign root:
`/pitagora_scratch/userexternal/vgopakum/jorek-residual-transfer-sweep-20261105`.
Shared scratch holds this campaign because the work project is above its soft
storage quota; the original datasets remain on `pitagora_work`. No previous
campaign or data files were removed. `campaign.json` fixes the protocol,
cohorts, source policy and code hashes; `jobs.json` records Slurm submissions.
The repository copy is `reports/residual_transfer_sweep_submission.json`.

The source is a frozen copy of the residual pilot's **last.pt**, trained on
eight ES trajectories for 64 updates. It is **not** the ES-validation winner:
that winner has zero corrections and is represented by the analytic control.
This campaign therefore tests exploratory transfer from the fixed trained
source. Seeds vary target subsets, untransferred initial weights and window
order; they do not represent independently pretrained ES sources. Scratch
initializes zero output heads but shares the ES operator-interface scales.
The advection arm transfers scalar advection and fluid momentum; the diffusion
arm transfers particle and thermal diffusion. Magnetic coupling stays analytic.

All trained arms use the existing full native protocol: **250 epochs**, all
admissible five-step sliding windows in the first 50 frames, effective batch
20, Adam lr0.001/weight decay0.0001, StepLR50/gamma0.5, gradient clip1, and
normalized correction-L2 anchoring with weight1. The correction bound remains
0.1 times ES training term RMS. Arms within each seed/budget have identical
windows and updates; different budgets have equal epochs, not equal updates.
This extends the short pilot's training horizon from two to five steps.
Validation selects the best epoch using all admissible 49-step validation
prefixes and normalized primary-state relative L2. The initialization is not
an eligible trained checkpoint; analytic/frozen performance is reported
separately. These full-training settings differ from the pilot's short-window
optimization and five-step observed-field validation.

Each task first audits every admissible training window and all validation
prefixes, including backward through the worst training window per trajectory.
The audit now uses the same zero correction anchor as residual training.
One audit worker per GPU and activation recomputation bound memory. Training
starts only after an exhaustive passing audit with identical code, checkpoint,
data IDs and substeps. Each allocation requests one GPU for 24 hours; timed
training exits after 22 hours, checkpoints, and requeues the same task.
Audit, training and test stages use separate allocations. Training concurrency
is capped at three GPUs; controls use at most one additional GPU.

All ten EM test families are evaluated independently at horizons **10 and 100**
with GPU elliptic solves. Invalid release prefixes are reported separately
before model inference; prediction failures invalidate aggregate scores rather
than being silently dropped. `summary.json` contains each trial, paired gains
against scratch, mean/std across seeds, and missing/failed statuses. It can be
refreshed during the campaign with:

```bash
/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python \
  /pitagora_scratch/userexternal/vgopakum/jorek-residual-transfer-sweep-20261105/code/summarize_native_transfer.py \
  /pitagora_scratch/userexternal/vgopakum/jorek-residual-transfer-sweep-20261105
```

Launch validation: 44 targeted CPU tests passed, covering the solver,
selective correction transfer, nested matched cohorts, timed requeue/resume,
invalid release pressure and preservation of failures in reporting. All
generated shell scripts passed `bash -n`. At submission, GPU arrays were
pending the account's `QOSGrpNodeLimit`; no new sweep scores were available.
The older pilot recommendations below describe the decision before this
explicit launch request.

## Analytic operators with bounded learned corrections (2026-11-05)

`NativeOperators(correction_fraction=0.1)` predicts additive corrections:

\[
\mathcal O_i(q)=\mathcal O_i^{\rm analytic}(q)
  +m\,0.1\,s_i\tanh N_i(q),
\]

where `s_i` is the ES training RMS of that equation-term channel and `m` is the
existing interior mask and line-search fraction. The bound is relative to a
training characteristic scale, not the pointwise analytic term (which can be
zero). Particle/thermal corrections are coefficient-free before multiplication
by target `D_perp`/`ZK_perp`. Geometry, constraints, transport switches, implicit
diffusion, viscosity and magnetic coupling remain analytic.

Final projection layers initialize to zero; fresh models reproduce analytic
dynamics while their output layers receive gradients. The residual momentum
energy safeguard uses the complete analytic candidate as its reference. The
old force-only reference would modify analytic inertia even with zero network
output. Replacement-mode checkpoints retain their original safeguard and
semantics. Transfer rejects mixtures of replacement/correction models or unequal
correction bounds. CPU and GPU tests check zero-correction analytic recovery.

`train_native_corrections.py` trains new ES correction weights from trajectories;
old whole-operator weights are not relabelled as residuals. The old ES checkpoint
supplies only architecture and train-only scales. The bounded source pilot uses
eight ES training trajectories, two validation trajectories, four epochs of
sixteen two-step windows (64 updates), Adam LR1e-3/weight decay1e-4, and a
zero-centred correction penalty weighted one. Five-step source validation
selects the checkpoint, with the initial analytic model eligible. No EM or test
scores enter this selection. Corrections are constrained discrepancy models;
trajectory supervision plus regularization does not identify unique physical
term residuals.

`pilot_native_transfer.py` then runs the same six EM arms, seed907, budget8,
sixteen updates per trainable arm, windows, optimizer, validation/test cohorts
and five/ten-step evaluations as the replacement pilot below. In residual mode,
the term anchor becomes normalized correction L2 about zero. Source preparation
has changed (a new short rollout-pretrained residual model instead of the old
250-epoch whole-operator checkpoint), so comparisons across pilots are not an
isolated ablation of parameterization. Within the new EM pilot, arms remain
matched and scratch retains the shared ES normalization caveat.

Immutable snapshot and outputs:
`/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-residual-transfer-pilot-20261105`.
Job **2780287** runs GPU recovery tests, ES correction pretraining, and the EM
pilot sequentially, with one GPU and a one-hour cap. Submission evidence is
`reports/residual_transfer_pilot_submission.json`. Before submission, 47 targeted
CPU tests passed. Existing production snapshots and checkpoints are unchanged.

Source validation preferred the initial analytic model after all 64 ES updates:
shared error 0.04667 initially versus 0.06541/0.09083/0.08704/0.08088 after the
four epochs. The validation-selected transfer checkpoint therefore has zero
corrections. Its EM trainable arms start identically; any equality of their
scores is not evidence about transfer of learned corrections.

To measure actual learned transfer, job **2780309** evaluates a separate fixed
final-epoch ES checkpoint under the same six-arm EM protocol, saving to
`trained-source-results`. This diagnostic was selected because the validation
winner was the analytic model, before observing its EM scores. It is explicitly
not a source-validation-approved checkpoint. Both studies and their selection
rules are retained; the diagnostic does not overwrite the validation result.

Both jobs completed: **2780287 in 5m41s**, **2780309 in 3m47s**, exit 0. The
two new GPU recovery tests passed, and all 24 ten-step test rollouts completed.
The reporting check confirms identical EM cohort IDs and optimizer-update
ordering across both residual studies and the previous replacement pilot.

| Final-trained-source diagnostic | Step 5 shared error | Step 10 shared error | Step 10 magnetic perturbation error |
|---|---:|---:|---:|
| Residual scratch | 0.08860 | 0.14368 | 0.02181 |
| Full correction transfer, fine-tuned | 0.07104 | 0.11874 | 0.01959 |
| Advection/momentum correction transfer | 0.07103 | 0.11874 | 0.01961 |
| Diffusion correction transfer | 0.08801 | 0.14388 | 0.02073 |
| Frozen ES corrections | 0.07005 | 0.11412 | 0.02649 |
| Analytic equations | 0.06380 | 0.11012 | 0.02007 |

Residual scratch reduces ten-step shared error by **82.84%** relative to the
previous replacement scratch arm (0.83711). In the trained-source diagnostic,
full/advection transfer reduce residual scratch error by **17.35%**; diffusion
transfer does not help this aggregate. Frozen corrections reduce it by 20.57%,
but analytic equations remain better. Full transfer improves mainly temperature
(0.26715 to 0.16917) and potential (0.02680 to 0.02256), with slightly worse
density/vorticity. Magnetic perturbation error also improves slightly, including
relative to the analytic control, although the shared-field aggregate does not.

The validation-selected study has identical 0.14368 scores for every trainable
arm, since the selected ES checkpoint contains zero corrections. Its frozen arm
matches analytic equations exactly at 0.11012. This validates zero-correction
recovery on actual release rollouts and prevents treating trivial initialization
identity as a transfer result.

All diagnostic arms still activate the density safeguard on all 225 test
substeps, including analytic equations. Temperature limiting is inactive.
Residual scratch/full transfer activate energy relaxation on only 2/8 substeps,
with minimum fractions 0.999969/0.999961; previous replacement scratch activated
it on all 220 substeps, down to roughly 0.048. The residual dynamics are much
closer to the analytic candidate, but these remain guarded coarse-grid solves.

No full sweep was submitted. The correction formulation is substantially more
accurate, while useful transfer remains exploratory: one seed, two test
trajectories, and the learned-source diagnostic failed the ES validation
selection criterion. Source training and the energy reference also changed
across pilots. Preserve those distinctions when interpreting the improvement.
Full reports are `residual-pilot-report.json`, `residual-comparison.csv`, and
`residual-comparison.png/pdf` under the run root, with a repository copy at
`reports/residual_transfer_pilot_results.json`.

## Bounded native ES-to-EM transfer pilot (2026-11-05)

`pilot_native_transfer.py` reuses the completed ES term checkpoint and composes
its selected operators with the current constrained EM IMEX solver. This is a
separate experiment from the running within-regime four-arm comparisons.
Snapshot and results:
`/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-native-transfer-pilot-20261105`.
Submission evidence is `reports/native_transfer_pilot_submission.json`.

The authors' pinned adapter was fetched again and archived as
`author_coarse_solver.py` alongside `export_resolution.json`. It confirms the
negative potential conversion and mixed native/exported conventions. Existing
training-only checks give median initial density/temperature profile errors
of 0.030%/0.033% (ES) and 0.064%/0.088% (EM). Old viscosity gives a median
EM momentum-rate relative error of 2.99%, versus 269% for new viscosity.
This resolves the pilot's numerical convention; it does not recover the exact
generating executable or per-frame timestamps. The declared frame interval
remains five common-native units, and the static flux wall closure is approximate.

Pilot protocol: one seed (907), eight EM training trajectories, two validation
and two test trajectories, one epoch, sixteen fixed two-step training windows
(starts 0 and 4 in each trajectory), eight minimum IMEX substeps per frame,
Adam LR1e-3/weight decay1e-4, and independent term anchoring with weight one.
Validation uses five free steps; test uses ten. The four trainable arms are
random weights, all ES weights, ES advection/momentum weights only, and ES
particle/thermal diffusion weights only. Untransferred blocks start from the
same seeded random weights. Every trainable block is fine-tuned. Frozen full
ES transfer and analytic equations are additional inference-only controls.

All arms use the same source interface normalization and target loss scales.
Thus "scratch" means a matched **random-weight control**, not an experiment
with no ES information whatsoever. No EM term checkpoint trained on all 76
trajectories enters this budget-eight comparison. The source checkpoint must
match its target-cache provenance and the same family-separated manifest.

Reports retain fieldwise R-weighted relative L2, magnetic perturbation error,
persistence, failed trajectories, and positivity/energy/CFL limiter activity.
Failed trajectories invalidate aggregate scores rather than disappearing from
averages. The pilot tests integration and early transfer behavior; sixteen
updates and one seed cannot establish a sample-efficiency advantage or justify
an automatic full budget sweep. Job 2780074 requests one GPU for at most one hour.

Validation before submission: 43 targeted tests passed, including selective
weight transfer/freezing, source provenance, constrained native solves and
adjoints. All selected release prefixes passed finiteness/positivity checks.

Job **2780074 completed in 4m15s**, exit 0. All 12 ten-step test rollouts
completed; the four trainable arms used exactly the same 16-window order.
The macro score averages R-weighted relative L2 over rho, T, potential and
vorticity, then over the two test trajectories; lower is better.

| Arm | Step 5 shared error | Step 10 shared error | Step 10 magnetic perturbation error |
|---|---:|---:|---:|
| Random-weight scratch | 0.7161 | 0.8371 | 1.0614 |
| Full ES transfer, fine-tuned | 0.6015 | 0.8107 | 0.9736 |
| Advection/momentum transfer | 0.6726 | 0.8451 | 1.0458 |
| Particle/thermal diffusion transfer | 0.8451 | 1.0249 | 1.0586 |
| Frozen ES operators, analytic EM coupling | 0.5836 | 0.7727 | 0.9863 |
| Analytic equations | 0.0638 | 0.1101 | 0.0201 |
| Persistence | 0.6276 | 0.8247 | 1.0000 |

Full transfer improves the ten-step shared score by 3.15% over scratch; frozen
transfer performs better than this short fine-tuning run. Analytic equations
substantially outperform every learned composition. Density limiting activates
on 219/220 frozen-transfer and 225/225 analytic substeps. Scratch and diffusion
transfer additionally trigger the kinetic-energy limiter on every test substep.
The numerical safeguards are therefore material, even without rollout failure.

**Decision: do not launch a new full transfer-budget sweep from this pilot.**
Improve separate term accuracy and investigate conservative, small corrections
to the analytic operators before spending on a sweep. The conclusion concerns
this short pilot, not asymptotic transfer performance. Existing unrelated
within-regime campaigns remain unchanged. Full results, limiter counts, archived
adapter and provenance are in `pilot-report.json`, `pilot-comparison.csv`, and
`pilot-comparison.png/pdf` under the run root; a repository copy of the report is
`reports/native_transfer_pilot_results.json`.

## CFL correction and four-GPU relaunch (2026-11-05)

The current native snapshots are
`/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-matched-fast-20261105/{es,em}`.
They reuse the original matched data and completed term checkpoints. The older
native chains below are superseded; their logs and checkpoints remain archived.
AR and Neural ODE have completed 250 epochs and common test evaluation in both
cases. The separate composite experiments are not modified by this relaunch.

**Relaunch verified:** all four exhaustive audits completed with zero failures.
Each ES initialization passed all 4,484 windows, 100 worst-window backward checks
and 42 common 49-step validation prefixes; each EM initialization passed all
3,374 windows, 76 backward checks and 11 validation prefixes. All four full
training jobs below have started; finite optimizer updates and saved checkpoints
were observed in every arm (`reports/relaunch_observed.json`). These audits
certify the initializations, not every future trained weight; production failures
and limiter activity remain recorded.

**Remaining ES failure:** independently averaged cell-centred face velocities
produced a large artificial near-wall divergence when combined with the
impermeable boundary. The discrete divergence could exceed the physical
`-2*u_Z` by orders of magnitude, weakening the density bound and accelerating
CFL collapse. `upwind_transport` now derives both face fluxes from shared corner
potentials, so the mixed derivatives cancel consistently. Fluxes still cancel
pairwise between cells; the implicit transport operator remains an M-matrix.
The existing dual-wall closure is still a coarse-grid approximation. No new
source-window exclusions, fixed density floor, or larger CFL cap are introduced.
The known ES failures (`jorek_run0476`, start36; `jorek_run0432`, start44) now
complete forward/backward checks in both initializations.

**Memory and runtime:** audit workers now use distinct GPUs (one worker per
GPU), rather than placing four large backward graphs on one GPU. Sparse stencil
indices are precomputed; the momentum candidate and force-only reference share
one numeric factorization; unused explicit viscosity/resistivity calculations
are omitted from IMEX; analytic derivative kernels are compiled. cuDSS remains
FP64 with deterministic solves and per-RHS true residual checks. Its deterministic
mode supports one RHS, so two columns reuse the factorization with separate
solves ([NVIDIA 0.7.1](https://docs.nvidia.com/cuda/cudss/0.7.1/types.html)).
Whole-substep checkpointing is available but off by default: it reduced memory
in a diagnostic and increased runtime. The standard, nondeterministic cuDSS
kernels were also measured, but are not enabled in this campaign.

Training uses four GPUs with the **same global batch of 20**, window permutation,
summed loss, gradient clipping, Adam, learning-rate schedule and 250 epochs.
Gradients are summed across ranks before one common optimizer update; empty
shards and unused parameters are handled explicitly. Validation trajectories
are distributed and assembled in their original order. Checkpoints retain
per-rank RNG states and use a coordinated requeue marker. Final test evaluation
runs as a separate one-GPU dependency using the unchanged evaluator and best
checkpoint. Report wall time and GPU-hours separately: four GPUs are an
execution-resource change, not four times the effective batch.

Warm measurements use the first seeded matched batch of 20 pretrained windows,
with an isolated GPU or a four-GPU allocation (compilation warm-up excluded):

| Case | Previous solver, 1 GPU | Optimized solver, 1 GPU | Optimized solver, 4 GPUs |
|---|---:|---:|---:|
| ES | not benchmarked | 55.94 s | 16.72 s |
| EM | 217.59 s | 129.99 s | 42.99 s |

Four-rank versus serial gradient relative errors are ES `1.13e-7`, EM `7.04e-8`;
one-Adam-update parameter errors are ES `1.08e-7`, EM `5.02e-8`. EM wall time is
about 5.1x lower than the previous solver; its four-GPU batch costs about 172
GPU-seconds versus 218 previously. These are one-batch measurements, not
end-to-end training results. Linear training-only extrapolations are roughly
11 days ES / 21 days EM per pretrained arm, before validation, queue time and
changing CFL work. They are not scratch-run ETAs. Native integration still
performs at least 40 internal equation/FNO evaluations per five-frame window,
plus implicit and adjoint solves, versus five model calls for AR/NODE.

Validation: **57 CPU tests, seven GPU tests, eight targeted five-step backward
checks, and four 49-step prefixes pass**. GPU tests include compiled/reference
state derivatives and two-RHS implicit adjoints. CPU tests include distributed
complex gradients, empty shards, global-batch optimizer equivalence and timed
checkpoint/resume. Peak targeted-check memory was below 4.7 GB; measured batch
peaks were below 2.6 GB ES / 6.9 GB EM per rank. See `reports/final_cpu_tests.log`,
`reports/final_gpu_tests.log`, `reports/optimized_initializations.json`, and
`reports/matched_{serial,parallel}_{es,em}.json`.

| Case / arm | Four-GPU smoke | Exhaustive audit | Full training | Test evaluation |
|---|---:|---:|---:|---:|
| ES scratch | 2779850 | 2779851 | 2779852 | 2779853 |
| ES pretrained | 2779854 | 2779855 | 2779856 | 2779857 |
| EM scratch | 2779858 | 2779859 | 2779860 | 2779861 |
| EM pretrained | 2779862 | 2779863 | 2779864 | 2779865 |

Each chain is `smoke -> exhaustive audit -> training -> evaluation`, with
successful-completion dependencies. All 4,484 ES / 3,374 EM windows, worst-window
gradients for every training trajectory, and every common 49-step validation
prefix must pass for each exact code/data/initialization identity. The targeted
checks above do **not** replace that certificate. Live launch/coverage evidence
is recorded in `comparison_status.json`; submission alone does not mean a full
training run has started.

## Matched four-arm comparison (2026-11-04)

Use `orchestrate_comparison.py` for new comparisons. The previous AR/NODE and
anchored native runs are exploratory results, not this matched experiment.
Campaigns are under
`/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-matched-20261104/{es,em}`.
Data preparation completed: ES has 4,484 eligible training windows and 42 validation trajectories; EM has 3,374 windows and 11 validation trajectories. All four arms use these same lists. Source-invalid windows excluded are ES16/EM46. The common 200-step test cohorts contain ES51/EM9 trajectories.

Each has an immutable code snapshot, a shared `data/comparison.json`, cached
training/validation data, separate arm directories, audit reports and `jobs.json`.

| Setting | All four arms |
|---|---|
| Methods | AR, Neural ODE, native OpsSplit scratch, native OpsSplit term-pretrained |
| Data | Same family-separated IDs; ES100/EM76 training trajectories; first50 frames |
| Training windows | Identical six-frame windows (five predictions); finite exported/native fields and positive native density/pressure throughout; valid initial state for wall closure |
| Sampling | Every eligible window once per epoch, same permutation seed907+epoch; effective batch20 |
| Normalization | One train-only fieldwise minmax fit over the union of eligible training-window states |
| Rollout loss | Batch-summed five-step relative L2 on all normalized exported fields; **term weight zero** |
| Rollout optimization | 250 epochs; Adam lr0.001, weight decay0.0001; StepLR50/gamma0.5; clip1 |
| FNO budget | Width64, modes32, six total spectral layers; baseline6, native2/2/1/1; actual parameter counts recorded |
| Checkpoint selection | Same mean free-rollout relative L2 over admissible 49-step validation prefixes |
| Test | Same source-valid cohorts at 10/49/100/200 steps; no teacher forcing; failures retained |

`comparison.py` fixes eligibility **before training**, independently of any model.
It records excluded source windows and validation prefixes. A model-dependent
instability must fail its audit/run; it must never shrink that model's dataset.
Test eligibility is fixed separately for each horizon from source validity, so
long-horizon invalid data do not remove valid short-horizon examples. Neither
validation nor test data fit the common normalization.

`train_comparison.py` uses a single objective/evaluator for all four arms. AR and
NODE predict exported fields; native models decode rho,p,u,(psi) and reconstruct
omega/j through their constraints before scoring the same fields. Auxiliary-field
errors therefore include coarse-grid constraint mismatch. Native implicit/CFL
integration and wall closure remain explicit methodological differences from the
paper's NS Euler formulation. Exact original timestamps and the generating
revision remain unavailable; the native adapter uses the audited nominal timing.

**Term supervision** means separate FNO targets for scalar advection/compression,
fluid momentum, particle diffusion and thermal diffusion. Targets are finite-
difference approximations to source-derived equations evaluated on snapshots,
not residuals exported by JOREK. The pretrained arm receives 250 extra term epochs
before its 250 rollout epochs. Targets use stride5 states within the shared valid
training-state pool; term validation uses the shared valid validation prefixes.
The scratch arm loads only the same train-fitted interface scales and random
weights. It loads no learned checkpoint. Both rollout arms have **no term anchor**.
This is a JOREK pretraining ablation, not a claim that the NS paper used this same
term-supervision stage. Compare accuracy at matched rollout updates and report the
additional term-training time separately; these arms are not equal-total-compute.

Evaluation reports paper eq.(13) NRMSE with epsilon1e-6 in the common range-
normalized coordinates, separately from cylindrical R-weighted decoded relative
L2. Reports retain per-field and per-trajectory scores, persistence, temporal-
extrapolation scores (frames50 onward), EM delta-psi errors, and completion/failure
counts. A horizon's common-cohort mean is null if any eligible model trajectory
fails; it cannot silently become a survivor-only average. Held-out families are
not labelled physics-OOD, and the initial campaign uses one seed, not a multi-seed
uncertainty estimate.

Full native training requires a successful exhaustive audit tied to the exact
initial weights, shared data and code hashes. This checks every eligible training
window, the worst-loss window's gradients for each trajectory, and all common
validation prefixes. Each arm first runs a bounded GPU optimizer smoke test.
The coupled-guard revision (2026-11-04) reuses the **same** shared data and
term checkpoints at `jorek-matched-20261104/{es,em}`. Its immutable native runs
are at `jorek-matched-coupled-20261104/{es,em-conservative}`. AR/NODE runs are unchanged.
The original completed exhaustive audits failed on ES44/EM6 windows. Matched
initialization checks then reproduced failure in both ES variants and EM
pretrained; these old results remain archived.

The observed failure couples scalar transport to momentum: positive density can
still become nearly vacuum, and independently learned terms need not cancel
transport's kinetic-energy work. An ES ablation completes the known bad window
with learned momentum or diffusion alone, but accelerates with learned scalar
advection. Positivity alone therefore cannot certify the coupled update.
`native_rollout.py` now combines two differentiable safeguards:

1. The density lower bound follows the current discrete transport, rather than
   a fixed floor: `min(rho.min(), rho_wall)/(1+h*max(div_h(v),0))`. Here `div_h(v)`
   is the row sum of the implicit upwind transport matrix divided by cell volume
   and R. When needed, blend the high-order density update with the positive
   implicit upwind/diffusion update using one global correction fraction. The
   low-order bound is verified; only roundoff-sized adjustment is allowed.
2. Let `K(rho)` be the density-weighted momentum stiffness and
   `E(u,rho)=u.T K(rho) u/2`. Compute a reference momentum update using analytic
   pressure curvature, Lorentz force, conservative diffusion and viscosity, with
   fluid inertia omitted. Bound the candidate energy by the larger of the old
   kinetic energy and this forced reference's energy in the updated density.
   If necessary, scale the whole candidate potential by `sqrt(E_bound/E_high)`.
   This preserves its direction and homogeneous wall condition. All scalar
   updates and auxiliary constraints remain coupled to the primary state.

EM also includes the learned density-advection defect in conservative momentum.
The source term depends on `rho_t + div(rho*v)`. After replacing continuity's
advection by a learned term, its correction is `delta_A_rho + delta_D_rho`, not
only `delta_D_rho`. The added load is
`-div(R^3 * delta_A_rho * grad(u))`. A manufactured load/gradient test verifies
this identity. The ES conservative flag is false, so the ES numerical snapshot
is unchanged. The earlier `em/` guard-only snapshot and its pending training
chain are superseded by `em-conservative/` and are not certification evidence
for the corrected EM runs.

This kinetic envelope is a **numerical safeguard, not an exact discrete total-
energy law or a proof of universal stability**. It changes the rollout method;
old checkpoints and audit certificates do not certify the new composition.
The FNO architecture, term targets, initial weights, rollout objective, optimizer,
windows and metrics are unchanged. Guard counts and minimum relaxation factors
are recorded per batch, audit trajectory, validation trajectory and test rollout.
These activations belong in the comparison: constrained predictions must not be
presented as unmodified neural equation terms.

Validation so far: 50 CPU regression tests and six GPU checks pass. All four
actual matched initializations pass the known ES/EM five-step reproductions,
with finite nonzero gradients in every learned module. Each also passes its
first common 49-step validation prefix. Evidence is in
`reports/coupled_guard_targeted.json` (ES) and
`reports/check_conservative_guard.json` (corrected EM); these are **not**
exhaustive certificates. Two additional slow ES pretrained windows
(`jorek_run0432`, start36; `jorek_run0494`, start25) also complete their
five-step forward checks; see `reports/check_slow_es_guard.json`.
No model-dependent data exclusions or inflated CFL failure thresholds were used.
The failed single-guard experiments are retained separately in `reports/`.

| Case / arm | GPU smoke | Exhaustive audit | Full training, after successful audit |
|---|---:|---:|---:|
| ES scratch | 2776215 | 2776216 | 2776217 |
| ES pretrained | 2776218 | 2776219 | 2776220 |
| EM scratch | 2776467 | 2776468 | 2776469 |
| EM pretrained | 2776470 | 2776471 | 2776472 |

The new chains were submitted with successful-completion dependencies. GPU
partition capacity is currently limiting scheduling. Short audit chunks may
advance reports within existing campaign allocations while their queued audit
job is held against concurrent writes. The wrapper releases the queued audit
job when its chunk exits. The corrected EM dispatcher keeps at most four
trajectory tasks in flight, so prequeued work cannot overrun a checkpoint
budget. This dispatcher-only amendment was made before any EM replacement job
executed, with all jobs held and no output/certificate present. Each audit resumes completed trajectories only under
identical data/code/initialization hashes; partial reports cannot release
training. Exhaustive scope remains ES4,484/EM3,374 training windows **per arm**,
worst-window gradients per training trajectory, and ES42/EM11 full validation
prefixes. Current coverage and scheduler states are in `comparison_status.json`.

The superseded pending native chains (including `em/parallel-audit-v2` and
the guard-only EM chain) were
cancelled after replacement submission. Short diagnostic steps shared the
baseline GPUs; affected wall times include contention and must not be used as
clean throughput benchmarks. The energy guard adds a reference momentum solve;
its practical cost must be measured on a dedicated allocation.

Reproduce a campaign (omit `--submit` to prepare scripts only):

```bash
/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python \
  JOREK/zenodo_opsplit/orchestrate_comparison.py \
  --manifest /path/to/manifest.json --case es \
  --output /path/to/new-campaign --submit
```

Relaunch only the native arms while preserving a prepared matched campaign:

```bash
/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python \
  JOREK/zenodo_opsplit/orchestrate_comparison.py \
  --native-from /path/to/existing-case-campaign --case es \
  --output /path/to/new-native-campaign --submit
```

Audits also checkpoint completed trajectories and requeue at timed exits.

Rollout jobs checkpoint at timed exits and requeue with identical optimizer,
scheduler and sampler state. Test evaluation commits completed trajectories and
resumes at trajectory boundaries. Batch accumulation for native solves preserves
the summed batch20 objective; it does not make its serial elliptic work as fast as
a batched AR forward pass. Wall time and pretraining cost must accompany accuracy.
The native solver performs at least 40 RHS evaluations per five-frame window,
versus five model calls for AR/NODE, plus sparse solves and their adjoints. At
250 epochs the matched ES/EM arms process 1,121,000/843,500 training windows each.
Fixing pathological CFL collapse does not remove this intrinsic cost difference.


## Data and evidence

- [Electrostatic model003, record 11099659](https://zenodo.org/records/11099659): 50 data archives, 32.054 GB, the supplied 0001–0500 batch rather than all 2,000 ES simulations discussed in the paper.
- [Electromagnetic model600, record 11099527](https://zenodo.org/records/11099527): 10 data archives, 10.146 GB, parallel velocity disabled.
- [Neural-Parareal paper](https://doi.org/10.1016/j.cpc.2024.109391), [accessible preprint](https://arxiv.org/html/2405.01355v1). This paper supplies the datasets and physical context; the present additive split is our experiment, not the paper's Parareal algorithm.

All archives have been downloaded and checksum-verified: the extracted corpus contains 497 ES and 100 EM trajectory files. Full finite-value/schema validation runs before the full study.

The solver uses finite elements; the published samples are regular 100×100 arrays suitable for FNOs. Inspection of real release files establishes:

| Property | ES | EM |
|---|---|---|
| File fields | `rho,T,Phi,omega` | `rho,T,u,omega,Psi,zj` |
| Canonical order | `rho,T,u,omega` | `rho,T,u,omega,psi,zj` |
| Axis order | `[time,Z,R]` | `[time,Z,R]` |
| First trajectory frames | 201 | 211 |
| Physical timestamps / unit attributes | absent | absent |

Coordinates span R=9.501…10.499 and Z=−0.499…0.499 m. These samples lie inside the actual wall at R=9.5/10.5 and Z=±0.5. We do **not** clamp sampled edge values as exact Dirichlet nodes, or use periodic finite differences. The FNO uses domain padding; this does not guarantee exact physical wall conditions.

ES inputs explicitly warn of an unusual `1e4*mu0` normalization. EM input timesteps/temperatures use a different native convention. The historical whole-field adapter treats `Phi`/`u` as the release's electric-potential channels and preserves exported values. A preliminary interior check on frame 10 of run 1 gives correlation between saved vorticity and the cylindrical Laplacian of potential of −0.9927 (ES) and −0.9973 (EM), with ratios about −7.08e−8 and −6.98e−8. This supports matching signs/scales but is **not** a complete units or constraint audit.

Numerical preconditioning divides channels by `[1e20,1,1e3,1,0.1,1]`. These factors are **not SI conversions**. Model time is saved-frame index, with step one. Do not interpret a learned coefficient as a physical transport coefficient, claim a physical-time RHS, or substitute the old trainer's assumed units/timestep.

## Historical effective-split pilot

The physical decomposition to target after auditing exports is

\[
q_f=(\rho,T,u,\omega),\qquad q_b=(\psi,j),
\]
\[
\partial_t q_f=\mathcal D_f(q_f)+\mathcal A_f(q_f)
                 +\mathcal C_f(q_f,q_b),\qquad
\partial_t q_b=\mathcal I(q_f,q_b)+\mathcal D_b(q_b).
\]

Here `D` denotes dissipative terms, `A` fluid advection/compression/pressure coupling, `C` magnetic feedback, and `I` induction. The potential/vorticity and flux/current constraints must accompany a faithful JOREK implementation. In particular, a pressure equation is not a temperature equation; density-weighted viscosity and thermal conduction are not four independent constant scalar diffusivities.

The implemented first experiment is therefore an **effective additive RHS split**, trained through saved-frame rollouts:

\[
F_\theta(q_f)=N_\theta(q_f)+\mathrm{diag}(d_\theta)\Delta_{R,Z}q_f,
\quad
\Delta_{R,Z}=\partial_{RR}+R^{-1}\partial_R+\partial_{ZZ},
\]
\[
G_{\theta,\phi}(q_f,q_b)=\big(F_\theta(q_f),0\big)+M_\phi(q_f,q_b),
\quad q^{n+1}=q^n+G(q^n).
\]

`N` is a four-input/four-output FNO. `M` is a six-input/six-output FNO. The nonnegative effective diffusion coefficients are learned and capped below the scalar explicit-diffusion stability limit. This cap does not guarantee stability of the neural dynamics. The Laplacian uses centered interior stencils; the learned remainder handles sampled edges.

This follows the repository's additive operator/Neural-ODE convention; it is not Lie or Strang composition of exact subflows. Because only trajectory endpoints are provided, individual physical terms are not identifiable. `M` can absorb both magnetic effects and source/target mismatch. Neither `omega=operator(u)` nor `j=operator(psi)` is enforced. This stage tests whether a reusable fluid block improves EM sample efficiency; it does **not** establish recovery of independent physical operators. The scratch split has identical capacity to transfer; the monolithic FNO is a smaller whole-RHS control, not a parameter-matched ablation.

## Equation-based split: original source audit

The following formulation is separate from the completed empirical baseline above.
`equations.py` implements tested **interior diagnostic kernels**, not a replacement
trainer or a complete JOREK time integrator. Existing checkpoints cannot be
reinterpreted as independently learned advection/diffusion operators.

### Convention and evidence

The audit uses the installed JOREK source at
`/pitagora_work/FUPB2_UKAEA_ML/vgopakum/src/jorek/src`, Git HEAD
`01a52be37576f79c785f1f7a80dee296dda5c9c8`. This is a reference implementation;
the release does not establish that this revision generated the 2024 datasets.
Relevant source locations are:

| Source relative to that root | Evidence |
|---|---|
| `models/model003/mod_elt_matrix_fft.f90:553–601` | ES momentum, constraints, continuity, pressure RHS |
| `models/model003/mod_elt_matrix_fft.f90:699–760` | Density-weighted momentum and pressure mass matrices |
| `models/model600/mod_elt_matrix_fft.f90:1547–1668` | Induction, Lorentz force, constraints, anisotropic density diffusion |
| `models/model600/mod_elt_matrix_fft.f90:1999–2043` | Pressure transport, conduction and optional heating |
| `models/model600/mod_elt_matrix_fft.f90:1159–1166` | Old/new viscosity switch |
| `models/model600/mod_elt_matrix_fft.f90:1609–1620` | Conservative-momentum corrections |
| `models/preset_parameters.f90` | Defaults, including zero Ohmic/viscous heating coefficients |
| `docs/physics/normalization.md` | Native-to-SI conventions |

For axisymmetry and no parallel flow, define native variables and bracket by

\[
q=(\rho,p,u,\psi),\quad p=\rho T,\quad
[f,g]=f_R g_Z-f_Z g_R,\quad
\mathbf v_p=(-R u_Z,R u_R),\quad \nabla\cdot\mathbf v_p=-2u_Z.
\]

The two auxiliary fields obey

\[
\omega=\Delta_c u,\qquad j=\Delta^*\psi,\qquad
\Delta_c=\partial_{RR}+R^{-1}\partial_R+\partial_{ZZ},\quad
\Delta^*=\partial_{RR}-R^{-1}\partial_R+\partial_{ZZ}.
\]

Here `u` is the **native velocity stream function**. It is not automatically the
HDF5 `u`/`Phi` channel. The negative potential–vorticity fit reported above must
be reconciled with the native positive constraint before using these equations
on release arrays. Native `j` is also not SI toroidal current:
`j_phi,SI = -j/(R*mu0)`. Channel standardization is applied only at network
interfaces; differential operators act on fields in one common physical/native
convention.

### Three operator groups, with constrained momentum

Write the evolution as a differential-algebraic system

\[
\mathsf M(q)\dot q=\mathcal A(q)+\mathcal D(q;c)+\mathcal C(q),
\qquad \omega-\Delta_cu=0,\quad j-\Delta^*\psi=0.
\]

`M` contains the density-dependent momentum solve. It is not the identity on
six independently evolved channels. A native-consistent elliptic solve/projection
must accompany each split stage; if evolving `u,psi` directly, recompute the
auxiliary fields rather than independently predicting them.

**A: fluid advection, compression and pressure curvature.** Its scalar equations are

\[
\mathcal A_\rho=R[\rho,u]+2\rho u_Z,
\qquad \mathcal A_p=R[p,u]+2\gamma p u_Z.
\]

Pressure curvature drives the ES blobs and belongs to this shared group. To
avoid replacing native momentum with a guessed passive-vorticity equation, its
model003 reference is most directly specified in weak form. With test function
`chi`, measure `dR dZ`, `rho_hat=R^2 rho` and `V^2=R^2|grad u|^2`,

\[
\langle\chi,\mathsf M_u\dot u\rangle
=-\int \rho R^3\nabla\chi\cdot\nabla\dot u\,dR\,dZ,
\]
\[
\langle\chi,\mathcal A_u\rangle
=\int\left(-\tfrac12 V^2[\chi,\hat\rho]
-\hat\rho R^2\omega[\chi,u]+R^2[\chi,p]\right)dR\,dZ.
\]

For compactly supported test functions the pressure load is `-2R p_Z`.
The code diagnostic returns this load, not `du/dt` or `domega/dt`.
The installed model600 additionally has conservative-momentum terms involving
`rho_t` and density transport. Audit the generating revision before sharing the
entire momentum flow. Shared scalar transport has a clearer starting point.

**D: particle diffusion, heat conduction, viscosity and resistivity.** Define
`L_K f = div(K grad f)` with the cylindrical divergence. The basic terms are

\[
\mathcal D_\rho=L_{D_\perp}\rho,\qquad
\mathcal D_p=L_{K_\perp}T,\qquad
\mathcal D_\psi=\eta j.
\]

`K` here is native `ZK`, which already includes the normalization of the energy
equation. Recover temperature using
`T_t=(p_t-T*rho_t)/rho`; a density-diffusion stage therefore changes temperature
even when its pressure increment vanishes. Variable coefficients require flux
divergence, not `K*Laplacian(f)`.

The ES viscous momentum load is
`-integral(mu R grad(chi).grad(omega))`. Current model600 recovers this only with
`visco_old_setup=true`. Otherwise its axisymmetric load is
`-integral(mu R^3 grad(chi).grad(omega) + 2 mu R^2 omega chi_R)`.
Share constitutive kernels only after matching this convention; equal numerical
`visco` values do not establish equal viscous operators.

In EM, use full diffusion tensors

\[
\mathbf K=K_\perp\mathbf I+(K_\parallel-K_\perp)\mathbf b\mathbf b^T,
\quad
\mathbf B=(-\psi_Z/R,F_0/R,\psi_R/R),\quad \mathbf b=\mathbf B/|\mathbf B|,
\]

and the analogous particle tensor. `K_parallel=0` does **not** eliminate the
field-direction correction when poloidal field is nonzero. Assemble this with
the isotropic part inside D; the correction alone need not be dissipative.
The flux/divergence machinery can transfer, while the magnetic-direction
dependence is new. Resistivity and any enabled Ohmic heating belong to D too.
Do not infer heating from nonzero `eta`: the installed source uses separate
`eta_ohmic`, default zero.

Hyperdiffusion, hyperviscosity, hyperresistivity, low-density/temperature
coefficient increases and numerical stabilization must be retained as explicit
additional terms when matching native trajectories. Several are nonzero in
the released inputs. The diagnostic kernels deliberately do not implement
these, boundary closures, or momentum inversion.

**C: ideal magnetic induction and Lorentz coupling.** The axisymmetric terms are

\[
\mathcal C_\psi=R[\psi,u],\qquad
\langle\chi,\mathcal C_u\rangle=\int\chi[\psi,j]\,dR\,dZ.
\]

There is no direct ideal-C contribution to density or pressure for this reduced,
no-parallel-flow setting. They respond through the changed velocity in A.
The Lorentz term is a projected momentum load and uses the same mass solve.
This structure prevents an unrestricted EM network from rewriting all fluid
equations. Physics known from C can be evaluated without any ES magnetic labels.

### Learning and transfer protocol

Use separate, coefficient/time-conditioned modules for shared scalar advection,
fluid momentum, particle diffusion and thermal diffusion. Keep geometric
derivatives, coefficient factors and auxiliary constraints explicit. FNOs may
learn subflow maps or bounded discretization corrections; do not freely learn
several unrestricted RHS fields whose errors can cancel. Use an analytic C
initially, and add a separately supervised EM correction only if needed.

| Component | ES pretraining | EM use |
|---|---|---|
| Scalar advection/compression | Yes | Transfer independently; freeze or fine-tune |
| Pressure/inertial momentum | Yes | Transfer after mass-form audit; separate ablation |
| Particle/thermal diffusion | Yes | Condition on target coefficients; transfer separately |
| Viscosity | Yes | Transfer only after old/new formulation is resolved |
| Induction/Lorentz/resistivity | No ES supervision | Analytic reference or EM-trained module |
| Elliptic constraints | Deterministic | Shared geometry with verified native units/BCs |

If using subflow composition, one symmetric candidate is
`A(dt/2), D(dt/2), C(dt), D(dt/2), A(dt/2)`, with consistent constraint handling
after each stage. For ES, omit C and magnetic D. Second-order accuracy requires
appropriately accurate subsolvers and consistent boundary treatment; learned
maps do not inherit that guarantee automatically. A checkpoint trained only at
one full saved-frame step is not automatically a half-step solver.

Full-trajectory supervision alone cannot identify individual terms: adding H
to A and subtracting H from D leaves their summed RHS unchanged. The release
supports approximate interior RHS targets from spatial derivatives **after**
the unit audit. These are discretized equation targets, not ground-truth native
split-step trajectories. For learned subflows, generate short isolated-operator
reference solves with correct boundary data, or export JOREK's separate weak
residual contributions. Anchor term losses, then train composed free rollouts.
Validate residuals against native evaluations before treating labels as physics.

Compare scratch, advection-only transfer, diffusion-only transfer, combined
transfer, frozen transfer and fine-tuning at the existing nested EM budgets
8/32/64 and seeds 0/1/2. Use identical capacity, optimizer schedules, physical
scales, splits and update counts. Report per-field errors, `delta psi` as well as
total flux, constraint residuals, and long-rollout divergence. Native known C/D
with frozen ES modules gives a meaningful **hybrid zero-EM-training** baseline;
ES alone cannot supply a learned magnetic operator.

### Original blockers (superseded by the native implementation below)

1. Recover export signs/scales, density reference, temperature convention and
   physical timestamps. ES input explicitly changes `mu0` by `1e4`; its listed
   coefficient rescaling is evidence to verify, not an implemented adapter.
2. Resolve the original solver revision, momentum/viscosity switches and active
   stabilization. Current source defaults are not proof of 2024 run settings.
3. Recover wall data or a validated interior boundary closure. Sampled edges
   are 1 mm inside the wall; future true edge samples cannot supply rollout BCs.
4. Validate term residuals and split integration before training or submitting
   a new budget sweep.

The verified raw EM `jorek_input_run001.txt` says `nout=1`, `nstep=50`, and
`restart=true`, whereas record metadata says every tenth timestep and the HDF5
has 211 frames. This is a concrete reason not to infer elapsed time from that
input alone. The native implementation below uses a declared nominal interval and preserves this provenance limitation.

Run the kernel checks with the same Python environment:
`python -m pytest -q test_equations.py test_workflow.py`.
Manufactured tests cover geometric Laplacians, bracket signs, compression,
variable-coefficient flux divergence, anisotropic cancellation, gradients and
absence of periodic wrapping. They do not validate the release's unit mapping.

## Historical effective-split transfer comparisons

| Mode | Source knowledge | EM training | Meaning |
|---|---|---|---|
| `source` | none | none | ES shared block pretraining |
| `zero-shot` | frozen ES block and ES statistics | none | shared-field transfer, magnetic channels persist |
| `frozen` | frozen ES fluid block/diffusivities | new EM block only | reuse without changing the fluid operator |
| `finetune` | ES initialization | all blocks | adaptation with lower learning rate |
| `scratch` | none | both split blocks | capacity-matched control |
| `monolithic` | none | whole RHS FNO | conventional baseline |

Full EM zero-shot prediction cannot be inferred from ES alone: there are no induction targets in ES. The implemented zero-shot baseline is explicitly fluid-only with magnetic persistence. It never fits target statistics. Fine-tuning and frozen transfer preserve source statistics on all four shared channels; their two magnetic statistics use only the selected EM training subset.

Splits use hashes of sorted initial blob `(R,Z,width)` tuples, rounded to eight decimals. The same family always goes to the same split in both regimes, regardless of file numbering or blob order. This conservative grouping prevents an ES counterpart of an EM test family entering pretraining. All windows from a trajectory remain together. Within each seed, EM budgets select nested training subsets. Manifests fix identities and source provenance; transfer refuses a different manifest.

Training uses four-step free rollouts and validation-only checkpoint selection. Test evaluation starts from frame zero and runs 100 free steps by default. Reports retain per-trajectory, per-field, per-frame relative L2 errors, standardized MSE, persistence comparisons, and nonfinite failures. Errors use R weighting. Relative L2 is comparable across models on the same test trajectories; standardized MSE is not directly comparable when scratch/source normalizers differ. Near-zero fields can make relative errors large. Finite rollout completion is not physical validity or stability certification.

## Run the historical effective-split pilot

Dependencies already exist in the neighboring environment: Python 3.11, PyTorch 2.7.1+cu126, neuralop 1.0.2, numpy, h5py, pytest. The empty local `Neural_PDE` submodule is not needed. The small compatibility shim removes neuralop's root constructor metadata before strict parent-model checkpoint loading.

```bash
cd /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/zenodo_opsplit
PYTHON=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
DATA=/pitagora_work/FUPB2_UKAEA_ML/vgopakum/data/jorek_zenodo

# 42.2 GB archives plus roughly the same amount extracted; keep data off HOME.
python3 download.py --root "$DATA"
"$PYTHON" data.py --root "$DATA" --output "$DATA/full-manifest.json"

# A fresh output directory is required. Rendering is the default; --submit launches.
python3 orchestrate.py --manifest "$DATA/full-manifest.json" \
  --output /pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-transfer-new \
  --account fupb2_ukaea_ml --seeds 0 1 2 --budgets 8 32 64 \
  --epochs 20 --steps-per-epoch 128 --time 12:00:00

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PYTHON" -m pytest -q test_workflow.py
```

Downloads resume `.part` files, verify Zenodo MD5 and byte counts, reject unsafe archive paths/links, and refuse conflicting extracted files. A file lock prevents competing downloaders. HDF5 files are read lazily; no extra full-corpus training cache is created. Metadata snapshots, archive checksums, code hashes, selected training IDs, source checkpoint hashes, and job commands remain available.

The first bounded pilot uses two archives per regime: 19 ES and 19 EM trajectories, each split into 14 train / 3 validation / 2 test. EM training uses eight of the fourteen training trajectories. One seed, ten epochs, 64 batches per epoch, and one GPU for at most two hours. It is a plumbing/sample-efficiency pilot, not enough data or seeds for a scientific conclusion. The full-data job renders a three-seed, 8/32/64-trajectory plan after download and validation. A separate full study has been queued behind that validation job, using three one-GPU jobs capped at two hours each (20 epochs and 128 batches per epoch). It retains the pilot protocol; the pilot test results were not used to retune the architecture or rollout loss.

Current machine-readable launch and validation evidence is in `status.json`. Run outputs live under `/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-transfer-20261104`. Future physical claims require resolving historical units/time export, auditing both differential constraints, and testing a source-verified term split against this effective baseline.

## First pilot result

All six modes completed after the neuralop checkpoint fix; nine focused tests pass. The two EM holdouts are too few for an inference about transfer. Mean relative L2 across the four shared fields at frame 10 was 1.092 for zero-shot, 1.214 for frozen transfer, 0.831 for fine-tuning, 0.854 for scratch split, and 0.667 for monolithic FNO; persistence was 0.826. By frame 100 every model was worse than persistence, with substantial finite drift. No useful zero-shot or long-rollout advantage has been demonstrated. The full-data seed/budget study tests whether this persists with a better-trained source and more independent holdouts. Detailed per-field curves and the preserved failed first attempt are in the run directory.

## ES-only comparison using the OpsSplit paper protocol

`train_es.py` and `orchestrate_es.py` implement the requested **AR → Neural ODE → OpsSplit** sequence independently of the earlier transfer pilot. Each model starts from the same seed, uses the same training trajectories and shuffled window order, and trains from scratch. The protocol is saved in `es_paper_protocol.json`, based on the attached **2602.23113v2**, section 5 and appendices J.2/K.2.

| Setting | ES comparison |
|---|---|
| FNO | 32 modes per axis, width 64, six layers, GELU; lifting 64 / projection 256 |
| Data | 100 ES training trajectories; existing 44 validation / 53 test trajectories, family-separated |
| Training interval | First 50 saved frames; no later frame enters training or normalization |
| Epochs / batches | 250 full epochs; batch 20; every sliding window once per epoch |
| Loss | Batch-summed relative L2 on flattened normalized fields, summed over exactly five predicted steps before backpropagation |
| Normalization | Fieldwise training-prefix range to [-1,1], no clipping of held-out values |
| Optimizer | Adam, LR 0.001, weight decay 0.0001, default betas (0.9,0.999) |
| Scheduler | StepLR: halve every 50 **epochs**, following the NS code's scheduler placement |
| Gradient clipping / seed | Norm 1.0 (effective NS trainer setting); seed 907 |
| Selection | Best free-rollout validation relative L2 over frames 1–49 |
| Evaluation | Held-out free rollouts through frame 200 where available; frames 50 onward are temporal extrapolation |

The published prose takes precedence over stale YAML defaults (LR 0.005, scheduler period 100), the old trainer's [0,1] normalization, and its `rollout_length-1` off-by-one. Batch size and seed come from the spectral incompressible NS configuration. The old trainer actually clips at 1.0: its YAML value of 2.0 is not passed into `Train_Setup`. The paper says “every 50 steps”; this workflow interprets those as scheduler/epoch steps, matching the repository's scheduling convention.

For normalized state `x`, the three maps are:

- AR: `x_next = FNO(x)`.
- Neural ODE: `x_next = x + FNO(x)` (explicit Euler, saved-frame dt=1).
- OpsSplit: `x_next = x + FNO(x) + diag(d) * Laplacian_RZ(x)` (same Euler step).

The single six-layer neural block follows the incompressible NS split. The compressible paper uses two three-layer blocks to divide neural capacity between its convection and divergence operators; those equations are not the JOREK equations. ES currently uses the effective fluid remainder described above, with four extra diffusion parameters. It does **not** establish physically identifiable JOREK operators. Fieldwise affine normalization commutes with the linear Laplacian because its stencil annihilates constants. Four ES channels, nonperiodic padding, saved-frame time, the selected 50-frame training interval, and independent family-held-out validation/test are explicit adaptations. The installed neuralop backend and actual parameter counts are recorded; exact equality to the NS tables is not asserted.

The range-normalized MSE is stored under the legacy `standardized_mse` metric keys and is labelled accordingly. The existing decoded, R-weighted fieldwise relative L2 and persistence comparison are also saved; these are not the paper's unweighted epsilon-regularized NRMSE. Incomplete/nonfinite test rollouts remain visible. No ES OOD parameter distribution has been defined, so held-out-family tests must not be labelled physics OOD.

Run preparation/submission:

```bash
PYTHON=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
$PYTHON JOREK/zenodo_opsplit/orchestrate_es.py \
  --manifest /pitagora_work/FUPB2_UKAEA_ML/vgopakum/data/jorek_zenodo/full-manifest.json \
  --output /pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-es-paper-20261104/full \
  --submit
```

Omit `--submit` to render only. Each submitted method receives one H100 allocation, with `afterok` dependencies enforcing the order and a 24-hour allocation cap per method. The current default requests the native JOREK split: AR and NODE are submitted, while the separate native workflow is recorded in `opsplit-native-workflow.json`; launch it with `orchestrate_native.py`. `--split-formulation effective` explicitly reproduces the historical pilot split. A fresh output directory is required. The launcher snapshots code, protocol and manifest before submission. Each run records the selected IDs, hashes, parameter count, loss/LR history, best validation checkpoint, last epoch checkpoint, and test metrics. Training data are cached in host RAM rather than reopening HDF5 for every window.

`--smoke --time 00:20:00` retains the full FNO and batch size but uses two trajectories, two optimizer steps and one epoch. Smoke outputs are labelled and cannot be used as scientific results. For interrupted training, resubmit the saved `.sbatch` script with the same allocation options: it resumes `last.pt` including Adam, scheduler, sampler and RNG states. Dependent jobs cancelled by a failed prerequisite must be resubmitted with new dependencies. Protocol/data/code changes require a fresh run.

## Native training implementation

The native ES/EM workflow is implemented in `native.py`, `native_imex.py`,
`native_rollout.py`, `native_terms.py`, and the `*_native_*` training scripts.
Use `orchestrate_native.py`; `train_es.py` deliberately redirects native requests
because its whole-field `SplitFNO` is the historical effective model.
`native_readiness.json` records current evidence, jobs, limitations and results.
The earlier export-only audit is preserved in `native_export_audit.json`.

### Resolved conventions and numerical implementation

The authors' [surrogate adapter at commit 5e4203a](https://github.com/spamela/neural_parareal/blob/5e4203a3edff9aa8e78cca954a8c2b39fcc8ec85/coarse_solver/coarse_solver.py#L112)
provides the mixed export conventions. With `n0=1e20`, `mu0=1.257e-6`,
`e=1.602176487e-19`, `F0=10`, and `tau=sqrt(3.32e-27*n0*mu0)`, decode

```
rho_native = rho_export / n0
T_native   = T_export_eV * n0 * mu0 * e
u_native   = -potential_export * tau / F0
psi_native = Psi_export
p_native   = rho_native * T_native
```

Saved `omega` and `zj` already use native conventions. Training evolves only
`rho,p,u` (ES) or `rho,p,u,psi` (EM), reconstructing auxiliary fields. Initial
compact-blob profiles independently check the density/temperature mapping.
The ES input's stated conversion gives time ×100, temperature /1e4, and
transport/stabilization coefficients /100 in these common units.

The declared saved-frame interval is **5 common-native units**. It combines
input time steps with the published every-tenth-step sampling and is supported
by trajectory-rate comparisons. Exact original timestamps and the generating
revision have **not** been recovered; do not claim exact historical physical
elapsed times. Current source HEAD is `01a52be37576f79c785f1f7a80dee296dda5c9c8`;
historical April-2024 source `56335ba115d59a0ac6961dd253945dafe55d7b36` was also
inspected. Neither identifies the generating build.

Training-only residual comparisons select **old viscosity for both releases**:
median momentum-rate relative error is about 3% with this convention, compared
with roughly 270% in EM using the present source's default new viscosity.
This is an inferred release profile, not recovered run metadata. EM retains its
conservative-momentum correction and explicit magnetic coupling.

The solver uses a density-dependent finite-volume momentum inversion,
physical walls outside the sampled grid, and first-order IMEX stepping.
Particle/thermal diffusion, viscosity, hyperdiffusion and resistivity are
implicit. Low-density/temperature transport enhancements are also implicit;
EM switches can increase their coefficients 100-fold. Induction, Lorentz force
and anisotropic corrections are explicit. Wall density/temperature come from
inputs; initial flux is extrapolated across the unsampled 1 mm strip. Flux/current
closure is approximate and never consumes future true boundary samples.

`NativeRollout` now supports GPU FP64 sparse LU and adjoints through cuDSS 0.7.1; the CPU SciPy implementation remains an independent reference. Native states, constraints, finite differences and matrix assembly stay on the GPU. State and coefficient gradients are checked against finite differences and CPU results.
The split is a first-order IMEX approximation, not a Strang or exact JOREK FEM flow.

### Learning, positivity and protocol differences

Four independently supervised FNOs predict scalar advection, fluid momentum
load, coefficient-free density Laplacian and coefficient-free temperature
Laplacian. Their spectral-layer allocation is **2/2/1/1**, with width64,
modes32, GELU, seed907 and six total spectral layers. Exact parameter equality
to the paper's monolithic six-layer model is not asserted. Geometry, target
coefficients, viscosity/stabilization, mass inversion and magnetic physics
remain explicit. Only the four-cell interior has term targets; the boundary
strip retains analytic terms.

Term pretraining uses 100 ES or 76 EM training trajectories, first50 frames
sampled every5 frames, train-only interface normalization, 250 epochs, batch 20,
Adam LR1e-3/weight-decay1e-4, clipping1, and StepLR50/gamma0.5. The loss is the
mean of four separately normalized operator MSEs. Targets are **coarse FD
approximations to audited equations**, not ground-truth FEM term exports.
Both 250-epoch runs completed; both real-data native forward/backward and held-out five-step smoke checks passed (jobs ES2774902 / EM2774903), and 47 regression tests passed. Smoke checkpoints are not completed full-rollout training results. Their checkpoints and test metrics are under
`/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-native-20261104/`.

Rollout tuning adds five-step, batch-summed relative L2 loss on train-range
normalized primary states, plus independent term anchoring at each window
start. It retains the paper optimizer settings and uses batch 20 through
single-trajectory gradient accumulation. Epochs traverse all admissible sliding
windows. Checkpoints commit after every optimizer update and can resume within
an epoch. Validation uses 49-step free rollouts; smoke validation uses five.
`--train-trajectories` enables target budgets, and `--freeze` selects modules
whose source weights stay fixed. Source interface normalizers remain fixed in
transfer. An EM scratch term checkpoint must use the same target budget as its
fine-tuning comparison; a checkpoint pretrained on all76 is not a budget8 control.

Unrestricted composition of the first pretrained models produced negative
pressure. Each substep therefore backtracks the **common learned correction**
by factors of two, down to 1/256, with an analytic IMEX fallback. Every trial
re-solves the equations; fields are not clipped. The selected fraction is fixed
when differentiating that accepted step. Reports retain limited-step counts,
minimum fractions and analytic fallbacks. A successful guarded rollout is not
evidence that the unrestricted learned operator is stable or accurate. The
original failed attempts are retained alongside successful runs.

These are explicit adaptations to the NS paper: term pretraining/anchoring,
primary-state constraints, IMEX substeps and positivity backtracking. Existing
AR/NODE snapshots and their protocols are unchanged. Current evidence supports
training experiments; it does not establish useful ES-to-EM transfer or long-term
stability. Frozen transfer evaluation reports magnetic **perturbation** error,
reconstructed observed-field errors, persistence, failures and limiter activity.

Zero-density release frames make `p/rho` and the momentum coefficient undefined.
Term caches record excluded states; rollout training records excluded windows.
No clipping or unreported substitution repairs these samples. Term-cache
manifests also list nonpositive-pressure observations and all split identities.

### Commands

```bash
PYTHON=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
DATA=/pitagora_work/FUPB2_UKAEA_ML/vgopakum/data/jorek_zenodo/full-manifest.json
RUN=/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-native-20261104

# Fresh output: target preparation -> 250-epoch terms -> five-step rollout smoke.
# Change es to em for the electromagnetic case. Omit --submit to render scripts.
$PYTHON JOREK/zenodo_opsplit/orchestrate_native.py --case es \
  --manifest "$DATA" --output "$RUN/new-es-workflow" --submit

# Reuse completed term pretraining. Default remains a bounded rollout smoke.
$PYTHON JOREK/zenodo_opsplit/orchestrate_native.py --case em \
  --manifest "$DATA" --pretrained "$RUN/em-term-training/best.pt" \
  --output "$RUN/new-em-rollout" --submit

# Render a full 250-epoch rollout campaign only after considering smoke timing.
$PYTHON JOREK/zenodo_opsplit/orchestrate_native.py --case em \
  --manifest "$DATA" --pretrained "$RUN/em-term-training/best.pt" \
  --output "$RUN/new-em-full-rollout-plan" --full-rollout

# Frozen ES operators plus analytic EM physics; no EM training or fitted EM stats.
# Execute inside a GPU allocation. --modules can isolate transferred operators.
$PYTHON JOREK/zenodo_opsplit/evaluate_native_transfer.py \
  --manifest "$DATA" --checkpoint "$RUN/es-term-training/best.pt" \
  --case em --split test --trajectories 0 --horizon 200 \
  --output "$RUN/frozen-es-to-em.json"
```

Launchers snapshot code and manifests. Resubmit a saved training script to
resume unchanged checkpoints. Data/code/protocol changes require a fresh run.
`evaluate_native_transfer.py` can omit `--checkpoint` for an analytic reference;
its `--modules` switch supports separate advection/diffusion/momentum ablations.
The evaluator exits nonzero if any trajectory fails and preserves each failure.
Long-horizon commands above are prepared experiments, not completed results. The earlier CPU plans remain in `es-full-rollout-plan` and `em-full-rollout-plan`. The GPU full runs are submitted separately in `es-gpu-full` and `em-gpu-full`; see the GPU launch record below. The two-family frozen ES-to-EM check completed, but magnetic-perturbation relative errors near 1 do not demonstrate useful zero-shot transfer.


### GPU elliptic solves and full training

`gpu_sparse.py` binds the pinned [NVIDIA cuDSS 0.7.1 API](https://docs.nvidia.com/cuda/cudss/0.7.1/functions.html)
without changing the shared Python environment. Runtime installation:

```bash
$PYTHON -m pip install --no-deps \
  --target /pitagora_work/FUPB2_UKAEA_ML/vgopakum/software/jorek-cudss-0.7.1.6 \
  nvidia-cudss-cu12==0.7.1.6
```

The default library path points to that installation; `JOREK_CUDSS_LIBRARY` can
select another installation of the same ABI. Training provenance records the
library path and SHA256. Numeric factorization, solutions and adjoint solves
use GPU float64; symbolic reordering may use CPU inside cuDSS. Symbolic analyses
are cached by device, stream and exact sparsity pattern. Numerical factors are
recomputed for changed coefficients. The adjoint factorizes the transposed
matrix because this cuDSS version does not support transpose solves. Every solve
checks cuDSS errors and the true residual; there is no silent CPU fallback.

`native_gpu_benchmark.json` records full-resolution five-frame loss/backward
comparisons on one training trajectory per regime. On the allocated H100:

| Regime | CPU elliptic backend | GPU elliptic backend (warm) | Speedup |
|---|---:|---:|---:|
| ES | 36.07 s | 3.06 s | 11.8× |
| EM | 32.80 s | 3.78 s | 8.7× |

Losses agree to about 1e-12 absolute, and measured relative FNO-gradient error
is at most 6.5e-10. Positivity decisions agree in these checks. These are
single-window timings, not whole-epoch timing or a stability guarantee.
The existing 48 CPU tests and four CUDA-specific tests pass, covering
nonsymmetric implicit adjoints, cache reuse, zero RHS, singular-system rejection
and ES/EM state/gradient parity.

The first full training jobs, ES **2775369** and EM **2775370**, failed early with
nonpositive density. Their logs and checkpoints are preserved, with
immutable snapshots in `es-gpu-full` and `em-gpu-full` under the run directory.
They reuse the completed term checkpoints and retain 250 epochs, ES100/EM76
trajectories, first50 frames, five-step loss, effective batch 20, Adam1e-3,
weight decay1e-4, clipping1 and StepLR50/gamma0.5. GPU speedup does not remove
the cost of eight implicit substeps and single-trajectory gradient accumulation;
full campaigns can still take weeks. Use observed batch times to refine that
estimate rather than treating these short benchmarks as completion forecasts.

The full launcher requests a 24-hour allocation and checks a 22-hour runtime
budget between batches and validation trajectories. It saves an exact
optimizer/scheduler/RNG checkpoint, exits75, and the generated script requeues
its own Slurm job. A restart resumes the saved epoch/window offset; ordinary
training errors remain failures and do not cause a retry loop. This preserves
250 epochs across allocations without changing the mathematical protocol.

```bash
# A fresh full GPU run (existing launches above need not be duplicated):
$PYTHON JOREK/zenodo_opsplit/orchestrate_native.py --case es \
  --manifest "$DATA" --pretrained "$RUN/es-term-training/best.pt" \
  --output "$RUN/new-es-gpu-full" --full-rollout --submit
```

Use `--elliptic-device cpu` in the trainer for CPU/GPU validation. Default
elliptic placement follows the FNO device; the launcher explicitly selects CUDA.

### Positivity and explicit-wave correction (4 November 2026)

The original ES failure is reproducible at `jorek_run0497`, window start42.
The original EM failure is reproducible at `jorek_run017`, start35, using the
saved checkpoint after its first optimizer update. Both windows now complete.
Smaller fixed steps alone did not repair the analytic centred scalar scheme:
the ES analytic window still failed at 128 substeps per saved frame. These are
coarse-grid discretization failures, not evidence that JOREK itself loses
positivity or that the cuDSS solve is inaccurate.

`NativeRollout` now constructs an implicit first-order upwind/isotropic-diffusion
scalar fallback only when the high-order scalar candidate is nonpositive.
Interior transport fluxes cancel pairwise; the physical wall has zero normal
transport. Its M-matrix gives a positive low-order candidate. A single convex
fraction limits the whole density or temperature correction, retaining 99% of
the largest admissible correction. Admissible high-order scalar updates remain
unchanged. This is numerical stabilization of the gridded approximation, not
an additional native JOREK equation term. It is more diffusive when active;
limiter frequency and retained fractions must accompany scientific results.
No individual field values are clipped. The implicit solves and convex fraction
are differentiated, including the dependence on the density-dependent mass.

EM `jorek_run016`, start28 exposed a second failure: the explicit magnetic/fluid
coupling could become unstable and make the low-order transport solve badly
conditioned. A local CFL<=0.4 controller now includes E-cross-B transport,
compression and poloidal Alfvén speed. Eight substeps is a minimum, not a fixed
count. The reproduced learned window completes with 128 substeps over five
frames, compared with the original 40. The full saved-frame interval is always
advanced; controller choices are held fixed in the adjoint. More than 4096
substeps per saved frame, nonfinite states or failed linear residual checks
remain hard failures. The 1e-8 linear residual tolerance was not relaxed.

Source-state screening now checks pressure as well as density and finiteness.
For the unchanged ES100/EM76 training selections, 4484 ES and 3374 EM five-frame
windows are admissible; 16 ES and 46 EM windows contain invalid source states.
The EM 46 includes the six density-invalid windows already excluded previously.
Excluded windows are listed by trajectory and start in the audit. Two ES and
three EM validation prefixes also contain nonpositive thermodynamic fields and
are reported separately, consistently with the training validation policy.

`validate_native_windows.py` checks every admissible five-frame training window,
backpropagates the worst-loss window of every training trajectory through the
rollout and all four term anchors, and checks each admissible 49-step validation
rollout. It uses four worker processes on one allocated GPU. The full launcher
now submits this audit first; training is an `afterok` dependency and additionally
requires a passed report matching exact code/data/checkpoint hashes, selected
trajectories and substeps. A partial audit cannot authorize full training.
Validation of initial weights cannot guarantee every future optimizer iterate.

The replacement campaigns, under `/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-native-20261104`, are:

| Case | Directory | Exhaustive audit | Full training after audit |
|---|---|---|---|
| ES | `es-positive-full` |2775515|2775516|
| EM | `em-positive-full` |2775517|2775518|

At submission the audits await GPU resources; training awaits successful audits.
This is a submitted validation/training chain, not a claim that the exhaustive
audits have already passed. `window-validation.json` in each directory records
live coverage, failures, invalid-source exclusions, limiter use and elapsed time.
The old failed runs remain intact. Training restarts from the completed term
checkpoints, preserving 250 epochs, five-step loss, effective batch 20 and the
paper optimizer/scheduler. Timed checkpoint/requeue remains enabled.

Validation so far: 56 CPU tests and 5 GPU tests pass, including active-limiter
CPU/GPU state/gradient parity, fallback conservation and continuum convergence,
CFL scaling, complete time advancement, and stale/partial audit rejection.
Reproduction and pilot evidence is under `reports/`; a pilot is not a full audit.

**Runtime accounting:** the old warm-window timings imply about 39.7 ES GPU-days
and 36.9 EM GPU-days for 4484/3374 windows times 250 epochs, before validation,
checkpointing, queueing or additional CFL substeps. Each window evaluates at
least 40 physics substeps, three/four implicit solves per substep, four FNO
modules and their adjoints. Effective batch 20 is serial gradient accumulation,
not a batched elliptic solve. A 1.1-million-window ES campaign is consequently
much more expensive than the AR/NODE baselines despite faster individual solves.
The new stability control can increase this cost. No epochs, windows or physical
terms were silently removed to obtain a shorter estimate. Substantial further
speedup requires batched solves/trajectory parallelism or a different numerical
training protocol; the latter changes the paper comparison.

The corrected EM pilot completed 84 admissible training windows, both 49-step
validation rollouts, and worst-window gradient checks without failures. Each
of the original ES/EM failure windows and the additional EM wave-instability
window also passed two optimizer updates with finite gradients. Fresh ES/EM
GPU smoke training, validation, checkpoint save and completed-run resume passed.
These bounded checks support the repair; the exhaustive jobs above remain the
required gate for full training.
