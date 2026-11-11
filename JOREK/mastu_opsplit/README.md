# MAST-U native-mesh OpsSplit

Implementation of the [study handoff](../MASTU_OPSSPLIT_HANDOFF.md), started 2 November 2026. **5 November update:** the native physical-RHS SBDF2 split failed temperature positivity at all three full-window timesteps. The independent native reference passed refinement (maximum dynamic-normalized field difference 0.329635%). A matched late restart completed with native JOREK and split Euler, while SBDF2 failed again. These results do not establish full-window Euler convergence.

The selected fluid-benchmark formulation is now wired into `scripts/train.py`: Euler rollouts of normalized field states, numerical spline gradients in the linear branch, and L-SplineFNO contributions trained through field-trajectory losses. The [normalized Euler protocol](configs/normalized_euler.yaml) and [pilot configuration](configs/euler_pilot.yaml) describe this separate experiment. Native nonlinear-term labels and native split convergence remain requirements of the physical-RHS experiment, not prerequisites for trajectory supervision. Reference-data convergence and learned-rollout validation remain required. Legacy [gate records](reports/gates.json) remain separate.

## Native physical defect (11 November 2026)

The new [physical-defect experiment](IMPLICIT_OPSSPLIT.md#native-physical-defect-11-november-2026)
retains the state-dependent native volume RHS and learns only
`G(q) - G(q*)`, anchored at the immutable native reference. It preserves the
nonzero reference forcing, treats the frozen tangent implicitly, and keeps the
native mass operator frozen. Both first-order IMEX Euler and two-stage ARS222
are available; algebraic constraints, iterated Mach wall relations and the
existing positivity/elliptic guard act at each stage. The physics control now
includes the full known nonlinear volume equations.

Initial CUDA checks passed at the reference (maximum absolute RHS error
`4.75e-13`) and four native training states. A broader 240-state check then found
22 late-state induction comparisons above the declared tolerance (worst
`1.58e-4` of training-label RMS), although every critical-point search succeeded.
The cause is native source normalization: its axis search can select a stationary
saddle, while the frozen tracker requires a minimum. A separate autonomous
source-axis correction now passes all 240 selected states at unchanged thresholds
(worst training-scaled RMS `2.38e-7`), with strict root diagnostics unchanged.
Original failures remain archived. These checks cover selected training states,
not unrestricted parity. Moving X-point derivative checks passed. Rollout sources
use current predicted states without native trajectory histories. Evidence is under
`/pitagora_scratch/userexternal/vgopakum/mastu_physical_defect_20261111/diagnostics/`.

The BE physics screen passed on one training trajectory through 16 intervals
(0.234 microseconds), with no network calls or positivity limiting. Its cumulative
normalized MSE was `1.354e-4`, `6.718e-5` and `4.511e-5` with 1/2/4 physical
microsteps per saved interval. All constraint, boundary and positivity audits
passed. See the [audited status report](reports/physical_defect_status_20261111.json).
The full-window training screens are complete. BE passed H96 and all audits on
both cases at 1/2/4 microsteps: H96 MSE ranged from `1.362e-3` to `2.906e-3`.
ARS222 failed axis tracking in three of four case/substep pairs. Broader searches
(up to 2,048 cells, multistart Newton) did not recover a qualified smooth axis;
acceptance gates remained unchanged. Both original CPU-budget interruptions and
subsequent numerical outcomes are preserved. These results support BE for the
current pilot and do not establish a general integrator ranking.

The original frozen 64-update CUDA pilot now completes its matched two-case
held-out comparison: the learned defect reduces H96 total MSE by **5.16–5.53%**
versus its full-physics control at 2/4/8 microsteps. Both cases improve, all audits
pass, and density-field error worsens slightly (0.18–0.19%). Guards remain active.
Most of the roughly 98.8% gain over the older affine baseline comes from physics,
before learning. Those pilot results retain the original source-parity caveat.
Separate corrected-source verification now passes all four matched rollouts at
two microsteps and confirms a **5.157%** H96 reduction with the same frozen weights
(`1.97816e-3` physics versus `1.87614e-3` learned). One independent CUDA training
update also passes; its weights were not used in evaluation. Corrected sources
have not repeated the full training campaign or four/eight-step evaluation.
This is one seed and two cases, not a 2,000-update/20-case study or live deployment.
The implemented profile path supports the pinned
`n=0`, baseline-coefficient, fixed-boundary double-null case on this mesh;
topology changes, coefficient/geometry transfer and nonaxisymmetric modes remain
unqualified. Frozen mass and identified boundary rows remain approximations.
See the [formulation, limits and commands](IMPLICIT_OPSSPLIT.md#native-physical-defect-11-november-2026).
[`configs/physical_defect.yaml`](configs/physical_defect.yaml) and
[`scripts/train_physical_defect.py`](scripts/train_physical_defect.py) provide
CUDA-only training with bounded resumable workers, exact optimizer/RNG recovery
and checked reuse of native setup caches.

## GPU one-step campaign (9 November 2026)

Every compared model takes one saved state to the next. AR, Euler NODE, explicit
OpsSplit, the one-step conditional L-SplineFNO and the three implicit arms are
trained with the same 80/0/20 split, four-step rollout windows, loss, guards,
2,000-update budget and seeds 17–19, and are evaluated from frame 0 at 16/32/96
intervals with 1/2/4 substeps (AR: 1). Each run uses one H100.

**One-step conditional L-SplineFNO** (`EulerSplineModel('conditional', ...)`):
`q' = G(q + h P N(q; [D/D0, (t - t0)/T]))`. This is the Euler NODE update with the
FiLM input widened by the time `t` of the state being advanced. `t0` is the
trajectory start and `T` is `time_conditioning_intervals` (96) intervals. Euler
substeps advance `t` with the state. The model sees exactly the windows, targets
and rollout feedback of the other families, and adds 64 FiLM parameters.
The earlier direct time-query model (below) queried up to 96 intervals from `q0`.
That gave it longer temporal supervision, so it is not part of this comparison.

**Device path.** `--device cuda` (config `device`) runs native sampling, the
consistent-mass projection (cuDSS LU of the symmetric scaled mass) and the
elliptic/positivity guards on the GPU. Implicit arms use the cuDSS tangent system.
`configs/gpu_onestep.yaml` equals `ensemble_training.yaml` except for `device`,
`threads` and `time_conditioning_intervals`.

**Campaign.** The study snapshot is at
`/pitagora_scratch/userexternal/vgopakum/mastu_gpu_onestep_20261109/study`. Array
`slurm/gpu_onestep.sbatch` (`--array=0-18`, `fupb2_fauna`) runs one task per
model/seed. Tasks 0–11 cover AR/NODE/OpsSplit/conditional and tasks 12–18 the
implicit arms. Each task trains, evaluates the test set, then runs the
fixed-conditioning D⊥ OOD set into `ood/`. The conditional model also runs the
true-D/D0-in-FiLM extrapolation into `ood_film/`. The implicit arms also run
coefficient physics into `ood_coefficient/`. `SMOKE=1 FAMILY=...` runs the
six-update smoke. Summarize with
`scripts/summarize_gpu_onestep.py --root <campaign> --output comparison.json`.
The two-page report (`Paper/results_one_page/build.py --refresh`) reads only this campaign.
The conditional OOD steps of tasks 9–11 failed because the pinned OOD loader has no `gamma_ood` attribute.
They were rerun as array 2812238 (`ood_conditional_rerun/`) with the fixed evaluator against the unchanged snapshot.

Results (all 19 runs complete, seed means, test MSE at 16/32/96 intervals): NODE 0.064/0.18/1.00,
explicit OpsSplit 0.064/0.18/0.99, conditional 0.061/0.18/0.94 (wins 25/60 paired trajectories at 96),
implicit labels+rollout 7.5e-4/6.5e-3/0.11 (wins 120/120 with rollout-only), physics-only 5.0e-4/5.9e-3/0.10.
AR diverges (1.3e12 at 96). OOD with conditioning 1: NODE 2.86 and implicit 1.49 at 96; implicit with the true
D⊥ in its physics 0.25; conditional with the true D⊥/D0 in FiLM 1.2e7.

## Larger remainder and learned tangent correction (10 November 2026)

The GPU campaign showed the 9,727-parameter remainder adds nothing over physics-only. Its label loss
plateaus at 0.86–0.92. This study gives the implicit model more room in two ways.

**Larger N_theta.** `configs/implicit_large.yaml` uses the upstream L-SplineFNO default: a 64×64 grid,
12 modes, width 32 and 4 layers, about 1.19M parameters. Everything else matches `implicit_opsplit.yaml`.
In the smoke test its label loss fell to 0.21 within two updates.

**Learned ΔL** (`TangentCorrection` in `implicit_opsplit.py`, `configs/implicit_dl.yaml`,
`implicit_large_dl.yaml`). The correction is `ΔL(q - q*) = assemble(Σ_p T*_p m_p(R,Z) (jet_p(q) - jet_p(q*)))`
over the 83 nonzero couplings of the frozen tangent:
- `m = 0.5 tanh(φ W)` on a 6×12 cubic-spline basis.
- `W = 0` gives exactly the frozen tangent.
- ΔL only re-weights couplings that model303 already has.

ΔL is treated implicitly by Picard iteration on the fixed factorization,
`q'_{k+1} = solve(N(q) + ΔL q'_k)`, so the matrix is never refactorized. Training uses 3 iterations.
Evaluation iterates to 1e-7 or fails loudly.

Multipliers are shared per (weak equation, field) (`share: block_field`, 22 fields, 1,584 parameters).
The contraction of the Picard map was measured on the production system
(`/pitagora_scratch/userexternal/vgopakum/mastu_tangent_correction_diagnostics/picard_contraction.json`).
- With per-equation or per-(equation, field) sharing, any sign or spatial pattern at |m| = 0.5 gives
  ρ ≤ 0.50.
- With independent multipliers per coupling, ρ reaches 30–61 and the iteration diverges.
- The cause: the value and gradient terms of one field cancel inside the stiff tangent, so they must
  share one multiplier.

The coefficient-physics OOD step rebuilds `F*, L*` for the true D⊥ and keeps ΔL as trained.

Campaign: `/pitagora_scratch/userexternal/vgopakum/mastu_tangent_correction_20261110` runs
`slurm/tangent_correction.sbatch --array=0-11`. It covers the larger network rollout-only, the larger
network with labels + rollout, ΔL with the small network, and the larger network + ΔL, each for seeds
17–19. Every run does test, fixed-conditioning OOD and coefficient-physics OOD. The references are the
small-network and physics-only runs of the GPU campaign. Summarize with
`scripts/summarize_tangent_correction.py --root <campaign> --output comparison.json`.

Outcome (`comparison.json` in that campaign):
- The larger-network runs pass every audit. On test they reach 0.097–0.101 at 96 intervals, against
  0.103 for physics-only, and z_j remains about 91% of the error. In the coefficient-physics OOD they are
  worse than physics-only, at 0.26–0.51 against 0.24.
- All six ΔL runs failed their audits. Evaluation allowed 12 Picard iterations, and on 2–8 of the 20
  cases per run at one substep the iteration stalled at a change of 1–2.5e-7 (it was still contracting).
- The 3-iteration training also left a 0.6–2.3% unconverged change, so the trained and evaluated maps
  differed.

Rerun of the ΔL arms with converged-Picard training: configs `implicit_dl_conv` and
`implicit_large_dl_conv`, run by `slurm/tangent_correction_conv.sbatch --array=0-5`. Each training step
iterates without gradients until the change is below `train_tolerance` = 1e-6, then backpropagates
through 3 iterations (Jacobian-free backpropagation), so the trained map is the converged solve.
Evaluation allows 40 iterations. Campaign:
`/pitagora_scratch/userexternal/vgopakum/mastu_tangent_correction_conv_20261111`. Summarize with
`--conv-root <rerun campaign>`, which also pairs each ΔL arm with the same network without ΔL.

## Direct time-conditioned baseline (superseded for the comparison)

`configs/conditional_time.yaml` adds an initial-field L-SplineFNO baseline alongside
the implicit experiment. It conditions the seven initial fields through upstream
FiLM on `[D_perp / D_perp_baseline, elapsed_seconds / T]`, where `T` is the training
query duration (96 observation intervals by default). Each prediction is
`q0 + (elapsed_seconds / T) * learned_correction(q0, parameters, time)`, projected
to native coefficients and passed through the existing elliptic and whole-cell
positivity guards. A zero-time query returns the initial coefficients exactly.
Every query uses the original initial state; predictions never feed later queries.
There is no time integrator, but native mass projection and elliptic guards still
require spatial solves. This adapts the paper's conditioning to input fields;
it is not the geometry-only conditional lifting used in its steady-state examples.

The baseline uses the current ensemble/implicit protocol: the original 60 training
and 20 validation trajectories become 80 training trajectories, normalization is
pooled over those 80 only, and the 20 test trajectories remain held out. Checkpoint
selection is the final fixed-budget update. Each of 2,000 updates samples a training
trajectory and four distinct future times uniformly from intervals 1–96, always
starting at frame zero. The saved sampling schedule and optimizer/RNG checkpoint
support exact resume. Only the requested frames are read for training.

Evaluation queries every saved time independently and reports cumulative-through-
horizon errors at 16, 32 and 96 intervals using the shared volume-weighted metrics,
constraint defects, positivity bounds, boundary checks and guard intervention counts.
Both raw and guarded prediction MSEs are recorded. `evaluation_substeps` must be
`[1]`; a substep-refinement comparison has no meaning for this model.
Equal updates do not imply equal compute or parameter counts, and the direct
baseline sees longer-time targets than the four-step rollout training. Its default
test establishes held-out-initial-condition performance within the supervised time
range; it does not establish withheld-time interpolation or future-time extrapolation.

The accepted baseline ensemble has one PDE coefficient setting. Its conditioning
parameter is therefore always one: this run tests time/initial-condition learning,
not learned coefficient generalization. The model interface also accepts the
existing two-parameter `[D/D0, gamma]` convention, but the baseline runner rejects
that configuration because this dataset supplies only one parameter. Parameter-varied
training requires a separate accepted dataset; existing coefficient OOD test cases
must not be repurposed as training cases.

From this directory, run:

```bash
python scripts/train_ensemble.py --config configs/conditional_time.yaml \
  --model-config configs/lsplinefno.yaml --family conditional \
  --output /absolute/fresh/conditional_run
```

Add `--resume` to resume an unchanged experiment. `results.json` and
`conditional/metrics.json` contain aggregate and trajectory metrics; checkpoints,
query schedules, normalization and source/dependency provenance are retained.
For cluster execution, use `scripts/freeze_ensemble.py --output /absolute/fresh/study`
to create a snapshot. Submit **`slurm/conditional_time.sbatch`** with `SMOKE=1`
and `--array=0-1` first: each arm runs three real-data updates with an interruption
after update one, checkpoint resume, and one-trajectory evaluation at 4/16 intervals.
Review `smoke_passed.json`, runtime and memory before the production array.
Production uses `SMOKE=0`, `--array=0-5%3`, and exported `STUDY_ROOT`/`OUTPUT_ROOT`.
Tasks 0–2 run the full-time model at seeds 17–19; tasks 3–5 run the matched variant.
The freeze script's printed commands target the existing AR/NODE/OpsSplit experiment;
use the conditional job script for this baseline.

`configs/conditional_time_matched.yaml` samples exactly the existing four-step
rollout windows and queries all four target frames directly from the window's
initial field. It uses four intervals for time normalization. At evaluation it
queries from frame zero; horizons 16/32/96 are explicitly labeled lead-time
extrapolation. This controls target exposure, but differs from autoregressive
composition and does not make the two architectures identical.
`scripts/summarize_conditional.py` checks data/normalization/budget compatibility,
matching sampled windows for the short-time arm, paired test cases, and physical
audits. It withholds three-seed averages unless every seed has a complete horizon.

### Coefficient extrapolation of conditional checkpoints

The accepted ten-case density-diffusivity OOD dataset spans `D_perp(1)/D0` from
2 to 10 (`D0 = 1e-5`), with initial conditions matched to held-out baseline cases.
`evaluate_ood_fixed_conditioning.py --family conditional --conditioning coefficient`
loads each final frozen checkpoint and supplies the true multiplier. The default
`--conditioning fixed` supplies one as a control. Both use the saved training
normalization, original time scale, direct time queries, and physical guards.
No retraining or OOD-based checkpoint selection occurs.

The campaign at
`/pitagora_scratch/userexternal/vgopakum/mastu_conditional_coefficient_ood_20261109`
tests both conditional variants, seeds 17–19, and horizons 16/32/96.
`slurm/conditional_ood.sbatch` maps the twelve production tasks to seed, variant,
and conditioning mode; its four-task smoke array checks both variants/modes first.
`summarize_conditional_ood.py` writes `comparison.json`, including raw and guarded
errors, guard interventions, per-coefficient scores, and the corresponding ten
baseline-case errors. Three-seed means require all ten cases and successful audits.

All checkpoints trained only at `D/D0 = 1`, so this tests coefficient extrapolation,
not learning across a parameter range. The four-step variant also extrapolates in
time. Each OOD coefficient has a different initial condition, so this dataset does
not isolate a coefficient-response curve at a fixed initial condition.

## Independent Euler dataset

`scripts/build_euler_dataset.py` prepares 100 independent density/temperature
perturbations on the same warm-start MAST-U equilibrium, with 60 training,
20 validation, and 20 test trajectories assigned before simulation. Location,
width, amplitude, and sign vary. Refinements of a trajectory keep its split.
The shared unperturbed control is used only for numerical diagnostics.

The submitted ensemble lives at
`/pitagora_scratch/userexternal/vgopakum/mastu_euler_ensemble100_20261105`.
Array job `2780105` runs at most 16 four-rank cases concurrently; job `2780106`
packages the accepted trajectories after the array finishes. `status.py` reports
current progress. `manifest.json` is produced only when all cases are accepted;
`completion.json` additionally records successful loader readback. A failed or
missing case produces `incomplete.json` instead of a complete release.

Each case covers 4.217739–5.623652 microseconds with 97 saved frames. The native
reference solver is unchanged; Euler is the surrogate integration rule. Initial
dt64/dt128 checks require <=1% dynamic-normalized error for all seven fields,
control and response, <=1% energy error, and positive Q8 density/temperature.
A failed convergence check triggers a dt256 run with the same initial state;
that comparison uses the common converged dt128 control for both responses.
Failed runs and checks remain in the provenance. Native float64 coefficients,
one shared bicubic geometry file, physical observation times, checksums, and
training-only normalization are exported without Cartesian resampling.

After `completion.json` exists, the common reader is:

```python
from mastu_opsplit.euler_dataset import EulerDataset

data = EulerDataset("/pitagora_scratch/userexternal/vgopakum/mastu_euler_ensemble100_20261105/manifest.json")
train = data.windows("train", horizon=4)
validation = data.windows("validation", horizon=4)
test = data.windows("test", horizon=4)
sample = train[0]  # state: [5, canonical_native_dof, 7], normalized float64
```

The reader verifies artifact hashes, split identities, shared times, and that
normalization used exactly the training trajectories. See
`configs/ensemble_dataset.yaml` for the input contract. This is a dataset build,
not a scientific model-training launch. It supports initial-condition
generalization on one fixed mesh/equilibrium; full spatial convergence and
cross-equilibrium generalization are not established. Archive the completed
dataset separately from shared scratch for long-term retention.

## Normalized Euler pilot

### Physical continuity tangent experiment

`configs/continuity_euler_pilot.yaml` changes only the OpsSplit linear branch
relative to the guarded 120-update pilot. It uses the axisymmetric model303
density and pressure advection/compression terms, with the background frozen
at the spatial mean of training frames `[0,24)` from both parents. Define

```
a = (R*u_Z - Vpar*psi_Z/R, -R*u_R + Vpar*psi_R/R)
c = 2*u_Z - (Vpar_R*psi_Z - Vpar_Z*psi_R)/R
rho_t = a . grad(rho) + c*rho
T_t   = a . grad(T) + (gamma-1)*c*T
```

The temperature equation follows from `p=rho*T` and
`p_t = a.grad(p) + gamma*c*p`. `continuity.py` evaluates the complete directional
derivative of these two expressions at the training mean, including the
couplings through `u`, `psi`, and `Vpar`. The pinned native model303 source has
the density/energy RHS at lines 887–960 and pressure temporal mass couplings
at lines 1510 and 1551 of `models/model303/mod_elt_matrix_fft.f90`.
`models/preset_parameters.f90:180` sets gamma to 5/3; the pilot input does not
override it. The implementation is checked against the independently expressed
native density/pressure bracket equations and their pressure-to-temperature
chain rule.

Perturbation values use native bicubic sampling; perturbation gradients use
analytic first derivatives of the L-SplineFNO scatter/synthesis. The background
values and gradients are native bicubic evaluations. Physical channel scales
are applied before the tangent and removed afterwards. The multiplier is the
observation interval divided by the restart's `sqrt_mu0_rho0`, with no fitted
amplitude. Native consistent-mass projection returns the result to the native
coefficient state. The branch is linear in that state for fixed learned scatter
weights and is exactly zero at the training mean. Its only nonzero output
channels are density and temperature; the neural branch handles the remaining
equations, diffusion, sources and residual nonlinear dynamics. The background
forcing is also part of the learned residual.

This is a **partial physical operator**, not the full JOREK Jacobian, a native
pressure-mass solve, or a conservative MHD solver. The coarse learned spline
gradients approximate the native field gradients; analytic differentiation does
not guarantee accurate field reconstruction or stable explicit integration.
All Bernstein positivity and elliptic-constraint corrections remain active.

The experiment fixes the architecture, seed, training-window schedule,
normalization, optimizer, learning rate, two-step training loss, validation
checkpoint selection, 16-step held-out horizon and 120 updates per family.
AR, NODE and this OpsSplit have identical trainable parameter counts; runtime
is measured separately rather than equated. No test score selects the operator
or its amplitude. `scripts/diagnose_euler_linear.py` performs validation-only
linear-on/off/sign-reversed ablations with the guards active, and compares
L-SplineFNO versus native gradients. These inference ablations do not substitute
for retraining the nonlinear operator.

The previous guarded 120-update checkpoint's validation-only ablation gives
MSE 0.22185 with its curvature branch and 0.22843 without it on the perturbed
parent; on the control it gives 0.06394 with and 0.06073 without. Thus the
branch helps one validation parent and hurts the other. Its parallel-velocity
MSE increases on both when enabled. Constraint recovery also changes the
effective potential contribution: on the perturbed validation initial state,
the raw linear `psi` RMS is 0.00555, whereas the change after closure is 0.01936
because the predicted current drives the reconstructed potential.

At those same two validation starts, the learned 16x32 spline representation's
volume-weighted gradient error relative to exact native bicubic gradients is
37–42% for the potentials, 97.6–97.8% for density, 67–69% for temperature,
76.5–77.6% for parallel velocity and approximately 100% for current/vorticity.
These are errors in the reconstructed field gradients, not failures of the
analytic derivative formula. They identify a substantial spatial approximation
limitation. The seven-step validation loss also does not directly optimize the
16-step test horizon; the preceding OpsSplit had slightly better validation
MSE than NODE but worse held-out MSE. These observations support competing
approximation and rollout-generalization explanations, not a unique causal
proof. See the immutable experiment's `previous_diagnosis.json` for per-field
and guard-aware results.

The physical-operator experiment completed as job **2785214** (25m05s,
exit 0). All **31 tests** and the actual-mesh forward/backward preflight passed.
The immutable run is
`/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_continuity_pilot_20261105`;
[the verified report](reports/continuity_pilot_20261105.json) includes the
protocol, diagnostics, guards, substep results and artifact paths.

| Model | Mean held-out normalized volume MSE |
|---|---:|
| AR | 5.62320450 |
| NODE | 0.76919436 |
| Previous signed-curvature OpsSplit | 0.79168474 |
| Physical continuity OpsSplit | **0.75460207** |

The physical branch improves the pilot's aggregate MSE by **1.897% relative
to NODE** and **4.684% relative to the preceding OpsSplit**. Both parent
rollouts improve relative to NODE: 1.18070509 to 1.17274955 for the perturbation
and 0.35768363 to 0.33645460 for the control. The control's density MSE improves
by about 21.2%; temperature and several other individual fields do not all
improve. All three selected checkpoints are at update 120. AR/NODE training
losses and saved predicted trajectories reproduce the previous run bitwise.
Each new family has 9,673 parameters. Measured training/evaluation time is
327.6s for AR, 359.3s for NODE and 587.1s for physical OpsSplit: equal updates
and parameter counts do not mean equal computational cost.

Keeping the newly trained neural weights fixed, removing the physical branch
changes validation MSE from 0.22792441 to 0.22800892 (perturbation) and from
0.05923531 to 0.06076830 (control). The control density component changes from
0.03271358 with the branch to 0.04425320 without it, while temperature worsens
with the branch on both parents. At the validation start, its density RHS RMS
is 67.2%/47.5% of the neural density RHS RMS; it is a measurable contribution,
not a numerically negligible add-on. Gradient approximation errors remain
large. This supports the selected coupling's usefulness in this pilot, not a
claim that every physical term or every field benefits.

All 16-step rollouts retain positive subdivided Bernstein bounds for density
and temperature; independent Q12 and vertex checks also pass. The maximum
interior current/vorticity defect is 5.244e-13. The physical model's limiter
acts on 26 of 64 density/temperature field-updates, with minimum retained
fraction 0.91086. Positivity is therefore still enforced by the shared guard;
the raw neural proposals are not all positive. The fixed-weight 1/2/4-substep
checks over four test intervals give squared-difference reduction factors
4.17 and 5.28, with positive sampled fields. These short checks do not prove
full-horizon convergence of the learned model or the physical JOREK split.

The result is an encouraging **single-seed, same-parent chronological pilot**,
not evidence of statistically reliable superiority or independent-trajectory
generalization. The partial linear operator and shared guards do not enforce
full MHD dynamics or conservation.

### Guarded matched pilot

`configs/guarded_euler_pilot.yaml` runs 120 updates per family with shared
native elliptic closure and positivity handling. The preceding 30-update
checkpoint's validation-only ablation improves rollout MSE from 0.41863 to
0.35769 (perturbation) and 0.14803 to 0.13453 (control) when its linear branch
is disabled with nonlinear weights held fixed. The imposed positive curvature
term opposes the observed increment in several channels. Constant-field
synthesis error is only 2.38e-7; this check does not implicate a large constant
reconstruction error as the cause. These are inference ablations, not a
controlled attribution of all training effects.

The revised linear coefficients start at zero and use a bounded signed `tanh`
gate. OpsSplit therefore equals NODE at initialization and learns the sign and
strength of its analytic spline curvature contribution from training data.
These signed coefficients are effective learned corrections, **not physical
diffusivities**, and have no general explicit-stability guarantee.

All families predict the seven fields and then apply the same correction:

- Recover interior `psi` and `u` by native elliptic solves from predicted `zj`
  and `w`. Use the R^-2 weighted current identity and ordinary toroidal-volume
  vorticity identity. Boundary nodal values remain fixed to the previous state;
  boundary derivative DOFs remain learned because they evolve in native JOREK.
  Interior potential predictions are replaced by this closure.
- When required, blend density/temperature coefficients toward the positive
  **training-only spatial mean** until all subdivided Bernstein coefficients
  exceed a small positive floor. Their convex hull bounds the entire native
  bicubic cell, including edges and vertices. A 1% interior margin prevents
  repeated near-floor numerical trapping.
  This preserves native continuity/aliases but can damp dynamics; intervention
  counts, minimum retained fraction and raw proposed minima are saved.
- Add the same small penalty on negative proposed Bernstein bounds during all three models'
  training. The final rollout metrics include the corrections. No future fields
  enter either correction.

This is Euler followed by algebraic closure and a positivity limiter, not an
unconstrained Euler RHS. Substep checks include these operations. The sufficient
Bernstein condition is conservative: it may modify a positive polynomial whose
coefficient lower bound is negative. One de Casteljau subdivision reduces that
restriction; the training mean has positive bounds (rho >=0.000484568,
T >=0.0000879352). The audit independently checks Q12 samples and vertices on
every saved rollout frame. The closure enforces the declared interior weak constraints, not every
physical MHD boundary equation. All-field rollout errors remain part of the
assessment, so admissibility alone is not an accuracy or generalization claim.

The actual-mesh preflight changes a native reference by only 2.31e-21 normalized
volume MSE and gives corrected constraint defects below 5e-13. Exact adjoint,
positivity, zero-initialization parity and end-to-end regression tests pass.
The initial Q8-only guarded run (`2784793`) was stopped when independent checks
found negative AR density/temperature between its limiter points. Its partial
results and failure evidence remain under
`/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_guarded_pilot_20261105/`.
The whole-cell replacement, including the diagnosis and frozen source, is
`/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_bernstein_guarded_pilot_20261105/`
(job `2784836`, completed in 17m33s). All 28 tests pass. Mean held-out normalized
volume MSE is **AR 5.62320, NODE 0.76919, OpsSplit 0.79168**. OpsSplit remains
2.92% worse than NODE despite slightly lower validation error; no superiority
claim is supported. All validation-selected checkpoints are at update 120.

Every predicted field passes the density/temperature Bernstein lower-bound
check, and independent Q12/vertex checks are positive for all six rollouts.
Maximum interior weak current/vorticity defect over predicted frames is
5.17e-13 (the independent final-frame audit remains below 6e-13). OpsSplit's
fixed-weight 1/2/4-substep squared differences decrease by 4.01 and 3.91 on the
two four-frame checks, consistent with first-order behavior. This is not a
full-horizon timestep-convergence or full-MHD conservation certificate.

Corrections remain active: AR/NODE/OpsSplit require 15/26/27 limited field
updates out of 64 density/temperature updates across the two test rollouts.
The smallest retained proposal fractions are 0.386/0.889/0.885 respectively.
These are guarded surrogate results; the neural operators have not independently
learned positivity or the algebraic constraints. The positivity blend can change
integrated density/thermal content, and the sufficient Bernstein condition can
damp physically positive fields. Same-parent chronological splits and one seed
remain insufficient for an independent generalization claim.

`completion.json` records checks and limitations; `pilot/results.json` contains
per-field errors and timestep sensitivity; `pilot/audit.json` contains the dense
positivity/constraint audit. Full native-coefficient prediction arrays, selected
checkpoints, normalization and the source snapshot are retained. The concise
repository record is `reports/guarded_pilot_20261105.json`.

### Earlier unguarded pilots

`src/mastu_opsplit/euler.py` retains the canonical native bicubic coefficients,
including derivative DOFs and verified aliases. Its consistent mass projection
maps the L-SplineFNO outputs back into this space. The default linear branch now
uses the **L-SplineFNO learned scatter and analytic cubic synthesis derivatives**:
`L(q) = d * P_native [(dRR+dZZ) B_spline S_theta(q)]`. It shares the nonlinear
operator's scatter kernels but bypasses FiLM, lift, FNO layers and output MLP,
so it remains linear in the normalized state for fixed learned parameters.
The derivatives include physical-coordinate chain rules and derivatives of
the truncated basis normalization at the latent-grid boundary. No Cartesian
finite differences or native JOREK stiffness derivatives enter this branch.

This preserves the Euler/additive split used in `Expts/operator_splitting.py`
and `Utils/explicit_time.py`; the fluid source currently uses finite differences,
so its derivative backend is adapted rather than copied. The MAST-U linear
term remains a per-channel R/Z diffusion prior, **not the full linearized MHD
operator**. Positive learned diffusivities are scaled by latent-grid spacing
per saved frame, not physical model303 transport. That scale is a heuristic,
not a proof of Euler stability for the learned scatter. Physical wall conditions
and coupled current/vorticity constraints are not imposed. The audit reports
positivity, interior constraint defects and fixed-weight timestep sensitivity.
`linear_branch: native_bicubic` retains the previous derivative prior for replay.

AR predicts the next state directly. NODE uses the same L-SplineFNO in an Euler
residual update. OpsSplit adds the spline diffusion branch to that update.
All three use the same backbone, initialization seed, window order, rollout loss,
and optimization budget; OpsSplit adds seven trainable coefficients. The FNO's
internal Cartesian latent grid does not replace the physical unstructured mesh.
Sparse solves are on CPU with tested transpose actions for first-order autograd.

The bounded diagnostic uses two converged native reference trajectories (one
perturbation and its paired control). Stride-two sampling gives 49 observations,
29.289856 ns apart, across 4.217739–5.623652 microseconds. Frame ranges are
`[0,24)` training, `[24,32)` validation, and `[32,49)` testing. Normalization is
fit to training frames only. No window crosses a split boundary, and rollouts
use their own predictions. These are **same-parent chronological holdouts**;
they cannot establish generalization to independent simulations.

Run inside an allocated CPU job using the existing `slurm/train.sbatch`, with
`TRAIN_CONFIG` pointing to `configs/euler_pilot.yaml` and `MODEL_CONFIG` to
`configs/lsplinefno.yaml`. The direct command is:

```sh
python scripts/train.py --config configs/euler_pilot.yaml --model-config configs/lsplinefno.yaml --output /absolute/fresh/output
```

The trainer saves source/data hashes, normalization, validation-selected
checkpoints, per-field held-out errors, Q8 positivity checks,
and fixed-weight 1/2/4-substep comparisons. Run `scripts/assess_euler_pilot.py
--output /absolute/output` for the comparison plot and linear/constraint audit.
Production training still requires independent simulation families and a
declared evaluation protocol.

The first, native-gradient pilot completed on 5 November in 303 seconds (Slurm 2779845):
30 updates per model, two-step training rollouts, and 16-step held-out rollouts.
It used 9,673 parameters for AR/NODE and 9,680 for OpsSplit. Mean normalized
volume MSE across the perturbation/control tests was 6.72977 for AR, 1.15339
for NODE and 1.15338 for OpsSplit. OpsSplit is effectively tied with NODE.
NODE and OpsSplit develop negative density; temperature remains
positive at Q8 samples. Their current/vorticity constraints are not enforced.
The linear branch is below 0.67% of the nonlinear RMS in every channel at the
validation starts, explaining why this conservatively bounded diffusion prior
barely changes the NODE result. Fixed-weight substep differences decrease by
approximately four in squared error on halving the Euler step, but that does
not validate the learned trajectory. No benchmark-ready or superiority claim
is made.

Artifacts: `/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_euler_pilot_20261105/pilot/`
contains `results.json`, `audit.json`, `comparison.png`, normalization, and all
three validation-selected checkpoints. The sibling `study/` directory is the
immutable training-source snapshot. Later reference-provenance guards were
independently rerun by the final audit against the same hashed input files.

The L-SplineFNO-gradient rerun completed on 5 November (Slurm `2784732`,
8m24s including 22 passing tests and the audit). Same seed, data, 30 updates,
parameter counts and rollout horizons; AR/NODE test metrics reproduce exactly.
Mean held-out normalized volume MSE is **AR 6.72977, NODE 1.15339,
OpsSplit 1.31014**: OpsSplit is 13.59% worse than NODE in this bounded pilot.
The linear/nonlinear RMS ratio at validation starts is now 5.24–84.12% across
fields/cases. OpsSplit Q8 density/temperature remain positive for the perturbed
case, but control density reaches -0.00150203. Final normalized interior weak
current defects are 0.04064/0.00624 and vorticity defects 0.96483/0.95455
(perturbation/control), worse than NODE; physical constraints are not satisfied.
With fixed weights, squared 1-vs-2 / 2-vs-4 substep differences decrease by
3.86/3.74, consistent with first-order Euler. This does not establish physical
accuracy or independently held-out generalization. No positivity clipping or
constraint correction was applied. The comparison excludes persistence.

Rerun artifacts and immutable source snapshot:
`/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_lspline_gradient_pilot_20261105/`.
See `completion.json`, `tests.xml`, and `pilot/{results.json,audit.json,comparison.png}`.

## What runs now

- Immutable pinned equilibrium snapshots, explicit namelist defaults, dry-run/Slurm rendering, bounded solver execution, false-zero GMRES/HDF5-error rejection, preserved attempts, restart-horizon checks and native equilibrium comparison.
- Native FE value/derivative evaluation, exact curved-element physical queries, provisional boundary quadrature, sparse scalar/flux weak assembly with physical volume weights, residual reconstruction, differentiable assembly and HDF5 exchange validation.
- Imported L-SplineFNO current-field adapters, independent density/pressure OpsSplit modules, transport-FiLM whole-RHS/autoregressive builders, direct-term loading/loss and checkpointed synthetic smoke training.
- Candidate stratified sampling, parent-family leakage checks, training-only scaling, weighted errors, family bootstrap, failure-inclusive completion metrics, resource accounting and gate-protected release manifests.

The exchange schema remains provisional. Native residual coefficients are never treated as pointwise fields. Model303's pressure equation couples density and temperature; a pressure residual cannot be used as `T_t`. The original in-process bridge remains unfinished; the new native subprocess bridge is documented in [full rollout](reports/full_rollout.md).

Current numerical evidence: the combined solver passes all five native probes and tight Picard field/gradient parity. Newton converges in **10 iterations** on the tight original control, **6** on radial160, **11** on wall80 and **7** on core-poloidal440. Every refined probe completes three healthy steps with positive q8 geometry, density and temperature. Bounded MUMPS refinement passes independent single-step interior balance and field/gradient repeatability. The suite passes **236 tests**. Full radial80/160 comparison has **100% domain coverage** but still fails the 1% target for perturbed vorticity and paired current/vorticity; radial320 now passes eight Newton iterations and three healthy steps. The legacy core-poloidal comparison also fails fields and full coverage. Matched combined-solver trajectories and further core/wall/divertor probes are running. See [qualified solver evidence](reports/cpu_combined_refined_qualification_20261103.json), [current numerical status](reports/numerical_status.md), and the [completion sequence](reports/science_completion_20261103.json). No scientific gate has passed.

The isolated combined executable and source/input hashes are bound by the qualification report and its manifest. Its validated Newton controls retain the original normalized-pressure profiles:

```fortran
newton_GS_fixbnd = .true.
amix = 0.0
delta_psi_GS = 1e30
equil_accuracy = 1e-10
```

The tight parity control uses `equil_accuracy = 1e-12`. The value `delta_psi_GS = 1e30` selects the existing normalized-profile branch; the pressure multiplier remains one. Patch 0025 is qualified only with the cycle guard and MPI synchronization in 0026/0027. Fresh-start qualification does not establish continuation-state persistence through restart.

## Environment and local checks

Use the existing dependency environment; no installation or external-checkout edit is needed:

```bash
cd /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit
export CFO=/pitagora/home/userexternal/vgopakum/Conditional_Fusion_Operator
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
"$CFO/.venv/bin/python" -m pytest -q
```

The upstream commit and imported source hashes are pinned in `configs/lsplinefno.yaml`. Initial package versions and dirty-checkout provenance are recorded in `reports/preflight.json`; the initial installed package inventory is in `reports/environment_packages.json`. Final checks used the existing `NOs_for_POs/.venv/bin/python` environment after a shared-filesystem import stall in the CFO environment; `reports/preflight_final.json` and `reports/environment_packages_final.json` record that environment. The upstream MIT implementation remains in its existing checkout. The proposed JOREK derivative patch remains LGPL-3.0-or-later. The study's runner attributes the MIT toolkit adaptation in its module header.

Run synthetic training with a fresh output directory:

```bash
"$CFO/.venv/bin/python" scripts/train.py --config configs/train.yaml \
  --model-config configs/lsplinefno.yaml --output /tmp/mastu-new-smoke --smoke
```

This uses a tiny CPU model and analytic fields, writes intermediate/final checkpoints and a provenance report, and proves plumbing only. Resume into a different fresh output directory using `--resume /tmp/mastu-new-smoke/checkpoint-step-5.pt` with the same configs. Scientific training without `--smoke` remains gated on numerical acceptance and independent, fully labeled data families. The full native rollout is available separately through `scripts/run_full_split.py`.

## Snapshot and job review

All commands accept explicit configuration/output paths and reject overwriting accepted outputs. Dry runs do not launch scientific work:

```bash
"$CFO/.venv/bin/python" scripts/preflight.py --config configs/case.yaml \
  --model-config configs/lsplinefno.yaml --output /tmp/mastu-new-preflight.json
"$CFO/.venv/bin/python" scripts/prepare_case.py --config configs/case.yaml \
  --output /tmp/mastu-new-case --dry-run
"$CFO/.venv/bin/python" scripts/run_case.py --config configs/equilibrium_request.yaml \
  --prepared runs/prepared_equilibrium_20261102_v3 \
  --output /tmp/mastu-new-reference --dry-run
```

The user approved continuation on CPU using `fupa2_fauna/dcgp_fua_dbg`. The unchanged equilibrium regression passed as job **2767720**. The earlier migrated job 2767445 was cancelled before allocation; the separately approved BOOST comparison 2767719 also passed. Original requests and immutable snapshots remain preserved as history. See [regression status](reports/equilibrium_regression_status.json) and [CPU continuation](reports/cpu_continuation_20261102.json) for effective requests and measured results.

CPU continuation has produced short unperturbed and perturbed refined-mesh evolution through 96 steps and 1.406 microseconds. Every job has explicit caps and a fresh output directory. The earlier coarse-mesh 48/96-step discrepancy is 0.7632%; the refined baseline discrepancy is 0.2154%. Refined equilibrium construction succeeds after isolated mesh-capacity, shared-attachment and axis-initialization fixes; coarse fields and geometry remain exactly unchanged. A further isolated single-writer patch removes an intermediate-equilibrium MPI HDF5 race and passes exact four-rank regression. See [paired pilot diagnostics](reports/cpu_short_pilot_validation_20261102.json) and [output/timestep continuation](reports/cpu_io_continuation_20261102.json). Dynamic mesh convergence, consistent energy diagnostics and the native residual/temporal/constraint bridge remain incomplete; Gates A/B/C and scientific training/data release remain blocked.

The [initializer patch](patches/0001-model303-axisymmetric-perturbation.patch) is disabled by default and unapplied to production. A complete isolated JOREK build compiled and linked successfully; [build provenance](reports/initializer_build.json) records source/archive/patch/configuration and binary hashes. Native derivative product-rule tests also pass. A one-rank CPU regression with default zero amplitudes matches the original native fields exactly; [runtime evidence](reports/initializer_runtime_validation_20261102.json) is separate from immutable build provenance. Actual enabled initialization also passes at one and four ranks: fields match exactly, boundary coefficients and unselected fields are preserved, sampled density/temperature are positive, and affected-element support stays away from walls/separatrix. See [enabled runtime checks](reports/cpu_enabled_initializer_validation_20261102.json). Short perturbed evolution completed with positive sampled density/temperature and a measurable response; its baseline-subtracted density/temperature/pressure RMS falls by 11%/24%/23%. Historical intermediate-file HDF5 errors remain preserved. [Energy diagnostics need reconciliation](reports/cpu_energy_diagnostic_audit_20261102.json), and [interior weak-constraint checks](reports/cpu_constraint_diagnostics_20261102.json) exclude boundary and axis rows. Full physical acceptance remains pending; the standard preparation path still rejects general perturbed campaigns.

## Data and evaluation interfaces

`scripts/export_dataset.py` consumes an explicit native geometry/trajectory NPZ exchange plus metadata JSON; it cannot infer missing quadrature labels from an ordinary restart. See [data contract](reports/data_contract.md). Non-synthetic exchange remains provisional until native mapping and physical acceptance are verified. `manifest.release_manifest` requires all scientific gates, one geometry, matching frozen protocol identities and preserved failures.

`scripts/evaluate.py --config configs/evaluation.yaml --results INPUT.json --output REPORT.json` aggregates supplied trajectory metrics. The input declares `synthetic`, `metric_definition` (including the common physical horizon), `trajectories`, and optional measured resource `attempts`. Each trajectory declares `model`, `seed`, `case_id`, `parent_family_id`, `regime`, boolean `completed`, and either finite `perturbation_error` or `failure_reason`. The report labels accuracy conditional on completion and includes every supplied failure. This aggregation is not a surrogate rollout evaluator; actual native rollouts, balance diagnostics and final scientific figures remain dependent on the bridge/data/models.

## Progress and resource accounting

Run beads from the repository root:

```bash
bd list --limit 30
bd ready
bd show mastu-native
bd show mastu-grid
bd show mastu-resources
bd export -o .beads/issues.jsonl
```

Beads contains M0–M6, implementation tasks, assignments, dependencies, evidence and blockers. `.beads/issues.jsonl` is a reviewable interchange export; Dolt remains the live tracker. No tracker or Git remote was pushed. The timeline is dependency-based until build/pilot cost is measured. [Resource ledger](reports/resource_ledger.json) distinguishes the proposed first regression, unapproved pilot ceiling, and unbudgeted production. Requested process core-hours and actual allocated/billable resources must be tracked separately. Production bulk storage and retention remain unconfirmed.

## Native CPU restart diagnostics

`restart.py` reconstructs the unrefined, axisymmetric model303 bicubic field and physical gradients, including native signed element-size factors. Analytic tests and a compiled comparison against the pinned Fortran basis check the interpolation. Real density/pressure integrals agree with JOREK's printed values to their rounding precision. The full wall-extended volume differs from the closed-plasma volume reported by JOREK.

From this study directory, use a fresh output path:

```bash
/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python scripts/diagnose_restart.py \
  --input runs/cpu_baseline48_20261102/attempt_1/work/jorek_restart.h5 \
  --reference runs/equilibrium_regression_dcgp_20261102/attempt_1/work/jorek_restart.h5 \
  --quadrature-order 8 --output /tmp/mastu-baseline-new.json
```

`--dry-run` renders the request. Native shared DOF aliases are checked and condensed to canonical native IDs. The reader rejects incompatible models, malformed arrays, singular/folded elements and adaptively refined/constrained mesh layouts lacking a native elimination map. Patch IDs and native solver constraints remain unavailable. Diagnostics retain these limitations and cannot approve a native training dataset. Positivity at sampled quadrature points is not a proof between samples; integrated changes are not source/flux balance residuals.

### Coefficient OOD extension (6 November 2026)

The ten-case campaign in
`/pitagora_scratch/userexternal/vgopakum/mastu_coefficient_ood10_20261106`
varies **`D_perp(1)`**, the constant density-diffusivity amplitude, from 2 to 10
 times its original `1e-5` value. Ten distinct original **test** initial states
are reused to isolate coefficient changes. The original 100-case release is
unchanged. This adapts the coefficient-shift idea of Table 7 in
[the original OpsSplit paper](../../2602.23113v2.pdf); that experiment changes
viscosity and initial-condition distributions, so this is not an exact reproduction.

`build_coefficient_ood.py` freezes coefficient values, source hashes, initial
states and paired baseline-coefficient controls before simulation. It uses the
same 97-frame window and coarse/fine timesteps as the parent release, with a
fine/finer fallback. Acceptance requires <=1% dynamic-normalized error for all
fields in the full state, matched ID control, and **coefficient-shift response**,
<=1% energy error, and positive Q8 density/temperature at every saved frame.
The JOREK runtime coefficient echo must match the requested value. Failures
remain visible and prevent release; no failed case is silently resampled.
`completion.json` with `ood_dataset_ready` is the release gate.

The opt-in `DensityDiffusionEulerModel` explicitly adds
`D_perp * (rho_RR + rho_R/R + rho_ZZ)` to the density component of the existing
continuity split. It differentiates the L-SplineFNO synthesis analytically,
includes the physical training mean, applies the native code-time interval,
and divides by the density normalization scale. Its coefficient is supplied,
not learned. The magnetic anisotropy correction, boundary closure and remaining
physics stay in the coefficient-conditioned neural residual. All three model
families receive the same scalar input `D_perp/1e-5`; baseline training uses 1.
The learned scatter remains an approximation of physical fields, and the
explicit Euler update still requires rollout and substep checks.

[ensemble_density_diffusion.yaml](configs/ensemble_density_diffusion.yaml)
opts the existing trainer into this model definition. It needs baseline training
with that split; inserting the new term into an old continuity-only checkpoint
would change its dynamics without training. This OOD generation task does not
launch or modify training jobs.

[coefficient_ood_dataset.yaml](configs/coefficient_ood_dataset.yaml) describes
the evaluation input. `CoefficientOODDataset` exposes only test windows and
returns the actual coefficient multiplier. Its default normalization is the
byte-identical parent 60-case training normalization. The existing 80-case
training experiment pools parent train+validation; evaluation of that experiment
must instead pass its saved `normalization.npz` as `training_normalization` and
its verified SHA256 as `training_normalization_sha256`. The loader rejects
normalization involving any parent test case. Neither default nor checkpoint
normalization is refitted on OOD data.

### Stronger coefficient OOD pilot

The separate four-case campaign in
`/pitagora_scratch/userexternal/vgopakum/mastu_coefficient_ood_strong4_20261106`
uses density-diffusivity multipliers **20, 40, 70 and 100** (`D_perp(1)` values
`2e-4`, `4e-4`, `7e-4`, `1e-3`). Each case has a distinct original test initial
state and a matched baseline-coefficient reference. The original ten-case
campaign remains separate. These are exploratory coefficient stress tests,
not changes to the training distribution or to the acceptance tolerances.

The same coarse/fine/finer progression, 1% convergence criteria and saved-frame
Q8 positivity checks apply. Automatic postprocessing renders native-mesh
original/OOD/difference density videos for accepted cases and records all
cases, including failures, in `activity.json`. The final volume-weighted density
RMS difference relative to ID quantifies visible change; a 10% activity target
is reported as a diagnostic, never used to hide failed or less-active cases.
The observation window remains 4.218–5.624 microseconds. Larger coefficients
may require refinement or fail the pilot; increased visual activity alone does
not qualify a trajectory for model evaluation.

Use [coefficient_ood_strong_dataset.yaml](configs/coefficient_ood_strong_dataset.yaml)
only after its completion record exists. The coefficient-aware OpsSplit model
uses the same physical `D_perp/1e-5` input as in the first OOD campaign.

### Gamma OOD pilot (2026-11-06)

The current requested sweep is **gamma=1.2, 1.4, 1.8, 2.0**, keeping D_perp(1)=1e-5. Each case reuses a distinct held-out parent initial state. Original training simulations use gamma=5/3. Files live in `/pitagora_scratch/userexternal/vgopakum/mastu_gamma_ood4_20261106`. The briefly requested gamma=2/3 pilot in `mastu_gamma_2over3_ood1_20261106` was cancelled at the user's request; its outputs are excluded from the dataset.

Patch `0005-model303-gamma-namelist.patch` exposes the existing MPI-broadcast gamma in model303's namelist; the default and equations remain unchanged. Before OOD execution, a two-step baseline-gamma regression compares original and patched binaries on the same restart, requiring maximum training-scale-normalized native quadrature difference <=1e-8. The first such regression passed with exactly zero difference. Coarse/fine (and finer if necessary) OOD runs retain the existing 97-frame, all-field state/control/response and magnetic/kinetic-energy 1% gates, including Vpar, and Q8 sampled density/temperature positivity. Packaging and native-mesh density comparison videos require accepted cases. The input sheath coefficient stays fixed and JOREK derives gamma_stangeby using each new gamma.

`GammaEulerModel` uses `[D_perp(1)/1e-5, gamma]` for equal FiLM conditioning in OpsSplit, AR and NODE. OpsSplit additionally inserts gamma explicitly into the analytic spline temperature-compression tangent; density diffusion and density continuity retain their original coefficients. Other gamma-dependent nonlinear/boundary terms remain in the learned residual. Baseline ID windows map to `[1,5/3]`. `CoefficientOODDataset` returns each requested gamma and retains parent training normalization. Opt-in `configs/ensemble_gamma.yaml` trains this two-input architecture; existing one-input checkpoints are not interchangeable. This submission generates data only.

Validation: 27 coefficient, continuity and training tests passed, including the native pressure-to-temperature chain rule, signed compression scaling, mixed-gamma batches, all three model families' backward passes, and held-out loader normalization. Current jobs: build 2795119, prepare 2795137, baseline regression 2795138, four-case simulation array 2795139, packaging 2795140, videos 2795141.
