# Implicit Physical OpsSplit with the full native tangent

A fourth model family for the MAST-U 80/0/20 ensemble comparison. The existing
Physical OpsSplit adds a linearised continuity branch for density and
temperature only, evaluated through learned spline derivatives. This family
instead carries the complete linearised model303 volume operator for all seven
fields with exact native bicubic derivatives, integrates it implicitly, and
learns only the nonlinear remainder.

## Native physical defect (11 November 2026)

The new experiment in [`configs/physical_defect.yaml`](configs/physical_defect.yaml)
retains the known nonlinear volume equations and learns an additional defect.
The historical learned-remainder formulation and results below remain separate.
Define the assembled affine implicit part and explicit part as

```text
I(q)       = F* + L*(q - q*)
D_theta(q) = assemble[G_theta(q) - G_theta(q*)]
E_theta(q) = F_native(q) - I(q) + D_theta(q)
M* dq/dt  = I(q) + E_theta(q) = F_native(q) + D_theta(q).
```

`F_native` evaluates the complete volume RHS for the supported native case,
including temperature-dependent resistivity, viscosity and parallel heat
conduction, analytic current/particle/heat sources, and their state dependence.
`L*` remains the exported tangent with profiles fixed; the explicit remainder
therefore includes profile variation as well as the other nonlinear terms.
`M*` remains frozen at the warm native reference. The formulation does not replace
the state-dependent temporal operator of the original JOREK solver.

The immutable `q*` is the native reference restart, **not** the training mean.
The subtraction makes the learned defect vanish there for every network weight.
It retains `F* = F_native(q*)`: this warm reference is evolving and is not assumed
to be a stationary equilibrium. A zero-initialized output gives exactly the
matched full-volume-physics control. In this experiment `--arm physics` means
`D_theta = 0`, whereas the historical physics control below sets the entire
nonlinear remainder to zero. Remainder labels supply the same training-only
channel scales; label pretraining is disabled because the exact remainder is
already present. The learned arm uses rollout supervision only.

**Revision qualification:** the frozen pilot below passes reference and initial
four-state parity, but the broader 240-state audit found an induction-source
normalization mismatch. Its numerical results remain evidence for that frozen
revision. A separately versioned correction now passes all 240 selected states;
its separate frozen-weight rollout verification also passes on two held-out
cases at two actual microsteps. The old pilot is not silently reinterpreted as a
test of corrected code. Details follow below.

### Time integration and constraints

`scheme: be` is first-order IMEX Euler: backward Euler for `I`, explicit Euler
for `E_theta`. It is not a fully nonlinear backward-Euler solve:

```text
(M* - h L*) q_next = M* q + h [F* - L* q* + E_theta(q)].
```

`scheme: ars222` uses `gamma = 1 - 1/sqrt(2)` and
`delta = 1 - 1/(2 gamma)`:

```text
M* (Y - q)      = gamma h [I(Y) + E_theta(q)]
M* (q_next-q)   = h [(1-gamma) I(Y) + gamma I(q_next)
                     + delta E_theta(q) + (1-delta) E_theta(Y)].
```

Both stages reuse the coupled factorization at `gamma h`; `M*` is never inverted
alone. The affine reference forcing appears with total weight one. Independent
affine, coupled singular-mass tests verify first-order BE and second-order ARS
convergence. Every explicit stage reevaluates the physical remainder and learned
defect. One BE microstep uses one physical-RHS evaluation and two network calls;
one ARS microstep uses two physical-RHS evaluations, one affine-RHS evaluation,
and four network calls. Physics-only makes no network calls. The configured
`internal_substeps` multiplies the requested 1/2/4 evaluation substeps.

Current/vorticity constraints and training-identified held boundary rows enter
each coupled solve. The nonlinear Mach wall relation is iterated to a normalized
residual tolerance of `1e-10` (at most 12 iterations by default). The existing
whole-cell positivity and elliptic guard then acts at every stage; the Mach
relation is updated again if the guard changes temperature. Reports expose guard
changes, held-DOF changes, wall residuals and actual stage counts. The guard can
alter the numerical stage and its accuracy; formal ARS order applies where these
corrections are inactive. Raw scores use the final proposal from the actual
previously guarded history, not a separate unguarded rollout. The identified wall
relations and held rows retain the historical boundary approximation.

### Native profile and derivative qualification

[`reports/physical_profiles_20261111.json`](reports/physical_profiles_20261111.json)
pins the native input, source/default branches, frozen export and reference
restart. `NativeCritical` recomputes the magnetic axis and both X-points from the
current predicted bicubic coefficients, including differentiable X-point motion.
It reads no future restart or trajectory diagnostics. Reference locations only
initialize the local search. The boundary flux is the native selected minimum of
the two X-point fluxes; its derivative is piecewise smooth and changes branch
when the two fluxes cross.

Passed CUDA evidence is preserved in
`/pitagora_scratch/userexternal/vgopakum/mastu_physical_defect_20261111/diagnostics/`:

| Check | Result | Report |
|---|---|---|
| Full volume RHS at native `q*` | maximum absolute coefficient error `4.75e-13`; maximum channel-scaled error `6.42e-11`; remainder `4.07e-15` | `mastu_rhs_reference_validation.json` |
| Native training remainder labels | four states: `mastu_000:0/48/96`, `mastu_060:96`; worst volume-weighted channel RMS error / training RMS `7.48e-8`, consistent with float32 label storage | `mastu_training_rhs_parity_20261111.json` |
| Moving critical-point derivatives | late-frame CUDA autograd versus finite differences: scaled error `6.97e-9`, including both X-point heights; a larger perturbation crosses the boundary-flux branch | `mastu_late_critical_gradient_20261111.json` |

The reproducible label check validates label/restart hashes and exact equality
between stored trajectory coefficients and the native exported restart. It uses
only the pooled 80 training trajectories. These checks qualify the implemented
volume RHS and its local derivatives, not learned rollout accuracy.

The broader audit subsequently scored all **240 states** (80 training cases at
frames 0/48/96). All axis/X-point searches succeeded, but the declared parity gate
did **not** pass: 8 frame-48 states and 14 frame-96 states exceeded `1e-5` RMS /
training scale in the induction channel. The worst was `1.58e-4`; five states
also exceeded the `1e-3` pointwise / training-scale threshold. Frame zero passed
for all 80 cases. Thresholds were retained. The initial four passing states must
not be presented as full-pool parity.

Native `find_axis` accepts the first Newton-converged candidate in its
minimum-Q4-gradient search without checking that the stationary point is a
minimum. In the diagnosed `mastu_039:48` state it selects a saddle, whereas the
frozen autonomous tracker selects a strict minimum. The differing axis flux
changes the imposed current source. Substituting only the correctly aligned
native axis flux **offline** reduces induction RMS from `4.62e-11` to
`1.30e-14` (training-scaled `9.51e-5` to `2.67e-8`, the float32 label floor).
An independent local reproduction of native stopping and priority matches the
native saddle. Native histories were used only for this diagnosis; sources in
rollouts remain autonomous.

The source audit also confirms native local-coordinate stopping rules and an
approximate-point fallback; these differ from the strict physical-gradient and
minimum-Hessian acceptance gates. The confirmed exemplar is a converged saddle,
so early stopping alone is not its explanation. The corrected source revision
(`frozen_source_axis_v2/study`) computes a separate autonomous `source_psi_axis`
using native stationary-point priority/stopping, after the unchanged strict
minimum/X-point existence checks. It implements no failed-search fallback.

The corrected revision passes all **240/240** selected states at the original
thresholds: worst training-scaled RMS `2.38e-7`, pointwise maximum `9.81e-5`.
All 22 earlier failures are resolved. Qualified root values are bitwise unchanged
and valid-candidate counts identical across all 240 states. Its focused suite
passes 83 tests, with the CPU optimizer test excluded. This qualifies those
selected states, not every possible trajectory or topology. The original frozen
pilot and failed parity reports remain intact. Evidence:
`diagnostics/parity_all80_frame*.json`,
`diagnostics/parity_axis_source_audit_039_48.json` and
`diagnostics/parity_native_axis_roots_v2.json` and
`diagnostics/source_axis_resolution.json` in the campaign.

The completed BE physics screen scored **one training trajectory**, `mastu_000`,
from frame zero with no training updates or network calls. Its cumulative,
volume-weighted normalized MSE (mean over seven fields and all frames through
H) was:

| Physical microsteps per saved interval | H = 4 | H = 16 |
|---|---:|---:|
| 1 | `4.07607e-5` | `1.35407e-4` |
| 2 | `1.74877e-5` | `6.71834e-5` |
| 4 | `9.90380e-6` | `4.51094e-5` |

All six horizon/substep audits passed. No positivity limiting was needed;
maximum constraint defect was `2.92e-13`, boundary drift was zero, and maximum
Mach-wall residual was `7.10e-12`. Elliptic/wall corrections still occurred:
maximum normalized guard change was `2.33e-6`, while held-DOF change was only
`5.68e-14`. Raw and guarded MSE differed by at most `2.43e-14`; raw predictions
still use the actual guarded history. The 16-interval window is 0.234 microseconds.
Refinement reduced error on this window; this alone is not a full-window
convergence result or a matched learned-model comparison.

The [audited status report](reports/physical_defect_status_20261111.json) records
raw evidence hashes, frozen physics-source identity, arithmetic/stage-count
checks and detailed metrics. The BE evidence is `physics_be_screen/report.json`
under the campaign directory. Its 339-second total includes initial data
verification, setup, factorization and audits; it is not an inference-throughput
benchmark. Shared release verification can inspect all dataset splits, but only
the selected training window was scored.

The subsequent full-window screens cover two **training** trajectories only.
Cumulative H96 MSE (1.406 microseconds) is:

| Method / training case | 1 physical microstep | 2 physical microsteps | 4 physical microsteps |
|---|---:|---:|---:|
| BE / `mastu_000` | `2.90569e-3` | `2.34810e-3` | `2.20029e-3` |
| BE / `mastu_028` | `1.66654e-3` | `1.42379e-3` | `1.36171e-3` |
| ARS222 / `mastu_000` | axis-search failure | axis-search failure | not tested |
| ARS222 / `mastu_028` | `1.39781e-3` | axis-search failure | not tested |

BE passed every requested horizon and audit in all six case/substep pairs.
ARS passed H16/H32 before the listed failures, but only one of four pairs reached
H96. Failed trajectories are not averaged away. This evidence favors BE for the
current pilot; it does not establish a universal preference for first-order or
implicit integration.

The original `physics_be_full` and `physics_ars222_full` drivers both reached the
450 CPU-second worker budget during their last case/substep pair. Their records
remain intact. Separate `physics_be_remaining` and `physics_ars222_remaining`
runs completed those attempts: BE passed; ARS encountered the numerical axis
failure. Repeated H16/H32 prefixes match exactly for BE and within `9.02e-15`
absolute MSE for ARS. Configuration, source and normalization identities match.
A CPU-budget interruption is recorded separately from a numerical failure.

The independent capture audit in `critical_capture/independent_verdict.json`
reproduced the two-substep `mastu_000` failure. Expanding to 2,048 candidate cells,
up to 100 Newton iterations and multiple starts still found no qualified smooth
minimum axis, while both X-point groups remained valid. The best classified
minimum in those multistart checks had gradient `1.81e-4`, versus the unchanged
`1e-9` tolerance, near element edges within the reference neighborhood. Gates
and frozen source were unchanged. This rules out the tested simple search
expansions as a repair; finite local searches do not prove physical topology
loss or identify whether frozen mass, stage/guard interaction or spatial
artifacts caused the failure. No native restart simulation of the failed field
was run. The associated regression suite passed 80 tests, with the CPU optimizer
test deliberately deselected; optimizer-updating tests ran on CUDA.

### Completed 64-update exploratory learning pilot

The original frozen source revision completed **64 CUDA optimizer updates**,
seed 17, BE with two internal microsteps, four-interval training windows,
learning rate `1e-4`, and final-update checkpoint selection. The protocol was
recorded in `learning_pilot64_protocol.json` before scoring. Both existing
held-out cases (`mastu_004`, `mastu_010`) completed H16/H32/H96 with matched
physics controls. All 12 full rollouts and 18 arm/horizon/substep aggregates
passed their audits. Requested substeps 1/2/4 mean **2/4/8 physical microsteps**.

| Actual microsteps per interval | H96 physics MSE | H96 learned MSE | Reduction |
|---|---:|---:|---:|
| 2 | `1.97815e-3` | `1.87613e-3` | 5.157% |
| 4 | `1.84249e-3` | `1.74130e-3` | 5.492% |
| 8 | `1.82176e-3` | `1.72103e-3` | 5.529% |

Both cases improve in total MSE at every tested horizon/substep setting. The
H96 density-field MSE nevertheless worsens by 0.18–0.19%; the improvement is not
uniform across fields. Positivity guards remain active late in both arms
(48 limited field-steps per arm at each H96 substep setting; minimum evaluation
alpha about 0.979). During training, 13 of 64 updates activated limiting. Raw
metrics remain proposals along guarded histories, so these are gains for the
guarded system.

Matched historical affine-physics context gives H96 MSE about `0.162` at two/four
microsteps, versus `0.00198`/`0.00184` for the new full-volume-physics controls.
Most of the roughly 98.8% reduction comes from the physical formulation and
stage/boundary policy, before learning; the additional learned reduction is the
5.2–5.5% above. This context is descriptive rather than an isolated architectural
or equal-runtime comparison. Shared-GPU timing, setup, factor caches and audits
prevent a clean speedup claim. The source report is
`learning_pilot64_comparison.json`.

These original pilot numbers retain the **22/240 source-parity failures** of
that frozen revision. The separate corrected-source verification now also
completes: the same frozen 64-update weights and a matched physics control pass
all four rollouts on the same two held-out cases at **two actual microsteps**.
Corrected H96 mean MSE is `1.9781579e-3` for physics and `1.8761411e-3` for frozen
learned weights, a **5.15716% reduction**. Each case improves (5.47% and 4.93%).
Across all checked prefixes, the largest MSE change from the original-source
result is `1.52e-8` absolute (`6.79e-6` relative). Checkpoint, source, normalization,
case pairing and metric arithmetic were independently checked.

An independent single CUDA update on `mastu_039:48` passed with finite gradients;
those smoke-update weights were **not** used for evaluation. This demonstrates
that the revised source path supports training and preserves the observed
frozen-weight pilot gain, not a new 64-update training campaign. Evidence is
`source_revision_verification/report.json` and the independent
`source_revision_comparison.json` audit. Four/eight-microstep learned results
above remain original-revision evidence and have not been rerun on corrected
sources.

The current candidate for a larger study is BE with two internal microsteps,
known native volume formulas, autonomous native-compatible source normalization,
strict independent root gates, and the reference-anchored learned defect. This
remains a one-seed, two-case, 64-update exploratory pilot, not the default
2,000-update, 20-test-case study or an independent fresh holdout. Guard activity,
the slight density regression and the original ARS failures remain material
limits; this sample does not justify a confidence interval or robust deployment
claim.

Scope is the pinned axisymmetric `n=0`, model303, fixed-boundary double-null case,
on its existing mesh/equilibrium and baseline coefficient setting (`D/D0 = 1`).
The profile factory rejects unsupported branches and parameters. The autonomous
critical-point search is local to the reference axis/X-point neighborhoods and
fails if a required root is lost; limiter transitions, different topology,
free-boundary evolution, nonaxisymmetric modes, cross-equilibrium transfer and
coefficient generalization are not qualified. The earlier coefficient-physics
OOD implementation is not automatically a validation of this new profile path.
This is an offline native-mesh surrogate research implementation; sustained live
JOREK coupling remains a separate qualification task.

### Multi-seed qualification submitted — 11 November 2026

The corrected-source formulation is now submitted for **three fresh training seeds,
17/18/19, with 2,000 CUDA updates each**, plus one shared deterministic nonlinear
physics control. Every learned run starts from zero output; no pilot checkpoint is
reused. Training uses BE with two internal microsteps, four-interval windows,
learning rate `1e-4`, the existing 80/0/20 split, and final-update checkpoint selection.
Each array task requests one H100, eight CPUs, 120 GB host memory and 12 hours.

All 20 existing test cases are evaluated at H16/H32/H96 and requested substeps
1/2/4, giving **2/4/8 actual microsteps**: 240 complete rollouts, 720 horizon
records and 36 arm/horizon/substep aggregates. The primary H96/two-microstep gate
requires a ratio of pooled learned mean MSE to matched physics mean MSE of at most
**0.95**, each seed's ratio below one, and a paired trajectory-bootstrap 95% upper
ratio below one. The bootstrap uses 10,000 draws, seed 4242, with identical case
resampling across all learned seeds and the shared control. Its interval is
conditional on the three fitted seeds; seed dispersion is reported separately.
Incomplete coverage, nonfinite metrics or failed audits suppress campaign averages.
Secondary refinements, per-field regressions and guard activity remain visible.

Slurm accepted preflight **2829030**, training/control array **2829055** (tasks
0–2 learned seeds, task 3 physics) and summary **2829056**. At submission, preflight
was pending at `QOSGrpNodeLimit`; the array waits for successful preflight and the
summary waits for the array to finish, including failures. Invalid preflight
dependencies cancel the array instead of leaving it pending indefinitely.
**Training has not yet started; no multi-seed qualification result is claimed.**

Artifacts are under
`/pitagora_scratch/userexternal/vgopakum/mastu_physical_qualification_20261111`:
`campaign.json` pins the protocol, `frozen/study` holds the immutable source,
`launch.json` records exact submission commands, and `qualification.json` is the
automatically generated report. Its initial status is `incomplete_or_failed`
because runs have not started. The [launch record](reports/physical_qualification_20261111.json)
records hashes and scheduler state. Summary checks passed 17 unit tests; all 37
numerical modules match the previously validated corrected-source snapshot. The
new GPU preflight remains pending.

These are reused exploratory test cases, including the two pilot cases. This is
expanded qualification, not independent confirmation. The existing baseline
`n=0`, fixed mesh/equilibrium and coefficient-one scope remains unchanged.

### Reproduction and bounded GPU workers

From `JOREK/mastu_opsplit`, use fresh output paths. Existing evidence is never
replaced. The profile and reference hashes in the configuration must match the
pinned inputs. The configuration selects the small matched L-SplineFNO backbone,
2,000 rollout updates at learning rate `1e-4`, two internal BE microsteps,
final-update selection, and the existing 80/0/20 split. This is a research configuration, not a completed run.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
PY=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
PHYSICS_CAMPAIGN=/pitagora_scratch/userexternal/vgopakum/mastu_physical_defect_20261111
SETUP_CACHE=/pitagora_scratch/userexternal/vgopakum/mastu_gpu_onestep_20261109/seed17_physics
PREPARED_CACHE="$PHYSICS_CAMPAIGN/reproduce_prepared_data"
REFERENCE=/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_refinement_20261104/runs/baseline_settle1/work/jorek000144.h5

$PY scripts/verify_physical_defect.py \
  --plan /pitagora_scratch/userexternal/vgopakum/mastu_implicit_opsplit_20261108/labels/plan.json \
  --profile-manifest reports/physical_profiles_20261111.json \
  --reference-restart "$REFERENCE" \
  --normalization "$SETUP_CACHE/normalization.npz" \
  --output "$PHYSICS_CAMPAIGN/diagnostics/recheck_training_parity.json"

# Reproduce the short training-only screen using the frozen implementation.
FROZEN_STUDY="$PHYSICS_CAMPAIGN/frozen/study"
$PY "$FROZEN_STUDY/scripts/diagnose_physical_defect.py" \
  --config "$PHYSICS_CAMPAIGN/frozen_pilot_config.yaml" \
  --model-config "$FROZEN_STUDY/configs/lsplinefno.yaml" --scheme be \
  --case-indices 0 --horizons 4 16 --substeps 1 2 4 \
  --setup-cache "$SETUP_CACHE" --output "$PHYSICS_CAMPAIGN/recheck_be_screen"

# Physics control first; neural training only after its rollout checks.
$PY scripts/train_physical_defect.py --config configs/physical_defect.yaml \
  --model-config configs/lsplinefno.yaml --arm physics \
  --setup-cache "$SETUP_CACHE" --prepared-data-cache "$PREPARED_CACHE" --cpu-budget 300 \
  --output "$PHYSICS_CAMPAIGN/reproduce_physics_be"

$PY scripts/train_physical_defect.py --config configs/physical_defect.yaml \
  --model-config configs/lsplinefno.yaml --arm implicit_rollout --seed 17 \
  --setup-cache "$SETUP_CACHE" --prepared-data-cache "$PREPARED_CACHE" --cpu-budget 300 \
  --output "$PHYSICS_CAMPAIGN/reproduce_defect_be_seed17"
```

All training, native step/adjoint solves and physical-RHS evaluations use CUDA;
the launcher rejects CPU fallback. Under the local 600 CPU-second process limit,
the launcher now defaults to a 300 CPU-second budget, leaving reserve for a
complete 96-interval, eight-microstep evaluation before the hard limit. Checks
occur between updates/trajectories, not inside each GPU operation. Saved optimizer,
sampling schedule and CPU/CUDA RNG states support exact training continuation;
completed evaluation trajectories are reused. Add `--resume` to resume an
unchanged output. Failure to make checkpoint/evaluation progress stops the
launcher rather than looping indefinitely. Host-side setup and sparse assembly
still consume CPU time; the CPU budget is not a GPU wall-time budget.

`--setup-cache` reuses the training-identified fixed and wall rows after checking
source manifest/tangent hashes, `F0`, split counts and boundary tolerance; the
trainer also verifies the training case IDs. Experiment provenance records source
hashes, configuration, data and normalization identity. Cached setup is an
optimization for the same pinned release, not permission to reuse boundaries
across a different equilibrium or dataset. `--prepared-data-cache` separately
shares verified data, normalization and geometry setup across workers. Initial
verification performs the full checks; reuse checks source/manifest identity,
cache hashes and source-file fingerprints, while rebuilding LU factorizations.
A source change invalidates this cache: choose a fresh cache path for each code
snapshot. To compare ARS or internal refinement,
copy the configuration to a fresh path, change `physical_defect.scheme` or
`internal_substeps`, and use separate output directories with matched controls.

## Historical learned-remainder formulation

State: canonical bicubic coefficients q of (psi, u, zj, w, rho, T, Vpar) on the
native mesh (46,203 DOFs per field, 323,421 unknowns). The native model303
volume right-hand side is split about the common warm baseline q*:

    F(q) = F(q*) + L*(q - q*) + N(q)

* L* is the complex-step tangent of the complete pointwise model303 volume RHS
  at q*, exported by the patched native build (`native/full_split_procedures.f90`,
  `opsplit_tangent`) with transport and sources fixed. Its finite-difference
  check passes to 1.2e-10. Frozen at
  `mastu_full_opsplit_20261104/frozen_tangent/baseline_terms.h5`.
* M* is the native temporal operator (the mass action used by the validated
  SBDF2 split) frozen at q*, including JOREK's `corr_neg_dens1` low-density
  correction and F0 = 0.64:
  psi: v dpsi/R^2; u: -R^2 rho_c grad v . grad du; rho: v drho;
  pressure: v (T drho + rho_c dT); Vpar: v (rho Vpar grad psi* . grad dpsi + rho_c F0^2 dVpar)/R^2.
* One step of h saved-frame intervals (theta = 1, as configured):

      (M* - h L*) q' = M* q + h (F(q*) - L* q* + N_theta(q))

  theta in [1/2, 1] is supported for ablations.
* Current and vorticity rows are the weak definitions (R^-2 weight for current),
  tested on every DOF that evolves; they are identical to the PhysicalGuard
  constraint matrices, so the evaluation constraint audit holds by construction.
* Dirichlet rows: boundary DOFs whose coefficient never changes (relative to the
  field scale, tolerance 1e-7) in the 80 training trajectories. Interior DOFs are
  never Dirichlet. On the real data the separation is clean (fixed <= 4.7e-9,
  smallest evolving boundary change >= 4.4e-6).
* Vpar wall rows. JOREK replaces the Vpar equation at wall nodes with a penalised
  Mach-1 (Bohm) condition (`models/model303/mod_boundary_conditions.f90`):
  Vpar = (direction/B) f sqrt(gamma T) + ... on the node value and its along-wall
  derivative. Using the volume weak form there drove Vpar wrong by ~3 normalized
  MSE per step. Each free Vpar wall row is tested against the native semi-discrete
  identity M(q_k)(q_{k+1}-q_{k-1})/2h = c(q_k) with exact labels on six training
  states. Rows that fail (1,383 of 1,878) and whose exact JOREK functional form
  (c0 + c1 sqrt(T_v) for values; c0 T_d/sqrt(T_v) + c1 sqrt(T_v) for along-wall
  derivatives) fits half the training trajectories and the other half to < 1e-3
  become algebraic rows (870; worst fit 8.4e-4 on both halves). They are
  linearised at the training-mean temperature in the matrix, with the sqrt
  nonlinearity evaluated explicitly (and differentiably) at q_n. The remaining
  513 non-weak rows (mostly cross derivatives) are held at their current value
  within a step. With exact labels, the one-step Vpar error falls from 2.94 to
  ~1e-4 (persistence 1.2e-3).
* One SuperLU factorisation per step size is cached and reused; the adjoint is
  the transpose solve with the same factors (tested exact against dense A^-1 B).
  Unknowns are ordered by METIS nested dissection of the DOF graph with all seven
  fields of a DOF adjacent: 66-80 s per factorisation and ~0.63-0.67 billion LU
  nonzeros, versus 1,178 s and 2.8 billion with SuperLU's COLAMD. One LU solve
  takes ~0.25 s (forward) / ~0.38 s (transpose). Iterative refinement runs only
  while the normwise backward error |r|/(|A||x|+|b|) exceeds 1e-14 and keeps
  halving (at most 3 corrections); on the production system the first solve
  already reaches <= 1e-18 (forward) and <= 1.1e-16 (adjoint), so every solve is
  one LU solve. Unit (Dirichlet/held) rows are set exactly after each solve.
  The earlier rule (always one correction, then refine to a 1e-11 relative
  residual) doubled the forward cost and could not converge for the
  ill-conditioned adjoint (|x| >> |b|); dropping it changes states by <= 2e-9
  relative per step and the training update from 7.7 s to 5.0 s (validation:
  `/pitagora_scratch/userexternal/vgopakum/implicit_profile_20261109/refinement_validation.json`).
* N_theta is the same L-SplineFNO backbone and size as AR/NODE (16x32 latent grid,
  4 modes, width 8, 2 layers), with 13 output channels instead of 7: weak
  coefficients [strong, test_R, test_Z] for blocks [psi, u, rho, pressure, Vpar]
  minus the two identically-zero induction flux channels. 9,727 parameters versus
  9,673 for AR/NODE (projection head only). Outputs are scaled by the training
  label RMS per channel. The output projection starts at zero, so the untrained
  model is exactly the physics-only model.
* The shared PhysicalGuard (Bernstein positivity, elliptic closure) is applied
  after every implicit step, exactly as for AR/NODE/OpsSplit.

## Exact remainder labels

`scripts/export_remainder_labels.py` runs one native export step
(`OPSSPLIT_MODE=export`, `prepared_tangent/jorek.exe`, physics namelist identical
to the dataset runs) from each selected dataset restart. The native build checks,
at every quadrature point, that the closed-form physical coefficients reproduce
the untouched native RHS before writing them. The label is

    N(q) = c(q) - c(q*) - J*(jets(q) - jets(q*))

per quadrature point. Each label is verified against the dataset frame (values to
~1e-15, gradients to ~1e-12 relative). Only the 80 training trajectories are
exported: frames 0, 4, ..., 96 (2,000 states). `finalize` fits the per-channel
training label scale.

## Arms and protocol

| Arm | Training | Purpose |
|---|---|---|
| `implicit` | 2,000 exact-label updates, then 2,000 rollout updates | primary |
| `implicit_rollout` | 2,000 rollout updates from the physics-only start | isolates label pretraining |
| `physics` | none (N = 0) | what the frozen physics alone achieves |

Rollout fine-tuning calls the ensemble study's own `train_family`: same 4-interval
windows and sampling seed, loss, positivity penalty, Adam 1e-3, clipping 2, batch
one, 2,000 updates, final-update selection. Evaluation calls the ensemble's
`evaluate`: 20 test trajectories from frame zero, 16/32/96 intervals, 1/2/4
substeps, identical audits. Label pretraining is additional compute and is
reported separately; it is not a compute-matched comparison with AR/NODE.

OOD: `scripts/evaluate_ood_fixed_conditioning.py` feeds every network the
training conditioning value 1 (all training windows used 1) and withholds the
D_perp multiplier. L* and M* are frozen at the baseline coefficient, so this
family carries no explicit knowledge of the OOD coefficient either. Apply the
same script to AR/NODE checkpoints for a matched OOD table.

## Coefficient physics for the D_perp OOD set

The OOD trajectories change only `D_perp(1)` (multiplier m = 2-10 about 1e-5). With
`D_perp(2) = D_par = 0`, JOREK's `get_dperp` gives `D_prof = D_perp(1)` at every point
(`models/diffusivities.f90`), apart from `D_prof = D_prof_neg` where rho < 0. In the closed-form
coefficients, D appears only in the density diffusion tensor, so the native volume RHS is affine in D:

    c(q; D) = c(q; D0) + (D - D0) g(q),    g = -(I - b b^T/B2) grad(rho)   (density flux channels only)

`implicit_coefficient.py` uses this exactly. F(q*) and L* gain dD g(q*) and dD dg/djets(q*), and are
refactorised once per coefficient. The rest, dD [g(q) 1(rho >= 0) - g(q*) - dg*(jets - jets*)], is
applied explicitly in closed form. N_theta keeps its training conditioning (1) and its checkpoint.

* Native check (`scripts/verify_coefficient_operator.py`, two training restarts, m = 2 and 10, job
  2802708): native exports at the changed `D_perp(1)` reproduce the predicted forcing change to 4e-16
  and the tangent change to 7e-16, relative to the change. All other channels are bitwise unchanged.
  No OOD data is read.
* Evaluation (`scripts/evaluate_ood_coefficient_physics.py`) reuses the frozen campaign snapshot,
  checkpoints, normalization and evaluator. Only the coefficient module is new, and its hash is recorded.
  Float32 loader parameters are mapped back to the exact manifest multipliers.
* Information asymmetry, stated plainly: AR/NODE/OpsSplit received the multiplier through FiLM conditioning
  that was only ever trained at 1, and their fixed-conditioning OOD numbers are the comparator.
  This family receives it through the native equations instead. The old ensemble OpsSplit's linear
  branch has no diffusion term, so it has no physical route for D.

## GPU backend (torch + cuDSS)

`device: cuda` in the config (or `train_implicit.py --device cuda`, `slurm/implicit_train_gpu.sbatch`, FAUNA
GPU account `fupb2_fauna`) runs the whole step on one GPU; the default stays the CPU SciPy/SuperLU path.

* `ImplicitNativeSystem(backend='torch', device=...)` builds the same row-scaled A, B per step size and keeps
  them on the device. LU is NVIDIA cuDSS 0.7.1 (`gpu_native.CudssSolver`, ctypes, deterministic mode; library at
  `$JOREK_CUDSS_LIBRARY` or the work-area install). 0.7.1 has no transpose solve, so the adjoint factorises A^T on
  first use; it is released before evaluation (`release_adjoints`). Refinement policy is shared with the CPU path.
* Fixed sparse maps (spline sampling, weak assembly, Bernstein sampler, elliptic closure) have device kernels
  (`apply_torch`), dispatched by `_FixedNative` for non-CPU tensors. Products use a sliced-ELL gather kernel
  (`gpu_native.DeviceCSR`): bitwise repeatable, unlike cuSPARSE CSR products on the H100. The closure solves
  also use cuDSS.
* `configure_cuda_numerics()` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` (deterministic cuBLAS) and disables TF32:
  cuDNN's default TF32 put the remainder network 2e-4 (outputs) / 6e-4 (gradients) away from the CPU float32
  reference; with it off the devices agree to 1e-7 / 2e-6.
* Production-system check (job 2812139, seed17 checkpoint, same H100 node for both; `gpu_validation.json` in
  `/pitagora_scratch/userexternal/vgopakum/implicit_profile_20261109/`): factorisation 98 s -> 6.2 s (650M -> 499M
  LU nonzeros); training update 9.3 s -> 0.38 s, bitwise repeatable; 32-frame rollout 32.8 s -> 1.66 s. CPU vs
  GPU: loss 7e-7, gradient 1.3e-4, rollout states <= 5e-7 relative (float32 network level, not growing); every
  solve needed one LU solve (backward error <= 1.5e-15 forward, 6.3e-15 adjoint). The CLI smoke with
  `--device cuda` passed all audits.
* GPU memory: each cuDSS factorisation of the 323k system holds ~13 GB; training uses A and A^T at h,
  evaluation A at h, h/2, h/4.
* Limitation: a frozen snapshot must contain `gpu_native.py` (campaign_v2 does not), so earlier checkpoints are
  evaluated on the GPU only through a new snapshot of this code.

## Commands

```bash
PY=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
ROOT=/pitagora_scratch/userexternal/vgopakum/mastu_implicit_opsplit_20261108
# 1. physics-only go/no-go on training trajectories (no test data)
$PY scripts/diagnose_implicit_physics.py --config configs/ensemble_training.yaml \
  --model-config configs/lsplinefno.yaml --tangent <baseline_terms.h5> --output $ROOT/diagnostics/<new>
# 2. labels: plan, array of workers (inside a frozen snapshot), finalize
$PY scripts/export_remainder_labels.py plan --root $ROOT/labels --manifest <manifest> \
  --tangent <baseline_terms.h5> --executable-dir <prepared_tangent> --frame-stride 4
sbatch --array=0-3 --export=ALL,STUDY_ROOT=<snapshot>/study,LABEL_ROOT=$ROOT/labels,TASK_COUNT=4,CONCURRENCY=16 \
  <snapshot>/study/slurm/implicit_labels.sbatch
$PY scripts/export_remainder_labels.py finalize --root $ROOT/labels
# 3. training: freeze a snapshot, smoke, then the 7-task array
$PY scripts/freeze_ensemble.py --output $ROOT/campaign
sbatch --partition=dcgp_fua_dbg --time=00:30:00 \
  --export=ALL,STUDY_ROOT=$ROOT/campaign/study,OUTPUT_ROOT=$ROOT/campaign,SMOKE=1 $ROOT/campaign/study/slurm/implicit_train.sbatch
sbatch --array=0-6 --export=ALL,STUDY_ROOT=$ROOT/campaign/study,OUTPUT_ROOT=$ROOT/campaign \
  $ROOT/campaign/study/slurm/implicit_train.sbatch
```

Array tasks: 0-2 `implicit` seeds 17-19; 3-5 `implicit_rollout` seeds 17-19; 6 `physics`.

## Limitations

* M* and L* are frozen at q*; their state dependence is not re-linearised. N_theta
  absorbs volume-forcing error but not mass-matrix or time-discretisation error,
  which rollout fine-tuning can only partly compensate.
* Native boundary operators are reproduced only as Dirichlet rows, the identified
  Mach-1 Vpar wall relations and held wall rows; all other free boundary rows use
  the volume weak form.
* An earlier native split (`full_rollout.md`) failed temperature positivity late
  in the full window; here the shared guard bounds positivity, and its activity is
  reported.
* The SciPy LU runs on CPU, single-threaded.

## Status (8 November 2026)

| Item | Result / job |
|---|---|
| Exact labels | 2,000 training states, 0 failures; state identity 1.5e-15 (values), 1.9e-12 (gradients); array 2802451 |
| Unit tests | 16 implicit tests (opsplit + training); full MAST-U suite 373 passed |
| Smoke (v2) | job 2802487, 4-step test MSE 6.4e-5 / 2.4e-5 (1 / 4 substeps) after 3+3 updates |
| Physics-only diagnostic (training cases) | job 2802486; mastu_000: 0.0012 / 0.013 / 0.24 at 16 / 32 / 96 (persistence 0.17 / 0.52 / 2.28) |
| Production array | 2802490 (tasks 0-6 above), snapshot `campaign_v2/study`, ~9 h per task |
| Fixed-conditioning OOD | baselines 2802496 (NODE) 2802497 (AR) 2802498 (OpsSplit); implicit arms 2802499-2802505 (afterok on each training task) |
| Coefficient-physics OOD (9 Nov) | native check 2802708 passed (4e-16 / 7e-16); smoke 2802719 passed; arms 2802739-2802745 -> `$ROOT/coefficient/seed*_*` |

Summarize when complete:

```bash
python scripts/summarize_implicit.py --campaign $ROOT/campaign_v2 \
  --ensemble /pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_ensemble80_comparison_20261106 \
  --gradient /pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_gradient_retraining_20261106 \
  --ood $ROOT/ood --coefficient $ROOT/coefficient --output $ROOT/comparison.json
```
