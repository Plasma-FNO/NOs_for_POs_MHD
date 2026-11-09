# MAST-U ensemble: L-SplineFNO AR, NODE and Physical OpsSplit

Prepared 6 November 2026. Dataset and implementation status were inspected on
that date. This is a handoff for setting up ensemble training, not a record of
completed ensemble model runs.

## Objective

Compare L-SplineFNO as an autoregressive predictor (AR), an Euler neural ODE
(NODE), and Physical OpsSplit on independent MAST-U initial-condition
trajectories. Keep data splits, normalization, architecture, training budgets,
and physical corrections matched. Do not include persistence in this comparison.

The dataset and model components exist. The ensemble training loop still needs
wiring: the current Euler trainer is a bounded two-parent chronological pilot.
Do not present that pilot's results as ensemble results or assume that changing
its input path is sufficient.

## Locations and environment

Repository:

```text
/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD
```

Study directory:

```text
/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit
```

Existing Python environment:

```text
/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
```

L-SplineFNO dependency:

```text
/pitagora/home/userexternal/vgopakum/Conditional_Fusion_Operator
```

The dependency commit is `a3ca59bba100ca56e535fc35c32967bdeb04d8d9`.
[configs/lsplinefno.yaml](configs/lsplinefno.yaml) pins the commit and imported
source hashes. Reuse the installed dependency and environment.

## Accepted dataset

Dataset root:

```text
/pitagora_scratch/userexternal/vgopakum/mastu_euler_ensemble100_20261105
```

Entry point: `manifest.json`. Completion record: `completion.json`.

| Property | Value |
|---|---|
| Accepted trajectories | 100 |
| Training / validation / test | 60 / 20 / 20 |
| Frames per trajectory | 97 |
| Total frames | 9,700 |
| Time range | 4.2177393336842355–5.623652444912314 microseconds |
| Saved-frame interval | 1.4644928241958522e-08 seconds, approximately 14.645 ns |
| Canonical native DOFs | 46,203 |
| Native elements | 11,257 |
| State shape per trajectory | `[97, 46203, 7]` |
| Fields, in order | `psi, u, zj, w, rho, T, Vpar` |
| Training normalization frames | 5,820 |
| Maximum reported timestep-refinement dynamic error | 0.003730644817469726, approximately 0.3731% |

The reference simulations use unmodified native JOREK model303. Euler is the
surrogate integration rule. All 100 accepted cases selected the `fine` reference
rate. The dataset varies density/temperature perturbation location, width,
amplitude and sign on one fixed axisymmetric MAST-U equilibrium and mesh.

This tests independent initial conditions, not independent equilibria,
nonaxisymmetric tearing, spatial convergence or cross-equilibrium transfer.
The common unperturbed control is diagnostic only and is not a training,
validation or test trajectory.

`native_state` contains canonical bicubic value and derivative DOFs, not
ordinary nodal field samples. Preserve their geometry and coefficient semantics.
Do not resample the simulation state onto a Cartesian grid. L-SplineFNO's
internal regular latent grid does not change the native simulation mesh.

Preserve the manifest's trajectory and parent-family splits. Never randomly
divide overlapping windows across training, validation and test.

The manifest SHA-256 verified against the completion record is:

```text
f0f0f56c611c9f9d48081d4a611bfa44a0709e5cf4b691fc219c69396c81b87c
```

The handoff inspection also verified geometry and normalization hashes, all 100
accepted status records, and positive ensemble-mean Bernstein bounds. It did
not rehash every trajectory. Run the loader's full verification before training.
Shared scratch requires a separate archival copy for long-term retention.

## Dataset loader

Reuse [src/mastu_opsplit/euler_dataset.py](src/mastu_opsplit/euler_dataset.py).
With the study's `src` directory on `PYTHONPATH`:

```python
from mastu_opsplit.euler_dataset import EulerDataset

data = EulerDataset(
    "/pitagora_scratch/userexternal/vgopakum/"
    "mastu_euler_ensemble100_20261105/manifest.json"
)
train = data.windows("train", horizon=4, stride=1)
validation = data.windows("validation", horizon=4, stride=1)
test = data.windows("test", horizon=4, stride=1)

sample = train[0]
# sample["state"]: normalized float64 tensor [5, 46203, 7]
# Other entries: case_id, time_seconds, parameters, start_frame.
```

Four-interval windows number 5,580 / 1,860 / 1,860 across the three splits.
The loader verifies artifact hashes, trajectory identities, split membership,
geometry, observation times, and training-only normalization provenance.

Use `data.geometry`, `data.mean`, `data.scale`, and `data.time_seconds`.
Normalization was fitted to all frames of the 60 training trajectories only.
Do not recompute it using held-out cases or reuse the old pilot's normalizer.
Center native coefficients in float64 before neural float32 conversion.

Stream windows instead of loading the full ensemble into memory. Start with
batch size one and a simple loader, then profile before increasing concurrency.

## Model definitions

Reuse `EulerSplineModel` and `NativeSplineSpace` from
[src/mastu_opsplit/euler.py](src/mastu_opsplit/euler.py).

Let `x` be the normalized native coefficient state. Let `N_theta(x)` denote the
L-SplineFNO output projected into native coefficient space. Let `G(previous,
proposal)` be the shared positivity and elliptic-constraint correction.

```text
AR:                x_next = G(x, N_theta(x))
NODE:              x_next = G(x, x + h*N_theta(x))
Physical OpsSplit: x_next = G(x, x + h*(L_star(x) + N_theta(x)))
```

One saved-frame interval is one pseudotime unit. Euler substeps use
`h = 1 / substeps` over that interval, with the same guard at each substep.
AR predicts the next state directly and has no Euler substep refinement.

Euler advances normalized native field coefficients. L-SplineFNO has an internal
latent grid, but this implementation does not evolve a separate hidden latent
state. Training is supervised through field-trajectory rollout losses; separate
nonlinear-RHS labels are unnecessary.

All families share the neural architecture. The current conditioning input is
the same constant `parameters` tensor supplied by the loader. The initial state
contains the perturbation; perturbation metadata is not an extra conditioning
input in this comparison.

## Physical OpsSplit linear branch

Select `linear_branch: lspline_continuity`. Reuse `FrozenContinuity` from
[src/mastu_opsplit/continuity.py](src/mastu_opsplit/continuity.py), not the older
signed-curvature `lspline_analytic` branch or replay-only `native_bicubic` branch.

The physical branch linearizes the following axisymmetric model303 advection
and compression terms around the training-only spatial mean:

```text
a = (R*u_Z - Vpar*psi_Z/R, -R*u_R + Vpar*psi_R/R)
c = 2*u_Z - (Vpar_R*psi_Z - Vpar_Z*psi_R)/R
rho_t = a . grad(rho) + c*rho
T_t   = a . grad(T) + (gamma-1)*c*T
gamma = 5/3
```

The temperature equation follows from `p=rho*T` and the native pressure equation.
The tangent includes couplings through `u`, `psi`, `Vpar`, `rho`, and `T`.

- Background values and gradients use native bicubic evaluations of the mean.
- Perturbation values use native sampling.
- Perturbation gradients use analytic first derivatives of the L-SplineFNO
  scatter/synthesis, sharing its learned scatter kernels.
- Physical channel scales are restored before the tangent and removed afterward.
- Native consistent-mass projection returns the result to coefficient space.
- The branch has no fitted amplitude or additional trainable parameters.

Its only nonzero output channels are density and temperature. Diffusion, sources,
other equations, background forcing and nonlinear residuals remain learned by
`N_theta`. The branch is linear in the normalized state for fixed scatter weights
and vanishes at the training mean. This is a partial physical operator, not a
full JOREK Jacobian, native pressure-mass solve, or conservative MHD integrator.

### Critical change: observation interval

The ensemble stores frames at half the interval used in the previous pilot.
Derive the physical multiplier from the selected stride and native metadata:

```python
code_interval = observation_interval_seconds / sqrt_mu0_rho0
# sqrt_mu0_rho0 = 4.686377037426928e-07 seconds for these references.
# stride=1: code_interval approximately 0.03125
# stride=2: code_interval approximately 0.0625
```

Read the time normalization from a verified native initial restart, such as the
manifest's `initial` path, and check compatibility across cases. Use the window
loader's effective observation interval when changing stride. Do not carry the
old `0.0625` into stride-one training. `FrozenContinuity` already includes this
physical multiplier; Euler substeps divide the frame update without applying
the physical interval a second time.

## Shared positivity and constraints

Reuse [src/mastu_opsplit/physical_guard.py](src/mastu_opsplit/physical_guard.py)
identically for all three families:

1. Recover interior `psi/u` from predicted `zj/w` using native weak elliptic
   constraints. The current identity uses the `R^-2` weight; vorticity uses
   ordinary toroidal volume weighting.
2. Preserve potential boundary nodal values from the previous state. Boundary
   derivative DOFs remain learned because native references evolve them.
3. Enforce density/temperature positivity through subdivided Bernstein bounds
   over entire native bicubic cells, including edges and vertices.
4. Retain the common penalty on negative proposed Bernstein bounds.

Construct the guard with ensemble geometry and normalization. Its positivity
anchor must remain strictly positive. The checked ensemble training-mean lower
bounds are approximately `3.01286167e-4` for density and `8.92926990e-5` for
temperature, so the existing anchor requirement passes.

For independent denser audits, reconstruct Q8/Q12 geometry from a verified native
initial restart using `read_restart`; confirm it matches the shared geometry.
Check fixed potential boundary nodal values over the new trajectories before
assuming the old pilot's boundary check carries over.

Save limiter intervention counts, minimum retained fraction, raw proposed
minima, whole-cell lower bounds and constraint defects. These corrections can
damp dynamics and do not enforce full MHD conservation. Positivity of corrected
predictions does not mean raw neural proposals are positive.

## Trainer work still required

[scripts/train.py](scripts/train.py) currently routes normalized Euler runs to
[src/mastu_opsplit/euler_training.py](src/mastu_opsplit/euler_training.py).
That trainer loads two old references, uses fixed chronological frame splits,
samples two-step training windows, and caps diagnostics at 200 updates.
[configs/continuity_euler_pilot.yaml](configs/continuity_euler_pilot.yaml) still
targets those old references. Do not use it unchanged for ensemble training.

Implement a distinct ensemble entry point or explicit trainer dispatch while
keeping the old pilot reproducible:

- Connect `EulerDataset` and replace hardcoded references, frame splits and
  sampling schedules with the manifest's disjoint trajectory splits.
- Construct native projection, the physical guard and frozen physical background
  once. Reuse them across windows without recomputing sparse factorizations.
- Use ensemble normalization and the effective observation interval.
- Add resumable checkpoints, saved sampling/RNG state and immutable provenance.
- Adapt rollout evaluation and auditing to arbitrary held-out trajectories.
- Record manifest and source hashes, architecture, seed, optimizer, sampled
  windows, update budget, physical interval and selected checkpoint.
- Add focused integration tests for split isolation, training-only background,
  stride/time scaling, resume behavior, guards and all three model families.

The older native physical-RHS experiment and its failed full-window SBDF2 split
are separate. Do not accidentally route this trajectory-supervised ensemble
comparison through that experiment's nonlinear-label or native-split gates.

## Matched experiment protocol

Start from the validated small architecture and shared settings:

```yaml
architecture:
  grid_shape: [16, 32]
  modes: 4
  width: 8
  n_layers: 2
learning_rate: 0.001
physical_guard: true
positivity_basis: bernstein_subdivided
positivity_penalty: 0.001
gamma: 1.6666666666666667
```

Use Adam, gradient clipping at 2, and native volume-weighted normalized field
MSE. Start with four-step training windows and batch size one for integration
checks. Keep autoregressive rollout training rather than teacher-forcing every
predicted step.

For each matched seed, give all three families identical initial neural weights,
sampled windows, batch size, update count, training horizon, optimizer, validation
frequency and checkpoint-selection rule. Select checkpoints on validation
trajectories only. Compare multiple matched seeds, for example 17, 18 and 19.
Choose the longer training budget after measuring ensemble throughput; the old
120-update run was a small diagnostic, not a converged ensemble training budget.

Evaluate autonomous 16-, 32- and 96-step rollouts, reporting physical duration
and the chosen starting frames. A 16-step stride-one rollout now covers half
the physical duration of the old 16-step stride-two pilot. Report:

- Aggregate and per-field native volume-weighted normalized MSE.
- Decoded and dynamic-relative errors, with denominator definitions recorded.
- Whole-cell density/temperature positivity and independent Q12/vertex checks.
- Current/vorticity defects, limiter interventions and numerical failures.
- NODE/OpsSplit fixed-weight 1/2/4-substep sensitivity over matched durations.
- Training and evaluation runtime, parameter count and resource use.
- Variation across held-out trajectories and training seeds. Overlapping windows
  are not independent samples for uncertainty estimates.

The 16x32 learned representation previously had large errors in reconstructed
gradients despite mathematically correct analytic differentiation. Explore finer
spline grids through matched validation experiments; do not tune resolution on
test scores. Improvements need not be uniform across fields, and the comparison
must report outcomes even if OpsSplit does not win.

## Existing evidence, not ensemble results

The previous 120-update, single-seed chronological pilot produced:

| Model | Mean held-out normalized volume MSE |
|---|---:|
| AR | 5.62320450 |
| NODE | 0.76919436 |
| Previous signed-curvature OpsSplit | 0.79168474 |
| Physical continuity OpsSplit | 0.75460207 |

Physical OpsSplit improved aggregate MSE by 1.897% against NODE and 4.684%
against the previous OpsSplit. Both parent rollouts improved against NODE,
mainly through density; not every field improved. All three new models had
9,673 trainable parameters. AR and NODE replayed their previous predictions
bitwise. Physical OpsSplit took about 1.63 times NODE's training/evaluation time.

All 31 tests and the actual-mesh preflight passed. Corrected predictions passed
whole-cell positivity and independent Q12/vertex checks. Maximum interior
constraint defect was approximately `5.244e-13`. The physical model's limiter
still acted on 26 of 64 density/temperature field-updates.

See [reports/continuity_pilot_20261105.json](reports/continuity_pilot_20261105.json)
and [README.md](README.md) for diagnosis, ablations and limitations.

The completed immutable experiment is:

```text
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_continuity_pilot_20261105
```

It contains `study/`, source hashes, `run.sbatch`, test results, preflight,
checkpoints, rollout predictions, audits and completion metadata. Job 2785214
completed successfully in 25m05s. These results establish neither independent
trajectory generalization nor statistically reliable model superiority.

## Execution and acceptance

The working CPU setup used Slurm account `fupa2_fauna`, four allocated CPUs,
two Torch threads and 24 GB RAM. The pilot ran on `dcgp_fua_dbg`; longer runs
need a suitable production partition/time limit, such as the previously used
`dcgp_fua_prod`. Check current scheduler availability and limits before launch.

Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=2` and
`PYTHONDONTWRITEBYTECODE=1`. Native projection and constraint solves currently
use CPU SciPy with exact-transpose custom autograd. Profile before assuming a
GPU will accelerate the entire rollout. Preserve dependency hashes and use
fresh output directories and frozen source/config snapshots for submitted jobs.

Before a longer run, require full loader verification, unchanged native geometry,
valid training-only background/time scaling, finite forward/backward passes for
all families, and passing positivity/constraint checks on the actual mesh.
Checkpointed batch jobs should survive client disconnection, but monitor their
scheduler completion and exit codes rather than assuming launch means success.

No ensemble model training was launched while preparing this handoff. The next
work is trainer integration, a bounded matched smoke run, then the validated
multi-seed comparison. Preserve the existing dataset and completed pilot outputs.
