# Implicit Physical OpsSplit with the full native tangent

A fourth model family for the MAST-U 80/0/20 ensemble comparison. The existing
Physical OpsSplit adds a linearised continuity branch for density and
temperature only, evaluated through learned spline derivatives. This family
instead carries the complete linearised model303 volume operator for all seven
fields with exact native bicubic derivatives, integrates it implicitly, and
learns only the nonlinear remainder.

## Formulation

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
  nonzeros, versus 1,178 s and 2.8 billion with SuperLU's COLAMD. A forward solve
  takes ~0.5 s; iterative refinement drives forward and transpose residuals
  below 1e-11.
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
