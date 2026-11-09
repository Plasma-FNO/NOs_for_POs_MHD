# Generic implicit OpsSplit: a stiffness-independent neural PDE step

A PDE-agnostic version of the MAST-U implicit OpsSplit (`JOREK/mastu_opsplit/IMPLICIT_OPSSPLIT.md`), set
against the "unconditionally temporally stable neural operators" question of the upgrade-viva
future-work chapter (sections 4.1.1 and 4.1.2), and tested on standard CFD and time-dependent benchmarks.

## Formulation

Any spatial discretisation gives a semi-discrete system with a mass operator and (optionally)
algebraic rows:

    M dq/dt = F(q; mu),        C(q) = 0

Split the right-hand side about a reference (or the current state) into an implicit linear part and an
explicit remainder:

    F(q; mu) = f(mu) + L(mu)(q - q*) + N(q; mu)

One linearly implicit theta-step of size h (no Newton iteration; one linear solve):

    (M - theta h L) q' = (M + (1-theta) h L) q + h [ f - L q* + N_theta(q) ]
    q' = E_h q + h A_h^-1 [...],   A_h = M - theta h L,   E_h = R(h M^-1 L),   R(z) = (1 + (1-theta) z)/(1 - theta z)

| L | N | model |
|---|---|---|
| 0 | (G(q) - q)/h | autoregressive (a learned A^-1: eq. 4.14 of the chapter) |
| 0 | G(q) | NODE / Euler |
| 0 or known, explicit | sum_k alpha_k NO_k(q) | explicit OpsSplit (eqs. 4.3-4.4) |
| known (native tangent) | exact N | classical IMEX / physics-only |
| known | learned | MAST-U implicit OpsSplit |
| L0 + mu L1, known | learned | coefficient physics (affine in mu) |
| learned J - D (+ mu-affine) | learned | **structured implicit neural operator** (section 4.1.2) |

Keeping A explicit and *solving* with it is the inductive bias; learning A^-1 directly is just AR.

## Stability statements

Assume M is symmetric positive definite and use the M-norm.

1. **Linear part.** If x^T L x <= -gamma x^T M x with gamma >= 0 (L "dissipative"), then for theta >= 1/2
   and *every* h > 0, ||E_h||_M <= 1, and for theta = 1, ||E_h||_M <= 1/(1 + h gamma). Proof: with
   qbar = theta q' + (1-theta) q,
   (||q'||^2 - ||q||^2)/2 = h qbar^T L qbar - (theta - 1/2)||q' - q||^2 <= 0.
2. **With the learned explicit remainder** (theta = 1): q' = E_h (q + h M^-1 N(q)). If M^-1 N has
   Lipschitz constant l, the step has Lipschitz constant <= (1 + h l)/(1 + h gamma). The bound does not
   depend on how stiff L is, and the step is contractive for every h when l <= gamma.
3. **Parameterisation that guarantees (1).** L = J - D with J skew-symmetric (transport, dispersion,
   ideal/Hamiltonian coupling) and D symmetric positive semidefinite (viscosity, resistivity, diffusion,
   drag). With a local stencil (tridiagonal in 1D, the native element graph in general) A stays sparse
   and factorisable. Sparsity buys cost; the sign structure buys stability.
4. **Physical instability.** If the true linearisation has growth rate omega > 0 (KS band, tearing,
   Rayleigh-Taylor), no norm makes it dissipative. Either the growth is carried by N (bounded by l) or L
   is allowed sym(L) <= omega M, which makes stability conditional on resolving 1/omega but still
   independent of the stiff stable modes. Backward Euler damps genuine growth when h omega > 2.

## Where L comes from, by PDE class

| class | examples | stiff part | natural L | structure |
|---|---|---|---|---|
| parabolic / gradient flow | heat, reaction-diffusion, Allen-Cahn, Cahn-Hilliard | diffusion (k^2, k^4) | diffusion operator | -D; stabilised semi-implicit schemes (adding S(q'-q)) are exactly "move Lipschitz from N into L" so l <= gamma |
| dispersive | KdV, NLS | dispersion (k^3, k^2 i) | dispersion operator | pure J (gamma = 0): theta = 1 damps, theta = 1/2 conserves |
| viscous transport (low/moderate Re) | Burgers, advection-diffusion | viscosity at fine grids | viscous operator | -D, advection in N |
| advection-dominated CFD | high-Re NS, Euler | advective CFL (nonlinear) | linearised advection about q* or q_n (Oseen/Picard) | J (+ -D); needs re-linearisation (Rosenbrock-W) and an iterative solve |
| low-Mach compressible, MHD | acoustics, Alfven/magnetosonic waves | linear waves about a background | linearised wave operator (native tangent) | J - D about a stable equilibrium; plus constraint rows (div B, current definition) |
| physically unstable | KS, tearing, RT | unstable band + stiff damping | linear operator | J - D + omega I (omega bounded) or growth in N |

## Testbed

`gos/` implements the family for five problems (`gos/problems.py`); every one is generated on a finer
grid than the model grid, so a known L still leaves a closure to learn.

| problem | grid | h | coefficient (train / OOD) | what it tests |
|---|---|---|---|---|
| `burgers` | Fourier 256 (ref 1024) | 0.05 | nu in [0.02, 0.06] / 0.012, 0.1 | viscous CFD, shocks, stiff diffusion |
| `kdv` | Fourier 256 (ref 512) | 0.02 | delta^2 in [4.8e-4, 1.2e-3] / 2.6e-4, 2e-3 | purely skew L, gamma = 0 |
| `ks` | Fourier 128 (ref 256) | 1.0 | - | physically unstable band, chaos, k^4 stiffness |
| `ns2d` | Fourier 64^2 (ref 128^2) | 0.5 | nu in [1e-3, 4e-3] / 6e-4, 7e-3 | 2D incompressible NS (vorticity), Li et al. forcing |
| `burgers_fem` | P1 FEM, 127 nodes (ref 511) | 0.01 | kappa0 in [4e-3, 1.2e-2] / 2e-3, 2.4e-2 | Dirichlet walls, M != I, variable kappa(x), a(x), tridiagonal L |

Arms (`gos/models.py`): `ar`, `node`, `explicit_known`, `imex_exact` (no learning), `implicit_known`,
`implicit_known_cn` (theta = 1/2), `implicit_struct` (learned J - D, affine in mu), `implicit_free` (learned,
same locality, no sign constraint). All learned arms share one FNO backbone with log(mu) as an input
channel; learned operators are L = O0 + mu_t O1 (section 4.1.1's alpha NO_1 + beta NO_2 with known
coefficients). Training: 8-step rollout MSE on the first half of each trajectory, validation selection.
Evaluation: test rollouts to the full (2x training) horizon, 2h/4h/h-half step transfer, OOD coefficients,
5x-length stability rollouts, step-Jacobian sigma_max, and recovery of the true operator.

```bash
PY=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
ROOT=/pitagora_scratch/userexternal/vgopakum/generic_opsplit_20261109
$PY -m pytest -q tests                                   # 52 tests
$PY -m gos.train data --problem burgers --root $ROOT
$PY -m gos.train run --problem burgers --arm implicit_struct --seed 0 --root $ROOT
$PY summarize.py $ROOT                                   # -> $ROOT/summary/summary.{json,md}
```

## Results (9 November 2026; 210 runs = 14 variants x 3 seeds x 5 problems)

Full tables: [results/summary.md](results/summary.md) (machine-readable: `results/summary.json`). Figures
(`fig_error_vs_horizon.png`, `fig_step_transfer.png`, `fig_operator_recovery.png`) are not tracked (the repo
ignores `*.png`); they are in `$ROOT/summary/` and are regenerated with `python plots.py $ROOT`. Checkpoints,
per-run JSON and generated data stay on scratch under `$ROOT` (runs/, data/; the transient-dominated first
NS2D sweep is in archive/). Errors are test relative L2 (mean over trajectories, clipped at 2), mean over
seeds, at the longest horizon (2x the training horizon) unless stated.

**Main comparison (single training step h):**

| arm | Burgers H=200 | KdV H=200 | KS H=50 | NS2D H=80 | FEM H=100 |
|---|---|---|---|---|---|
| ar | 0.062 | 0.46 | 0.36 | 0.29 | 0.071 |
| node | 0.0059 | 0.35 (bounded 0.77) | 0.33 | 0.26 | 0.0080 |
| explicit_known (4.1.1 explicit Euler) | diverged | diverged | diverged | diverged | diverged |
| imex_exact (no learning) | 0.011 | 1.81 (unbounded) | unbounded | 0.91 | 0.018 |
| implicit_known | 0.0071 | 0.19 | 0.44 | 0.27 | **0.0038** |
| implicit_known_cn | 0.0073 | 0.25 (bounded 0.56) | 0.40 | 0.27 | **0.0036** |
| implicit_struct | 0.0052 | 0.45 | 0.31 | 0.25 | 0.0054 |
| implicit_free | **0.0035** | 0.22 | 0.32 | 0.27 | 0.0073 |
| implicit_free_tf | **0.0033** | **0.14 (bounded 1.0)** | **0.30** | 0.28 | 0.056 |

Findings:

1. **Implicit L is what makes an OpsSplit trainable at stiff steps.** The explicit Euler split of section
   4.1.1 with exact operators diverges on all five problems (sigma_max of the step Jacobian 30-1200). Every
   implicit arm trains and its 5x-length rollouts stay bounded on Burgers, KS, NS2D and FEM.
2. **The stability bound predicts which classical IMEX steps fail.** With exact N, IMEX Euler is stable
   where L is strongly dissipative (Burgers, FEM: sigma_max ~ 1.0) and unstable where it is not: KdV (skew L,
   gamma = 0), KS and NS2D (advective h*l >> 1 + h*gamma).
3. **The structured guarantee on L does not show up as a measurable gain.** Correction (follow-up): the
   one-step amplification of the near-exact flow map is itself 1.40 (KdV), 2.2 (KS), 1.15 (NS2D) at the
   largest test state (`measure_amplification.py`), so the learned models' sigma_max of 1.33-1.41 on KdV is
   physical, not a defect of N. KdV long-horizon unboundedness is therefore not explained by one-step
   amplification. theta = 1/2 is less bounded than theta = 1 (0.56 vs 0.94), consistent with backward
   Euler's extra numerical damping.
4. **Learned operators are not identified by rollout loss.** The FNO absorbs most linear response. Partial
   exceptions: free L recovers Burgers diffusion for k <~ 20 (low-band error 0.50; 0.40 with a tangent-free
   remainder); struct L recovers about half the FEM advection profile (0.53; 0.35 with multi-step training).
   No learned arm finds the FEM conductance. Where a learned implicit L is identified, it moves toward the
   operator whose backward-Euler resolvent reproduces exp(hL) (KdV: Re h*lambda > 0 at low k), not toward
   the PDE operator. Caveat: struct L starts at 1% of the scale through exp(), which slows it.
5. **Coefficient OOD gains come from a known L(mu), as in MAST-U.** FEM kappa0 = 0.024: implicit_known 0.018,
   CN 0.012 vs node 0.157, AR 0.30. Burgers nu = 0.1: 0.036 vs node 0.105. A mu-affine learned operator
   carries the coefficient only where it was partly identified: Burgers free L 0.015 (tangent-free 0.010) at
   nu = 0.1, but on FEM, where the conductance was not found, struct/free are no better than node.
6. **Step-size robustness comes from the training protocol, not the implicit structure.** Single-h models of
   every kind degrade at 2h-4h (Burgers 4h: free 0.074, known 0.15, node 0.72; KdV/KS fail). Training on
   {h, 2h, 4h} flattens error across steps for every arm, node included (Burgers 4h ~0.015-0.018, KS 4h 0.32
   vs >= 1.2), but costs ~5x accuracy at h: a step-independent remainder in a first-order scheme carries the
   O(h) error of IMEX Euler. Accurate and step-robust needs a higher-order linearly implicit integrator
   (SBDF2, Rosenbrock-W, ETD), the implicit analogue of the Euler/midpoint/RK4 table in section 4.1.1.
7. **NS2D (advection-dominated CFD): all learned arms tie (0.25-0.29).** The binding stiffness is advective
   and sits in N, so implicit viscosity cannot help in-distribution (it does stabilise 2h transfer: known
   0.39 vs node 1.6). Exploiting the formulation here needs linearised advection in L (Oseen/Picard about
   q_n, iterative solve). An initial-transient protocol issue (rollouts from the decaying GRF initial
   condition) was fixed with a 5-time-unit burn-in; the first NS2D runs are archived.

Next experiments: supervise N with exact remainder labels (available for all five problems; the MAST-U
recipe) to test identifiability of learned L; penalise sigma_max(DPhi) or spectrally normalise N; SBDF2 /
Rosenbrock-W steps with multi-step training; warm-start struct L; NS2D with linearised advection in L.

## Follow-up experiments (9-10 November 2026; 171 runs, 3 seeds each)

Flags: `--labels` (exact remainder labels: 40% label-only pretraining of N, then rollout loss + 0.01 x label
loss), `--jac-weight` (penalty on step-Jacobian amplification above the near-exact flow map's own value,
`measure_amplification.py`), `--init-known` (learned L initialised from the true operator), arms `ros2_*` /
`node_rk2` (second order, trained on {h, 2h, 4h}), `oseen*` (NS2D with advection re-linearised every step and
solved implicitly by batched GMRES). Run names carry suffixes `_lab`, `_lip`, `_init`, `_mdt`.

1. **Warm-started learned operators are the strongest variant.** The structure-constrained L keeps the physics
   and learns a correction. KdV H=200: 0.10 (free) / 0.12 (struct) vs 0.19 known L and 0.22 free from scratch,
   bounded 0.96-1.0. FEM: struct_init 0.0039 (known 0.0038) with conductance recovered to 5% (was 97% wrong) and
   skew to 15%; kappa0 = 0.024 OOD 0.026 vs 0.163 untrained-init struct. Burgers OOD nu = 0.1: 0.014-0.022 vs
   0.036 known L. NS2D: propagator error 5 -> 0.7, accuracy unchanged. KS: no change.
2. **Exact remainder labels improve operator recovery but cost accuracy at the coarse step.** Recovery: KdV free
   0.93 -> 0.44, KS struct 8.6 -> 2.0, FEM skew (free) 0.92 -> 0.44; FEM conductance still not found. Test error
   worsens (FEM 0.005 -> 0.14; Burgers struct 0.005 -> 0.08): pinning N to the exact nonlinearity leaves only L to
   absorb the time-discretisation error. KdV OOD improves (free 1.99 -> 0.56).
3. **The amplification penalty is a null result.** With the threshold at the physical amplification it rarely
   activates; results match the unpenalised runs on every problem.
4. **Second-order steps recover most of the accuracy multi-step training costs.** At h / at 4h: Burgers ROS2-known
   0.012 / 0.012 vs implicit-Euler 0.026 / 0.018; FEM 0.017 / 0.0087 (best 4h of any arm) vs 0.039 / 0.013; KdV
   ROS2-free 0.53 / 0.27 (best 4h). Explicit Heun with a learned RHS does as well or better elsewhere (Burgers
   0.009 / 0.009; FEM 0.012 / 0.011), because the network does not represent the stiff modes. ROS2 with the known
   KS operator is unbounded.
5. **NS2D with implicit linearised advection.** Without learning (`oseen_exact`) it stabilises the classical
   scheme (bounded 1.0 vs 0.5; H=80 error 0.52 vs 0.91; 4h 0.66 vs 1.17) and is the most accurate model at H=10
   (0.108 vs ~0.15 learned), but less accurate than learned models at H=80 (~0.26). The learned `oseen` arm did not
   train: losses diverged in 2/3 seeds (best checkpoint = initialisation) and reached val 0.67 in the third, so
   this result is inconclusive. Likely fixes: keep the exact forcing in the physics part so training starts from
   `oseen_exact`, adjoint (implicit) differentiation instead of backpropagating through 40 GMRES iterations, lower
   learning rate.
