# Generic OpsSplit research findings — 10 November 2026

**Status: independent confirmation complete, with all 49 frozen models evaluated.** The clear learned-model win is FEM: an equilibrium-preserving defect reduces fresh H=100 error by **75% against the matched classical solver** and **67% against the historical implicit baseline**. Across the four Fourier problems, retaining exact nonlinear physics and using adequate internal steps produces the major gains without learning; the nominated ETDRK4 controls are more accurate still. These results support a conditional strategy, not universal implicit superiority.

Retain known physics, choose an adequate integrator, and add a learned defect only where it improves the matching classical solver. The FEM support gate prevents the observed high-viscosity extrapolation failure, but loses useful extrapolation at the newly tested coefficient 0.003. KS improves strongly at H=50, while its longer-horizon confidence intervals and spectrum errors limit the claim. Detailed confirmed comparisons and inference costs appear below.

Evidence and reproducibility records:

- [Compact confirmed results and provenance](research_confirmation_20261110.json), [full fresh report](../research_runs/20261110/fresh_holdout_results.json), and [comparison figure](research_20261110.png).
- [Frozen selection](research_frozen_selection_20261110.json), [selection protocol](research_selection_20261110.json), and [frozen figure roles](../research_runs/20261110/selected_roles.json).
- [Exploratory comparison tables](research_20261110.md) and [machine-readable results](research_20261110.json).
- [Initial campaign protocol and dataset hashes](research_protocol_20261110.json). Later refinements are recorded in each run's configuration; the initial protocol is not an exhaustive list of subsequent experiments.
- [Detailed benchmark audit](research_audit_20261110.md).
- [Original 381-run comparison](summary.md) and [original machine-readable summary](summary.json), preserved unchanged.
- [Research model implementation](../gos/research_models.py), [training runner](../gos/research.py), and [independent holdout workflow](../gos/research_holdout.py).

## What was audited across all methods

The historical campaign contains 381 runs: 25 variants × three seeds × five problems, plus the two NS2D Oseen variants × three seeds. The audit covers the following families and their combinations of training flags.

| Family | What it tests | Interpretation retained in this campaign |
|---|---|---|
| AR; NODE/Euler; learned Heun | Learned flow map or learned rate without a known stiff operator | Necessary data-only baselines; give NS2D the same fixed-forcing input as implicit arms |
| Explicit known-operator split | Exact linear physics advanced explicitly | Its coarse-step failure establishes a limitation of explicit stepping with those stiff modes, not superiority over every explicit learned model |
| Classical IMEX Euler | Known linear operator and exact nonlinear remainder | Necessary no-learning control, but first-order coarse integration is a weak numerical baseline |
| Known-operator implicit Euler and CN | Learned remainder with fixed physical L | Strong historical FEM baselines; CN is not uniformly robust on dispersive long rollouts |
| Learned structured/free operators | Affine coefficient dependence with or without sign constraints | Rollout fit alone does not identify the true operator; warm starts help some cases and can fail OOD |
| Tangent-free remainder | Removes the network's linear response about zero | Strong historical Burgers/KS variants; extra JVP work must appear in inference costs |
| Physical warm starts | Initialize learned L from known physics | Often useful, but the unconstrained warm start has severe low-coefficient failures |
| Exact-remainder labels and amplification penalties | Identification and amplification control | Labels trade coarse-step accuracy against operator recovery; the historical amplification penalty was a null result and used test-derived thresholds |
| Multiple training step sizes; ROS2 | Transfer across h, 2h, 4h and higher-order stepping | Protocol and numerical order matter; the historical ROS2 propagator diagnostic incorrectly used the theta-method formula |
| Oseen/Picard advection | Put NS2D's state-dependent transport in the implicit solve | Classical stabilization is useful; old learned runs often retained initialization. The repaired residual/adjoint version remains a costly exploratory alternative |

The headline historical KS metric is H=50, although optimization uses the first 100 steps. It measures generalization to unseen trajectories within the training time span. Fresh confirmation also reports H=100 and H=200, plus spectrum/energy statistics, to distinguish short-time predictability from eventual chaotic trajectory divergence.

## Candidate formulation

In normalized model coordinates, retain the semidiscrete physical equation and add a learned load:

```text
M dq/dt = L(mu) q + N_exact(q; mu) + C_theta(q; mu).
```

The network no longer has to rediscover all of `N_exact`. Its final layer starts at zero, so the known-operator model initially equals the matched classical solver. Fourier corrections have their spatial mean removed. For KS, the positive part of the known linear symbol is moved into the explicit physical remainder; the sum still reproduces the original semidiscrete equation, while the implicit operator has no positive real modes.

The main implicit candidate uses ARS(2,2,2), with `gamma = 1 - 1/sqrt(2)` and `delta = 1 - 1/(2 gamma)`. For one internal step of length tau:

```text
(M - gamma tau L) Y = M q + gamma tau F(q)
(M - gamma tau L) q_next = M q
    + tau [(1-gamma) L Y + delta F(q) + (1-delta) F(Y)].
```

Here `F` includes the exact nonlinear load and the selected correction policy. Both stages use the same shifted matrix. The known FEM factorization can be reused within a rollout with fixed coefficients. This is the ARS IMEX family introduced by [Ascher, Ruuth, and Spiteri (1997)](https://doi.org/10.1016/S0168-9274(97)00056-1). Linearly implicit midpoint and a low-gamma ROS2 variant were screened as alternatives; the Rosenbrock/W-method reference is [Verwer, Spee, Blom, and Hundsdorfer (1999)](https://ir.cwi.nl/pub/1252/).

For FEM's stationary zero state, the refinement constrains the learned defect to vanish at zero:

```text
C_theta(q; mu) = M [G_theta(q, mu, x) - G_theta(0, mu, x)] / h.
```

Spatial coordinates and coefficient conditioning can otherwise let the network produce an additive load even at `q=0`. Removing that load is a physically justified way to prevent a spurious long-time offset. It costs two network evaluations per correction. An improvement should be attributed to equilibrium-preserving defect correction; without applying the same constraint to an old learned-remainder baseline, it does not isolate the contribution of retaining exact N.

### Frozen versus stage-wise correction

The economical default evaluates the correction at the start of each requested macro step and holds it fixed through the physics stages. That is a learned discrete defect. A nonzero frozen correction generally loses second-order consistency for the complete learned vector field as the macro step varies.

`--closure-each-stage` recomputes the correction at both stages of every internal step, preserving the complete method's second-order conditions. It costs `2 × internal_steps` network evaluations, doubled again with `--zero-equilibrium`. The classical zero-correction method is second order. These policies must remain separate in the accuracy/cost comparisons.

### What capped internal steps establish

With `--max-step`, a requested macro step dt uses

```text
internal_steps = max(1, ceil(substeps * dt / h)),
internal_dt = dt / internal_steps <= h / substeps.
```

At 4h this normally means four times as many physics steps. The frozen correction still runs once per macro step, or twice with zero-equilibrium subtraction. Matching errors across macro step sizes therefore demonstrate consistent internal stepping; they do not demonstrate accurate physics integration with a four-times-larger internal step or a free speedup. Explicit nonlinear stages still impose an advective restriction. Implicit L alone does not establish unconditional nonlinear stability.

## Exploratory evidence used before freezing selection

Training checkpoints were selected using the original validation split only. Outer research decisions—including stricter NS2D internal steps and the coefficient support gate—responded to the original exploratory test, OOD, and stability results. The entire design process was therefore not validation-only. The final 49 checkpoints, method roles, source hashes, and extra FEM coefficients were frozen before any fresh holdout data were generated.

The following are completed results on the original test trajectories. Historical entries average three training seeds; the listed new controls use one deterministic classical run. Errors are trajectory relative L2 means after clipping each error at 2. None of these rows is an independent confirmation result.

| Problem / horizon | Best historical learned error | Refined classical implicit control | Error | Classical ETDRK4 control | Error |
|---|---:|---|---:|---|---:|
| Burgers / 200 | 0.003289 | ARS222, 2 capped internal steps at h | 5.117e-5 | ETDRK4, 1 step at h | 1.584e-7 |
| KdV / 200 | 0.09990 | ARS222, 16 capped internal steps at h | 0.001129 | ETDRK4, 16 capped steps at h | 3.037e-6 |
| KS / 50 | 0.2988 | ARS222, 16 capped internal steps at h | 0.05649 | ETDRK4, 4 capped steps at h | 0.04756 |
| NS2D / 80 | 0.2462 | ARS222, 32 capped internal steps at h | 9.678e-5 | ETDRK4, 16 capped steps at h | 6.802e-8 |
| Burgers FEM / 100 | 0.003545 | ARS222, 2 internal steps at h | 0.004122 | Fourier-only control; not applicable | — |

These are numerical improvements with exact nonlinear physics. They are not evidence that a network produced the gain, nor comparisons at equal runtime.

The learned-defect pilots add a more specific picture:

- **Burgers:** the lower-learning-rate closure selected iteration zero and exactly reproduced the classical error. A higher-learning-rate fit worsened the final-horizon error to 0.0002429. The current evidence favors retaining the classical solver here.
- **KdV:** with four internal steps, a seed-0 learned correction reduced error from 0.02988 to 0.01860. Increasing classical ARS222 to 16 steps gave 0.001129 without learning. The matched-step gain is promising, but its cost tradeoff and generalization still need confirmation.
- **KS:** completed four-step pilots gave 0.08038 at learning rate 1e-4 and 0.1450 at 1e-5, versus 0.23997 for the matching classical four-step solver. These are exploratory seed-0 results. Classical 16-step ARS222 still gave the lower reported H=50 error, 0.05649. The fresh comparison below tests its longer-horizon behavior.
- **FEM:** the completed zero-equilibrium defect, with learning rate 1e-4 decaying to 1e-7, gives H=100 error **0.0009493 ± 0.0000810** across seeds 0/1/2. That is approximately 77% below the matching classical error 0.004122 and 73% below the historical best 0.003545. The 4h error is 0.001135 ± 0.0000706, and all tested long rollouts are bounded. These runs use two internal steps per requested macro step (`max_step=False`). At the lower OOD coefficient the learned mean is 0.01848, close to the classical 0.01856; at the higher OOD coefficient it worsens to **0.007626**, versus classical **0.0006196**. This original-test learning gain motivated the frozen confirmation below; the extrapolation cost motivated a separate support-gated inference policy.

### Coefficient support gate — selected before fresh confirmation

The optional `closure_in_range=True` policy applies the learned correction only inside the original coefficient training interval. For FEM that interval is `[0.004, 0.012]`; outside it, the model uses the matched exact-physics solver with zero learned defect. The default remains false, so existing checkpoints retain their original behavior unless this policy is explicitly selected.

This is a conservative inference fallback, not evidence that the neural correction extrapolates. It is discontinuous at either coefficient boundary when the learned correction there is nonzero. Do not assume smooth parameter dependence or reliable coefficient gradients across those boundaries. Accuracy still depends on the classical discretization and internal time steps outside the support interval.

To test the selected policy beyond the two previously inspected OOD coefficients, fresh FEM evaluation included preregistered coefficients **0.003** and **0.018**, corresponding to 0.75 times the training minimum and 1.5 times the training maximum. Original coefficients 0.002 and 0.024 remained in the evaluation. The fresh results below show both the benefit and the cost of this fixed fallback; the gate was not retuned afterward.

### NS2D forcing correction changes the baseline

The original NS2D FNO lacked spatial coordinates or the fixed forcing field as input, despite a spatially fixed forcing that breaks translation symmetry. New `*_forced` arms receive the same known forcing channel. The seed-0 NODE error falls to 0.01514, known-L implicit to 0.01227, and warm-started structured implicit to 0.01418. This correction explains much of the improvement over the old 0.25-level learned models; comparing only against the old baseline overstates the implicit-specific gain.

The stronger exact-physics controls remain much more accurate, but their step size matters. The eight-step ARS222 control has low in-distribution error and nevertheless fails at the lower OOD viscosity, with clipped error 2 and a long bounded fraction below one. The capped 32-step control reports low OOD error and bounded original-test long rollouts. The repaired Oseen residual model has a completed 500-iteration pilot, not a full-budget confirmation; it should not be grouped with 4000-iteration baselines as equal-budget evidence.

### Why ETDRK4 must remain visible

ETDRK4 is an exponential method, not an implicit linear-solve method. It uses the exact Fourier linear symbol and exact nonlinear load. Those controls have more physical information than the historical models that learned all of N, although the new exact-physics implicit models receive the same N. Their double-precision internal arithmetic is also stronger than the mainly single-precision implicit implementation.

The contour-stabilized ETDRK4 implementation follows [Kassam and Trefethen (2005)](https://people.maths.ox.ac.uk/trefethen/fourth-order.pdf). This control is especially relevant because the Fourier reference generator uses the same method family on a finer grid. Shared numerical structure, available exact physics, and arithmetic precision all help explain why errors can approach the stored-data floor. The [audit's deterministic precision check](research_audit_20261110.md#numerical-review-of-the-new-candidate-family) shows that arithmetic differences matter at small errors. The extremely small Fourier errors therefore describe this smooth testbed and its reference alignment, not universal real-world accuracy or an equal-precision comparison. These results do not establish universal implicit superiority or justify deleting the exponential control.

## Execution and validation safeguards

All campaign training uses CUDA on an NVIDIA H100 PCIe. Local processes have a hard CPU-time limit of 600 seconds; this is separate from GPU compute time and wall time. Some long runs were interrupted by that process limit. They must not be labeled numerical divergence solely because the process was killed.

The runner now checkpoints before a 450-CPU-second worker budget and starts a new CUDA worker automatically. Checkpoints preserve model parameters, Adam state, learning-rate schedule, best validation state, history, iteration, random-number generator states, and failure counters. A dedicated CUDA regression test verifies exact parameters, history, and CUDA RNG after interruption/resumption against an uninterrupted controlled run. This validates the recovery mechanism; it is not a guarantee of bitwise equality across different hardware or software stacks.

Training wall times include concurrent GPU sharing and resumable worker chunks, so they are descriptive rather than controlled speed comparisons. One legacy KS CPU-time counter lacks its first checkpoint segment and must retain an incompleteness annotation. Scientific metrics and checkpoint selection do not depend on that timing counter.

Final inference timing ran after training stopped, on the idle GPU: identical batches of eight trajectories, eight autoregressive macro steps, h and 4h, two warmups and five synchronized repeats. Timing uses `no_grad`, not `inference_mode`, because tangent-free baselines require JVPs. Reports retain internal-step counts, neural calls, and JVP counts. Timing is not used to select models on the fresh holdout.

## Completed independent confirmation protocol

The 49 selected checkpoint identities, method roles, support-gate policy, result records, relevant source hashes, and additional FEM coefficient cases were frozen before generating fresh data. The holdout workflow preserved the original training scale. Independent seeds were 7103 for test trajectories and 7203/7303 for the original coefficient OOD, with 32 test and 16 trajectories per OOD value. The additional FEM coefficients 0.003/0.018 used seeds 7403/7503; values and seed mappings are retained in data and evaluation provenance. All generation and evaluation used CUDA on the NVIDIA H100 PCIe.

Selected learned seeds, their matched classical solver, historical checkpoints, corrected NS2D data-only baselines, and ETDRK4 controls were evaluated on identical fresh trajectories. Reports retain primary and uncapped errors, nonfinite fractions, coefficient transfer, 4h behavior, and long boundedness. KS includes H=48 for matched 4h comparison, H=100/H=200, and spectrum/energy statistics.

The paired bootstrap resamples held-out trajectories after averaging the nominated training seeds; matching-seed comparisons are also retained. These intervals quantify trajectory sampling uncertainty, not all possible training-initialization uncertainty. Report the training-seed spread separately. The figure reads a preselected roles file and does not choose the best model using fresh test errors.

Generation and evaluation ran one PDE per process to respect the local CPU limit. The merge required identical frozen manifests, all expected runs exactly once, consistent checkpoint/configuration/data hashes, and consistent timing settings. The complete report contains all 49 unique nominated models. An independent audit verified the frozen role and protocol hashes, eight source hashes, every nominated checkpoint and original result hash, all five fresh dataset hashes, error-array aggregates, and timing aggregates. An independent calculation reproduced the reported selected bootstrap intervals to numerical precision.

The broader new campaign contains 44 result records: 21 completed training runs, three gated inference variants, 18 controls without learned parameters, and two partial screens. A gated inference variant reuses its ungated checkpoint; it is not an additional training run. The final test suite reports **149 passed, 20 skipped**.

## Fresh confirmation and recommendation

All primary errors below are means of per-trajectory relative L2 error, clipped at 2 as in the historical benchmark. Learned rows average the three frozen training seeds unless explicitly stated otherwise; each classical control has no training randomness. No method was selected from these fresh errors. The intervals are paired 95% bootstrap intervals over trajectories after averaging the selected seeds, using 2,000 resamples; they do not include uncertainty over new training seeds or correct for the full set of exploratory comparisons.

| Problem / native-step horizon | Historical implicit | NODE | Selected implicit strategy | ETDRK4 physics | Strategy / historical ratio [95% CI] |
|---|---:|---:|---:|---:|---:|
| Burgers / 200 | 0.002897 | 0.005811 | 3.748e-5, physics only | 1.422e-7 | 0.01294 [0.01035, 0.01586] |
| KdV / 200 | 0.06237 | 0.6475 | 0.0008324, physics only | 1.613e-6 | 0.01335 [0.00989, 0.01847] |
| KS / 50 | 0.2840 | 0.2556 | 0.04641, physics only | 0.03379 | 0.1634 [0.09499, 0.2406] |
| NS2D / 80 | 0.1941 | 0.01234, forcing supplied | 8.520e-5, physics only | 7.615e-8 | 0.0004389 [0.0003189, 0.0005964] |
| Burgers FEM / 100 | 0.002335 | 0.006266 | 0.0007704, learned defect | — | 0.3300 [0.2827, 0.3918] |

The four Fourier strategy rows use ARS222 with 2/16/16/32 capped internal steps at h, respectively. Their improvements are **classical numerical gains**, not neural gains. ETDRK4 uses 1/16/4/16 capped steps and double-precision internal arithmetic; its access to exact physics and alignment with the fine-grid reference generator must accompany any accuracy claim.

### FEM: confirmed learning gain, limited extrapolation

The learned defect gives **0.0007704 ± 0.0001482** across three training seeds, against matched physics-only error **0.003087**. The paired learned/physics ratio is **0.2496 [0.1891, 0.3457]**. Against historical implicit CN, the ratio is **0.3300 [0.2827, 0.3918]**. This is the strongest confirmed evidence that learning adds value to the chosen physical integrator. The refinement combined equilibrium preservation with a lower learning-rate floor of 1e-7. It does not isolate these changes or the contribution of retaining exact nonlinear physics separately.

At 4h, the defect error is **0.0009665**, versus **0.003191** for matching physics and **0.09302** for historical CN. The corresponding ratios are **0.3029 [0.2411, 0.3893]** and **0.01039 [0.00473, 0.03650]**. These FEM arms use two internal steps per macro step with `max_step=False`, so this result does test a larger internal step. All native-h long rollouts were bounded for every FEM seed.

The fixed support gate exactly reproduces the ungated learned model on the fresh in-distribution test trajectories. At every tested OOD coefficient, it exactly reproduces the matched physics-only solver, including all recorded h/4h horizons. Its OOD behavior is therefore fallback behavior, not learned extrapolation.

| Fresh FEM coefficient | Historical CN | Ungated learned defect | Gated defect = physics | Gated / ungated ratio [95% CI] |
|---|---:|---:|---:|---:|
| 0.002 | 0.05431 | 0.03278 | 0.01959 | 0.5976 [0.2951, 1.440] |
| **0.003, newly preregistered** | 0.01048 | **0.005824** | **0.01343** | **2.306 [1.572, 3.073]** |
| **0.018, newly preregistered** | 0.006386 | 0.004416 | 0.0009014 | 0.2041 [0.1565, 0.2515] |
| 0.024 | 0.009716 | 0.007593 | 0.0004878 | 0.06425 [0.05473, 0.07479] |

The gate substantially helps at high coefficients but discards useful extrapolation at **0.003**, where its error is 2.31 times the ungated model's. Its ratio to historical CN there is 1.282 [0.7590, 1.973], so superiority over CN is also unconfirmed. At 0.002 its apparent benefit over the ungated model has a wide interval crossing one. The selected gate remains unchanged: this holdout establishes a concrete limitation rather than supplying data to retune the boundary. A smooth or uncertainty-based policy would need a separate future selection and confirmation cycle.

### KS: short-window gain does not establish long-time fidelity

At H=50, ARS222's ratio to NODE is **0.1815 [0.1127, 0.2609]**, a clear short-window improvement. H=50 lies within the optimization time span; H=100 reaches its end and H=200 extends beyond it.

| KS horizon | Historical implicit | NODE | ARS222 physics | ETDRK4 physics | ARS / historical ratio [95% CI] |
|---|---:|---:|---:|---:|---:|
| 100 | 0.5511 | 0.5053 | 0.4466 | 0.3502 | 0.8104 [0.5763, 1.022] |
| 200 | 0.5363 | 0.5597 | 0.4691 | 0.4068 | 0.8746 [0.6695, 1.055] |

Both longer-horizon intervals cross one. The fresh data therefore do not establish superiority over the historical implicit model at these horizons. ARS222 is below NODE at H=200 with ratio 0.8380 [0.6625, 0.9715], but trajectory divergence in this chaotic system should not be confused with attractor fidelity.

The recorded mean absolute log10 spectrum errors are **1.793** for historical implicit, **1.116** for NODE, **2.119** for ARS222, and **1.990** for ETDRK4; lower is better. Corresponding mean energy ratios are 1.020, 1.002, 0.960, and 0.993. The classical controls' improved H=50 trajectory prediction does **not** improve this spectrum statistic; ARS222 is worse than NODE. The experiment does not establish a universal long-time KS winner.

### NS2D: correct the forcing input before attributing an implicit benefit

With the forcing supplied to both models, fresh known-L implicit error is **0.01101 ± 0.001400**, versus NODE **0.01234 ± 0.0002525**. The paired ratio **0.8922 [0.8052, 1.013]** crosses one. A frozen, secondary seed-0 learned-Heun control gives 0.01091, but it is not a three-seed comparison. Correcting the forcing input removes most of the historical learned-model gap; a distinct learned implicit advantage is not confirmed at native h. The exact-physics controls remain far more accurate with adequate internal steps.

### Step transfer and boundedness

| Problem / target horizon | Historical implicit at 4h | Selected implicit strategy at 4h | ETDRK4 at 4h |
|---|---:|---:|---:|
| Burgers / 200 | 0.03344 | 3.748e-5 | 9.446e-8 |
| KdV / 200 | 1.974 | 0.0008324 | 1.380e-6 |
| KS / **48** | 2.000 | 0.04029 | 0.02948 |
| NS2D / 80 | 1.055 | 8.520e-5 | 7.484e-8 |
| Burgers FEM / 100 | 0.09302 | 0.0009665 | — |

KS uses H=48 here because the requested target must be divisible by four; this is not the H=50 native-step comparison. Capped Fourier controls take four times as many internal steps at 4h, so their matched accuracy is expected and costs more per requested macro step. Corrected NS2D learned implicit error at 4h is 0.7725; corrected NODE and learned Heun both reach clipped error 2. The larger-step evidence favors the physical integrators, while capped internal stepping remains part of that benefit.

All selected implicit strategies and ETDRK4 controls have bounded fraction **1.0** on the native-h long test: 1,000 steps for Burgers/KdV/KS, 400 for NS2D, and 500 for FEM. The check uses the first 16 fresh trajectories and a norm threshold ten times each trajectory's maximum reference norm. Every displayed historical implicit and NODE role also has fraction 1.0 except KdV NODE, whose three seeds give 0.6875, 0.375, and 1.0. Repeated seeds share those 16 initial conditions; they are not independent trajectory draws. This is a finite native-h boundedness check, not proof of stability or an OOD/4h long-time guarantee.

### Accuracy costs time

Measured idle-H100 costs below are **milliseconds per macro step for batch eight**, averaged over the selected training seeds. Each measurement uses eight autoregressive steps, two warmups, and five CUDA-synchronized repeats. A 4h macro step covers four times the physical time of an h macro step.

| Problem | Historical implicit, h / 4h | Selected implicit strategy, h / 4h | ETDRK4, h / 4h |
|---|---:|---:|---:|
| Burgers | 3.92 / 4.24 | 1.69 / 4.72 | 1.06 / 2.61 |
| KdV | 1.13 / 1.01 | 10.38 / 44.94 | 11.18 / 47.39 |
| KS | 4.31 / 3.90 | 14.84 / 59.49 | 2.86 / 11.07 |
| NS2D | 5.66 / 5.28 | 36.77 / 141.03 | 23.82 / 97.40 |
| Burgers FEM | 1.89 / 1.64 | 3.99 / 3.50 | — |

FEM physics alone costs **1.37 / 1.36 ms**, so the learned defect costs about **2.92 times** matching physics and **2.11 times** historical CN at h. Its benefit is accuracy, not a runtime speedup. It makes two network calls per macro step because of zero-equilibrium subtraction. The Fourier implicit controls use 2/16/16/32 internal steps at h and 8/64/64/128 at 4h, with no network calls. ETDRK4 uses 1/16/4/16 and 4/64/16/64 respectively. These observed timings are specific to this GPU, batch size, implementation, and short measurement regime; repeat variation and host overhead remain. Shared-GPU training wall times are not controlled speed comparisons.

The resulting recommendation is conditional: use exact-physics integration as the first baseline; for this Fourier testbed keep ETDRK4 visible as the stronger accuracy control; for FEM use the equilibrium-preserving defect when its measured accuracy improvement justifies its cost. The frozen support gate is a conservative high-coefficient fallback with a demonstrated low-coefficient limitation. Further claims about learning, coefficient smoothness, chaotic statistics, or equal-cost superiority require new experiments rather than reinterpretation of this holdout.

## Reproduction commands

Run from `GenericOpsSplit` with the recorded environment. Re-evaluation uses the existing frozen manifest and fresh datasets, and writes to a new directory so the campaign artifacts remain unchanged. Stop other training before requesting comparable timing. Each PDE runs in a separate process to stay within the local CPU-time limit.

```bash
PY=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
ROOT="$PWD/research_runs/20261110"
OUT=$(mktemp -d "$PWD/research_runs/recheck_XXXXXXXX")

for problem in burgers kdv ks ns2d burgers_fem; do
  "$PY" -m gos.research_holdout evaluate --manifest "$ROOT/frozen_selection.json" \
    --holdout-root "$ROOT/fresh_holdout" --problem "$problem" --timing \
    --out "$OUT/holdout_${problem}.json"
done

"$PY" -m gos.research_holdout merge \
  --reports "$OUT/holdout_burgers.json" "$OUT/holdout_kdv.json" "$OUT/holdout_ks.json" \
    "$OUT/holdout_ns2d.json" "$OUT/holdout_burgers_fem.json" \
  --out "$OUT/fresh_holdout_results.json"
"$PY" research_plots.py --report "$OUT/fresh_holdout_results.json" \
  --roles "$ROOT/selected_roles.json" --out "$OUT/research_20261110.png"
```

To regenerate the same preregistered cases into a separate directory, use the same frozen manifest and seeds. These are reproduction samples, not a second independent confirmation. Then point evaluation's `--holdout-root` to the new directory. Generation refuses to overwrite existing data.

```bash
BASE=/pitagora_scratch/userexternal/vgopakum/generic_opsplit_20261109
REGEN=$(mktemp -d "$PWD/research_runs/regenerate_XXXXXXXX")
for problem in burgers kdv ks ns2d burgers_fem; do
  extra_ood=()
  if [ "$problem" = burgers_fem ]; then extra_ood=(--extra-ood 0.003 0.018); fi
  "$PY" -m gos.research_holdout generate --manifest "$ROOT/frozen_selection.json" \
    --data-root "$BASE" --holdout-root "$REGEN" --problem "$problem" \
    --test-count 32 --ood-count 16 --seed 7103 "${extra_ood[@]}"
done
```

For a new FEM training replication, retain the original train/validation data and choose fresh run tags; the launcher resumes bounded CUDA workers automatically. Run each selected seed separately, then derive its support-gated variant without retraining. These commands create new run records; they do not alter the frozen selection.

```bash
RUN_TAG=$(date +%Y%m%d_%H%M%S)
for seed in 0 1 2; do
  "$PY" -m gos.research run --root "$ROOT" --problem burgers_fem --scheme ars222 \
    --substeps 2 --zero-equilibrium --lr 1e-4 --lr-min 1e-7 --iters 5000 \
    --seed "$seed" --tag "reproduce_$RUN_TAG" --cpu-budget 450
  "$PY" -m gos.research_gate --root "$ROOT" --runs \
    "$ROOT/runs/burgers_fem/physics_ars222_n2_known_closure_zero_reproduce_${RUN_TAG}_s${seed}"
done
"$PY" -m gos.research run --root "$ROOT" --problem burgers_fem --scheme ars222 \
  --substeps 2 --no-closure --seed 0 --tag "reproduce_control_$RUN_TAG"
"$PY" research_compare.py "$ROOT" --out "$OUT/research_comparison"
```
