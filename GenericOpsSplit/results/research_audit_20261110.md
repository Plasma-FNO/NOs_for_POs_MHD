# Generic OpsSplit benchmark audit, 10 November 2026

This audit reads `gos/train.py`, `gos/models.py`, `gos/problems.py`, `measure_amplification.py`, and the 381-run historical summary. It precedes new model selection.

## Historical reference points

These are the strongest reported mean final-horizon errors across all historical methods, including follow-ups. The metric averages trajectory relative L2 errors after clipping each at 2.

| Problem | Horizon | Best historical method | Mean error | Long bounded fraction |
|---|---:|---|---:|---:|
| Burgers | 200 | implicit_free_tf | 0.003290 | 1.000 |
| KdV | 200 | implicit_free_init | 0.09990 | 0.958 |
| KS | 50 | implicit_free_tf | 0.2988 | 1.000 |
| NS2D | 80 | implicit_struct_tf | 0.2462 | 1.000 |
| Burgers FEM | 100 | implicit_known_cn | 0.003551 | 1.000 |

Warm-starting the learned linear operator is therefore not a universal accuracy winner. The unconstrained warm start also fails at the smaller out-of-distribution coefficient: Burgers and KdV both have clipped errors of 2.0 there. Every candidate must retain both coefficient extremes in reporting.

## Findings that affect interpretation

1. **KS is not tested beyond the training time span by the headline accuracy metric.** Training uses the first 100 steps of 201-frame trajectories, while reported pointwise accuracy ends at step 50. Test trajectories are independent, so this remains a legitimate trajectory-generalization test. The README's blanket description of the headline horizon as twice the training horizon does not apply to KS. Spectrum statistics use later steps and should accompany any KS improvement.
2. **NS2D has a representational mismatch.** Its fixed spatial forcing breaks translation symmetry, but `needs_coords=False` gives learned models a translation-equivariant FNO without forcing coordinates. The learned remainder must infer spatial phase from the state. Supplying the known forcing as part of the physical step is a strong, testable correction.
3. **Existing ROS2 propagator diagnostics do not measure the ROS2 propagator.** `propagator_error` always computes the theta-method rational function. Recovery of L remains meaningful, but reported propagator errors must not support ROS2 claims until corrected.
4. **The amplification penalty uses test-derived thresholds.** `measure_amplification.py` derives the constants from `data['test']`. This is test-informed hyperparameter selection. The penalty was a null result, but future thresholds should use training/validation states.
5. **Temporal transfer compares multiple numerical costs.** A model at h/2 performs two substeps per data interval; ROS2 performs two remainder evaluations. New methods with internal substeps need their substep counts and measured inference times reported. Accuracy alone does not establish speedup.
6. **The reference problem admits highly accurate classical solvers.** New exact-physics/learned-defect models must include the identical integrator with zero learned defect. A large improvement over old learned models can be an integrator improvement rather than evidence for learning.
7. **Clipped relative errors conceal failure severity.** An error of 2 can mean moderate inaccuracy or explosive divergence. Retain long bounded fractions and, where available, uncapped errors/nonfinite counts. A zero solution is bounded but inaccurate, so boundedness alone is not stability plus fidelity.
8. **Normalization sees the full training trajectories.** The scalar standard deviation is computed over all training frames although optimization uses their first half. No test trajectories leak, but strict temporal-extrapolation experiments should compute normalization from the optimized time span.
9. **A completed training status does not establish successful learning.** Two of three Oseen runs report `completed` but retain the initialization checkpoint (`best_iteration=0`). Report the selected checkpoint iteration alongside status when diagnosing new variants.

## Confirmation protocol

- Select the formulation, integrator, substep count, and training settings on validation data. Existing test results are exploratory after repeated adaptive inspection.
- Confirm the selected strategy with three training seeds and fresh held-out trajectories, using independent generation seeds such as 7103 and distinct seeds for each OOD coefficient. Preserve the original generator distribution and reference resolution.
- Compare with the best historical implicit method, the strongest learned method without a known implicit operator, and the same new classical integrator with zero learned correction. Retrain or reevaluate the relevant baselines on the fresh holdout.
- A substantial primary win means at least a 25% mean error reduction at the same reported horizon, supported across seeds. Report both OOD coefficients, h/2, 2h, 4h, long boundedness, and KS statistics separately. Do not hide regressions in an aggregate score.
- Use matching dataset, network capacity, rollout supervision, and iteration budgets for learned ablations. Give measured training and inference costs separately. A costlier method may still win on accuracy, but it has not established equal-cost superiority.
- All training must use CUDA. Record the device, GPU name, software versions, configuration, checkpoint iteration, and data provenance in each result.

`research_compare.py` preserves configurations and seed counts, compares new results with the best historical error, and keeps missing/nonfinite metrics visible. It deliberately does not declare a confirmed winner from exploratory results.

## Numerical review of the new candidate family

The ARS222, linearly implicit midpoint, and low-gamma ROS2 formulas satisfy the second-order conditions. The known-operator KS split removes positive linear growth from the implicit solve and adds exactly that growth to the explicit remainder, preserving the original semidiscrete equation. Unit tests check nonlinear order, nonidentity mass matrices, the KS split identity, zero-closure equivalence, mean preservation, and closure evaluation counts.

A correction evaluated once at the beginning of a macro step is a learned discrete defect. Holding that correction fixed through the internal stages generally reduces the complete learned method to first-order consistency as the macro step varies. Recomputing the correction at each stage restores second-order conditions but uses two network calls per internal step. These are different accuracy/cost strategies and should remain separate in comparisons.

The nonlinear physics remains explicit in these methods. Implicit treatment of the stiff linear operator does not remove the advective restriction of the complete method. Adding substeps can improve accuracy and nonlinear stability, but it does not establish unconditional stability.

The ETDRK4 control uses double precision internally while the implicit candidates primarily use single precision. A deterministic arithmetic check integrated 20 data intervals with eight internal ETDRK4 substeps, against 64-substep double precision on the same grid. It used coefficient mu_ref, unit normalization, and smooth prescribed fields (Burgers/KdV: `0.5 sin(2 pi x) + 0.2 cos(4 pi x)`; KS: `0.25 cos(8 pi x)`, with x the normalized grid coordinate).

| Problem | Single-precision state/weights error | Double-precision error |
|---|---:|---:|
| Burgers | 4.16e-6 | 1.76e-10 |
| KdV | 5.16e-7 | 1.05e-8 |
| KS | 5.95e-6 | 2.23e-6 |

This was a CPU numerical unit calculation, not training or holdout evaluation. It establishes that arithmetic can matter near the error floor; it does not estimate the final benchmark difference. Small ETDRK4 advantages near that floor cannot be attributed solely to the integrator family without matching precision.

Any improvement over the old FNO methods combines additional known nonlinear physics, more numerical stages, conservation, and a changed integrator. Compare the learned method against the identical zero-closure solver to isolate the contribution of learning. If the zero-closure solver wins, the supported recommendation is to retain the physical solver and learn only when validation demonstrates an additional benefit.
