# Numerical implementation status — updated 2026-11-04

**Newton convergence and the closed-core crossing repair are verified on the combined solver. Full spatial convergence and Gates A/B/C remain open.** The [completion sequence](science_completion_20261103.json) separates completed solver proofs from the remaining science checks.

## Qualified combined solver

[Qualification](cpu_combined_refined_qualification_20261103.json) binds one executable, frozen inputs, completed native probes, MUMPS diagnostics and an independent tight Picard comparison. Jobs 2771564/2771571 completed successfully.

| Probe | Newton iterations | Final raw update | Healthy time steps | Minimum q8 Jacobian (m²) |
|---|---:|---:|---:|---:|
| Original80, tight control | 10 | 8.948e-13 | 1 | 1.923e-08 |
| Original80, matched tolerance | 7 | 4.005e-11 | 3 | 1.923e-08 |
| Radial160 | 6 | 8.482e-11 | 3 | 4.791e-09 |
| Wall80 | 11 | 2.533e-11 | 3 | 1.923e-08 |
| Core-poloidal440 | 7 | 8.162e-11 | 3 | 9.592e-09 |

Every probe has finite fields, positive sampled density/temperature and positive signed q8 Jacobians. Only wall80 activates the stagnant-two-cycle fallback. The tight control preserves every field and gradient against corrected-mesh Picard; maximum relative gradient difference is **9.031e-8**, below **1e-6**, with pressure derivative multiplier exactly one. MUMPS takes at most two refinement steps; maximum reported backward error is **4.820e-16**.

The combined source adds **0022 and 0024–0029** to the previously validated capacity/MPI/neighbor fixes. Patch 0022 solves closed-core intersections against the original bicubic flux and geometry. Patch 0024 supplies the missing normalized-pressure span derivatives. Patches 0025–0027 continue valid nearby saddle roots only after a stagnant Newton two-cycle and synchronize the state across ranks. Patch 0028 repairs reversed interior wall tangent charts while preserving the original-control patches exactly. Patch 0029 enables bounded MUMPS iterative refinement. Production installation remains unchanged.

The bounds-only candidates 0019/0020 remain rejected. Broad 0025 without its cycle guard fails control gradient parity and remains rejected. Patch 0021 is diagnostic; 0023 only raises the tight Picard reference iteration cap. The separate fold in the fully doubled divertor-leg mesh remains unresolved. Root continuation is qualified for these fresh starts; restart-state persistence is not established.

![Qualified native raw-update convergence](cpu_combined_solver_convergence_20261103.png)

## Full-horizon spatial result

Both corrected radial80 and radial160 baseline/perturbed pairs completed **48 steps at 1.4059131112280784e-6 s**. The [radial comparison](cpu_exact_radial_convergence_20261103.json) covers **all 1,161,200 physical queries (100% of volume)**. All baseline fields and density/pressure integrals pass. Three final checks exceed the unchanged **1%** field/response target:

- Perturbed vorticity: **1.395%**.
- Paired current response: **1.392%**.
- Paired vorticity response: **3.016%**.

These are preserved legacy-linear-solver trajectories. The [radial320 probe](cpu_combined_radial320_20261104.json), job **2771635**, completed with the qualified combined executable: **8 Newton iterations, raw update 8.891e-11**, three healthy steps and minimum q8 Jacobian **1.196e-9 m²**.

The [legacy core-poloidal220/440 comparison](cpu_exact_core_poloidal_convergence_20261103.json) also completed. Baseline/perturbed parallel velocity differs by **4.871%/4.869%**; perturbed vorticity by **1.446%**; paired current/vorticity by **1.571%/3.044%**. Integrals pass, but **64 of 1,431,520** physical queries are uncovered. [Bounded inverse-map and error-localization diagnosis](cpu_spatial_error_diagnosis_20261104.json) recovered none; 63 are demonstrably outside the coarse boundary. Almost all parallel-velocity squared error is outside the core, while 99.75% of paired-vorticity squared error is in the inner core. No coverage or tolerance waiver is applied.

New matched **48-step baseline/perturbed trajectories** use the same qualified combined Newton/core/wall/MUMPS binary on control80, radial160, radial320, wall80 and core-poloidal440. Dependency-guarded radial, wall and core comparisons are submitted. Both control80 trajectories have completed healthy. Core-poloidal880 converges its equilibrium (7 iterations, raw update 2.406e-11) and has positive q8/q16 initial geometry, but loses density/temperature positivity at its first MHD step. Its failed run and dependent comparison were cancelled; the failure remains preserved. [One-step timestep diagnoses](cpu_poloidal880_timestep_diagnosis_20261104.json) show that halving and quartering dt do not cure it. The largest response lies near the straight outer wall. [Stronger MUMPS scaling](cpu_mumps_scaling8_qualification_20261104.json) rescues that first step at the unchanged dt: q8 minimum density 0.00928064 and temperature 9.89670e-5, backward error 1.108e-16. Both original baseline/perturbed controls retain field/gradient parity; maximum relative gradient difference is 4.990e-8. [Fresh poloidal440/880 and wall80 three-step probes plus MPI4/16 replay now pass](cpu_scaling8_refined_qualification_20261104.json). Poloidal880 final q8 minimum density is 0.00740781 and temperature 9.69836e-5. Matched 48-step stronger-scaling poloidal440/880 pairs run as jobs 2773383/2773385/2773387/2773389; comparison 2773393 depends on all four. These probes do not establish full spatial convergence.

Further anisotropic probes also fail: divertor160 has a q8 fold, wall160 folds only after joining otherwise positive patches, and exterior-flux doubling aborts on an extension attachment. All failures remain recorded. [MPI16 qualification](cpu_combined_mpi16_parity_20261104.json) passes both baseline/perturbed initial and final field/gradient parity (maximum relative gradient difference 2.377e-9). A capacity-only 240001-node executable also passes control parity at 16 ranks; its balanced twofold full-grid probe stopped at extension11 before equilibrium. [Attachment diagnosis](cpu_balanced_attachment_diagnosis_20261104.json) proves the rounded guide selects the adjacent non-corner after refinement. The guide-corrected probe passed that attachment but failed native final-mesh Jacobian checks at several other cells. Physical wall endpoints were unchanged; the failed mesh is retained. [Live campaign status](cpu_spatial_campaign_status_20261104.json) separates these probes from the ongoing full trajectories. Full spatial acceptance, refined timestep sensitivity and physical balance remain open.

The [qualified-binary full wall40/80 comparison](cpu_combined_wall_convergence_20261104.json) is complete: baseline/perturbed Vpar errors **1.006813%/1.006330%** narrowly exceed 1%; all paired responses and density/pressure integrals pass. **One of 1,289,984** fine physical queries remains outside the coarse domain, so coverage also fails. Stronger-scaling wall80 parity and wall120 health probes test an intermediate refinement because wall160 has a merged-patch fold. All matched radial160/radial320 and core-poloidal440 trajectories have completed healthy. The [combined-binary core220/440 comparison](cpu_combined_core_poloidal_convergence_20261104.json) confirms Vpar errors 4.8714%/4.8688%, perturbed vorticity 1.4460%, paired current/vorticity 1.5710%/3.0439%, and 64 uncovered queries.

The [poloidal440/880 geometry comparison](cpu_poloidal440_880_geometry_coverage_20261104.json) leaves **57 of 2,528,320** queries uncovered (0.000102647 m³). [Bounded inverse diagnosis](cpu_poloidal440_880_missing_diagnosis_20261104.json) confirms all 57 lie outside the coarse boundary; none were recovered. Coverage remains a separate rejection even if later field errors pass.

Wall120 also fails at the wall160 patch corner. New isolated patch **0031** limits only a shared interior-edge tangent handle at a folded three-cell boundary corner, accepting a trial only when all three adjacent cells have positive sampled Jacobians; failed trials roll back. Compiled tests on both saved failed meshes show positive independent q32 Jacobians, exactly unchanged physical boundary curves, shared-edge agreement, and no change to healthy wall80. A collapsed-geometry test verifies rollback. **245 tests pass** ([JUnit](cpu_xpoint_handle_suite_20261104.xml)). [Native baseline/perturbed/wall80 parity qualification](cpu_corner_handle_qualification_20261104.json) passed job **2773412**. Guarded wall120/160 probes **2773436/2773438** passed three healthy steps, strict Newton stopping and global q16 geometry checks. Wall160 takes 13 Newton iterations to raw update **1.675e-11**. The wall80 full pair **2773488/2773490** completed healthy. Wall160 full baseline/perturbed jobs **2774854/2774856** and dependent wall80/160 comparison **2774880** are running/pending.

A separate, isolated **0033** candidate limits ordinary interior handles adjacent to an X point. Five compiled tests remove the saved divertor160 fold with a 20% contraction while preserving healthy original/poloidal880/wall80 geometries and rolling back unrepairable trials. Native controls and divertor160 **2773463** all passed: strict reference parity, 12 Newton iterations to **9.306e-11**, three healthy steps and positive global q16 geometry. [Qualification](cpu_xpoint_handle_qualification_20261104.json) binds this separate binary. Matched full divertor80/160 pairs **2774866/2774868/2774874/2774876** feed comparison **2774883**. Capacity-only **0032** raises storage to 400001 nodes/elements and 8001 boundary elements for the next poloidal level; **2773441** passed original baseline/perturbed and poloidal880 parity and is constructing the poloidal1760 probe. [Capacity qualification](cpu_scaling8_capacity400_qualification_20261104.json) records those completed controls. Geometry comparison **2773445** depends on that qualification. These are experiments, not accepted convergence.

The [final radial160/320 comparison](cpu_combined_radial_convergence_20261104.json) covers all **1,719,280** fine physical queries. Every individual field and density/pressure integral passes. Paired current is **0.7224%**, but paired vorticity is **1.25495%**, above the frozen 1% target. The [three-level radial80/160/320 result](cpu_combined_radial_order_20261104.json) gives paired-vorticity order **1.265** and paired-current order **0.946**; some much smaller errors are nonmonotone, so global asymptotic convergence is not asserted. New same-capacity-binary radial320 parity and radial640 health probes are **2774872/2774878**. Their matched full comparison remains required.

The [full poloidal440/880 comparison](cpu_scaling8_poloidal_next_convergence_20261104.json) now completes with positive q8 fields through 48 steps. Baseline/perturbed parallel-velocity differences remain **2.76921%/2.76772%**. All other individual fields, all paired responses (maximum **0.80688%** vorticity) and integrals pass. The same **57 uncovered queries** reject full-domain acceptance. The matched capacity-binary poloidal880 reference pair runs as **2774908/2774910**, preparing the poloidal880/1760 comparison.

[Local cached-flux diagnosis](cpu_balanced_flux_diagnosis_20261104.json) finds reversed adjacent flux rows at seven of eight concave balanced-mesh cells; one cell has an unresolved local cache query and is not classified. Patch **0034** corrects only reversed adjacent intersections on an identical polar curve, requiring exact original-cell flux roots, polar-segment bounds and restored order, otherwise preserving both input points and rejecting the native attempt. Its five compiled checks plus five existing exact-crossing checks pass. Isolated native job **2775165** rejected this candidate on the original healthy control: central row 49 triggered the ordering guard inside the already-qualified closed core. That failed attempt is preserved. The open-only candidate **2775323** also failed its healthy control at endpoint row82/column1. The third candidate **2775359** requires an adjacent concave corner polygon and excludes endpoints and closed-core rows. Compiled guard checks detect all eight actual concave cells and none on healthy original/wall80/poloidal880 meshes. Native qualification remains pending. Patch **0036** independently caches distinct incident-element counts for boundary labeling; exact old/new boundary arrays agree on five compiled fixtures, including two saved balanced wall patches. Other curved-cell folds remain separate; this is not a convergence claim.

## Native labels and software checks

Native selected exports verify on all **55,135 physical elements** in both cases after identifying **976 byte-identical cross-rank halo copies**. Legacy balance and gradient-repeatability failures remain in [the original evidence](cpu_native_halo_validation_20261103.json). Bounded MUMPS refinement passes both independent interior balances (maximum absolute residual **8.65e-14**) and both independent field/gradient repeats; see [refined repeat validation](cpu_mumps_repeat_v2_20261103.json). Gradient parity against the inaccurate legacy solve still fails. These proofs do not cover full trajectories, boundary/axis equations or the autonomous numerical backend.

**259 software tests passed** ([JUnit](cpu_concave_crossing_suite_20261104.xml)). No training data or scientific gate is accepted. Spatial field/response tolerance remains **1%**, integral tolerance **2%**, and raw equilibrium tolerance **1e-10** or tighter, with full-domain coverage required.

## Earlier implementation and validation history

Gate B is **not passed**. No validated standalone all-numerical model-303 trajectory exists in this study. Gate A is also not passed. Both short refined baseline and perturbed JOREK trajectories reached 96 steps and 1.4059131112280784e-6 s. Enabled initialization is validated, but dynamic mesh convergence and full physical acceptance remain incomplete. The final restarts are readable and finite; historical runs have a diagnosed intermediate-equilibrium MPI output race and remain diagnostic only. See `cpu_short_pilot_validation_20261102.json`, `cpu_energy_diagnostic_audit_20261102.json`, and `cpu_io_continuation_20261102.json`. Production data generation and neural training remain blocked by these scientific dependencies.

Implemented and exercised on explicitly manufactured fixtures (final focused result: **13 tests passed**):

- Sparse local-basis evaluation of canonical value/derivative coefficients and axisymmetric scalar/flux weak assembly, including shared global DOFs and physical `2*pi*R` weighting.
- Source-derived local n=0 density/pressure advection/compression algebra; cylindrical and parallel compression fixtures.
- Same-state selected-plus-retained residual reconstruction with cancellation reporting.
- Differentiable torch weak assembly and finite-difference gradient check through a fixture state-dependent linear solve. This does not validate the missing JOREK temporal solver.
- HDF5 fixed-geometry hashing and native trajectory roundtrip, schema/positivity/horizon/FE identity/weak residual validation, exclusive output creation, explicit export CLI.

Focused validation command:

```
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /pitagora/home/userexternal/vgopakum/Conditional_Fusion_Operator/.venv/bin/python -m pytest -q tests/test_numerical.py tests/test_data.py
```

The exact native bridge, source entrypoints, pressure temporal coupling and boundary treatment are recorded in `equations.md`. Real exchange stays provisional; passing synthetic fixtures cannot approve a dataset release.

## Next required evidence

1. Isolated build instrumenting native quadrature/basis, the selected spatial blocks, total/remainder and temporal/constraint objects; unchanged equilibrium regression must pass.
2. Physically reviewed, boundary-consistent localized n=0 initializer with all dependent variables consistent. Existing model303 perturbation loop only affects harmonics `in=2..n_tor` and cannot seed the requested n=0 perturbation.
3. Baseline and perturbed same-state export checks, term strength and constraint/balance diagnostics, then mesh/timestep refinement.
4. Autonomous numerical evaluation at its own current q, matching baseline and perturbed JOREK within the measured discretization envelope, with differentiable state-dependent solves verified independently.
5. Only then approve the schema/label units, gate B, data release, and real neural training.

No production source, existing executable or dependency checkout was modified. No fake numerical evaluator was supplied: calling `require_native_backend()` raises a concrete missing-capability error.

## Independent integration review

The initializer candidate's nodal product-rule algebra is consistent with bicubic value/s/t/st DOFs. Its analytic shape's compact support does not guarantee that the resulting FE interpolant vanishes at every point before the separatrix: an element with perturbed nodal DOFs can extend across the analytic cutoff. Gate A therefore needs element/quadrature support-clearance, boundary and positivity checks on the actual initialized mesh, not only the standalone shape test.

A narrowly scoped optional quadrature/selected-block diagnostic export can be prepared without a compute campaign. It needs an explicit accepted-timestep export path, rank/element/DOF identities, and isolated-build regression. Such an exporter remains distinct from the full same-state spatial/temporal/constraint bridge. No small patch currently supplies the complete retained evaluator and autonomous state-dependent solve.

Coordinator integration review found two release/protocol weaknesses: accepted trajectories were not bound to the frozen protocol identity, and candidate sampling allowed overlapping train/OOD ranges or unknown split names. The coordinator corrected both with an explicit shared `frozen_protocol_hash`, split allowlist, finite ordered bounds and disjoint OOD envelope validation plus regression tests. Simulation review found missing runner cap enforcement and incomplete output-cadence type validation; those findings were referred to the simulation owner. No reviewer check passes Gate A, B or C.

Native restart diagnostics now verify and condense shared native DOF aliases. Coarse/fine equilibrium geometry, sampled positivity and printed-integral checks are recorded in `cpu_native_mesh_validation_20261102.json`. Neither this mapping nor the source initialization/grid fixes supplies the missing same-state spatial/temporal/constraint backend.

## Short refined pilot and diagnostic corrections

Paired final-minus-baseline perturbations lose 11.31% density RMS, 23.76% temperature RMS and 23.06% pressure RMS over the short horizon. All four initial/final states have positive sampled density and temperature. Baseline full-field density and temperature drift by 2.24% and 2.63%, respectively; background evolution remains part of the physical case.

Native printed particle-balance mismatch is about 0.4% by a component-scaled L2 diagnostic. The approximately 99% printed energy mismatch cannot approve or reject physical evolution by itself: `mod_integrals3D.f90` computes `ohmic_tot_t` from the thermal heating coefficient, which is zero here, while induction still dissipates magnetic energy. In addition, generic diagnostic source power applies the enabled Coulomb-log resistivity correction; legacy model303 element assembly does not. Independent endpoint estimates of the model303 resistive sink are 317.77 MW for baseline and 316.43 MW for the perturbed state. q4/q8 estimates differ by less than 5e-9 relatively. These estimates do not provide a complete source-consistent energy balance. No physics settings were changed to reduce the mismatch.

Source-derived interior current/vorticity weak relations now have a manufactured polynomial check. Native q8 final residuals, scaled by the sum of the two component norms, are 2.63e-5/4.25e-5 for baseline and 2.62e-5/2.80e-5 for perturbed evolution. Boundary and magnetic-axis rows are excluded; the native constraint map and autonomous solve remain unvalidated. See `cpu_constraint_diagnostics_20261102.json` and `cpu_constraint_diagnostics_tests.xml`.

Isolated patch `0004-equilibrium-restart-single-writer.patch` restricts the serial R/Z equilibrium restart to rank zero. Four-rank regression 2768291 has zero stderr and exactly matching fields, geometry, indices and boundary arrays. New runner classification rejects HDF5 errors even when the process returns zero. Sequential four-rank 48-step baseline/perturbed checks are recorded in `cpu_io_continuation_20261102.json`; they test the fine-mesh timestep sensitivity against the older 96-step diagnostics without accepting those older outputs as production references.

Final timestep evidence: both clean 48-step jobs completed. Full-field maxima against the older 96-step diagnostics are 0.2154% baseline and 0.1861% seeded; the largest paired-perturbation difference is 0.06627%. All short-horizon timestep targets pass. Coarse/fine baseline density/pressure integrals agree within 0.074%, but full-field and seeded mesh convergence remain unchecked. The final integrated suite passes 64 tests with no failures/skips. Gates A/B/C remain pending.

## Current energy and mesh evidence

Diagnostics-only patches 0005/0006 reproduce equilibrium and both 48-step trajectories with exactly unchanged physical fields. Corrected magnetic loss matches independent q8 integration within 5e-9 relatively. Full energy remains unclosed: native net L2 mismatch is about 1.53, even though component-scaled mismatch is about 1.58%. Source-derived viscous boundary work and a legacy/generic perpendicular-viscosity difference explain part of the gap; sampled remaining mismatch reaches 6.13 MW. Do not promote physical acceptance using the smaller normalized number.

Physical coarse/fine queries now measure final stream-function/vorticity/parallel-flow discrepancies of 19.56%/6.69%/16.63%. Density and temperature agree within 0.268%/0.132%. Fine-volume query coverage is 99.9509%; unmatched queries remain explicit. The proposed spatial field target therefore fails. Native boundary/axis modified rows, seeded refinement, complete work accounting and the autonomous numerical backend remain unfinished. The current software suite passes 73 tests. Evidence: `cpu_energy_balance_validation_20261102.json`, `cpu_viscous_work_audit_20261102.json`, `cpu_mesh_field_comparison_20261102.json`, and `validation_cpu_energy_continuation_20261102.json`.


## Legacy model303 temporal-mass audit — 2026-11-02

`cpu_temporal_mass_audit_20261102.json` uses dense exact-geometry four-rank snapshots and centered physical-time rates to separate generic diagnostic energy derivatives from selected legacy temporal coefficients. The positive low-density region still activates smooth `rho_1=0.01` mass correction: setting `rho_min_neg=-1e12` does not disable this correction. Nonroundoff corrected mass occupies 13.17–20.74% of full-wall quadrature volume over the sampled interval. The generic parallel kinetic energy also includes poloidal-field terms absent from the corresponding toroidal-only parallel temporal diagonal. Their partial power contributions, including pressure and perpendicular density-mass corrections, sum to 0.1163–0.5533 MW. These are centered finite-rate diagnostics, not an exact discrete-energy identity. Independent parallel kinetic energy agrees with native diagnostics to the frozen tolerance, and algebraic decompositions are checked. Remaining temporal/cross terms, stabilization and boundary work prevent complete energy closure; no physical parameters or acceptance thresholds changed.


## Refined cached-R/Z probe recovery — 2026-11-03

Job 2769623 completes the guarded three-step refined probe on 46,415 elements/47,067 nodes at 8.78695694517549e-8 s. The fault came from restart import reducing node storage to the saved 32,832-node extent before grid growth. Isolated patch 0013 preserves full grid capacity and the saved active count in both restart readers. Patches 0011/0012 also retain bounded MUMPS workspace retries and fatal-error checks. Ordinary and cached eight-step parity pass exactly; no MUMPS errors or retries occur in the corrected refined probe. All final restart datasets are readable/finite, and native q8 fields, positive density/temperature and positive signed Jacobians pass. The related suite passes 60 tests, including compiled before/after capacity and MUMPS failure-injection checks. See `validation_cpu_restart_capacity_recovery_20261103.json` and `cpu_mumps_crash_diagnosis_20261103.json`.

The three-step output remains diagnostic only. Full spatial/seeded convergence, energy closure, the autonomous numerical bridge and Gates A/B/C remain unvalidated. Production generation and scientific training remain disabled.


## Matched-horizon refined pair and spatial comparison — 2026-11-03

Both refined cases now complete **48 steps at 1.4059131112280784e-6 s**, matching clean existing baseline/perturbed references at dt=0.0625. Jobs 2769727/2769729 use the validated binary on 46,415 elements/47,067 nodes. Each final restart has 131 readable finite datasets, positive q8 density/temperature/Jacobians, no MUMPS retries and empty stderr. Enabled refined initialization preserves geometry, boundary coefficients and unselected fields; support remains away from the separatrix and boundary.

**Full spatial convergence does not meet the candidate 1% field target.** Common physical q4 queries cover 99.96447092610835% of refined volume, with missing volume reported explicitly. Baseline u/w/Vpar discrepancies are 10.787%/2.682%/10.444%; perturbed discrepancies are 2.650%/16.379%/10.438%. Full-field density/temperature/pressure differ by at most 0.172%. Their paired-response differences are 0.426%/0.161%/0.156%, but paired current/vorticity differ by 8.284%/35.214%. Independent q8 density/pressure integrals agree within 0.060202%, meeting the 2% integral target. Smooth thermodynamic fields and integrated moments therefore do not establish convergence of the full coupled state.

`cpu_spatial_convergence_20261103.json` and `validation_cpu_spatial_pair_20261103.json` bind the clean references, new outputs, frozen analysis, coverage, health evidence and thresholds. CSV/PNG/PDF artifacts are saved alongside them. This remains a two-resolution comparison at fixed timestep; refined timestep sensitivity, asymptotic order, native boundary/axis constraints, complete energy closure and the autonomous numerical backend remain unvalidated. Gates A/B/C, production generation and scientific training remain pending.


Update 2026-11-04: the full native subprocess OpsSplit rollout and five weak-output NO blocks are implemented. See [full_rollout.md](full_rollout.md) for the current implementation, actual validation evidence, and remaining gates. The legacy two-block experiment and in-process backend are separate. Scientific training has not started.
