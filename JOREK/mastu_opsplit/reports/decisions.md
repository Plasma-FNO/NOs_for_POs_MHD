# Decision log

## 2026-11-02 — Study placement and scope

Implement under `NOs_for_POs_MHD/JOREK/mastu_opsplit`, within the existing authorized workspace. The handoff's sibling project root was a suggestion and did not exist. Keep this independent of the transport toolkit's JAX runtime. Do not modify the dirty Conditional_Fusion_Operator checkout or existing production JOREK source/build. Store small prepared cases in ignored `runs/`; production bulk-data placement and retention remain unconfirmed.

## 2026-11-02 — Pressure equation and labels

Pinned model303's sixth row evolves pressure `rho*T` through a coupled temporal operator. The second neural block predicts the selected pressure advection/compression residual. It does not directly predict `T_t`. Native current/vorticity constraints and momentum temporal couplings remain numerical. See `equations.md` for exact source references. Strong-form labels are supported for the chosen undifferentiated-test-function blocks; other weak-form terms remain numerical. Their native toroidal normalization and boundary treatment still require verification.

## 2026-11-02 — No silent axisymmetric initialization

The upstream perturbation loop begins at harmonic two and applies no requested perturbation at `n_tor=1`. Prepare a separate disabled-by-default source patch with native derivative-DOF product rules and boundary taper. The existing executable is not declared compatible with that patch. Perturbed production remains disabled until isolated build/regression, physical admissibility, and refinement checks pass.

## 2026-11-02 — Numerical gate remains open

Ordinary restart files do not expose a same-state selected/full spatial residual or a standalone temporal/constraint solve. Existing JOREK element solve vectors mix dt-scaled spatial terms, temporal/history contributions, stabilization, and boundary treatment. Do not relabel these as RHS training targets. Manufactured FE tests and synthetic learning smoke prove plumbing only. The native bridge is an explicit remaining bead; real training is blocked until all-numerical autonomous agreement is demonstrated.

## 2026-11-02 — Resource and evaluation freeze

Slurm account membership and partition availability were queried read-only. Available relevant account: `fupb2_ukaea_ml`; `boost_fua_dbg` has a 30-minute maximum. A single unchanged-equilibrium regression is the first proposed request. This is separate from the unapproved 16-trajectory pilot ceiling and the unbudgeted 280-trajectory campaign. Requested-rank core-hours do not imply billable core-hours: record actual allocated TRES and node-hours.

Sampling ranges, accuracy thresholds, latent resolution, and 280 trajectories remain **candidates**, exactly as in the handoff. Physical centers, widths, valid envelope, and training horizon are unresolved. Do not generate final-test realizations or select numerical thresholds from their outcomes. Freeze these choices using pilot/validation evidence before final evaluation. Final deadlines depend on measured pilot/build cost; no unsupported calendar completion dates are assigned.

## 2026-11-02 — Isolated initializer build

The proposed initializer patch compiled and linked in a fresh pinned git-archive source copy under `/tmp`, with two compiler processes and an explicit 15-minute cap. Actual build wall time was 88.17 seconds. Production source and executable were untouched. `initializer_build.json` records source/archive/patch/recipe/environment/configuration and binary hashes. This supersedes the initial syntax-only build status; MPI and physical regression still require authorized compute.

Final input review also found that PyYAML treats exponent literals lacking an explicit exponent sign as strings. The proposed namelist therefore quoted three real parameters. Correct those literals, reject numeric-looking strings, preserve older snapshots and regenerate the reviewed equilibrium snapshot before any launch. No invalid namelist was executed.

## 2026-11-02 — Single regression approval and submission

The user explicitly approved one unchanged equilibrium regression with the reviewed account/partition and caps. Job2767445 was submitted once, without resubmission. Authorization-bound snapshot physical hashes match v3 exactly. The job is queued with QOSGrpNodeLimit and no estimated start; no alternate account/partition or quota bypass is attempted. The pilot/training budget remains unapproved. Scheduler result, native-field comparison and actual allocation accounting must precede any success claim.

## Approved queue move — 2026-11-02

The user explicitly approved moving the queued regression to FAUNA. Existing job 2767445 was updated in place to account `fupa2_fauna`, partition `dcgp_fua_dbg`, QoS `normal`; no second submission or attempt was created. The caps remain one node, one MPI rank/thread, 8 GB, ten minutes, one attempt. BOOST-only tmpfs GRES was cleared. The original configuration, submitted script and prepared physical snapshot remain preserved; effective scheduler overrides and approval are recorded in `equilibrium_queue_move_20261102.json`. Runtime success and physical regression acceptance remain unverified until outputs are checked.

## Two-partition regression approval — 2026-11-02

The user requested "Run it in both partitions", extending authorization to one unchanged regression per debug partition. The migrated job 2767445 was held and cancelled while still pending; accounting confirms zero allocated CPUs and zero runtime. Two fresh requests avoid carrying BOOST-only tmpfs resource metadata into CPU scheduling: job 2767719 on `fupb2_ukaea_ml/boost_fua_dbg` and job 2767720 on `fupa2_fauna/dcgp_fua_dbg`, both QoS `normal`. Each requests one node, one MPI rank/thread, 8 GB, ten minutes, and one attempt, with distinct immutable snapshots and output/log paths. Physical snapshot hashes match the original exactly. Both scripts pass shell syntax, runner dry-run and scheduler test-only checks. No pilot or training authorization is inferred. Full submission commands and current states are in `equilibrium_both_partitions_20261102.json`.

CPU regression acceptance: job 2767720 completed with exit 0:0 and 27 seconds of scheduler allocation. Rechecking the actual restart reproduces maximum `values` difference 8.432857492834955e-05 against tolerance 1e-4, with matching job/config/reference identities and finite restart diagnostics. Mark only equilibrium regression passed; retain all physical-evolution and numerical gates as pending. BOOST job 2767719 remains authorized and queued.

## CPU continuation and numerical checks — 2026-11-02

Honor the explicit user instruction to continue the study on CPU. Keep per-job caps, preserve failed attempts, and leave scientific training/production gated. Developments so far use one CPU per job; the bounded current batch consists of seven ten-minute requests, one twenty-minute timestep check and one thirty-minute mesh construction (two node-hours at all requested caps, excluding earlier regression requests). Actual accounting is recorded separately.

Do not accept the coarse timestep from successful process exit alone. The 24/48-step stream-function discrepancy is 1.507%; refine rather than relaxing the 1% target. Likewise, narrow localized initializer candidates are poorly represented on the coarse mesh, motivating a mesh check before physical acceptance. The source default gamma remains 5/3; no machine operating parameters, transport coefficients or wall geometry were changed. Native restart interpolation is a diagnostic component; it does not replace the missing residual, boundary/constraint and self-state temporal interfaces.

## Storage approval and refined-grid capacity — 2026-11-02

Automatic approval review initially rejected moving/deleting original raw outputs because destination retention/ownership was unverified and explicit relocation approval was absent. No rejected command executed. Read-only checks verified that the shared base is owned by the current user. A checksum manifest bounded the request to 1.57 GB of terminal-job output. The user then explicitly approved the verified move. All listed files were checked against the manifest before removing Home duplicates; symlinks retain the original repository paths. Retention remains unconfirmed and cannot be claimed for production data.

The refined-grid failure is a compile-time boundary capacity issue, not evidence against CPU execution. Build an isolated derivative with `n_boundary_max=4001`, preserve physical equations, then require coarse equilibrium regression and refined-grid validation. Preserve the failed 1001-capacity run. Do not silently reduce mesh resolution or change physics to pass a gate.

## 2026-11-02 — Preserve native geometry while fixing refined construction

The first refined failure was boundary array capacity; the next was independent snapping of exact shared patch anchors; the final RTree failure read copied axis flags before initialization. Isolated derivative builds raise capacity to 4001, propagate exact snapped aliases only to future attachment points, and populate the RTree after axis/geometry finalization. No geometry checks/assertions are disabled. The original production binary remains unchanged. All failed runs remain available. Final coarse restart fields and mesh arrays are exactly equal to the capacity-only baseline; refined construction succeeds with positive sampled Jacobians. This is not Gate A mesh convergence or Gate B native-solver validation. Immutable build reports and patch hashes bind each prepared derivative executable.

## 2026-11-02 — Bounded enabled initializer and short diagnostic evolution

A compiled-helper mesh screen selected the 5% density/temperature shape at (0.95,0) m with (0.08,0.16) m widths and (0.55,0.75) normalized-flux taper. Actual zero-step native outputs at one/four ranks agree exactly, preserve boundary and unselected-field coefficients, remain positive at q8, and keep affected-element control hulls below psi_n=0.809 with conservative flux-rounding intervals. One manually frozen short perturbed diagnostic is submitted after the refined baseline, with at most four ranks/16G/30min/one attempt. The standard case-preparation path remains closed to general perturbed campaigns. These controlled diagnostic snapshots and explicit provenance do not approve a dataset, an ensemble, physical Gate A, or neural training.

## 2026-11-02 — Single-writer output and physical diagnostic audit

The completed refined 96-step baseline and seeded trajectories have readable, finite final restarts but historical HDF5 errors. Source inspection identifies the serial intermediate R/Z equilibrium export being called by every MPI rank. Isolated patch 0004 guards that call with `my_id == 0`; regression 2768291 has zero stderr and exactly matching native fields, geometry, indices and boundary arrays. Keep historical raw outputs unchanged and diagnostic only. Future runner classification rejects HDF5 errors even when JOREK exits zero. No retry is enabled for an HDF5 failure.

Do not tune physical coefficients to hide the printed energy mismatch. The model303 induction row uses temperature-dependent resistivity even with `eta_ohmic=0`; generic native diagnostics then omit the magnetic resistive loss and apply a Coulomb-log source correction that legacy model303 does not use. Endpoint resistive-loss estimates and source hashes are recorded in `cpu_energy_diagnostic_audit_20261102.json`. They are explanatory diagnostics, not a complete energy-balance acceptance result. Preserve the pinned physical configuration until the source/sink and boundary-work accounting is reconciled.

Use the source-derived current/vorticity weak relations only on interior rows, excluding boundary and magnetic-axis DOFs. The new manufactured polynomial check verifies signs, radial weights and sensitivity to a wrong current. Small native interior residuals do not validate modified rows, the native elimination map or the missing autonomous temporal/constraint solve. Gates A/B/C and production/training remain pending.

## 2026-11-02 — Preserve negative spatial and energy evidence

Fix diagnostic laws without changing the physical model: patched magnetic source/loss columns agree with independent integration, and native evolving states remain exactly unchanged. Preserve the failed build and superseded unused snapshots. Total energy remains unclosed after the magnetic correction; source-derived viscous boundary work is an audit, not a verified replacement for native modified-row work.

Keep the original spatial acceptance target. Exact physical queries reveal 16–20% flow discrepancies despite sub-percent density/temperature agreement and small integrated-moment errors. Do not hide the flow errors, extrapolate unmatched points, or restrict the metric domain after seeing the result. The next bounded refinement probe should measure memory and runtime before a larger reference request. Current measured batch RSS is 8.787 GiB; full spatial doubling is unlikely to fit the present caps under the explicitly unvalidated scaling estimates. No enlarged compute request or production campaign is approved by these diagnostics.
