# Project progress and timeline

Updated 3 November 2026. Newton and radial160 equilibrium convergence are now demonstrated; full spatial convergence remains open. See [current numerical status](numerical_status.md) and [completion sequence](science_completion_20261103.json). **The research study is not complete.** Tested infrastructure and source-audited provisional contracts are delivered. Gates A, B and C remain pending. No new scientific dataset, checkpoint, accuracy result or speedup claim exists.

## Completed implementation tasks

| Bead | Delivered result | Evidence |
|---|---|---|
| `mastu-sim` | Immutable pinned snapshots, bounded runner, explicit resource caps, abort/completeness/regression checks, reviewed request and candidate initializer patch | `simulation_audit.json`, `case_contract.md`, sixteen targeted tests, five isolated Fortran syntax checks and successful full isolated build |
| `mastu-data` | Native value/derivative FE evaluation, weak assembly, selected source algebra, differentiable assembly, provisional HDF5 exchange | `equations.md`, `data_contract.md`, `numerical_status.md`, thirteen targeted tests |
| `mastu-learn` | Three neural family builders, current fields/FiLM, pinned imported source hashes, direct term loading/loss, synthetic training and exact checkpoint resume | `learning_status.md`, eleven local tests, 23 selected upstream tests |
| `mastu-eval` | Candidate LHS parameters, family-split and OOD-envelope checks, training-only scaling, weighted errors, failure accounting, family bootstrap, protocol-bound release manifests | Eight evaluation tests and two manifest tests; full-suite validation below |

Infrastructure tasks are narrower than scientific milestones. Closing these tasks does not close M1–M6 or approve native label conventions.

## Scientific timeline and dependencies

| Milestone | Actual status | Next required result |
|---|---|---|
| M0 contracts/preflight | In progress | Finalize native normalization/constraint/boundary interface after the isolated bridge is validated; existing executable, equilibrium and spline provenance are recorded |
| M1 physical pilot / Gate A | CPU continuation authorized; both equilibrium regressions and short baseline timestep target pass | Refined evolution, enabled perturbation, useful horizon, advection/positivity/balance and mesh checks |
| M2 native numerical assembly / Gate B | Open substantive implementation | Exact current-state spatial exporter, retained evaluator, temporal/constraint solve, and autonomous agreement on baseline and perturbed dynamics |
| M3 versioned dataset / Gate C | Depends on M1/M2 and measured budget | Ratified envelope, family-safe final splits, physical horizon, acceptance and failure manifest; bulk retention confirmed |
| M4 scientific training | Depends on M3 and all numerical gates | Three primary neural families × three seeds, matched budgets, direct labels and autonomous rollout fine-tuning |
| M5 independent evaluation | Depends on frozen M4 | ID/OOD/time accuracy, failures, balances, bootstrap uncertainty and end-to-end cost |
| M6 final scientific handoff | Depends on M5 | Fresh real end-to-end reproduction, dataset/checkpoint locations, valid-domain and reference-uncertainty report |

Calendar deadlines are not assigned without measured native-build/pilot costs. The remaining critical path is physical pilot/refinement and native bridge → all-numerical agreement → frozen release → scientific training → independent evaluation. CPU development is authorized; ensemble budgets and production retention remain unfrozen.

The user approved exactly one ten-minute equilibrium regression on `fupb2_ukaea_ml/boost_fua_dbg`, one node, one MPI rank/thread, 8G, one attempt. Job **2767445** is submitted and currently pending with `QOSGrpNodeLimit`; Slurm provides no estimated start. The debug partition QoS has a two-node group limit. `reports/equilibrium_authorized.sbatch` and `configs/equilibrium_authorized.yaml` record the authorized request. A fresh immutable authorization-bound snapshot has physical hashes identical to v3. Earlier proposed configs/snapshots remain preserved. `equilibrium_regression_status.json` records live status. The pilot and 280-trajectory production campaign remain unauthorized and unmeasured.

## Independent verification

Final integrated suite: **50 passed in 17.66 s**. Command from study root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python -m pytest -q
```

Upstream relevant 2D spline/basis tests: **23 passed, 57 deselected in 55.96 s**. Exact selection and revision are in `learning_status.md`. Native patch: all five modified Fortran files passed isolated syntax checks, and its standalone mixed-derivative helper matched finite differences. A full isolated JOREK build subsequently compiled and linked in 88.17 seconds with at most two compiler processes: 272 objects, 141 modules, a hashed model303/n_tor=1 executable. `initializer_build.json` records the exact archive, patch, recipe, environment and binary hashes. No MPI/physical test has run.

Synthetic twelve-step CLI training reduced loss from `0.21913614869117737` to `0.0796082615852356`. The local fixture at `runs/synthetic_smoke_20261102` includes checkpoint/provenance and is excluded from Git. Separate tests verify both spline axes, FiLM/state/FNO gradients and exact resumed final weights. These are analytic fixtures, not plasma results.

Review found and fixed protocol hashes missing from release checks, potentially overlapping OOD envelopes, nonfinite metadata leaving partial HDF5 outputs, incomplete runner caps, and missing cleanup margin. The remaining FE-support caveat is explicit: nodal taper zeros cannot prove clearance or positivity within support-straddling elements.

Final checks and source fingerprints are in `validation_final.json`, `preflight_final.json` and `environment_packages_final.json`; earlier `validation.json`/`preflight.json` preserve the initial checks. `resource_ledger.json` records proposed/actual cost. The original dirty spline checkout, production source and executable were preserved. No tracker or Git remote was pushed.

## Initial paper-summary delivery (superseded)

`mastu-paper-summary` delivered `Paper/results_one_page/results_one_page.pdf`, a standalone one-page summary of published arXiv:2602.23113v2 Tables1–3 and selected operator-transfer findings. It matches the supplied paper layout; editable LaTeX, numeric plot data and rebuild instructions are alongside it. Five focused checks passed, including a second-page TeX hard error, an independent post-render page-count rejection and source protection. Overflow never replaces the previous accepted PDF. This is a summary of original-paper results, not new MAST-U evaluation evidence or a GateA/B/C result. Job2767445 was also rechecked read-only and remains PENDING/QOSGrpNodeLimit.

## Current JOREK one-page record

At the user's clarification, the same one-page artifact now covers the active MAST-U/model303 experiment. Published Navier–Stokes benchmark scores were removed. It shows pinned upstream equilibrium nodes (explicitly not a new run), the proposed native FE/spline/weak-assembly update, verified fixture/build evidence, queued reference status and an unmeasured comparison table. Data and evidence hashes are recorded by the build; both one-page guards remain active. This supersedes the initial published-paper summary, without claiming physical accuracy or passing any scientific gate.

Final JOREK-page checks: exactly one US Letter page at10pt; seven focused checks passed in2.30s, including evidence integrity, original/source protection, both overflow safeguards and a regression check excluding the old published benchmark content. Evidence: Paper/results_one_page/validation.json.

## One-page MHD equations and neural replacement

The current experiment page now includes four numbered blocks: selected reduced-MHD density/pressure/induction/constraint balances; exact candidate axisymmetric advection/compression formulas; the two independent L-SplineFNO substitutions; and native weak assembly with coupled temporal/constraint treatment. Other magnetic, momentum, transport, boundary and stabilization contributions remain numerical. Pressure rho*T is explicitly distinguished from a temperature tendency. Physics source/audit hashes are included in build provenance. Both compact bracket targets and physical divergence/advection forms matched the source-derived implementation across128 randomized states, maximum absolute error7.11e-15. This is algebraic checking, not physical-rollout validation. The final PDF is exactly one page; seven focused checks passed in2.58s.

## Current regression requests — two debug partitions, 2026-11-02

This supersedes the single-job queue statements above. After explicit approval to run in both partitions, job 2767445 was cancelled before allocation. Job 2767719 is submitted on UKAEA BOOST debug, and job 2767720 on FAUNA CPU debug. Each has the same one-node, one-rank/thread, 8 GB, ten-minute, one-attempt cap; input/executable/reference hashes match, and outputs are isolated. Live recorded states and both submission records are linked from `equilibrium_regression_status.json`. Neither submission alone establishes regression success or passes Gate A.

CPU-partition result: job **2767720 completed successfully** in 27 seconds of allocated wall time (25.697 seconds inside the runner), exit 0:0. The restart `values` comparison against the pinned upstream reference passed: maximum absolute difference **8.432857492834955e-05**, tolerance **1e-4**. Independent re-reading of the restart reproduces the comparison and confirms n_tor=1, zero evolution steps and finite diagnostics. BOOST job 2767719 remains pending under QOSGrpNodeLimit. This is an equilibrium-only regression; baseline/perturbed time evolution and Gates A/B/C remain pending.

## CPU study continuation — 2026-11-02

The user explicitly requested continuing the full study on CPU. Current compute uses `fupa2_fauna/dcgp_fua_dbg`; bounded development requests are recorded in `cpu_continuation_20261102.json`. This supersedes earlier statements that no time evolution has run. It does not establish a production budget or pass any scientific gate.

Completed CPU jobs include the three-, six-, twelve- and forty-eight-step unperturbed baselines at one common 1.4059131112280784e-6 s horizon and the default-amplitude isolated initializer regression. The initial 24-step request hit the known pre-step RTree assertion; the preserved one-time replacement succeeded. Default initializer fields match the original exactly. On the 24/48-step comparison, stream-function error remains 1.507%, above the 1% target; a 96-step comparison is therefore running. A separate construction doubles mesh-resolution counts while preserving physical profiles, topology and wall control points. Source/flux balances, useful perturbation dynamics, mesh convergence and Gate A remain incomplete.

Added native restart diagnostics with signed bicubic scaling and cylindrical volume weighting. Basis values/derivatives match compiled pinned Fortran; real density/pressure integrals reproduce the native printed values to rounding. Unknown patch IDs and missing native physical constraints stay explicitly diagnostic; native shared DOF aliases are now checked and condensed. These are not the autonomous Gate B backend. The shared namelist parser now handles whitespace inside array indices and rejects equivalent duplicate assignments; this was required by the existing wall-mesh input formatting.

CPU timestep update: 48 versus 96 steps now gives a maximum full-field difference of **0.7632%**, meeting the short-horizon 1% target. This does not qualify the perturbation, spatial mesh or useful physical horizon. The doubled-resolution equilibrium construction failed explicitly because `n_boundary_max=1001` is too small. A separate pinned source build raises only this capacity to 4001; production source/executable remain protected.

The integrated suite now passes **59 tests**. Its separate XML/JSON reports preserve the earlier 50-test record. The new namelist test rejects duplicate array assignments despite index whitespace; restart diagnostics have analytic field/gradient/volume tests, malformed input checks and compiled Fortran parity.

Home storage reached 189 MB free. After explicit user approval, **1,574,892,504 bytes** of completed raw outputs were copied to shared work, every file checksum was verified, and only verified Home duplicates were replaced with symlinks. Both failures remain preserved. Repository paths and raw-file bytes are unchanged. Shared-storage retention is still unconfirmed for a production release; transfer request and completion manifests are in `reports/cpu_storage_transfer*_20261102.json`.

## Current CPU evidence after mesh fixes

Both original partition regressions completed successfully: BOOST 2767719 and CPU 2767720 reproduce the pinned reference within 1e-4 (maximum 8.432857492834955e-5). Historical pending/request paragraphs above describe earlier states.

The isolated mesh development resolves three distinct failures without modifying production: boundary capacity 1001→4001, consistent reuse of already-snapped shared patch attachments, and initialization of axis flags before RTree construction. All geometry guards and RTree assertions remain enabled. Capacity-only versus final coarse arrays (`x`, `values`, `vertex`, `size`, `boundary`, `constrained`, `n_sons`) are exactly equal. The refined equilibrium (2768117) builds all eleven wall patches and completes. At q8, all sampled Jacobians are positive, density/temperature are positive, and density/pressure integrals reproduce native output rounding. The mesh has 11,257 elements, 11,581 nodes and 46,203 canonical DOFs; this is mesh construction evidence, not dynamic mesh convergence.

Four-rank refined equilibrium equivalence (2768142, 10 minutes maximum) and dependent 96-step refined unperturbed evolution (2768143, 30 minutes maximum) are submitted, each with one attempt and 16G. Both may use either FAUNA CPU partition; no duplicate execution is requested. Their outputs go directly to the approved work location.

Current integrated validation: **62 passed, zero skipped/failed**, 17.676 seconds (`cpu_continuation_native_tests.xml`). The software suite includes compiled native basis and shared-attachment parity, canonical alias validation, malformed restart rejection and earlier learning/evaluation fixtures. Gates A/B/C remain pending.

The four-rank refined equilibrium (2768142) passed with **zero native-field difference** from one rank. Refined baseline evolution (2768143) is running. Preliminary compiled-helper screening selected a 5% density/temperature bump centered at (R,Z)=(0.95,0) m with widths (0.08,0.16) m and psi_n taper (0.55,0.75). Its fine-mesh perturbation interpolation errors are 0.265%/0.183% versus 2.47%/2.81% on the coarse mesh. Affected-element control hulls remain below psi_n≈0.809 and avoid boundary vertices. The screen uses rounded native flux values and is not enabled-runtime acceptance. Separate one-/four-rank zero-step initialized cases (2768146/2768147) will test actual fields, boundary preservation, positivity and MPI exchange; the general perturbed-campaign preparation path remains disabled.

The regenerated one-page report passes all seven protection/content checks and remains one page. `validation_cpu_continuation_20261102.json` records the distinct 62-test study and 7-test report suites.

Actual enabled initialization completed at one rank (2768146) and four ranks (2768147), 88 seconds each. Fields match exactly across ranks. All boundary coefficients and unselected fields are exactly unchanged; sampled density and temperature are positive. The affected 560 elements have a conservative psi_n control-hull maximum 0.808843, including stdout-rounding uncertainty, and touch no boundary vertices. Maximum nodal relative bumps are 4.86196% for a requested 5% analytic peak. `cpu_enabled_initializer_validation_20261102.json` binds native outputs, diagnostics and parameters. This validates initialization only; balances and independent solver constraints remain unchecked.

The next single short perturbed evolution, **2768153**, is queued with `afterok:2768143`, four MPI ranks, 16G, 30-minute maximum, one attempt, 96 steps at dt=0.03125 and the same 1.4059131112280784e-6 s horizon. Its manually frozen diagnostic snapshot is bound to the successful enabled-initializer report. General campaign preparation, production generation and scientific training remain disabled.

## Completed output correction and paired timestep checks — 2026-11-02

This supersedes earlier running/queued statements. Refined 96-step baseline 2768143 and seeded trajectory 2768153 both completed at 1.4059131112280784e-6 s. Every final restart dataset is readable and finite, and sampled density/temperature are positive. Paired density/temperature/pressure perturbation RMS decreases by 11.31%/23.76%/23.06%. Historical HDF5 errors are preserved and keep those trajectories diagnostic only.

Source inspection found all MPI ranks writing the same serial intermediate R/Z equilibrium restart. Isolated build 2768289 applies only the rank-zero guard in patch 0004. Four-rank regression 2768291 completed in 92 seconds with zero stderr and exactly matching fields, geometry, indices and boundary arrays. Runner tests now reject HDF5 errors despite a zero process exit status.

Clean 48-step baseline 2768294 and seeded evolution 2768296 completed sequentially in 457/458 seconds. Both reached the requested horizon, have zero stderr and 131 readable finite restart datasets. Against the historical 96-step diagnostics, maximum full-field differences are 0.2154% baseline and 0.1861% seeded. Baseline-subtracted density/temperature/pressure perturbation differences are 0.00431%/0.06627%/0.00713%. All meet the proposed short-horizon timestep targets. Same-time coarse/fine unperturbed density/pressure integrals differ by 0.06435%/0.07347%; full-field and seeded mesh convergence remain unchecked. See `cpu_fine_timestep_validation_20261102.json`, `cpu_dynamic_mesh_moments_20261102.json` and `cpu_io_continuation_20261102.json`.

The energy audit identified two native diagnostic inconsistencies: missing resistive magnetic loss when the thermal Ohmic-heating coefficient is zero, and generic source-power Coulomb-log scaling absent from legacy model303 evolution. Independent endpoint resistive-loss estimates are 317.77/316.43 MW; they do not close the complete balance. Source-derived interior current/vorticity weak blocks pass a manufactured polynomial test and have small native final residuals. Boundary/axis modified rows and native elimination remain unvalidated. These results do not pass Gate A or B.

Integrated software validation: 64 passed, zero failures/skips in 15.599 seconds. The updated one-page report retains all seven protection/content checks. This continuation used 1.12944 allocated CPU core-hours, including the build and three simulation jobs; the ledger now records 3.77056 allocated CPU core-hours across known development/regression requests. Production generation and scientific training remain disabled. Next scientific dependencies are source-consistent energy and boundary/axis checks, full spatial/perturbation convergence and term-strength ratios, followed by the autonomous native spatial/temporal/constraint bridge. No Gate A/B/C status was promoted.

## Energy diagnostic correction and spatial field comparison — 2026-11-02

This supersedes the previous energy-law and unmeasured full-field mesh statements. Isolated patches 0005/0006 make magnetic source/loss diagnostics follow legacy model303 resistivity and use one loss coefficient for total/inside/outside integrals. Thermal Ohmic heating remains disabled; no physical settings or evolution equations changed. The first build failed on a diagnostic variable name; its exact patch/source and 28 allocated CPU seconds are preserved. Corrected builds 2768312 and 2768317 passed. Equilibrium regressions 2768314/2768318 and sequential 48-step dynamic regressions 2768319/2768320 all completed with zero stderr. Both evolving cases match the previous clean physical fields, mesh, times and magnetic/kinetic energies exactly. All ten compared intermediate snapshots also match exactly.

Native final magnetic resistive losses are 317.742 MW baseline and 316.388 MW seeded, agreeing with independent q8 integration to less than 5e-9 relatively. This fixes the magnetic diagnostic inconsistency, not total energy conservation. Total energy net L2 mismatch remains 1.540/1.522, while normalization by summed component norms gives 1.578%/1.588%. These metrics must not be confused. Inferred exterior quadrature has 528 labelled edges, no unlabelled edges and 108 verified zero-measure axis edges; its manufactured cylindrical divergence check passes. Reconstructed particle boundary fluxes agree with native diagnostics within 5.9e-11 of the largest flux component. Source-derived parallel viscous boundary work contributes about 4.49–5.98 MW on the saved states, and model303 perpendicular viscous loss differs from the generic diagnostic formula. Including these contributions still leaves up to 6.13 MW mismatch on the sampled states. Native modified boundary rows, remaining temporal/stabilization work and full balance closure remain unvalidated. See `cpu_energy_balance_validation_20261102.json` and `cpu_viscous_work_audit_20261102.json`.

Exact curved-element queries now compare physical coarse/fine fields without nearest-node interpolation. Coverage is 99.9509% of fine q4 volume; unmatched points stay explicit. At the common final time, density/temperature/pressure errors are 0.268%/0.132%/0.140%, but stream function/vorticity/parallel-flow differences are 19.56%/6.69%/16.63%. Spatial convergence therefore does not pass the proposed 1% field target. Real native self-queries have maximum absolute field discrepancy below 2e-9. Tests cover curved geometry, shared-edge queries, explicit exterior points and an elongated element lying beyond the nearest 32 centers. See `cpu_mesh_field_comparison_20261102.json`.

The final integrated suite passes 73 tests, with zero failures/skips, in 15.996 seconds. This continuation used 1.23667 allocated CPU core-hours, including the preserved failed build; known development/regression accounting totals 5.00722 allocated CPU core-hours. Reported batch peak RSS reaches 8.787 GiB for the current mesh. `cpu_spatial_refinement_cost_20261102.json` records measured costs and explicitly unvalidated scaling heuristics; the next proposed cost probe increases linear resolution by roughly 10% under the existing four-rank, 16-GiB, 30-minute cap. No larger simulation was submitted. Gates A/B/C, production generation and scientific training remain pending.


## Budget-backed CPU refinement — 2026-11-02

The user authorized raising caps when CPU credit permits. `saldo -b --dcgp` verifies FUPA2_FAUNA CPU credit: 26,481,874 local resource-hours remaining overall and 2,531,431 monthly at the recorded snapshot. These shared-account units include requested memory and differ from rank CPU-hours. The frozen `cpu_budget_authorization_20261102.json` permits one node, at most 16 MPI ranks, one thread, 128 GiB, two hours and one attempt per job; the phase envelope is 128 requested CPU-hours, eight requested node-hours, six simulation allocations, concurrency one. Existing untagged requests retain their old caps. Production and scientific training remain disabled.

Capacity-only builds extend intermediate grid storage to 160,001 nodes/elements. The 16-rank eight-step parity run completed native evolution but failed both its original coefficient acceptance and the subsequently frozen physical-gradient protocol (relative gradient error 9.95e-6 versus 1e-6 tolerance); its failure remains preserved. Four-rank parity matches reference fields and gradients exactly, so refinement retains four ranks. A modest probe failed in lower-leg construction; the nested 2x mesh probe failed before evolution, using about 4 GiB, in contour construction. Bounds-check reproduction 2768423 reuses its saved R/Z equilibrium and identifies a write to `R_outer_contour(10001)` above the fixed 10,000-point capacity. Patch 0009 raises only contour capacity to 40,000 in an isolated build, retaining contour bounds checks. Its guarded job verifies exact eight-step reference arrays before the cached-R/Z nested three-step probe. Live jobs, failures and reservations are in `cpu_refinement_continuation_20261102.json`; unique allocation accounting is in `cpu_budget_refinement_usage_20261102.json`. A submission is not a successful refined reference or spatial-convergence result.

The integrated software suite passes 81 tests (`cpu_budget_refinement_tests.xml`). No scientific gate has passed.

Guarded contour-capacity fix job **2768425** is active. Its exact old-mesh regression must pass before the nested probe starts. The rebuilt experiment PDF remains exactly one page, with seven report protection/content checks passed in 2.52 seconds.

Contour-capacity parity update: job 2768425 completed the guarded eight-step old-mesh run in 157.4 process seconds. All nine checked native reference arrays, including fields, geometry, times and energies, match exactly. The cached-R/Z nested three-step probe has started. This is capacity-regression evidence; no spatial convergence or scientific gate is claimed.

The doubled-R/Z probe in job 2768425 stopped cleanly at the native strategic-point guard: six strikes instead of the required four. The guard remains enabled. A distinct final phase allocation, **2768438**, reuses the validated existing R/Z equilibrium and doubles only final flux-aligned resolution counts, holding physical inputs fixed. This isolates mesh refinement from changed preliminary equilibrium discretization. It uses four ranks, 128 GiB, two hours maximum, one attempt; all six permitted phase allocations are now submitted. Total requested reservations including builds are 28.667 CPU-hours and 6.333 node-hours, below the 128/eight-hour envelope. No full-length refined reference is accepted.


## Cached-R/Z boundary startup diagnosis

Job 2768438 failed after 310 allocated seconds, after all eleven patches and a successful native mesh check (46,415 elements; 47,067 nodes), before evolution. Both input boundary switches are false. The cached-R/Z path skips the only initialization of local `freeb_equil2`, then tests that uninitialized logical before invoking equilibrium vacuum-response loading. Patch 0010 initializes the saved flag from the requested `freeboundary_equil` before either startup branch; ordinary initialization and the guard are preserved. No response file or physical-boundary change is proposed. This patch is prepared and needs compilation, unchanged ordinary-start parity, cached-start parity and then another bounded refined probe. The original six-allocation phase is exhausted; any next simulation must be recorded under a fresh verified budget authorization rather than bypassing the existing submission ceiling. Evidence: `cpu_cached_rz_boundary_diagnosis_20261103.json`.


## Explicitly authorized cached-R/Z recovery allocation

The user explicitly approved the next job. Fresh `saldo -b --dcgp -n` verification still shows 26,481,874 shared CPU-project resource-hours remaining overall and 2,531,431 monthly. A separate one-job authorization preserves the exhausted earlier phase unchanged. Job **2768489** is submitted on `fupa2_fauna/dcgp_fua_prod`: one node, four MPI ranks, one thread, 128 GiB, two hours maximum, at most two compiler processes and one attempt per stage. Maximum reservation is eight rank CPU-hours/two node-hours; conservative full-node credit envelope is 512 local resource-hours.

The isolated source copy applies patch 0010. Its single guarded sequence builds the executable, verifies exact ordinary eight-step reference arrays, verifies cached eight-step fields/gradients under the frozen q8 parity protocol with exact geometry/time, and runs the refined three-step probe only if both pass. A fixed-boundary STARWALL load, solver/HDF5 error, incomplete final time, or invalid sampled refined fields fails the job. Startup files, worker, patch, protocol, budget and submitted script are hash-bound. Neither production nor scientific training is enabled. Preparation and execution scripts passed syntax checking; native validation awaits the allocation. Live record: `cpu_cached_rz_recovery_20261103.json`.


## Refined three-step recovery completed — 2026-11-03

This supersedes earlier pending cached-R/Z recovery statements. Initial job 2768489 encountered MUMPS workspace errors, ignored those errors, and crashed in Poisson. Patch 0011 adds bounded workspace recovery and checks all MUMPS initialization/analysis/factorization/solve returns, aborting the world communicator before an invalid solution can be used. Job 2769482 preserved exact ordinary/cached eight-step parity and demonstrated clean failure after the initial workspace bound was exhausted. Patch 0012 extends that bounded ladder; job 2769505 exposed the same Poisson delta-write crash without any MUMPS error.

The actual memory-corruption cause is restart capacity. Both binary and HDF5 restart readers allocated only the saved active node count. The cached R/Z file has 32,832 nodes, but the refined mesh needs 47,067. Grid finalization writes beyond the imported node array; the later `node_list%node(i)%deltas` write faults. The original and later crashes have the same faulting instruction. Patch 0013 reserves `max(n_nodes_max, saved_node_count)` in both readers and restores the active saved count before reading data. It preserves saved fields and the grid's allocation contract. Twelve compiled before/after tests reproduce the original bounds failure and verify both corrected readers, initialized delta/value storage, active counts and deep-copy behavior. The final related suite passes 60 tests.

Guarded job **2769623 completed successfully** in 811 allocated seconds (0.901111 rank CPU-hours), with four ranks and the unchanged 128-GiB/two-hour cap. The isolated executable passes exact ordinary eight-step reference arrays and cached-start field/gradient parity with zero discrepancy. The refined probe completes exactly three steps at **8.78695694517549e-8 s** on **46,415 elements and 47,067 nodes**. Seven Poisson calls complete, with zero MUMPS errors, zero workspace retries and empty refined stderr. Every final restart dataset is readable and finite. Native q8 checks find positive Jacobians (minimum 7.754036608537834e-8 m²), density (minimum 0.008866882220878397 code units) and temperature (minimum 9.742912025167962e-5 code units). All three inputs are byte-identical to the preceding attempt. Batch peak RSS is 38,530,264 KiB.

Final evidence: `validation_cpu_restart_capacity_recovery_20261103.json`, `cpu_mumps_crash_diagnosis_20261103.json`, `cpu_restart_capacity_recovery_20261103.json` and `cpu_restart_capacity_recovery_tests_20261103.xml`. The resource ledger counts each failed/successful allocation once; known development/regression accounting now totals 9.827777777777778 rank CPU-hours. Original failed outputs remain preserved. This is a short diagnostic probe, not spatial-convergence or complete physical-acceptance evidence. Gates A/B/C, production generation and scientific training remain pending; production source was not modified.


## Matched-horizon refined baseline/perturbed pair — 2026-11-03

The user requested the refined pair and a spatial-convergence check. Fresh CPU credit was verified and recorded in `cpu_spatial_pair_authorization_20261103.json`. Baseline job 2769727 and perturbed job 2769729 run concurrently, each with four MPI ranks, one thread, 128 GiB, two hours maximum and one attempt. The validated patch-0013 binary is snapshotted unchanged. Both run 48 steps at dt=0.0625 to 1.4059131112280784e-6 s, matching clean existing jobs 2768319/2768320. The pair differs only in the two 0/5% perturbation amplitudes; center, widths, taper, physical inputs and geometry controls are frozen.

Actual refined initial states pass unchanged geometry/unselected-field/boundary checks. The perturbation affects 2,210 elements, touches no boundary element, and has conservative support-hull psi_n maximum 0.8095523438469315. Maximum nodal density/temperature bumps are 4.949009681769212% for the same requested 5% analytic peak. Initializer/runner preflight has 21 passing tests. The comparison suite has 22 passing tests, plus a separate successful end-to-end report test with a known 10% full-field/response mismatch.

Dependent analysis job 2769740 starts only after both validated runs succeed. It compares initial/final baseline, perturbed and baseline-subtracted response fields at common refined q4 physical-volume points, reusing inverse maps only after exact geometry checks. Pressure is formed as rho*T before baseline subtraction. Full-domain density/pressure moments use independent q8 integration. Existing candidate limits remain 1% for fields and 2% for integrals. Coverage gaps, zero-reference channels and fixed-timestep/two-resolution limitations stay explicit. No scientific gate is passed by submission. Live run/accounting record: `cpu_spatial_pair_20261103.json`; planned comparison artifact: `cpu_spatial_convergence_20261103.json`.


### Matched-horizon pair completed; spatial targets fail

Baseline **2769727** and perturbed **2769729** both complete 48 steps at **1.4059131112280784e-6 s**, using dt=0.0625 on the 46,415-element/47,067-node mesh. Scheduler allocations take 2,603 and 2,426 seconds. Each final restart has 131 readable finite datasets; q8 samples have positive density, temperature and signed Jacobians. Final minimum density is 0.0029677505170918397 baseline and 0.002940291405720145 perturbed; final minimum temperature is 8.861544437781217e-5 and 8.862203621443264e-5, in code units. Both runs have empty stderr, zero MUMPS errors/retries, and exact initial geometry/unselected-field/boundary checks.

The first analysis launcher, 2769740, fails after two allocated seconds because a relative script path resolves against the batch directory. Its outputs/accounting remain preserved. Corrected absolute-path analysis **2769889** completes in 76 seconds using the unchanged comparison source and thresholds. A separate replacement-analysis authorization records the one additional bounded analysis allocation; no additional simulation runs.

Clean 11,257-element references are compared with the refined pair at 742,640 refined q4 physical queries. The inverse map covers 741,859 points and **99.96447092610835%** of refined physical volume. Unmatched volume is 0.007694674994376021 m³; it remains explicit and is not extrapolated. Native self-query checks pass, and mapping coordinate error stays below 1e-10 m. All comparisons below are relative physical-volume L2 differences on the covered domain.

| Field | Baseline (%) | Perturbed (%) | Paired response (%) |
| --- | ---: | ---: | ---: |
| psi | 0.097621 | 0.097474 | 0.140305 |
| u | 10.787471 | 2.650285 | 0.193665 |
| zj | 0.527543 | 0.535281 | 8.283527 |
| w | 2.681772 | 16.378644 | 35.214181 |
| rho | 0.171642 | 0.171029 | 0.426041 |
| T | 0.076307 | 0.076100 | 0.161298 |
| Vpar | 10.443823 | 10.438476 | 0.377672 |
| pressure | 0.084240 | 0.083873 | 0.156066 |

The frozen **1% full-field target fails** for stream function, vorticity and parallel flow. The paired density/temperature/pressure response meets 1%, but paired current/vorticity differences are 8.28%/35.21%. Independent q8 full-domain density/pressure integrals differ by at most **0.060202%**, below their 2% target. Small density/temperature/pressure errors do not establish convergence of the full coupled state. This is two-resolution consistency at fixed timestep; refined timestep sensitivity and asymptotic order are not measured. Energy closure and Gates A/B/C remain unvalidated. No production generation, scientific training or accepted-reference status is enabled.

Evidence: `cpu_spatial_convergence_20261103.json`, `validation_cpu_spatial_pair_20261103.json`, `cpu_spatial_convergence_fields_20261103.csv`, and matching PNG/PDF plots. Unique accounting includes both successful simulations, the preserved failed launcher and successful analysis: 5.609444444444444 additional rank CPU-hours. Live study accounting totals 15.437222222222221 rank CPU-hours and 4.18 node-hours. Software evidence remains 21 preflight tests, 22 focused comparison tests and one separate end-to-end report test; suites overlap and are not summed as a unique test count.
