# MAST-U model-303 case contract

Audit date: 2026-11-02. **Gate A has not passed.** The prepared artifact is an equilibrium-only snapshot, not a trajectory or dataset. After the initial audit, the user approved exactly one unchanged regression, submitted as job2767445. It is currently queued; see `equilibrium_regression_status.json`. No evolved physical pilot or training campaign has run.

## Pinned identity and available evidence

Installation: `/pitagora_work/FUPB2_UKAEA_ML/vgopakum/src/jorek`. Source `src` HEAD is `01a52be37576f79c785f1f7a80dee296dda5c9c8`, version `v2.26.09.0-16-g01a52be37`. Executable `builds/mastu_grid_303_n1/jorek_model303` SHA256 is `7c533e0b03b5ed8f87fe3a877080e313ee2caf043c35470a87f6fbfa51ea528c`. Its adjacent `build_config.txt` records model=303, n_tor=1, n_coord_tor=1, l_pol_domm=0, n_plane=1, n_period=1, n_coord_period=1. These are compile-time settings. The source checkout's generic `models/mod_settings.f90` is not the executable's settings: the build-local settings are snapshotted separately.

Final prepared local artifact: `runs/prepared_equilibrium_20261102_v3/case.json`. Its manifest records hashes of the original namelist, expanded namelist, equilibrium file, executable, environment script, original upstream equilibrium reference, build settings, source initializer, spatial residual and physical defaults. Each snapshotted input is read-only; attempts copy inputs into new directories. Raw snapshots and outputs remain ignored by Git.

`reg_tests/testcases/grid_with_patches_mastu_303/settings.sh:1–33` describes the fixed-boundary MAST-U equilibrium with wall-extended grid, n_tor=1 and nstep_n=0; its comparison uses absolute tolerance 1e-4 on native `values`. Existing toolkit evidence `spec/reviews/jorek-j1-build.json` records grid regression pass (25.5 seconds, max absolute difference 2.04194e-5). `jorek-j2-cost.json` records compute-node grid regression (19.4 seconds, no differences under h5diff tolerance). Neither is a new study regression or evidence of transport evolution. Prior step timings used n_tor=3 and must not estimate this axisymmetric campaign.

## State, units, signs, mesh and baseline

References below are relative to pinned `src/` unless stated otherwise.

- `models/model303/mod_model_settings.f90:8–43`: reduced MHD, evolving density, one combined temperature, parallel velocity, no neutral/impurity/two-temperature closure; seven variables ordered `(psi,u,zj,omega,rho,T,Vpar)`. Current and vorticity are constrained auxiliary fields. The pressure-like temporal operator in equation 6 couples rho and T; T is not an independent pressure tendency (see numerical equations report).
- `docs/physics/normalization.md:18–62`: R,Z are meters; magnetic field and poloidal flux retain SI dimensions. `n0=central_density*1e20`, mass density reference `rho0=central_mass*m_AMU*n0`, `t_SI=t_code*sqrt(mu0*rho0)`, `n_SI=rho*n0`, `p_SI=rho*T/mu0`, combined ion+electron `T_eV=T/(e*mu0*n0)`. Use pinned solver constants and recorded restart normalization, rather than rounded documentation constants, for production conversion.
- `docs/physics/base_fluid_models/RMHD/rmhd_model.md:20–115`: `B_R=psi_Z/R`, `B_Z=-psi_R/R`, `B_phi=F0/R`, physical `j_phi=-zj/(R*mu0)`, `v_R=-R*u_Z+Vpar*psi_Z/R`, `v_Z=R*u_R-Vpar*psi_R/R`, `v_phi=F0*Vpar/R`; velocity SI conversion divides by sqrt(mu0*rho0). Preserve these signs.
- Original example input: `F0=0.64` at line204, `central_density=0.5225575` at206 (`n0=5.225575e19 m^-3`), density profile lines207–213, combined temperature profile T0=.018/T1=.0001 lines216–222, rotation profile V0=V1=0 lines237–243. The header's machine/pulse comments do not establish experimental fidelity or a prescribed core electron temperature.
- Original input lines249–306 fixes wall extension and patch topology: n_R=45,n_Z=130,n_flux=10,n_tht=55,n_open=1,n_outer=8,n_inner=2,n_leg=3,n_leg_out=20,n_private=2,n_up_leg=3,n_up_leg_out=20,n_up_priv=2. Wall/limiter coordinates remain exactly in the snapshotted full namelist; no synthetic rectangle replaces that geometry. Mesh widths and perturbation-resolution limits still require native geometry measurements.

## Explicit evolution defaults

`configs/case.yaml:input_defaults` expands important inherited settings into the recorded namelist. Source `models/preset_parameters.f90` defines Crank-Nicholson at14; eta_T_dependent,visco_T_dependent,ZKpar_T_dependent true at27–30; eta=visco=visco_par=1e-5 at41–50; central_mass=2.01410177811 at57; fixed-boundary flags false at72–74; bc_natural_open=false at79; ZK_perp=(1e-5,0,0,99,99),ZK_par=1,ZK_par_max=1e20,D_perp=(1e-5,0,0,99,99),D_par=0 at235–255. These profiles are nonzero before any common perpendicular multiplier. Scaling must alter amplitudes, preserving shape parameters; multiplying all five entries would change shape.

Floors T_min=rho_min=1e-20, negative-correction thresholds -1e12 at292–298; numerical eta/viscosity/perpendicular diffusivity coefficients zero at305–310. Default shock capturing and VMS are disabled at318–335; model303 does not expose those switches in its namelist. Temperature-dependent parallel conduction, resistivity and viscosity remain nonlinear numerical physics.

Sources are not zero: particlesource=1e-5 and heatsource=1e-7 (`preset_parameters.f90:348–359`). Model303 overrides particlesource_psin to100 (`model303/initialise_parameters.f90:165`), making the inherited particle source broadly present. `model303/mod_sources.f90:42–51` supplies the tanh profiles and optional Gaussian source terms. Keep_current_prof=true at670 retains current-drive compensation. The experiment cannot assume source-free particle/energy conservation or a transport-steady Grad–Shafranov equilibrium.

Default tauIC=0 at525, no_mach1_bc=false at706. `initial_conditions.f90:213–246` constructs the parallel-flow boundary treatment; for grid_to_wall its direction is forced to zero at236. `model303/mod_boundary_conditions.f90` and `mod_boundary_matrix_open.f90` still require native boundary-row/flux verification before final evolution acceptance. The spline transfer cannot enforce these constraints.

`docs/numerics/time-integration.md:13–74,146–149` describes state-dependent temporal operators and the generalized implicit scheme. The study snapshot retains Crank-Nicholson, tstep_n=1, nstep_n=0. A useful timestep, output cadence, physical relaxation horizon, baseline restart and nonlinear-solver tolerances have not been validated for a perturbation. No explicit surrogate integrator has been accepted on the real case.

## Initializer patch prepared for review

The existing initializer's perturbation loop is `in=2,n_tor` (`model303/initial_conditions.f90:172–210`); it does nothing at n_tor=1. Existing Gaussian heat/particle sources are time-dependent drive profiles, not the requested localized initial rho/T perturbation. An undocumented restart edit is not a substitute.

`patches/0001-model303-axisymmetric-perturbation.patch` is an **unapplied-to-production** five-file JOREK derivative under LGPL-3.0-or-later. It adds disabled-by-default amplitudes `(A_rho,A_T)`, physical center/width pairs and a normalized-flux taper pair, plus matching MPI pack/unpack entries. Positive amplitudes are supported only for n_tor=1,n_order=3. Amplitudes and active shape parameters must be finite; amplitudes nonnegative; widths positive; taper `0<=a<b<1`.

The shape is an elliptical Gaussian multiplied by C2 quintic cutoffs in normalized flux and elliptical radius squared. Radial cutoff is flat through three widths and exactly zero at four widths. Both density and temperature receive multiplicative `1+A*G` factors. Native `(value,d/ds,d/dt,d2/dsdt)` DOFs receive full analytic product/chain rules. Physical boundary nodes and private-flux exclusion regions are skipped. The existing stream-function relation `u=-tauIC*rho*T` is recomputed only on eligible interior nodes; the existing inverse-Poisson vorticity initialization remains in place, and psi/current remain unchanged. Zero amplitudes preserve the original path.

Verified here: patch applies cleanly to copied pinned files; all five patched files pass gfortran syntax checks against existing native module interfaces; standalone helper amplitude/taper assertions pass; mixed derivative jets match central finite differences on a nonlinear coordinate fixture. `reports/initializer_syntax_check.json` records syntax evidence. This does **not** establish full compilation, MPI correctness, unchanged equilibrium regression, resolved support, physical admissibility, or Gate A. The standard prepare/run commands intentionally reject enabled perturbations until the isolated build and reference checks are accepted.

Before enabling the patch, measure native support against all boundaries/X-points and distorted elements; choose scientifically reviewed center/width/taper values; test default-amplitude equality against untouched regression; test enabled initialization and dependent constraints; then perform timestep/mesh refinement and term-strength checks. No center or physical width has been invented as a machine operating parameter.

## Bounded execution and proposed request

Runner code narrowly adapts the MIT-licensed existing `tkit.datasets.jorek_run` at revision `14fcecb664cbb084b86051370bc415d8446e1637`. Directly importing that runner imports toolkit provenance and enables JAX globally; the local standard-library adaptation retains its process-group termination and at-most-two RTree-only pre-step retries. It adds stderr abort detection, immutable prepared identity, required-output checks, native values regression and final physical time validation. A return code of zero with GMRES abort text is rejected. Completed process status never asserts physical acceptance. Failures and attempts remain recorded; no resume overwrites accepted raw output.

`configs/equilibrium_request.yaml` and `reports/equilibrium_request.sbatch` describe exactly one proposed equilibrium regression: account fupb2_ukaea_ml, partition boost_fua_dbg, one node, one MPI rank, one OpenMP thread, 8G, ten minutes, at most one submission. Coordinator verified current association/partition read-only. Authorization status remains `proposed_not_authorized`; the runner refuses execution while submission_allowed=false. No job was submitted.

Requested process resources are 1/6 CPU core-hour and 1/6 node-hour at the cap. A partition advertising64 CPUs/node could bill a full-node allocation differently (up to64/6 CPU-hours if charged for all CPUs). Actual allocated/billable usage must be read from `sacct` AllocCPUS/CPUTimeRAW/AllocTRES and site accounting, separately from the wrapper's process-core-hour estimate. Pilot/campaign costs remain unmeasured. The first request does not authorize a 12–20 trajectory pilot or the280-trajectory campaign.

## Open scientific gates

Gate A still needs a fresh equilibrium comparison, unperturbed transport trajectory, reviewed initializer regression, perturbed trajectory, physical-admissibility/balance checks, measured advection contribution and mesh/timestep refinement. Gate B additionally needs native same-state residual/geometry export and an autonomous retained numerical evaluator with temporal/constraint solves. Gate C, bulk retention, production budget and final split/envelope freeze remain unapproved/unmeasured. No dataset, trained checkpoint, speedup or accuracy claim is made.

The compact initializer shape is defined at native nodal DOFs. Finite-element interpolation can extend it into support-straddling elements. Native element/quadrature clearance from the separatrix, walls and private-flux cuts must therefore be checked, and positivity tested between nodes. Nodal taper tests alone do not establish these properties.

Final proposed request uses a540-second process timeout inside the600-second scheduler limit, reserving60 seconds for environment setup and report cleanup. The wrapper enforces the frozen study caps: at most4 MPI ranks,1 thread,2 attempts,1800 seconds per attempt; rendered jobs require one node, at most16G/30minutes and60seconds beyond all capped attempt timeouts. Total pilot/production budget remains disabled pending authorization and measured cost. The earlier local snapshot remains retained; the final request refers to `runs/prepared_equilibrium_20261102_v3` after the timeout/config identity change.


## Isolated initializer compilation evidence

A fresh archive of pinned JOREK source was extracted under `/tmp/mastu-opsplit-initializer-build-20261102/source`; the patch applied cleanly there. Full compilation and linking succeeded with `make -j2 jorek_model303`, GNU12.3/OpenMPI4.1.6/MUMPS and the pinned n_tor=1 configuration. The build took88.17seconds, created272 objects and141 module files, and returned0. Maximum two compiler processes, single OpenMP/BLAS threads and900-second process-group timeout were enforced. `/tmp` had6.96GB free before building against a1GiB reservation estimate. No simulation, MPI launcher or Slurm submission ran.

`reports/initializer_build.json` records exact commands, source/archive/patch/recipe/environment hashes, configured-source hashes and the actual compiler/link log. Persistent ignored artifacts are `runs/initializer_build_20261102/jorek_model303`, `build.log`, `build.sh`, `Makefile.inc`, `version.h`, and `build.json`. The7,630,368-byte binary has SHA256 `0a8547092540244f967096b169e9bc78a95e21767be2931c2d3fc397f33ccaf3`. The archive has no `.git` metadata, so embedded banner VCS fields may be empty; the explicit source commit/archive/patch manifest is authoritative. Production source/build remain unchanged.

Compilation is not runtime verification. The proposed first regression request still uses the untouched verified executable. Default-amplitude equality, enabled perturbations, MPI parameter exchange and Gate A remain unverified on the new binary.

The final request snapshot is v3: YAML literals now use explicit exponent signs (`1.0e+20`, `-1.0e+12`), so PyYAML loads Fortran real parameters as numbers. Numeric-looking strings are rejected before namelist rendering. Earlier v1/v2 snapshots remain retained and must not be used for submission; their quoted real defaults were invalid. The final v3 namelist is checked by targeted tests, and resource amounts remain unchanged.

## Authorized first regression

The explicit user approval covers one original-executable equilibrium regression only. `configs/equilibrium_authorized.yaml`, `reports/equilibrium_authorized.sbatch` and `runs/prepared_equilibrium_authorized_20261102` bind that approval to physical input hashes identical to v3. Job2767445 is pending under the debug partition QoS group-node limit, with no start estimate. Earlier proposed configs remain preserved as unauthorized drafts; they are not the submitted identity. No resubmission or pilot/training authorization is inferred.


## Legacy model303 temporal-mass audit — 2026-11-02

`cpu_temporal_mass_audit_20261102.json` uses dense exact-geometry four-rank snapshots and centered physical-time rates to separate generic diagnostic energy derivatives from selected legacy temporal coefficients. The positive low-density region still activates smooth `rho_1=0.01` mass correction: setting `rho_min_neg=-1e12` does not disable this correction. Nonroundoff corrected mass occupies 13.17–20.74% of full-wall quadrature volume over the sampled interval. The generic parallel kinetic energy also includes poloidal-field terms absent from the corresponding toroidal-only parallel temporal diagonal. Their partial power contributions, including pressure and perpendicular density-mass corrections, sum to 0.1163–0.5533 MW. These are centered finite-rate diagnostics, not an exact discrete-energy identity. Independent parallel kinetic energy agrees with native diagnostics to the frozen tolerance, and algebraic decompositions are checked. Remaining temporal/cross terms, stabilization and boundary work prevent complete energy closure; no physical parameters or acceptance thresholds changed.


## Budget-backed CPU refinement — 2026-11-02

The user authorized raising caps when CPU credit permits. `saldo -b --dcgp` verifies FUPA2_FAUNA CPU credit: 26,481,874 local resource-hours remaining overall and 2,531,431 monthly at the recorded snapshot. These shared-account units include requested memory and differ from rank CPU-hours. The frozen `cpu_budget_authorization_20261102.json` permits one node, at most 16 MPI ranks, one thread, 128 GiB, two hours and one attempt per job; the phase envelope is 128 requested CPU-hours, eight requested node-hours, six simulation allocations, concurrency one. Existing untagged requests retain their old caps. Production and scientific training remain disabled.

See `cpu_refinement_continuation_20261102.json` for the live bounded development phase.
