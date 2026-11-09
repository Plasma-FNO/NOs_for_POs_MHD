# MAST-U axisymmetric OpsSplit with L-SplineFNO agent handoff

Prepared 2 November 2026. Revised to use L-SplineFNO on unstructured meshes. Status: tested implementation infrastructure exists under [`mastu_opsplit`](mastu_opsplit/README.md); see [actual progress and remaining milestones](mastu_opsplit/reports/progress.md). Gates A/B/C remain pending. CPU continuation has produced short unperturbed baseline evolution, refined equilibrium construction, and default/enabled initializer validation (one and four MPI ranks); no accepted dataset or scientific training campaign exists. Synthetic plumbing checks are explicitly separate. Beads tracks M0–M6 and implementation evidence at the repository root. Two unchanged equilibrium regressions are explicitly approved: CPU job 2767720 and BOOST job 2767719 both passed the reference comparison. The user subsequently approved continuing the study on CPU; bounded timestep and mesh checks are tracked in `mastu_opsplit/reports/cpu_continuation_20261102.json`. These replace job 2767445, cancelled before allocation; see [live regression status](mastu_opsplit/reports/equilibrium_regression_status.json).

Current JOREK experiment: [two-page methods and interim-results report](../Paper/results_one_page/results_one_page.pdf), with [editable source, evidence snapshot and rebuild instructions](../Paper/results_one_page/README.md). It explains how AR, NODE, explicit and implicit OpsSplit are conducted on the 80/0/20 ensemble. The 2 November pre-pilot page is archived in `Paper/results_one_page/archive_prepilot_20261102/`.

Latest CPU continuation: CPU credit was checked with `saldo --dcgp`; shared project balances permit the requested cap increase. The opt-in development profile allows up to 16 MPI ranks, 128 GiB and two hours, with an aggregate 128 CPU-hour/eight node-hour phase cap. Four-rank parity is exact; 16-rank physical-gradient parity failed, so refinement stays at four ranks. Larger mesh attempts expose native grid failures. Bounds-check job 2768423 identifies a 10,001st contour point overflowing fixed capacity 10,000; isolated patch 0009 raises capacity to 40,000 and passes exact eight-step old-mesh parity (fields, geometry, times and energies); the doubled-R/Z nested probe then stopped at the unchanged native four-strike guard (six contacts found). Job 2768438 completed the refined grid (46,415 elements, 47,067 nodes; native mesh check passed), then failed before evolution requesting STARWALL despite both boundary switches being false. Source inspection identifies uninitialized `freeb_equil2` on the cached-R/Z startup path. Patch 0010 initializes it from the requested boundary mode; the user explicitly authorized the next allocation, and job **2768489** is submitted to build patch 0010, check exact ordinary-start parity, check cached-start physical parity under the frozen protocol, then run the refined three-step probe only if both pass. It uses four ranks, 128 GiB, two hours maximum and one attempt per stage; the fresh one-job budget is recorded in `cpu_cached_rz_recovery_authorization_20261103.json`. Failures and raw outputs remain preserved. See [live refinement phase](mastu_opsplit/reports/cpu_refinement_continuation_20261102.json), [budget](mastu_opsplit/reports/cpu_budget_authorization_20261102.json), and [usage](mastu_opsplit/reports/cpu_budget_refinement_usage_20261102.json).

Earlier diagnostics-only patches preserve both 48-step trajectories exactly and correct magnetic losses, but full energy remains unclosed. Physical coarse/fine stream-function and parallel-flow errors are 19.56% and 16.63%, so spatial convergence still fails. The temporal-mass audit exposes active low-density correction and selected legacy/diagnostic mass differences; it does not close the complete balance. Software validation passes 81 tests. Gates A/B/C, accepted data and scientific training remain pending; the autonomous native spatial/temporal/constraint backend remains unfinished.

## 1 Objective and scope

Build one reproducible dataset of axisymmetric pressure perturbations and their coupled relaxation in the existing JOREK MAST-U equilibrium. Train neural operators for selected physical RHS contributions, assemble them with numerical contributions in a time integrator, and evaluate closed-loop performance across perturbation and transport regimes. Extend the assembly method in *Learning Physical Operators using Neural Operators*, arXiv:2602.23113v2, using the L-SplineFNO architecture from *Learnable Splines for Unstructured Neural Operators*, arXiv:2610.22446v1. L-SplineFNO is the selected backbone for all three primary neural comparisons.

Use one equilibrium, one mesh topology, one physics configuration, and one perturbation family. Mesh refinement for verification does not create another physical case. Fields depend on R and Z; retain cylindrical/toroidal geometry factors and set n_tor=1, corresponding to the axisymmetric n=0 component. A localized R,Z perturbation is a toroidal ring. This experiment does not model nonaxisymmetric ELMs, tearing islands, localized 3D filaments, or L–H transitions. “Operating regimes” below means controlled changes in perturbation strength and transport within this case.

The primary research question is whether a modular physical-operator model has better accuracy and stability outside its training parameter range than an autoregressive model and a whole-RHS neural model. Speedup is a measured secondary outcome, not a promised result.

Deliver scripts, pinned configurations, a versioned dataset, trained checkpoints, evaluation tables/figures, and a reproducible report. Keep this separate from the integrated transport toolkit's JAX runtime and existing studies. Do not add a second device, new physics closure, architecture survey, or general workflow framework.

## 2 Verified starting assets

| Asset | Existing location and status |
|---|---|
| Paper | `/pitagora/home/userexternal/vgopakum/.codex/attachments/d0511a14-f05f-468a-a799-a181344b5410/2602.23113v2.pdf` |
| Paper code | `/pitagora/home/userexternal/vgopakum/NOs_for_POs` |
| L-SplineFNO paper | `/pitagora/home/userexternal/vgopakum/.codex/attachments/1ed36b35-0671-4754-85e3-171b29c60188/2610.22446v1.pdf` |
| L-SplineFNO repository | `https://github.com/gitvicky/Conditional_Fusion_Operator`; local checkout `/pitagora/home/userexternal/vgopakum/Conditional_Fusion_Operator` |
| Inspected local revision | `a3ca59bba100ca56e535fc35c32967bdeb04d8d9`; checkout contains modified and untracked files |
| Backbone implementation | That checkout's `Models/learnable_spline_fno.py:LearnableSplineFNO` |
| Factory and conditioning | `Models/factory.py:_build_learnable_spline_fno` and `Models/conditioning/film.py:FiLMConditioning` |
| Architecture tests | `Tests/test_learnable_spline_fno.py`, `Tests/test_bspline_basis_convention.py`; these were inspected, not executed for this plan |
| JOREK installation | `/pitagora_work/FUPB2_UKAEA_ML/vgopakum/src/jorek` |
| Pinned source | Installation subdirectory `src`; commit `01a52be37576f79c785f1f7a80dee296dda5c9c8`, describe string `v2.26.09.0-16-g01a52be37` |
| Starting example | Source subdirectory `reg_tests/testcases/grid_with_patches_mastu_303` |
| Ready executable | Installation subdirectory `builds/mastu_grid_303_n1/jorek_model303` |
| Local example inputs | Installation subdirectory `runs/j1-check/grid_with_patches_mastu_303`; includes `input` and `eqdsk.dat` |
| Runtime environment | Installation file `jorek-env.sh`; loads GNU/OpenMPI dependencies and sets `JOREK_ROOT` |
| Existing runner | `/pitagora/home/userexternal/vgopakum/IntegratedModelling/tokamak-toolkit/src/tkit/datasets/jorek_run.py` |
| Unit/input helpers | `/pitagora/home/userexternal/vgopakum/IntegratedModelling/tokamak-toolkit/src/tkit/adapters/jorek.py` |
| Regression evidence | Toolkit `spec/reviews/jorek-j1-build.json` and `spec/reviews/jorek-j2-cost.json` |

Preserve the L-SplineFNO checkout's existing edits. Record the actual imported source paths, commit, relevant file hashes and any patch used; the observed local HEAD is not a claim about the latest remote revision. Reuse the implementation through a pinned dependency or a narrowly scoped, attributed copy with its required modules and license. Do not rewrite the spline kernels. Keep the new study's data/configs separate from that repository's existing MAST-U engineering study.

The upstream example is explicitly a fixed-boundary MAST-U equilibrium with a wall-extended grid. Its regression uses nstep_n=0. Existing compute-node regression evidence validates equilibrium/grid construction, not this proposed time-dependent dataset. The available three-step cost measurement used n_tor=3 and is not a campaign estimate for this axisymmetric study.

Use model 303 initially because it matches the verified example and executable. Do not migrate to model 600 simply because it is newer; identify a concrete missing capability first. Preserve source, equilibrium, and executable hashes. Any exporter or initialization modification belongs in a separate build/worktree, never in an existing production build.

Reuse paper-code model and training components after inspection. The existing `Expts/operator_splitting.py:JOREK_ES_OS_RHS` is not a validated implementation: it uses periodic derivatives and hard-coded R,Z arrays, references an undefined `self.kappa`, and returns two RHS channels from three state channels. Do not adopt it unchanged. The inspected `Utils/explicit_time.py` implements explicit steppers; the README alone is not evidence of a working IMEX integrator.

## 3 Project boundaries and agent assignments

Suggested implementation root: `/pitagora/home/userexternal/vgopakum/IntegratedModelling/mastu-opsplit`. Suggested bulk-data root: `/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit`. These are proposed paths to create during implementation, not existing project artifacts. Reuse an existing suitable checkout where available; preserve unrelated changes. Keep raw data and checkpoints out of Git. Confirm filesystem write permissions and retention before production.

Use four roles, which can be separate agents or sequential assignments:

| Role | Owned work | Dependencies | Required handoff |
|---|---|---|---|
| Coordinator and evaluator | Scope, interface/config freeze, budget ledger, independent acceptance checks, final evaluation | Starts immediately | Decision log, gate records, locked evaluation protocol, final report |
| Simulation agent | JOREK baseline, perturbation initialization, parameter sampling, Slurm jobs, reference diagnostics | Baseline contract | Validated inputs/build, pilot cost, reference runs and failures |
| Numerical and data agent | Native state/term definitions, quadrature export, spline source/query contract, numerical assembly, dataset packaging | Works with simulation and learning agents from start | Schema, weak-form mapping tests, validated all-numerical rollout, dataset manifest |
| Learning agent | L-SplineFNO dependency, field-input/FiLM adapter, data loader, training, checkpointing | Can scaffold against synthetic fixtures; real training follows numerical/data gates | Three L-SplineFNO deployment families, spline/gradient checks, checkpoints, training records |

Coordinator owns shared schemas and configs. Other agents edit separate modules. Freeze interfaces before parallel implementation and coordinate any schema change. Do not duplicate runners, normalize data independently in different agents, or let training determine test ranges.

Minimal project layout:

```text
configs/          case.yaml, sampling.yaml, lsplinefno.yaml, train.yaml, evaluation.yaml, resources.yaml
scripts/          prepare_case.py, run_case.py, export_dataset.py, train.py, evaluate.py
slurm/            simulate.sbatch, train.sbatch
src/              case setup, data schema, numerical operators, L-SplineFNO adapter, model assembly
patches/          narrowly scoped JOREK initialization and residual-export changes
tests/            targeted numerical, data, runner and training checks
reports/          gate records, figures, final results
```

Use plain Python/YAML, HDF5, PyTorch, and existing repository libraries. Add a dependency only if the selected representation requires it and an installed implementation cannot supply it.

Each command must support explicit configuration and output paths, reject accidental overwrites, and return a nonzero status on failure. A dry run must render commands/resource requests without submitting or launching simulation/training.

## 4 Gate A Establish a useful physical case

The simulation and numerical agents jointly produce `reports/case_contract.md` and `configs/case.yaml` before an ensemble is generated.

1. Snapshot the example and reproduce its equilibrium regression. Record units, normalization, current and field sign conventions, mesh, wall geometry, model switches, sources, transport laws, boundary conditions, floors, numerical stabilization, and time scheme. Expand defaults that affect evolution into the recorded configuration.
2. Enable time evolution and establish an unperturbed reference trajectory. A Grad–Shafranov equilibrium need not be a transport steady state. If relaxation is needed, document its duration and reference restart. Do not conceal large baseline drift by subtracting it in plots.
3. Add one smooth positive perturbation to density and temperature, with fixed boundary taper. Define, for example, `delta_rho/rho_base = A_rho * G(R,Z)` and `delta_T/T_base = A_T * G(R,Z)`, with an elliptical Gaussian G smoothly tapered to zero before the separatrix. Both amplitudes are nonnegative initially. Record physical widths and their local mesh resolution.
4. Choose centers from a fixed admissible region inside the closed flux surfaces. Keep support away from the separatrix initially. Preserve boundary values and current/vorticity constraints; initialize dependent fields consistently. Use an existing supported initializer where possible. Otherwise add a small, tested source initialization routine rather than editing undocumented HDF5 internals.
5. Run the unperturbed case and a perturbed case to a useful relaxation horizon. Evolve the full selected model; do not silently freeze magnetic fields or flows to make the task easier.
6. Verify that nonlinear advection is resolved and materially contributes. Use perturbation evolution and term norms, not visually large background fields. If the response is essentially static or pure diffusion, revise the perturbation within this case before generating an ensemble.
7. Repeat representative trajectories with smaller timestep and a refined mesh. Check full fields, perturbation fields, and integrated diagnostics. Confirm that perturbation widths are adequately resolved, especially near distorted elements.

Gate A passes when the baseline is understood, the perturbed evolution is useful, all required fields remain physically admissible, and numerical error is quantified. A solver exit code of zero alone is insufficient. Transport/boundary choices not inferable from the example must be recorded explicitly, with scientific review where applicable; do not invent machine operating parameters.

## 5 Gate B Define the operators and validate numerical assembly

Keep the paper's structure: selected learned physical terms plus numerical terms, followed by time integration. Use a standalone assembly harness for the research comparison. Direct replacement inside the existing JOREK implicit solver is not required for the first result.

The initial learned blocks are density advection and temperature advection/compression, grouped according to the exact pinned model equations. Keep the remaining equations and contributions numerical. This is a valid partial replacement experiment; do not expand to every MHD nonlinearity before it works.

Write the semi-discrete system schematically as:

```text
M(q, geometry) dq/dt = F_kept(q, parameters, geometry)
                     + N_density(q, geometry)
                     + N_temperature(q, geometry)
                     + S
C(q, geometry) = 0
```

This is a design schematic, not a complete model-303 equation. Agents must derive and cross-reference each retained/replaced contribution against the pinned source. Record which fields are evolved and which are algebraic auxiliaries. Preserve the temporal operator, including state dependence and spatial coupling, and solve constraints instead of independently predicting inconsistent current/vorticity fields.

Required work:

- Export each selected spatial residual block at the same instantaneous state, together with the total spatial residual and retained remainder. Distinguish physical residuals from dt-scaled solve vectors, Jacobian entries, stabilization terms, and boundary-row modifications. Document signs and whether coefficient factors are included.
- Verify that the sum reconstructs the original discrete spatial residual. If only selected blocks are separated, the remainder must include every other contribution explicitly by identity. Check both assembled and mass-applied units; do not apply different incompatible mass inverses to separate blocks.
- Keep coefficients outside learned terms where the exact algebra permits. Temperature-dependent coefficients remain nonlinear numerical functions unless deliberately assigned to a learned block. Do not label evolving anisotropic diffusion as a constant linear stencil.
- Expose or port only the required numerical evaluator and temporal/constraint solves. Treat this as substantive work, not a free export from JOREK. Before training, replace the learned blocks with their exact numerical counterparts and reproduce reference evolution.
- Use an IMEX or implicit scheme if stiffness requires it; choose the split through stability tests. Include stiff couplings, not just diffusion, in that assessment. Explicit integration is acceptable only after timestep refinement demonstrates stability and accuracy.
- Differentiate through the chosen rollout implementation correctly. Check training gradients against finite differences on a small fixture, including numerical solves where they depend on learned states. Do not silently detach state-dependent solves to save implementation time.

Representation decision: retain the native JOREK finite-element state and mesh. Supply physical field values evaluated at an unstructured source point set to L-SplineFNO; use the native element quadrature points initially. Its learnable spline scatter builds a latent Cartesian grid internally, the FNO processes that latent grid, and analytic cubic B-spline decoding returns outputs at unstructured query points. No externally rasterized training dataset is required. A latent Cartesian grid is not the physical mesh.

The primary hybrid step is:

```text
native finite-element state q
  -> evaluate current physical fields at source points
  -> two L-SplineFNO modules for the selected physical contributions
  -> decode at native quadrature query points
  -> assemble those contributions using the original weak-form basis and weights
  -> combine with retained numerical residuals
  -> temporal/constraint solve to update native q
```

Define whether each learned output is a strong-form scalar contribution or a flux component needed by the weak form. Export matching pointwise training labels and separately retain the assembled residual for identity checks. A vector of assembled finite-element residual coefficients is not a scalar field sampled at node coordinates. Do not train or decode it as one. Derivative degrees of freedom and multiple mesh patches must remain represented by the native basis; boundary constraints and physical volume factors belong to the verified numerical assembly.

Use the same source and query points initially. The architecture supports independent query points, so later diagnostic queries and mesh-refinement checks do not require raster interpolation. Keep this a single-geometry study. Do not repeatedly compress the entire numerical state through an approximate spline scatter/decode between steps; only the learned contribution takes that route.

Learnable scatter is an approximation, not an inverse finite-element projection. Gate B's all-numerical baseline must bypass the learned transfer and use exact physical contributions through the same source/query and weak-assembly contract. Verify FE evaluation/assembly independently, then measure spline approximation and latent-resolution error separately. A learned scatter/decode round trip need not reproduce arbitrary fields exactly.

Use fixed affine coordinate bounds from the physical mesh, recorded once, to map R,Z into the latent-grid bounds. Do not normalize each trajectory or query cloud independently. Retain physical R and the inverse coordinate scales for cylindrical operators and derivatives. Preserve connectivity, quadrature weights, boundary labels, patch IDs and basis evaluations independently of the point-cloud feature arrays. Spline support and the latent FNO do not impose physical wall/separatrix boundary conditions or prevent communication across domain gaps; enforce those in the physical assembly and test near-boundary behavior.

Critical acceptance check: an all-numerical assembly using the final representation and integrator must match JOREK on both baseline and perturbed runs. Merely feeding exact labels from reference timestamps into a solver is not this check: numerical terms must be evaluated at its own current state. Failure blocks training and cannot be fixed by letting the network learn the discretization mismatch.

Once this passes, train direct term predictions and then fine-tune through assembled rollouts. Rollout-only training is an ablation; compensation between learned terms otherwise weakens the operator interpretation.

## 6 Gate C Freeze sampling and operating regimes

Use a single perturbation family and fixed geometry. Perturbation centers, widths, A_rho, and A_T provide initial-condition diversity. For regime transfer, vary only the existing perpendicular particle/thermal transport coefficients initially. Apply a common multiplier s_perp to their baseline profiles, preserving profile shape and the existing parallel transport law. Keep resistivity and viscosity fixed for this first study.

The following are candidate pilot ranges, not established stable operating limits:

| Quantity | Candidate training interval | Candidate held-out extension |
|---|---|---|
| A_rho and A_T | 0.02 to 0.15 | 0.20 to 0.30 |
| s_perp | 0.7 to 1.4 | 0.35 to 0.5 and 2.0 to 2.8 |
| Perturbation center and widths | Mesh-resolved admissible interior region | Held-out location/width bins within the verified numerical envelope |

Choose widths from resolved local scales, not a guessed machine-independent distance. If the baseline perpendicular transport is zero, scaling it is meaningless: define and document a justified nonzero baseline before sampling. Maintain positive coefficients and record dimensional as well as normalized values. When transport changes, either prepare a corresponding relaxed baseline or retain and document its unperturbed reference trajectory; do not confuse background equilibration with the seeded perturbation.

Use deterministic stratified or Latin-hypercube samples with a saved seed and explicit rejected-sample reasons. Do not use Gaussian amplitudes with an unbounded tail. Freeze the final envelope after pilot convergence checks, before model tuning or opening final test results.

Measure regime indicators from reference terms, such as the time/volume-weighted ratio of advection to diffusion contributions in each equation, with a documented denominator floor. Coefficient changes do not automatically imply that the simulation became advection dominated. Report the realized regime distribution and adjust the pilot if it covers only one effective regime.

Proposed full dataset allocation, subject to measured cost:

| Split | Trajectories | Purpose |
|---|---:|---|
| Train | 160 | Training envelope |
| Validation | 20 | Same envelope; model and hyperparameter selection |
| ID test | 20 | Untouched in-distribution test |
| Strong perturbation test | 20 | Held-out amplitudes only |
| Low transport test | 20 | Held-out low s_perp only |
| High transport test | 20 | Held-out high s_perp only |
| Combined test | 20 | Strong perturbation plus held-out transport |
| Total | 280 | One physical case and one dataset release |

Start with 12–20 pilot trajectories plus necessary baseline/refinement runs. These pilot runs select physics/numerics and must not later become untouched test data. If budget cannot support 280 trajectories, shrink training first and preserve explicit test groups, reporting uncertainty from the smaller tests. Counts are planning targets, not sample-complexity guarantees.

Split by parent initial-condition family: paired coefficient variants, shared perturbations, refinements, restarts, and adjacent windows stay in the same split. A common equilibrium is intentional; near-duplicate perturbation realizations across splits are not. Fit all preprocessing on training data only. Select checkpoints on validation only.

Choose a physical training horizon T_train from the pilot. Generate held-out trajectories to at least 2*T_train where reference evolution stays valid. Evaluate temporal extrapolation on held-out initial conditions; do not call an unseen tail of a training trajectory an independent test. Use physical times, actual output timestamps, and a documented sampling cadence that resolves the fastest retained dynamics.

## 7 Dataset and execution contract

Use a manifest plus one HDF5 file per trajectory, with immutable raw outputs retained separately. Include:

- Stable case ID, parent-family ID, split, random seed, parameter vector, configuration hash, source commit/patch hash, executable hash, equilibrium and restart hashes, modules, MPI/OpenMP layout, scheduler job ID and requested/actual resources.
- Monotonic time in seconds and original code time, named fields with units and exact normalization, dynamic magnetic and flow fields required to close the model, current/vorticity auxiliaries, and reference-baseline identity.
- Canonical FE mesh and degrees of freedom; source/query coordinates and IDs; element, patch and boundary labels; basis evaluations and quadrature weights; physical R and affine coordinate bounds; raw state data; pointwise selected-term/flux labels and corresponding assembled residuals; retained remainder; normalization and residual reconstruction checks. Store fixed geometry once with a hash reference from every trajectory.
- Particle, thermal, kinetic and magnetic diagnostics as supported by the pinned equations; source, dissipation and boundary terms needed to interpret balances. Preserve model conventions such as total versus species temperature.
- Solver outcome, convergence/health checks, physical-validity status, elapsed time, memory/storage measurements, and any retried attempt or stabilization/floor activation.

The field-input adapter must distinguish two existing APIs:

- Low-level `LearnableSplineFNO`: fields `[B,C_in,N_source]`, source coordinates `[B,2,N_source]`, optional `mu` `[B,K]`, query coordinates `[B,2,N_query]`, output `[B,C_out,N_query]`.
- Factory wrapper: source input `[B,N_source,2+C_field]` with coordinates first, query coordinates `[B,N_query,2]`, and output `[B,N_query,C_out]`. Its `in_vars` includes two coordinate channels; the low-level `in_channels` does not.

Store trajectory samples as `[time,N_source,C_state]` with separate geometry, and convert layouts at one tested adapter boundary. Native FE coefficients remain a separate canonical array. Document the final state channels, output physical terms/fluxes, weights, normalization and shape assertions in `reports/data_contract.md`. Batch a shared fixed mesh without padding initially. Reject unmasked padded points if variable point counts are later introduced: fake points would change the spline scatter and coverage channel. A zero-padded exterior is not physical plasma.

Wrap or reuse the existing bounded JOREK runner. It snapshots inputs and recognizes GMRES aborts even when JOREK returns zero. Retry only transient, identified grid-setup failures within a fixed limit. Preserve all attempts; do not label an aborted trajectory as successful because some output exists. Production quality checks enforce the requested final physical time and required outputs. Failed cases stay in the manifest and failure statistics; replacements use a predefined sampling rule. Do not silently truncate failed references into training successes.

Submission scripts must use an explicit current Slurm account, partition, time/memory requests, and bounded array concurrency. Do not blindly copy historic account names. Source the verified JOREK environment in each simulation job. Use checkpointing for training; a restart must restore optimizer, scheduler, random states and data split. Resume only matching configurations and never overwrite accepted raw results.

Track CPU core-hours, GPU-hours, peak memory, raw bytes and export bytes separately. Estimate campaign cost from measured pilot runtimes and variance, including failed/retried attempts, reference refinements and model seeds. The resource config must contain total and per-job caps. No unlimited resubmission loops.

## 8 Models and training

Use **L-SplineFNO for every primary neural model**, reusing `Conditional_Fusion_Operator`. The original L-SplineFNO paper learns steady fields from geometry and global conditions. This study instead supplies the current evolving field at every RHS evaluation, including intermediate integrator stages. That field-to-physical-operator adaptation must be implemented and verified; do not reuse a geometry-only steady-state pipeline or treat time as a third spatial dimension.

Start with two independent L-SplineFNO modules, one per selected physical block. Each gets the current fields its exact term depends on; document dependencies. The same full current state must be available to the overall model and to the comparison models. Fixed geometry metadata may be reused across the two modules, but a shared trainable encoder is a separate design change and not the starting implementation.

Use the factory's input FiLM conditioning on non-coordinate field features for parameters that genuinely enter a learned target, leaving stencil coordinates unmodulated. Important API detail: passing `mu` to the bare backbone does not by itself activate FiLM. The factory wrapper provides it; otherwise connect `FiLMConditioning` explicitly and test that connection. For coefficient-free advection blocks, s_perp acts in the retained numerical transport terms and need not condition the neural block. Whole-RHS and autoregressive models must receive s_perp because it affects their outputs. Do not use initial perturbation amplitude, case ID or time index as a substitute for current-state input. Keep a shared, explicit physical-parameter contract and record each model's use of it.

Use cubic splines in two spatial dimensions: at most 16 stencil taps per point per scale. Start with `scatter_scales=[1]`, `enforce_pou=True`, `density_channel=True`, `encode_refinement_steps=0`, and the existing sign-preserving denominator safeguards. The density channel describes sampling coverage, not plasma mass density. Multiscale branches, region-specific model weights, alternate spline families and certification machinery are outside the primary implementation. Record the inspected safeguard defaults explicitly: `density_eps=1e-6`, `pou_floor=1e-3`, and `density_floor_frac=0.1`; measure their activation rather than silently weakening them.

Candidate pilot hyperparameters: latent `grid_shape=(64,64)`, `modes=12`, `width=32`, `n_layers=4`, `kernel_n_control=9`, and `kernel_control_width=0.5`. These are starting values, not validated resolution choices. Compare a small set of latent resolutions on pilot/validation data, allow an anisotropic latent grid for the elongated domain, and lock one resolution before test evaluation. Match the total parameter budget across the two OpsSplit modules and each single-model baseline by adjusting width; record spline, FiLM and FNO parameters separately. Record actual memory and compute as well.

Retain sparse, bounded-memory scatter/decode. Never construct a dense N-by-G transfer or a dense G-by-G Gram inverse. Cache only immutable geometry information where the API supports it. Recompute learned weights and feature-dependent quantities under autograd; do not cache detached learned stencils, reuse an encoded state across different physical states, or freeze kernel gradients accidentally. `encode` followed by repeated `decode` is valid only for multiple queries of the same encoded state. Profile before adding a new cache.

Partition of unity and constant reproduction are transfer properties, with qualifications at uncovered or denominator-floor-affected grid points. They do not guarantee conservation, positivity or stability of the learned PDE. Preserve the native balance and stability tests. Analytic decoder derivatives are optional diagnostics initially; they must include coordinate scaling and boundary-basis normalization and be verified against manufactured functions before use. They do not replace the FE weak form by default. Bounds on network output from the spline paper do not certify physical accuracy or long-time rollout stability.

| Model | Definition |
|---|---|
| Persistence | Repeats the initial state; essential because backgrounds can dominate raw-field metrics |
| Numerical baseline | Verified all-numerical assembly at the ML representation, plus JOREK reference |
| Autoregressive L-SplineFNO | Predicts the next state on native sample points at a fixed saved-time interval, with a verified map back to the chosen state representation |
| Whole-RHS L-SplineFNO | Learns all differential tendencies after a consistent mass/constraint treatment, then integrates |
| OpsSplit with L-SplineFNO | Two spline operators learn the selected density and temperature contributions at quadrature queries; numerical weak assembly and retained equations evolve the full state |

All neural models receive equivalent initial information, geometry, parameter values, and boundary data. Enforce algebraic constraints consistently where the state representation requires them. OpsSplit intentionally receives more numerical physics; state this in the comparison. Do not supply future true velocity or magnetic fields to OpsSplit while claiming an autonomous rollout. Conditional tests using reference fields are allowed only as separately labelled diagnostics.

Train at least three independent seeds per primary neural model. Match total trainable parameter counts approximately, document the remaining mismatch, and use the same validation-based tuning budget. Also report wall time, memory and function evaluations because matched parameters do not imply matched work.

Training sequence:

1. Pass the upstream 2D spline checks and the new field/FiLM/weak-form adapter checks. Overfit a tiny training subset and confirm gradients reach state features, FiLM where enabled, both spline kernel banks and FNO weights.
2. Pretrain OpsSplit on selected term labels with channel-scaled, geometry-weighted losses.
3. Fine-tune through short autonomous rollouts, initially five output intervals as in the paper; extend only if validation supports it.
4. Combine rollout loss with an explicit operator-label loss so terms remain identifiable. Choose weights using validation; record them.
5. Train autoregressive and whole-RHS baselines with the same training trajectories and comparable rollout supervision. Give the whole-RHS model matching total-derivative supervision where valid, or report the label-access difference and include a trajectory-only comparison.
6. Include an OpsSplit rollout-only ablation to assess term identifiability. If budget allows, compare a frozen analytic cubic scatter against learned scatter at the same latent resolution; freeze the encoder correction at zero and retain the rest of the architecture. This is a normalized fixed-scatter ablation, not the dense-solve analytic SplineFNO. A nested 25%/50%/100% data-efficiency study is optional after the primary comparisons.

Normalize perturbations and operator labels with training-derived scales. Report full-field accuracy as well. Transform derivatives and coefficients consistently when using normalized variables and time; never apply a field-value normalizer naively to a dimensional coefficient. Check near-zero term denominators. Verify reduced precision against a float64 numerical reference before using it for solves or residual export.

## 9 Performance protocol and acceptance criteria

Freeze evaluation code and numerical thresholds before the final tests. Proposed thresholds below are development targets to ratify using the pilot; they are not achieved results. Any revision must precede final-test evaluation and remain in the decision log.

Numerical gates:

- Same-discretization residual reconstruction: relative discrepancy at most 1e-8 in float64 for non-negligible totals, with a scale-aware absolute tolerance and cancellation cases reported separately.
- Mesh/timestep refinement: at most 1% change in chosen perturbation-field norms and at most 2% in key integrated diagnostics on the pilot verification set, or a documented tighter reference if ML errors are smaller.
- All-numerical assembly: errors within the measured reference/discretization envelope; FE evaluation/weak-assembly and baseline state-projection errors must fit within that budget. Measure learned spline approximation separately rather than requiring the trained transfer to be an exact projection.
- L-SplineFNO checks: source/query layouts and common coordinate bounds; partition of unity; covered-node constant behavior and safeguard activation; finite field/kernel/FiLM gradients; empty-coverage handling; near-wall behavior; small latent-grid refinement study before freezing the architecture.
- Numerical error should be materially below the ML error being interpreted. Tighten the reference if they become comparable.

Report metrics for every regime and seed, including failures:

1. Volume-weighted relative L2 error per field and perturbation field, with epsilon tied to a training scale. Axisymmetric physical volume uses 2*pi*R weighting, or the equivalent native quadrature. Include absolute/scaled errors as perturbations decay toward zero.
2. Error versus physical time, at T_train and 2*T_train, plus median and 95th-percentile trajectory error. Do not let accurate equilibrium backgrounds hide poor dynamics.
3. Direct selected-operator error and total-RHS error; check cancellation and sign changes.
4. Density/temperature positivity, constraint residuals, finite values, time to failure, and completed-trajectory fraction. Report clipping/fallback frequency if used. Do not repair failures silently.
5. Particle/energy balance residuals including modeled sources, sinks and boundary fluxes. Do not demand constant energy in a dissipative, open or driven system.
6. Perturbation peak, width/spreading, centroid where meaningful, relaxation time, and relevant integrated transport. Define extraction consistently before testing.
7. End-to-end rollout wall time and memory for the same physical horizon/output requirements, including spline scatter, latent FNO, spline decode, FE assembly/solves, host/device transfers and initialization. Distinguish CPU JOREK versus GPU surrogate comparisons from same-hardware comparisons. Include training/data cost and, if useful, break-even evaluation counts.

Use paired comparisons on the same trajectories and uncertainty intervals bootstrapped by trajectory/parent family, not by frames. Show variation across training seeds; 20 test trajectories per regime give limited precision and should not be oversold.

Candidate accuracy targets: median perturbation-field error below 5% at T_train on ID tests and below 10% on each OOD group, with at least 95% completion within the declared valid envelope. Report tails and integrated diagnostics even if these targets pass. For a superiority claim, compare OpsSplit against the strongest neural baseline on prespecified regime/time metrics with uncertainty; do not require it to win every metric or change the claim after seeing results. There is no required speedup factor. A well-supported negative result still completes the study.

For a limited mesh-resolution check, decode the same encoded held-out state at refined quadrature/query points and compare with FE-evaluated reference terms. Report this separately from re-encoding a refined source mesh: changing source sampling changes coverage features and can change predictions. Preserve physical-volume weighting; sampling-density normalization is not volume-conservative integration. This is a numerical-resolution test within the same MAST-U case, not evidence of new-geometry generalization.

For timestep transfer, run continuous-time models at validated internal steps h/2, h and 2h while comparing at common physical output times. This tests integration consistency, not arbitrary timestep freedom. Evaluate fixed-step autoregressive models at their trained cadence; do not invent an interpolation capability for them.

## 10 Milestones and deliverables

| Milestone | Required result | Gate to proceed |
|---|---|---|
| M0 Contracts and preflight | Paths, versions, script skeletons, equations/fields table, proposed resource config | No unresolved interface ambiguity |
| M1 Physical pilot | Unperturbed and perturbed trajectories; refinement and term-strength checks | Gate A |
| M2 Numerical assembly and labels | Native source/query export, weak-form mapping, all-numerical rollout, 2D L-SplineFNO/FiLM adapter and gradient checks | Gate B |
| M3 Dataset release | Frozen envelope/splits, budgeted production, manifests and quality report | Gate C and dataset acceptance |
| M4 Training | Three L-SplineFNO deployment families with three seeds; checkpoint manifests and pinned architecture/config | Validation-selected models frozen |
| M5 Evaluation | ID/OOD/time results, failures, uncertainty and end-to-end timings | Independent checks reproduced |
| M6 Handoff | Reproduction commands, configs, data/checkpoint locations, report | Fresh small end-to-end reproduction succeeds |

The simulation and numerical agents must collaborate on M1–M2. Data production waits for the label and representation contract so expensive runs do not need repeating. The learning agent can work on small synthetic fixtures meanwhile. The coordinator can implement evaluation fixtures and split checks early, then verify actual results independently.

Required figures: geometry plus perturbation, native mesh and latent-grid coverage; sampled parameter space with splits; reference/predicted/error snapshots; error versus time by regime; selected operator errors; balance and stability diagnostics; accuracy versus total rollout cost. Required tables: data acceptance/failures by regime, numerical convergence, model size/training cost, and per-regime performance with uncertainty.

## 11 Validation checklist for implementing agents

- Unchanged upstream equilibrium regression still passes with any source instrumentation.
- Initialization preserves boundary values, normalization and dependent-variable consistency.
- Native data/export round trips preserve fields within the declared tolerance; no rasterization is required.
- Run relevant upstream 2D tests in `Tests/test_learnable_spline_fno.py` and `Tests/test_bspline_basis_convention.py`; record results against the actual dependency revision.
- Verify source/query independence, point-order equivariance, shared-mesh batch consistency, and correct layout conversion.
- Test learned-kernel gradients, finite sparse/low-coverage behavior, denominator safeguards, and constant reproduction only on the stated covered/unfloored domain.
- Check state sensitivity on a fixed mesh: two different current fields must reach the backbone as different features. Verify FiLM parameter plumbing and gradients; because FiLM initializes at identity, test output sensitivity after a controlled nonidentity modulation or a training step.
- Pointwise labels assembled with native basis/weights reproduce their corresponding selected residual blocks. Never equate residual coefficients with point values.
- Any coordinate or output rescaling is applied consistently to spline derivative diagnostics; physical weights remain independent of mesh sampling density.
- Selected terms plus retained remainder reconstruct the full spatial residual.
- Analytical manufactured-field checks exercise cylindrical derivatives, integration weights, boundary treatment, and coefficient placement.
- All-numerical assembly reproduces actual dynamics without reference-state injection.
- Split tests reject duplicate parent families and training/test normalization leakage.
- A fake JOREK process exiting zero with a GMRES-abort log is rejected; incomplete output is rejected.
- Restart/resume preserves identity and does not overwrite existing successful output.
- A tiny training fixture yields finite gradients and decreasing loss; finite-difference gradient checks pass where required.
- Evaluation includes failed rollouts rather than dropping them from aggregate accuracy.
- A clean environment can reproduce one short case, export, training smoke run and evaluation from documented commands.

## 12 Authority and completion rules

This handoff specifies future implementation work. It does not submit jobs, approve an unlimited compute campaign, or authorize publication of data. During execution, use any applicable existing authorization; do not ask for it again. Where resource authorization is absent, first prepare and validate scripts plus a concrete capped resource request, then obtain the required authorization before submission. The existing toolkit brief has an explicit HPC submission/physics escalation rule; it applies when working under that brief. Do not copy unrelated JAX or radial-profile conventions into this separate 2D PyTorch study.

Resolve routine script/layout choices autonomously. Escalate only unresolved material physics choices, an unmet required resource approval, or a verified blocker that would change the scientific experiment. Keep working on independent tasks while a decision is pending. Never bypass a failed numerical gate by loosening tolerances silently, dropping a difficult regime, or changing the case into a different PDE.

Completion means reproducible evidence, not a claimed positive outcome. Final delivery must state the actual valid domain, numerical/reference uncertainty, success and failure rates, measured performance, cost, and any unsupported claims. No claim of experimental MAST-U predictive fidelity follows from this synthetic fixed-equilibrium dataset.

## 13 Coordinator startup prompt

> Implement the MAST-U axisymmetric OpsSplit study described in this handoff using L-SplineFNO from the pinned Conditional_Fusion_Operator code for all primary neural models. Feed current physical fields on native unstructured source points, decode selected physical contributions at quadrature query points, and assemble them with verified numerical terms. Preserve the dependency checkout's local edits. Validate the field-input and FiLM paths; do not substitute the paper's geometry-only steady-state pipeline. Preserve the single-case scope and reuse the pinned JOREK example and existing paper code where verified. Start by reading repository instructions, auditing the environment, and creating the case/data/evaluation contracts. Assign the simulation, numerical/data, learning, and evaluation responsibilities with disjoint file ownership. Follow gates A, B, and C before production or training. Produce a validated all-numerical assembled rollout before replacing terms with neural operators. Keep a decision and resource ledger, preserve failed attempts, and report evidence at each milestone. Prepare bounded Slurm requests under applicable authorization. Finish with versioned data, checkpoints, reproducible scripts, and honest regime-by-regime performance results; do not expand to another device or 3D physics.

## 14 Source references

- OpsSplit paper arXiv:2602.23113v2 and existing `NOs_for_POs` sources listed in section 2.
- Attached *Learnable Splines for Unstructured Neural Operators*, arXiv:2610.22446v1, sections 3.1–3.3 for conditioning, learned scatter and analytic decoding. Its reported engineering benchmarks are not validation of this dynamical plasma application.
- [Conditional Fusion Operator repository](https://github.com/gitvicky/Conditional_Fusion_Operator), inspected through the matching local checkout at the revision recorded in section 2. Relevant sources: `Models/learnable_spline_fno.py`, `Models/factory.py`, `Models/conditioning/film.py`, `Models/region_routing.py`, `docs/learnable_spline_fno.md`, and the listed spline tests. The local implementation exposes 2D field inputs and independent queries; preserve its modifications and verify the selected revision during implementation.
- Pinned JOREK `reg_tests/testcases/grid_with_patches_mastu_303/settings.sh` and `input` for case identity and nstep_n=0 behavior.
- Pinned JOREK `docs/physics/base_fluid_models/RMHD/rmhd_model.md` for variable definitions and equations; verify actual enabled source terms against code.
- Pinned JOREK `docs/numerics/time-integration.md` for temporal operators and implicit discretization.
- Local toolkit `spec/reviews/jorek-j1-build.json`, `spec/reviews/jorek-j2-cost.json`, and `src/tkit/datasets/jorek_run.py` for installation verification, regression scope, and observed failure handling.
