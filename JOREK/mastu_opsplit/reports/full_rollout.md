# Full native OpsSplit rollout — implementation and validation scope

**5 November status correction:** the full-window perturbed SBDF2 runs failed
temperature positivity at dt64, dt128, and dt256. The independently refined
unmodified reference passed (maximum dynamic-normalized difference 0.329635%).
The later matched restart from `seed_dt256_full/jorek000528.h5` completed 96
steps with Euler and unmodified JOREK; SBDF2 failed again. This isolates a
split/integration problem but does not prove full-window Euler convergence.
Results are preserved in `/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_split_fix_20261105`.

The native variable-step mass predictor/history code has been corrected for
JOREK's already-rescaled previous increments. This does not explain the
constant-step failures. A compiled Fortran regression exercises the production
weights and injected mass predictor with alternating timestep ratios.

The user subsequently selected the fluid-benchmark normalized-field Euler
formulation. See `configs/normalized_euler.yaml`. The physical-RHS replacement
below remains a separate experiment; its nonlinear-term-label and native-split
acceptance gates do not apply to trajectory-supervised normalized-field Euler.
No scientific model has been trained. Statements below describing full-window
checks as running are historical; those runs have now failed as stated above.

Implementation: `native/full_split_procedures.f90`, `scripts/prepare_full_split.py`,
`src/mastu_opsplit/full_rollout.py`, `scripts/run_full_split.py`, and
`scripts/infer_full_split.py`. `build_family("opsplit_full", ...)` constructs the
complete model; the old `opsplit` family remains the legacy two-block experiment.

For the installed axisymmetric MAST-U model303 pilot, the fixed common warm state
is q*. The common-baseline spatial tangent, with native transport and source
coefficients held fixed during differentiation, forms a spatially varying but
**fixed linear operator L**. Native complex-step differentiation supplies its
coefficients; actual gradients still use the native bicubic geometry map. The split is

    M(q) dq/dt = F(q*) + L(q-q*) + N(q),
    N(q) = F(q) - F(q*) - L(q-q*).

Centering preserves the near-cancelling equilibrium current/source balance and
keeps that static balance out of the learned residual. L contains resistive
current coupling, perpendicular viscosity, density and temperature diffusion
tensors, parallel velocity viscosity, and the baseline magnetic, pressure and
advection couplings. Transport-only L was tested and retained as diagnostic
evidence; it leaves fast MHD couplings explicit and is not the final candidate. The bicubic native coefficients and
physical geometry map evaluate every linear gradient. The NO input is the
current seven-field state on the native quadrature points. Five independent
L-SplineFNOs predict complete nonlinear **volume** forcing for induction,
perpendicular momentum, density, pressure, and parallel velocity. Temperature
and field-dependent transport departures and source-profile changes are included.

Each block supplies three pointwise coefficients multiplying the test value,
R derivative, and Z derivative. They use the toroidal volume measure
2*pi*R*dR*dZ; derivative coefficients are **negative physical fluxes**.
They are not assembled residual coefficients or pointwise time derivatives.
Auxiliary current/vorticity rows, state-dependent mass couplings, and physical
boundary/sheath conditions are solved by native JOREK. Those known temporal and
boundary operators retain their nonlinear algebra; they are not learned.

The numerical and learned modes share an explicitly selected time scheme.
`scheme: euler` uses the coupled first-order IMEX update:
current-state native temporal/constraint matrix minus dt*L, with N explicit.
The original nonlinear **volume Jacobian is disabled**. Native boundary equations remain intact. Their implicit theta weight is set to
match the selected scheme: one for Euler/BDF2 and one half for CN–AB after
startup. Earlier diagnostic variants retained the pilot boundary theta; they
are not the final validation candidate.
`scheme: cnab2` uses a Crank–Nicolson linear step, Adams–Bashforth extrapolation
of N, and an extrapolated midpoint native mass. It starts with one Euler step;
its q and N histories come only from its own preceding step. A fresh restart
reinitializes this history (there is no claim of bitwise multistep restart).
A manufactured variable-mass test verifies second-order convergence of this
formula; MAST-U accuracy still requires native trajectory evidence.

`scheme: sbdf2` supplies the damped implicit BDF2 alternative. For a timestep
ratio r=h/h_previous it uses a0=(1+2r)/(1+r), a_history=r^2/(1+r), N extrapolated
as N_n+r*(N_n-N_previous), and M evaluated at q_n+r*(q_n-q_previous).
The native mass-history action is added only to evolution rows; algebraic
constraint rows retain their original form. The first step uses Euler. Its
manufactured variable-mass convergence test also passes; native validation is
still required. CN–AB shows large current/vorticity errors at the coarse pilot
step, so it is not accepted merely for being higher order.

These are additive IMEX splits, not claimed Strang factorizations. Changing the
integrator requires its own timestep study; previous Crank–Nicolson convergence
is not evidence for the new update.

The patch rejects non-axisymmetric modes, nonzero diamagnetic terms,
Taylor–Galerkin and hyperdiffusion terms, bootstrap, NEO, pellets, nonzero history
zeta, or quadrature other than the supported Q4. Extending to another physics
configuration requires extending and revalidating this implementation. Native
Q8 sampling is used independently for trajectory errors and positivity.

The full tangent is a 15-by-21 local map from the seven field values and their
R/Z gradients into the five weak forcing blocks. Its independent centered
finite-difference test passes 256 probes with maximum relative error 1.19e-10.
The native implicit assembly uses this same frozen map for trial functions.
The unused original nonlinear volume Jacobian is not evaluated in the optimized
rollout path; numerical parity is tested against the unoptimized assembly.

Native modes:

- `export`: one original native step, records current-state physical coefficients
  and transport at every element/query; a runtime identity checks against the
  untouched native RHS expressions before any replacement.
- `recombine`: replaces the complete volume RHS through L+N, retaining the
  original native Jacobian for **algebraic parity testing only**.
- `numerical`: complete numerical N, native M/constraints, fixed implicit L.
- `neural`: a checkpoint predicts N from the solver's own current restart every
  step. A stamped, finite, geometry-checked table replaces all five volume RHS
  blocks. No reference trajectory is fed back into the rollout.

`run_full_split.py --config CONFIG --output OUTPUT [--dry-run]` requires a frozen
prepared case, initial restart, frozen operator, mode, dt, steps and nout. Neural
mode additionally requires a `mastu_full_opsplit_v1` checkpoint. It runs inside a
four-rank Slurm allocation with the prepared JOREK environment sourced. The
checkpoint stores the five-block model/config, input/output normalization,
frozen operator SHA256 and transport parameter. Output directories are exclusive;
failed runs and logs remain available. No scientific checkpoint exists yet.
The native process solve is not currently differentiable through time; Torch
native evaluation and weak assembly are differentiable. Term-supervised training
and inference are supported by the model interface, not an end-to-end adjoint.

Validation artifacts live at
`/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_full_opsplit_20261104`.
The first attempts exposed cancellation in local identity checks; they are
preserved. No convergence or scientific training gate is passed by component
tests, successful compilation, or recombination alone. Required evidence:

1. Complete same-state weak residual identity and recombination trajectory parity.
2. Seed/control numerical rollouts at three timesteps, errors normalized by
   field temporal variation, including control-subtracted responses.
3. Q8 density/temperature positivity, native boundary/constraint solve, and
   agreement with the independent reference over the chosen dataset window.
4. Separately generated train/validation/test parent families before model
   comparisons; one pilot and adjacent frames are not independent families.

AR and NODE must use the same data splits, predicted state, physical evaluation
weights, constraints, horizons, and comparable parameter budgets. Report both
accuracy and rollout cost; no advantage over either baseline is established.

## Numerical evidence, 5 November 2026

The final candidate uses the full fixed baseline spatial tangent and SBDF2.
Twenty-four distinct component tests pass across the validation runs (22 rerun after the neural normalization change; the two unchanged native temporal-action tests passed previously). The native tangent passes 256 independent
finite-difference probes (maximum relative error 1.18331e-10). Complete native
recombination parity is 0.000991849% maximum dynamic-normalized error;
optimized temporal/constraint assembly differs from its unoptimized counterpart
by at most 0.000188301%. These checks do not replace trajectory validation.

On the 4.217739–4.334899 microsecond short window, the final build gives:

| Comparison | Maximum dynamic-normalized error |
| --- | ---: |
| SBDF2 7.322464 ns versus 3.661232 ns | 1.23398% |
| SBDF2 3.661232 ns versus 1.830616 ns | 0.301634% |
| SBDF2 1.830616 ns versus native reference at the same step | 0.942725% |
| Native reference 3.661232 ns versus 1.830616 ns | 0.982432% |

The maxima include all seven fields, control states and control-subtracted
responses over nine matched frames. Q8 density and temperature are positive.
Errors are normalized by temporal variation, not the large static equilibrium.
The finest pair and independent reference pass the preselected 1% field gate;
the coarsest pair does not. The full-window check remains separate.

Transport-only implicit L was insufficient: Euler remained at 3.00553% reference
error, CN–AB became unstable, and SBDF2 still had 7.86920% medium/fine discrepancy.
Those attempts and failures remain preserved. Adding the baseline magnetic,
pressure and advection couplings to L produced the accepted short-window result.

The final three-step neural/native handoff also passes. An **untrained
zero-residual** checkpoint gives exactly the numerical first step at q*, and
three distinct current-state hashes prove subsequent inference reads each
solver state. This tests the interface, not learned accuracy.

Full-window seed/control runs use 192/384 steps (7.322464/3.661232 ns) over
4.217739–5.623652 microseconds, with 97 matched frames. Their jobs are recorded
in `full_window_jobs.json`; analysis is `assess_full_window.py`, which checks
all seven field groups, Q8 positivity and energy errors against the original
independent native reference. Results are written to `full_window_report.json`.
They are running; no full-window pass is claimed yet.

Relevant machine-readable evidence in the artifact directory:
`final_short_report.json`, `final_interface_report.json`,
`fast_parity_report.json`, `component_tests_tangent.xml`, `tangent_test.txt`,
`implementation_sha256.json`, and the frozen candidate's
`full_split_protocol.json`. The original installed source and executable remain
unchanged. `frozen_tangent/diagnostic_correction.json` records a corrected
uncentered-linear HDF diagnostic; its old transport-only values were retained,
and the frozen runtime operator did not change.

No scientific model has been trained. Independently generated and fully labeled
train/validation/test parent families remain necessary before the AR/NODE
comparison. The current subprocess rollout has no native-solve adjoint; an
end-to-end rollout-loss trainer is not implemented.

## Additional checks while the full window runs

The zero-volume-remainder ablation uses the same SBDF2 step (3.661232 ns),
initial states, native mass and boundaries, but sets N=0 through an untrained
zero-output checkpoint. Over the short window it gives **68.0614%** maximum
dynamic-normalized error versus the complete numerical remainder. The largest
error is control current density; control vorticity is 23.5508%, control flow
16.3763%, and the temperature response 10.9296%. Density and temperature stay
positive. Thus the nonlinear volume remainder matters for this pilot; this is
not evidence that a trained NO will learn it accurately or beat AR/NODE.
The complete results are in `zero_ablation_report.json`.

Native states are now centered/scaled in float64 before conversion to neural
float32. Output denormalization also uses float64. A regression verifies that
small evolution on a large background survives this conversion. Existing
zero-remainder runs finished before this change; pre-change source identities
remain in `implementation_sha256_pre_precision.json`. A separate real native
handoff check uses the updated normalization.

A completed 29-frame prefix through 4.627797 microseconds has 0.246841%
coarse/fine split discrepancy, but 2.07128% error against the older native
reference at 7.322464 ns. This is not a full-window verdict and is not hidden
by the accepted short result against the finer reference. Two new native
reference runs use the unchanged JOREK executable at 3.661232 ns for all 384
steps. `refined_reference_jobs.json` records those jobs. The final comparison,
including independent native-reference refinement, writes
`full_window_refined_report.json` and `full_window_refined_summary.md`.
The initial comparison with the older reference remains separately preserved.

## Full-window failure and current refinement

The 7.322464 ns perturbed run failed at the native positivity check when
advancing step 152, after temperature became negative near 5.32 microseconds.
`runs/seed_dt64_full/{result.json,stdout.log,stderr.log}` preserve the failure.
The short-window convergence result does **not** establish full-window stability.
No full-window numerical acceptance or training release is claimed.

The full convergence pair is now 3.661232/1.830616 ns (384/768 steps, with the
same 97 output times), against the independent native reference at 3.661232 ns.
`fine_full_window_jobs.json`, `refined_reference_jobs.json`, and
`fine_full_analysis_job.txt` identify the active work. Automatic analysis writes
`full_window_fine_report.json` and `full_window_fine_summary.md`, including failed
members, coverage, energy, positivity and all field/response errors. Training
cannot be launched by this pipeline. The earlier coarser-pair analysis and
superseded plans remain preserved in `validation_replan.json`.

The updated float64-normalization neural/native interface passes its actual
three-step test (`precision_interface_report.json`); its first step still matches
the numerical baseline exactly. This is an untrained interface check only.
