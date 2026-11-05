# Composite JOREK OpsSplit training

This is the lightweight ES/EM experiment following the November 5 representation checks. It reuses the matched trajectory selections, valid windows, normalization, rollout objective, checkpoint selection and evaluation in `train_comparison.py`. Existing campaigns are unchanged.

## Model and numerical method

The observable state is `[rho,T,potential,omega]` in ES and additionally `[psi,j]` in EM. The adapter converts derivatives to common-native units. A six-layer fluid FNO is shared structurally across cases; EM adds a two-layer correction FNO. Both use width64/modes32 in the paper protocol. Projection outputs start at zero so the initial model contains only fixed linear dynamics.

The neural auxiliary outputs evolve the defects `omega - Delta_c(u)` and `j - Delta_star(psi)`. Fourth-order fixed stencils reconstruct the interior; the outer four sampled cells have zero reconstruction and therefore fully learned auxiliary values. Initial defects come from the input state, never from future targets. Defects can capture sampling/closure error and model mismatch; these are composite learned dynamics, not independently identified physical PDE terms.

Particle diffusion and EM resistivity use backward Euler with fixed cylindrical finite-difference operators. A precomputed separable eigenbasis gives a batched differentiable solve; no density-dependent momentum inversion or sparse solver is used. Physical time is the audited nominal common-native interval5 per saved frame. Neural increments use saved-frame time. Density and temperature now use softplus coordinates, followed by implicit density diffusion. Other fields use additive increments. The unguarded additive and exponential thermodynamic versions failed; their snapshots and failed jobs are retained.

For a positive native field `q`, reference `s = fluid_scale * native_factor`, and dimensionless neural increment `a`, the update is `q_new = s * softplus(softplus_inverse(q/s) + a)`. The inverse uses `x + log(-expm1(-x))`; the forward uses `logaddexp(0,x)` in float64. This behaves logarithmically near zero and additively for large fields, with `0 < d(q_new)/d(a) <= s`. It removes the exponential feedback when network rates grow with the state. No clipping, floor, validation-fitted scale or physical unit change is introduced. Source fluid scales transfer with the shared network. Extreme negative coordinates can still underflow, so runtime positivity checks remain mandatory.

`CompositeSplit.forward` now accepts and returns exported fields in float64. These uncentered fields are the recurrent state, including between internal substeps. Centered train-only normalization is applied only to FNO inputs and the shared rollout loss. Carrying normalized fields and then undoing their centering can erase a tiny positive temperature by cancellation, even in float64. Existing weights and physical coefficients remain compatible, but old callers that passed normalized fields must use exported fields; immutable campaigns retain their original interface.

The outer four cells evolve through neural rates and supply the sampled-interior solve's boundary values. They are not clamped to physical walls. This explicitly delegates the unresolved wall-strip dynamics to the network. The implicit linear branch is stable, but neither it nor the auxiliary representation guarantees nonlinear stability. Nonfinite fields and nonpositive density or temperature are hard failures; fields are not clipped.

## Training and gates

Training uses the existing ES100/EM76 common datasets, first50 frames, five predictions, effective batch20, 250 epochs, Adam1e-3, weight decay1e-4, StepLR50/gamma0.5, and gradient clipping1. No term pretraining or anchoring enters this arm. All eligible windows are traversed each epoch. Evaluation retains common cohorts, failures, persistence and magnetic-perturbation errors.

Each case has an immutable code snapshot and this dependency chain:

1. GPU smoke: two full optimizer batches, one 49-step validation prefix, checkpoint save and bounded test evaluation.
2. Exhaustive initialization audit: every eligible five-step training window, worst-window gradient per training trajectory, and all eligible 49-step validation prefixes.
3. Bounded optimizer check: twelve complete ES epochs or twenty-six complete EM epochs, with all eligible 49-step validation prefixes every epoch. EM's longer check exceeds its later epoch15 failure. Success saves `train/stability.json` and the checked optimizer state.
4. Full training resumes that state only after the optimizer check succeeds. Training checks code/data/initialization identity, audit coverage, stability coverage and the initial continuation checkpoint hash. Probe epochs count toward the same 250-epoch budget.

Full runs checkpoint and requeue at the existing runtime budget. A failed model must not remove additional source-valid windows. Successful checks do not certify every later optimizer iterate; runtime failures stay failures. Validation failures save `failed-validation.pt` and `failure.json`, including the trajectory, rollout step, update stage, failing field and grid location.

## November 5 failure diagnosis

CPU replay completed the remaining optimizer batches from the original saved ES epoch3 and EM epoch7 states. ES `jorek_run0298` failed at step48: neural density/temperature rates reached approximately 182418/286153 and overflowed before diffusion. EM `jorek_run083` failed at step33: its positive pre-diffusion density reached `5.10e38`, and cancellation in the eigensolve produced a negative minimum `-1.82e23`. These are CPU reproductions, not bitwise replays of the original GPU failures. Both show runaway exponential feedback; linear time-step stability does not prevent that feedback or guarantee floating-point positivity at extreme dynamic range.

With identical replayed weights and the softplus update, all 42 ES and 11 EM validation trajectories completed49 steps. Mean relative L2 was0.6051 ES and0.07839 EM. ES accuracy at these already degraded weights remains poor; the replay establishes removal of the observed numerical failure, not forecast quality. Fresh optimizer checks remain necessary before continuing full training. Reports: `reports/composite-es-fixed-replay.json` and `reports/composite-em-fixed-replay.json`. `diagnose_composite.py` reproduces the old model in its immutable snapshot; `check_composite.py` checks corrected rollouts at those weights.

The fresh GPU campaigns subsequently passed exhaustive audits (4484 ES/3374 EM windows, 100/76 gradient checks, 42/11 full validation prefixes) and twelve optimizer epochs with full validation every epoch. Best probe validation relative L2 was0.05965 ES and0.05870 EM, versus unchanged-state baselines0.07744 and0.09189. Full training resumed from these checked optimizer states as jobs2780392 ES and2780398 EM. The27 numerical/trainer regression tests also passed. The campaigns retain the original250-epoch budget; later-epoch stability remains monitored rather than guaranteed. Evidence and job locations are in `reports/composite-stability-setup.json`.

### Later EM failure: affine cancellation

EM job2780398 subsequently failed during its fifteenth epoch (zero-based14), on `jorek_run083` at rollout step44. Exact GPU replay of `failed-validation.pt` showed the step43 softplus update, diffusion and auxiliary reconstruction all retained positive native temperature `1.7145207234131832e-20`. This is exported temperature `8.513283719467987e-16`; subtracting its normalizer center `13.90527057647705` loses it because it is below half an ulp of that center (`8.881784197001252e-16`). Decoding the centered state produced zero. This was affine cancellation, not softplus underflow or a failed diffusion update.

Carrying uncentered exported fields fixes the representation loss without a floor, clipping, changed physics, or changed network/loss normalization. All11 validation trajectories at the exact failing weights then completed49 steps; their mean relative L2 was0.28517. This degraded checkpoint is a numerical regression case, not a claim of predictive quality. The31 regression tests include repeated tiny-temperature updates, gradients, internal substeps, and the stronger EM continuation gate. Trace and replay evidence: `reports/em-temperature-trace.json` and `reports/em-temperature-fixed-replay.json`.

Only EM is relaunched in `em-physical`, with a fresh seed907 initialization, exhaustive audit and26-epoch optimizer check before continuation. Existing ES job2780392 uses its untouched immutable code snapshot. The longer check counts within EM's original250-epoch budget. Later optimizer iterates and genuine floating-point underflow remain guarded rather than certified in advance.

The corrected EM campaign passed the exhaustive audit and all26 optimizer/validation epochs. Epoch15 reproduced the old training loss and completed validation with relative L2 of0.28517; the error subsequently recovered to a new best0.05343 at epoch26. Every eligible trajectory remained finite, with strictly positive density and temperature throughout the49-step validation rollouts. Full EM training resumed as job2780704 only after the26-epoch gate passed; ES job2780392 remained running unchanged. Full evidence, source checkpoint identity and job IDs: `reports/em-temperature-recovery.json`.

### EM epoch106 diagnostic: true underflow after runaway learned dynamics

The `em-physical` campaign completed105 epochs, then job2780704 failed during epoch106 validation on `jorek_run020`, step49. The first zero temperature occurred in the neural update at `[batch,Z,R]=[0,71,46]`, before diffusion or auxiliary reconstruction. Best validation remains the epoch51 checkpoint (GPU mean relative L2 `0.0466621`); validation had already spiked to `4.46228` at epoch102 and `0.935602` at epoch105.

A read-only CPU replay restored every checkpoint tensor exactly and checked frozen code/data identity. All11 trajectories completed49 steps at best, last and failed-validation weights, with mean relative L2 `0.0466646`, `0.1602481` and `0.3332922`. The latter two represent partial/final epoch106 weights, not the completed epoch105 history entry. CPU and GPU trajectories diverge at the unstable weights; this is not an exact replay of the GPU failure. At failed weights, CPU `jorek_run020` error was `3.014747` and normalized temperature/current input magnitudes reached214/224 by step49.

Extending that same CPU trajectory reproduced zero temperature at step54, with33 zero cells. At the most negative coordinate, positive native input `0.4372508` and reference `0.000280036` gave coordinate `1561.4092`. The shared fluid contribution was `+0.0738013`, while the EM correction was `-2743.7515`, producing coordinate `-1182.2685`. An80-digit Decimal evaluation gives approximately `9.8752e-518`, below float64's smallest positive value `4.9407e-324`; float64 returns zero. The input passed positivity checks, and failure occurred before this step's linear solve or auxiliary reconstruction. This confirms true underflow in the CPU reproduction, with substantial runaway dynamics already present. No source accuracy is claimed beyond the49-step validation prefix. Auxiliary feedback and boundary closure have not been causally isolated.

The five-step training objective and parameter-gradient clipping do not bound long-rollout neural rates. The earlier26-epoch pass cannot certify later iterates. EM should remain stopped pending stabilization of the learned EM update and coupled feedback, assessed with longer rollout/error/range checks; adding a temperature floor alone would conceal the failure. The epoch51 checkpoint is a candidate baseline, not a long-horizon stability certificate. No model/training change or restart was made for this diagnostic, and ES was untouched. Exact GPU replay is queued as job2783710, pending `QOSGrpNodeLimit` at report time.

Evidence: `reports/em-epoch106-diagnosis.json`, `reports/em-epoch106-cpu-restored/summary.json`, `reports/em-epoch106-cpu-extended/jorek_run020.json`, and `reports/em-validation-history.png`. Reproducer: `diagnose_em_underflow.py`; `--steps 64 --checkpoint failed-validation.pt --trajectory jorek_run020` extends the failing checkpoint without training.

Render a campaign without submitting it:

```bash
/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python \
  JOREK/zenodo_opsplit/orchestrate_composite.py \
  --data /path/to/existing/case/data --output /path/to/new/composite-campaign
```

Add `--submit` to launch the gated chain; `--smoke-only` limits the campaign to its optimizer smoke. `jobs.json` records submitted job IDs. `setup.json` records the data identity, protocol and limitations.

## EM z-score normalization trial

The separate `em-zscore` experiment uses `(x - mean) / population_std` for all six exported fields. Statistics are fitted in float64 over each eligible training frame and spatial cell once, using the union of training windows. Validation/test frames and repeated overlapping windows do not affect the fit. Constant channels use scale1. `prepare_zscore.py` saves the statistics and source provenance in a new metadata directory, links the unchanged source cache/term files, and leaves existing datasets and campaigns unchanged.

The model uses these means/scales for both FNO inputs and the rollout loss. The scales also determine additive neural output units and the rho/T softplus reference, as in the existing composite architecture. Thus this is an ablation of normalization and its associated neural step scaling, not only input preprocessing. Recurrent exported states stay uncentered float64; physical unit factors, linear coefficients, auxiliary reconstruction, positivity guards, architecture, seed and optimizer stay unchanged. Standardization does not remove skewness or bound outliers. Z-score relative losses must not be compared numerically with the prior min-max losses; decoded R-weighted relative L2 remains comparable.

The trial keeps250 total epochs and requires110 optimizer epochs with all11 eligible49-step validation prefixes checked every epoch, exceeding the prior epoch106 failure. A fresh smoke check and exhaustive initialization/gradient audit precede this probe. Full continuation resumes the same checked optimizer state only after the110-epoch gate succeeds. The gate establishes numerical completion through those iterates, not long-horizon accuracy or stability of future iterates.

Campaign data and checkpoints use `/pitagora_scratch/userexternal/vgopakum/jorek-composite-zscore-20261105/em/`, because work/home report full quotas. Reproduce data preparation with `prepare_zscore.py --source <matched-em-data> --output <new-data-directory> --stability-epochs 110`, then use `orchestrate_composite.py` with that derived directory. Both launcher and continuation guard read `protocol.composite_stability_epochs`; existing campaigns retain their default12 ES/26 EM gates.

## Frozen ES-to-EM transfer trial

The transfer trial initializes the shared fluid branch from the best ES `es-softplus` checkpoint, selected at epoch70 with ES49-step validation relative L2 `0.04378459235`. `prepare_transfer.py` checks the source checkpoint/data identity and rejects any overlap between ES training/selection geometry families and held-out EM families. The new EM dataset metadata references the unchanged original min-max cache and records the checkpoint hash. Existing datasets and campaigns remain unchanged.

The copied fluid weights, input normalizers and positive-coordinate reference remain frozen. The two-layer six-field EM correction starts with a zero output projection and is the only trainable branch. Its input/loss normalization uses the original EM min-max statistics; EM linear operators, coefficients, boundaries and auxiliary reconstruction remain the target's. Although the source checkpoint used the older centered recurrence, only its neural weights and normalization transfer: the EM target retains uncentered exported float64 recurrence and strict positivity checks. Architecture, native units, frame interval and grid compatibility are checked before use. The optimizer contains only trainable parameters, and resume reinstates freezing before restoring optimizer state.

The EM correction uses Adam1e-4, with the existing batch/window/scheduler settings and250-epoch target budget. Smoke and exhaustive initialization/gradient audits precede a110-epoch optimizer check with all11 eligible49-step validation rollouts each epoch. Full continuation resumes that same optimizer state only after the check passes. This is additional EM training after the ES campaign, not a matched total-compute or sample-efficiency claim.

Before EM training, the frozen transfer completed all11 EM49-step validation trajectories on CPU with mean relative L2 `0.06762092208`. This checks transfer initialization, not trained-GPU stability. There are13,442,116 frozen fluid parameters and4,496,198 trainable EM parameters. The transfer regression checks exact fluid weight/normalizer preservation through training/resume, unchanged target resistive operators, zero initial EM output, checkpoint identity and held-out-family separation. Evidence: `/pitagora_scratch/userexternal/vgopakum/jorek-composite-transfer-20261105/em/initial-validation-cpu.json`. The separate campaign resides beside that report in `campaign/`.

## Comparison and transfer limits

These first campaigns train ES and EM from scratch. `CompositeSplit.transfer` can copy the shared fluid weights and their source normalization to an EM model while preserving target linear coefficients, with optional freezing. The separate frozen-transfer trial above exercises that interface; a budget-controlled comparison remains necessary to establish sample-efficiency benefits.

The additional EM network makes the EM model larger than the existing six-layer AR/NODE baselines. Their data and objective match, but capacity does not. Parameter counts are recorded; do not present the existing EM comparison as parameter matched.

The implicit linear step, softplus thermodynamic increments, and learned auxiliary defects are declared adaptations from the original NS Euler formulation. Original release timestamps and the generating revision remain unresolved. Tests validate the linear solve against an independent dense system, gradients, stiff-mode damping, unchanged learned boundaries, auxiliary-defect preservation, transfer interfaces, positive-coordinate sensitivity, and training/resume/audit/stability wiring.
