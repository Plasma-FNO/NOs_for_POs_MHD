# Physical continuity branch contribution, seed 17, update 554

Slurm diagnostic job 2791752 completed successfully, exit 0:0, in 5m04s.
It used an immutable copy of the running Physical OpsSplit checkpoint at update
554 of 2,000. The comparison jobs, checkpoints, configuration and frozen sources
were not modified. The diagnostic decomposition and fixed-weight rollout ablation
passed a native rectangle integration test before the actual-mesh run.

Full numerical results and the copied checkpoint:

```text
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_ensemble80_comparison_20261106/linear_diagnostic_20261106
```

The probe uses 16 four-step windows from eight training trajectories, beginning
at frames 0 and 60. Training-only normalization and background are reverified
against the running experiment. No held-out score is used. Norms are native
volume-weighted RMS, pooled equally across the 16 starting states. Values are
normalized field increments per saved observation interval, not raw coefficient
Euclidean norms. The verified physical interval is approximately 0.03125 native
code-time units, applied exactly once.

## Branch magnitudes

| Field | Neural RMS at update 554 | Physical linear RMS | Linear/neural RMS | Linear/observed increment RMS |
|---|---:|---:|---:|---:|
| Density | 0.006305656 | 0.001845763 | 29.27% | 13.47% |
| Temperature | 0.011557774 | 0.001212434 | 10.49% | 3.014% |
| Other five fields | varies | exactly zero | zero | zero |

The branch is active and is not negligible within density/temperature. Combining
those two channels gives a linear/neural RMS ratio of 16.77%. Combining all seven
raw neural outputs gives 0.580%; that latter figure includes large potential
proposals subsequently changed by elliptic closure, so it must not be interpreted
as the fraction of the final guarded prediction attributable to physics.

At initialization, density and temperature linear/neural ratios were 2.188% and
1.110%. Physical RMS barely changes during training; the ratios grow because
the learned neural increment shrinks. The physical branch uses shared learned
scatter kernels, so its operator is linear in state at fixed network weights,
but is not entirely fixed across optimization updates.

The one-step guarded difference between full and linear-disabled proposals
retains essentially 100% of the linear RMS in both active channels. No limiter
interventions occurred on these starting-state probes. The guards are not
suppressing the observed physical contribution.

## Fixed-weight four-step rollout ablation

Both rollouts start from the same state, evolve autonomously for four intervals,
and apply the same guards at every step. The ablation removes only the physical
linear term while keeping trained neural weights unchanged. Reported errors are
normalized volume MSE without the auxiliary positivity penalty.

| Error | Full Physical OpsSplit | Linear term removed | Relative change on removal |
|---|---:|---:|---:|
| All-field mean | 0.00851533798 | 0.00851569079 | +0.00414% |
| Density | 0.00166698403 | 0.00169537505 | +1.703% |
| Temperature | 0.00805882319 | 0.00803362140 | -0.313% |

Removing the branch worsens density error but slightly improves temperature
error on these windows. Those effects nearly cancel in the aggregate. Moreover,
u and w account for 66.4% of full-model squared error here; density contributes
only 2.8%. The partial physical branch has no direct u or w output. Small changes
to other fields arise through neural coupling over multiple rollout steps.

These observations explain why aggregate training curves can overlap despite
a measurable physical contribution. They do not demonstrate that the linear
branch improves generalization, nor do they establish a causal comparison with
a retrained NODE at the same update. The ablation tests dependence of the current
trained model on its own linear branch.

## Native-gradient reference

Replacing only the learned spline perturbation gradients with exact native
bicubic gradients gives linear RMS of 0.002335721 in density and 0.001620360 in
temperature. The implemented linear increment has relative RMS differences of
37.46% and 40.06% against that native-gradient reference. Learned perturbation
gradient relative errors are approximately 37.3% for psi, 34.6% for u, 97.9% for
density, 64.0% for temperature, and 77.8% for Vpar.

This is evidence of spatial approximation error in the coarse learned spline
representation, not a missing physical multiplier or an accidentally disabled
branch. The native-gradient comparison retains the same partial continuity
tangent; it is not a full JOREK Jacobian or a replacement model validation.

Not checked here: final checkpoints, other seeds, long autonomous rollouts, or
held-out generalization. No architecture, physical amplitude, optimizer, budget,
or checkpoint selection was changed in response to these results.
