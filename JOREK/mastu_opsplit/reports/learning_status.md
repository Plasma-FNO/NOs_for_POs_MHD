# Learning implementation and evidence — 2 November 2026

## Actual status

The current-field L-SplineFNO adapter, three neural family builders, direct-term supervision utilities, validated term-data loader, synthetic training smoke command and complete-state checkpoint functions are implemented. No scientific model has been trained. Gates A/B/C have not passed. No MAST-U accuracy, stability, parameter-budget equivalence or speedup result is claimed.

Scientific training remains blocked on a useful verified physical pilot, a native current-state residual evaluator, correct coupled temporal/constraint solve, verified native mapping, and an accepted split dataset. `scripts/train.py` checks A/B/C before reporting the missing native rollout backend. Its `--smoke` mode is explicitly synthetic and cannot satisfy a scientific gate. A successful raw model forward is not a verified autoregressive native-state projection or an autonomous whole-RHS rollout.

## Dependency identity and preservation

The adapter imports the existing repository at `/pitagora/home/userexternal/vgopakum/Conditional_Fusion_Operator`, commit `a3ca59bba100ca56e535fc35c32967bdeb04d8d9`. The exact hashes of all 24 imported `Models` Python sources are locked in `configs/lsplinefno.yaml`. Any changed locked source or different HEAD is rejected. Runtime/checkpoint provenance additionally records absolute imported backbone/FiLM paths, every imported source hash, complete dirty status, and the `Models` patch plus its hash. The checkout already had unrelated modified/untracked files; they were preserved. No source was copied or spline kernel rewritten. The dependency's MIT license remains at its `LICENSE` path, recorded in provenance.

Use `/pitagora/home/userexternal/vgopakum/Conditional_Fusion_Operator/.venv/bin/python` (PyTorch `2.10.0+cu126`) with `PYTHONDONTWRITEBYTECODE=1`. Default `/usr/bin/python` lacks PyTorch. The upstream test run disabled pytest's cache provider and did not edit its checkout.

The existing `NOs_for_POs/Utils/explicit_time.py` was inspected: its explicit time-steppers and raster layout do not implement the required model303 coupled native temporal solve. Reusing that pipeline would bypass Gate B, so this study does not call it. Upstream L-SplineFNO and FiLM are reused directly.

## Adapter and target semantics

`FieldSplineOperator` consumes finite current fields `[B,N_source,C_state]`, coordinates `[N_source,2]` or `[B,N_source,2]`, and independent queries in the same layout. It emits `[B,N_query,C_output]`. Physical R,Z coordinates use one stored affine bounds matrix `[axis,min/max]` mapped to `[0,1]^2`; each source/query cloud is checked against those same bounds. Physical bounds are stored initially in float64, so exact float64 mesh endpoints cannot round inward before validation. Normalized source/query coordinates are converted to current-field dtype/device only at the neural boundary. No trajectory-local coordinate scaling is fitted. Physical coordinates, inverse scales and volume weights remain available in the native data contract.

Whole-RHS and autoregressive builders use input FiLM on field channels only and require finite positive `s_perp` values `[B,1]`. Coordinates never enter FiLM. A controlled nonidentity FiLM test verifies parameter sensitivity and gradients through the MLP. Bare-backbone `mu` is not mistaken for enabled FiLM.

OpsSplit owns two independent spline modules named `density` and `pressure`. Both see the current seven-channel native sampled state when configured for model303. The scalar outputs are candidate strong-form `density_advection_compression` and `pressure_advection_compression` residuals. **Pressure output is not temperature tendency**: equation 6 advances `rho*T` and requires the native coupled temporal treatment. Coefficient-free selected blocks do not receive transport FiLM. Current state dependencies may be narrowed only after native block verification.

Sparse upstream scatter/decode remains unchanged. The primary architecture uses one scatter scale, zero encode refinement, enforced partition of unity, the coverage channel, and original denominator safeguards. Diagnostics separately count point-stencil floors, uncovered grid nodes, covered absolute-epsilon floors, and signed-cancellation floors. Sparse near-zero support can activate absolute epsilon even at analytic initialization; that is reported rather than weakened. Constant reproduction applies only on covered/unfloored nodes. No physical conservation or wall boundary condition follows from spline partition of unity.

`parameter_counts` reports disjoint spline/FiLM/FNO/other totals. The 64x64 pilot settings remain candidates; no matched primary model budget or latent-resolution choice has been validated.

## Native term data and direct supervision

`load_term_trajectory` reuses `data.read_trajectory`, including FE evaluation and residual identity checks, and preserves native coefficients separately. It flattens element/quadrature in row-major order, returns physical source/query coordinates, unchanged axisymmetric volume weights, time, bounds, `s_perp`, geometry hash, parent/split metadata and strong labels. Scientific loading rejects synthetic data and provisional native mappings. Diagnostics require `diagnostics_only=True`.

`iter_term_batches` checks parent/split/case identities with the shared split validator before yielding time batches. It holds at most one decoded trajectory at a time, requires one fixed geometry and rejects duplicate cases. Native exchange reading still loads one complete trajectory; streaming within a single very large trajectory is not implemented.

`term_supervised_loss` uses physical-volume weights and explicit positive training-derived label scales. `pretrain_term_epoch` runs direct-label supervision and finite-gradient checks; callers must fit preprocessing on training data only. State normalizers are not independently fitted here. This utility is not an autonomous native rollout trainer.

Checkpoints preserve model, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG state, step, configuration/dependency identity, hashes of eight local adapter/training/data/numerical/helper/entrypoint sources, and data split. Intermediate smoke checkpoints are written every five steps. Resume rejects changed identity/splits and restores optimizer/scheduler/RNG state; an interrupted/resumed smoke run reproduces uninterrupted final weights exactly. Writes use exclusive creation. Load only trusted local checkpoints, since complete optimizer/RNG state uses PyTorch pickle loading.

## Validation

Relevant upstream 2-D spline/basis tests: **23 passed, 57 deselected in 55.96 s**. Command (from study root, replace `$CFO` with the dependency path):

```
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH="$CFO" "$CFO/.venv/bin/python" -m pytest -q -p no:cacheprovider "$CFO/Tests/test_bspline_basis_convention.py" "$CFO/Tests/test_learnable_spline_fno.py" -k 'sparse_2d or dense_2d or column_index or splinefno_roundtrip or encoded_grid_is_smooth or kernel_initialises or scatter_partition or scatter_compact or explicit_source_query or encode_once or independent_query_path or gradients_reach_kernel_coeffs or density_floor or source_and_query_point_counts or encoder_preserves_constants or raw_stencil_sum or no_linear_solve_attributes'
```

Local learning/training suite after precision/provenance review: **11 passed in 17.28 s** (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src /pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_learning.py tests/test_training.py`). The original CFO environment had shared-filesystem import stalls during this final run; the installed alternate environment (PyTorch `2.7.1+cu126`) passed all checks. The redundant stalled process was sent SIGTERM; its shared-filesystem wait can delay signal delivery.

Local checks cover float64 physical mesh endpoints with float32 fields, local-source checkpoint hashes, fixed coordinate bounds, independent queries, source-order equivariance, shared batch equivalence, current-field sensitivity, FiLM nonidentity sensitivity, finite/nonzero gradients to state, FiLM, FNO and both axis kernels, finite differences for state and both axis-kernel coefficients, independent OpsSplit modules, sparse coverage safeguards, exclusive output writes, RNG/optimizer/scheduler resume, tiny training loss decrease for all three families, and native label batching/split rejection.

The CLI twelve-step synthetic whole-RHS fixture reduced loss from `0.21913614869117737` to `0.0796082615852356`. This is analytic plumbing evidence, not plasma accuracy. Its tiny model has 4,762 trainable parameters: spline 12, FiLM 4,548, FNO 148, other 54. Measured default-safeguard activations were zero on that fixture's 64 fully covered grid nodes. The six-point sparse adapter fixture activates absolute epsilon at nearly empty nodes; signed-cancellation floors remain inactive at initialization.

Synthetic checkpoint and complete provenance remain at `/tmp/mastu-opsplit-learning-smoke-20261102/{checkpoint.pt,smoke.json}`; these temporary fixtures are not release checkpoints. Reproduce with a fresh output path:

```
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$CFO/.venv/bin/python" scripts/train.py --config configs/train.yaml --model-config configs/lsplinefno.yaml --output /tmp/mastu-new-smoke --smoke
```

`--dry-run` rendered the intended action without launching training; `bash -n slurm/train.sbatch` passed. No scheduler job was submitted. The upstream 2-D Fourier layer hardcodes complex64 spectral buffers; a full-double neural FNO path is not validated. Float64 physical coordinates with float32 model fields are supported without losing boundary precision. Native rollout gradients, model303 physical trajectory overfit, real normalization, three-seed scientific training, latent-resolution selection, budget matching, uncertainty and timing comparisons remain unvalidated.


Update 2026-11-04: the full native subprocess OpsSplit rollout and five weak-output NO blocks are implemented. See [full_rollout.md](full_rollout.md) for the current implementation, actual validation evidence, and remaining gates. The legacy two-block experiment and in-process backend are separate. Scientific training has not started.
