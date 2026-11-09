# Native exchange contract — provisional v1

This contract is executable through `mastu_opsplit.data` and `scripts/export_dataset.py`. It validates native-array structure and algebraic consistency. It is **not** an approved JOREK dataset schema or evidence that Gate B passed. Non-synthetic exchange metadata must state `acceptance_status: provisional_exchange`; production release acceptance is a separate numerical/physical gate.

## Geometry

One HDF5 geometry per fixed mesh, with schema `mastu-native-v1` and SHA-256 over each named array, shape, dtype and bytes. Each trajectory stores the same hash and a geometry reference. The initial source and query sets are the identical native quadrature set; row-major flattening of element then quadrature establishes point order. Shared element boundaries do not cause quadrature ID duplication.

| Array | Shape | Meaning |
|---|---|---|
| `dof_map` | `[element,local_dof]` | Zero-based global native coefficient indices, including derivative coefficients |
| `basis` | `[element,quadrature,local_dof]` | Physical native basis evaluation, with derivative DOF scales included |
| `basis_grad` | `[element,quadrature,local_dof,2]` | Physical R,Z derivatives of test basis |
| `quadrature_rz` | `[element,quadrature,2]` | Physical R,Z in metres; flattened source/query coordinates |
| `poloidal_weights` | `[element,quadrature]` | Positive `abs(Jacobian)*Gauss_weights`, m² |
| `bounds` | `[2,2]` | Fixed physical mesh bounds, axis rows `[min,max]` |
| `dof_ids` | `[global_dof]` | Stable unique coefficient IDs |
| `element_ids`, `patch_ids` | `[element]` | Element identity and mesh patch |
| `boundary_labels` | `[global_dof]` | Native boundary label per value/derivative coefficient |
| `source_ids`, `query_ids` | `[element*quadrature]` | Unique sample identity |

Physical volume weights are `2*pi*R*poloidal_weights`. They are independent of spline coverage density. Arrays remain sparse/local; no dense point-by-grid or global Gram inverse is built. Actual constrained/refined native nodes need effective basis/elimination mapping verified before this contract can represent them. The initial exchange format deliberately does not claim to handle arbitrary hanging-node constraints.

## Per-trajectory arrays

| Array | Shape | Meaning |
|---|---|---|
| `time_seconds`, `time_code` | `[time]` | Strictly increasing actual timestamps |
| `native_state` | `[time,global_dof,7]` | Canonical native FE coefficients |
| `state_samples` | `[time,element*quadrature,7]` | FE evaluated fields, checked against canonical coefficients |
| `pointwise_terms` | `[time,element,quadrature,2]` | Candidate strong density and pressure advection/compression blocks |
| `selected_residuals` | `[time,global_dof,7,2]` | Independently exported assembled blocks; density in equation 5 and pressure in equation 6 |
| `retained_residual`, `total_residual` | `[time,global_dof,7]` | Common-convention spatial residuals, excluding temporal history |

State order is exactly `psi,u,zj,w,rho,T,Vpar`. Term order is `density_advection_compression,pressure_advection_compression`. Strong selected blocks multiply an undifferentiated test basis; retained diffusion/derivative-test terms cannot be relabelled as scalar samples. Pressure labels are **not** standalone temperature tendencies. Native state updates need the coupled temporal/constraint solve.

Metadata requires `mapping_status` (`candidate`, `provisional` or `verified_native`), case ID, `parent_family_id`, split, seed, physical parameters, configuration/source/executable/equilibrium identities, baseline identity, seven field units, explicit normalization and residual units, solver success and required physical horizon. `synthetic` is an explicit boolean. Channel names, `pointwise_term_kind: strong` and `spatial_residual_convention: axisymmetric_volume_weak` are mandatory. A release manifest must additionally retain failed attempts, original restart/patch hashes, modules, MPI/OpenMP layout, scheduler/resources, health checks, memory/storage and balance diagnostics. Those production values do not exist yet; they must not be invented.

Validation rejects missing fields, nonfinite values, short horizons, incorrect FE evaluation, nonpositive density/temperature, selected-label assembly mismatches and total/remainder reconstruction errors. Norm-relative float64 residual tolerance is `1e-8` with `1e-12` absolute floor. Cancellation is flagged separately and assessed with a component-scale roundoff bound, not a tiny relative denominator. Production may need tighter tolerances after pilot uncertainty is measured. Output uses exclusive creation.

## Exchange command

Provide an explicit YAML config with `geometry_npz`, `trajectory_npz`, `metadata_json` and `geometry_h5`. Both NPZ bundles must use the exact array names above and may not use pickled object arrays. Run:

```
python scripts/export_dataset.py --config /absolute/export.yaml --output /bulk/trajectory.h5 --dry-run
python scripts/export_dataset.py --config /absolute/export.yaml --output /bulk/trajectory.h5
```

A dry run validates inputs and prints intended output without writing. This command consumes an explicit instrumented native exchange. It cannot recover missing labels, basis evaluations or temporal operators from ordinary JOREK restart files. No raw restart modification is used.

Training preprocessing must fit scales on training parent families only. Field-value normalization cannot be applied directly to dimensional coefficients. A numerical baseline bypasses spline transfer; spline approximation error must be measured separately. API layout conversion to `[B,C,N]` happens only inside the learning adapter.
