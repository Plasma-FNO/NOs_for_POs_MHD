# Pinned model-303 equation audit

Audited 2026-11-02 against `/pitagora_work/FUPB2_UKAEA_ML/vgopakum/src/jorek/src`, commit `01a52be37576f79c785f1f7a80dee296dda5c9c8`. References below are source-relative and refer to this checkout. This is a source mapping, not experimental validation.

## Native variables and temporal operator

`models/model303/mod_model_settings.f90:25–39` fixes seven channels: `(psi,u,zj,w,rho,T,Vpar)`. `datatypes/data_structure.f90:12–48` stores variable **values and derivatives**, geometry derivatives, global indices, boundary identifiers and parent/refinement constraints. Flattening only nodal field values loses required state. Local basis coefficients must include all native derivative DOFs and all patches; hanging-node constraints must either be eliminated into the exported effective basis or exported with their elimination mapping before a real geometry bundle is approved.

| Row | Variable / equation | Evolution and constraints | Source in `models/model303/mod_elt_matrix_fft.f90` |
|---|---|---|---|
| 1 | Poloidal flux `psi` | Differential induction; resistive, flow and diamagnetic terms | 822–831 |
| 2 | Stream function `u` | Differential perpendicular momentum with density-dependent gradient temporal operator | 838–862; history term 862 |
| 3 | Current auxiliary `zj` | Algebraic current definition, weighted elliptic relation to `psi` | 877 |
| 4 | Vorticity auxiliary `w` | Algebraic poloidal Laplacian relation to `u` | 883 |
| 5 | Density `rho` | Differential continuity, scalar mass weighting `R` | 890–910; temporal Jacobian 1385 |
| 6 | Temperature channel `T`, pressure equation | Evolves **rho*T**; row couples density and temperature increments | 925–964; off-diagonal temporal block 1510; diagonal 1551 |
| 7 | Parallel velocity coefficient `Vpar` | Differential momentum with density and magnetic dependence | 985–1004 |

The pressure row's history includes `zeta*v*r0*delta_T*R*xjac` **and** `zeta*v*T0*delta_rho*R*xjac` at 960–961. The temperature diagonal uses corrected density `r0_corr` at 1551. The momentum temporal operator is spatially coupled, not seven pointwise scalar masses. A pressure residual cannot be integrated directly as `dT/dt`, and independent neural predictions of `zj`/`w` cannot replace the constraint solve.

`docs/numerics/time-integration.md:11–72` defines the linearized implicit update with temporal Jacobian, spatial Jacobian, theta and history zeta. Runtime variable-step zeta is set at `mod_elt_matrix_fft.f90:173–176`. The native all-numerical bridge must preserve this scheme or demonstrate a validated alternative against the same reference; an explicit diagonal-mass update is not implemented here.

## Selected n=0 physical blocks

For `[a,b] = a_R*b_Z-a_Z*b_R`, source-coordinate change of variables gives `a_s*b_t-a_t*b_s = xjac*[a,b]`. `R=BigR=x_g` at lines 375–389. Axisymmetry removes toroidal derivatives; **parallel flow along the poloidal magnetic field remains**.

A candidate grouping, implemented only as local source-algebra diagnostics in `model303_advection_blocks`, is:

```
N_rho = R*[rho,u] + 2*rho*u_Z
        - (Vpar*[rho,psi] + rho*[Vpar,psi])/R

N_p = R*(rho*[T,u] + T*[rho,u]) + 2*gamma*rho*T*u_Z
      - Vpar*(rho*[T,psi] + T*[rho,psi])/R
      - gamma*rho*T*[Vpar,psi]/R
```

`N_rho` maps density advection/compression from source lines 891–899. `N_p` maps pressure advection/compression from lines 927–940. Every selected piece multiplies the undifferentiated test function. Therefore these **selected blocks** can use scalar strong labels and the native weak form `integral(v*N*R*dR*dZ)`; derivative-test diffusion and stabilization terms stay in the retained numerical residual. Our generic physical-volume convention includes `2*pi`, which the exporter must reconcile with JOREK's actual `HZ`/toroidal integration normalization for **all** blocks and temporal operators. No factor is inferred from ordinary restart files.

The density diamagnetic pressure-gradient source at line 901 remains numerical. All transport, source, Ohmic-heating and Taylor–Galerkin stabilization contributions remain numerical. Taylor–Galerkin expressions include `tstep*tstep` at 907–910 and 946–958; they require separate declared dt-dependent numerical treatment. Dividing an existing RHS vector by dt would leave history and incorrectly interpreted stabilization.

The local diagnostic expects current code-normalized fields, their derivatives with respect to physical R,Z, R in metres, and the pinned gamma. It outputs code-time spatial strong residuals. Conversion to SI physical-time labels must use the recorded case normalization; coordinate bounds used by the spline cannot substitute for that conversion. No automatic unit conversion is claimed.

## Required isolated source bridge

1. At native quadrature evaluation, export `eq_g`, physical basis and derivatives, `x_g,y_g`, signed Jacobian plus positive integration weights, element-to-global derivative DOF map, patch/boundary identifiers and refinement constraint elimination. Preserve field/geometry units and native toroidal basis normalization.
2. Factor selected physical scalar blocks out of `rhs_ij_5`/`rhs_ij_6` **before** dt and history are added. Export their pointwise values and independently assembled weak blocks. Keep an untouched-equations branch and check the assembled identity at the same instantaneous state.
3. Separately expose physical spatial totals, source contributions, dt-dependent numerical stabilization, temporal/history terms and boundary modifications. `RHS` already contains these different objects. `ELM` is the linearized solve matrix, not the mass operator or a spatial label.
4. `matrix/construct_matrix_mod.f90:61–80` calls element assembly; sheath boundary terms are then added at 180–188. Main construction begins at 298 and applies additional constraints/boundaries. Exporting an element residual alone omits these modifications. Keep the physical residual, boundary flux and algebraic constraint contracts distinct.
5. Supply a same-state callable numerical evaluator and temporal/constraint solve which accept the standalone integrator's **own** native q, including Newton/implicit stages. Saved reference-state labels cannot supply autonomous retained dynamics. Verify gradients through state-dependent solves with finite differences.
6. Build in an isolated source copy/worktree, reproduce unchanged equilibrium regression, then baseline and perturbed trajectories plus timestep/mesh refinement. Only this evidence can pass Gate B.

Existing `communication/mod_export_restart.f90` is a restart writer; its presence does not provide this bridge. No untested Fortran instrumentation has been added to the production source.


## Legacy model303 temporal-mass audit — 2026-11-02

`cpu_temporal_mass_audit_20261102.json` uses dense exact-geometry four-rank snapshots and centered physical-time rates to separate generic diagnostic energy derivatives from selected legacy temporal coefficients. The positive low-density region still activates smooth `rho_1=0.01` mass correction: setting `rho_min_neg=-1e12` does not disable this correction. Nonroundoff corrected mass occupies 13.17–20.74% of full-wall quadrature volume over the sampled interval. The generic parallel kinetic energy also includes poloidal-field terms absent from the corresponding toroidal-only parallel temporal diagonal. Their partial power contributions, including pressure and perpendicular density-mass corrections, sum to 0.1163–0.5533 MW. These are centered finite-rate diagnostics, not an exact discrete-energy identity. Independent parallel kinetic energy agrees with native diagnostics to the frozen tolerance, and algebraic decompositions are checked. Remaining temporal/cross terms, stabilization and boundary work prevent complete energy closure; no physical parameters or acceptance thresholds changed.
