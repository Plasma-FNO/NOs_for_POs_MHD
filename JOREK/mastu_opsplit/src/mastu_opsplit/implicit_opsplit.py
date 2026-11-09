"""Implicit Physical OpsSplit with the frozen full native model303 tangent.

The state holds canonical bicubic coefficients of (psi,u,zj,w,rho,T,Vpar).
Evolution rows (psi,u,rho,pressure,Vpar) use the native temporal operator M*
frozen at the common warm baseline q*, and the complete native volume RHS is
split as F(q) = F(q*) + L*(q-q*) + N(q). One step of length h is

    M* (q' - q) = h [F(q*) + L*(theta q' + (1-theta) q - q*) + N_theta(q)]

theta=1 is the requested form (M* - h L*) q' = M* q + h (F(q*) - L* q* + N_theta(q)).
L* is the complex-step tangent exported by the native build (fixed transport and
sources). Current/vorticity rows are the weak definitions at q', tested on every
DOF that evolves in the training trajectories; DOFs that never change in those
trajectories are Dirichlet rows. One sparse LU per step size is reused; the
adjoint is the transpose solve with the same factors.

M* is the native mass action used by the validated SBDF2 split (patched
mod_elt_matrix_fft.f90, opsplit_history_weight branch), evaluated at q* with
JOREK's corr_neg_dens1 low-density correction. Freezing M* and L* at q* is an
approximation; N_theta learns the remaining volume forcing, not the native mass,
boundary operators or the time discretisation.
"""
from dataclasses import dataclass
import hashlib
import time

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import torch
from torch import nn

from .learning import FieldSplineOperator

FIELDS = ('psi', 'u', 'zj', 'w', 'rho', 'T', 'Vpar')
BLOCKS = ('psi', 'u', 'rho', 'pressure', 'Vpar')
BLOCK_ROWS = (0, 1, 4, 5, 6)  # native row (field index) tested by each weak block
# Weak channels are [block][strong, test_R, test_Z]; the induction row has no flux channels.
ACTIVE_CHANNELS = tuple(index for index in range(15) if index not in (1, 2))


def corr_neg_dens1(rho, rho_1=0.01, coefficients=(0.5, 0.5)):
    """JOREK corr_neg_dens1 for rho_min_neg < 0 (models/corr_neg_include.f90)."""
    low, width = rho_1*coefficients[0], rho_1*coefficients[1]
    rho = np.asarray(rho, dtype=np.float64)
    return np.where(rho < low+width, low+width*np.exp(np.minimum(rho-(low+width), 0.)/width), rho)


@dataclass(frozen=True)
class FrozenTangent:
    tangent: np.ndarray    # [E,Q,21,15]: d c[block,coef] / d [values, d_R, d_Z]
    forcing: np.ndarray    # [E,Q,5,3]: complete native weak coefficients at q*
    values: np.ndarray     # [E,Q,7] at q*
    gradients: np.ndarray  # [E,Q,7,2] physical R,Z derivatives at q*
    sha256: str
    path: str

    @property
    def jets(self):
        return np.concatenate((self.values, self.gradients[..., 0], self.gradients[..., 1]), axis=-1)


def load_frozen_tangent(path, geometry):
    with h5py.File(path) as stream:
        if stream.attrs.get('measure') is None or 'fixed_coefficient_tangent' not in stream:
            raise ValueError('Expected a full-tangent baseline_terms.h5 export')
        record = {name: stream[name][()] for name in ('coordinates', 'fixed_coefficient_tangent',
                                                     'baseline_forcing', 'values', 'gradients')}
    e, q = geometry.basis.shape[:2]
    expected = dict(coordinates=(e, q, 2), fixed_coefficient_tangent=(e, q, 21, 15),
                    baseline_forcing=(e, q, 5, 3), values=(e, q, 7), gradients=(e, q, 7, 2))
    for name, shape in expected.items():
        if record[name].shape != shape or not np.isfinite(record[name]).all():
            raise ValueError(f'Invalid frozen tangent array {name}: {record[name].shape}')
    if not np.allclose(record['coordinates'], geometry.quadrature_rz, rtol=0, atol=1e-12):
        raise ValueError('Frozen tangent quadrature differs from the native geometry')
    if np.any(record['fixed_coefficient_tangent'][..., [1, 2]] != 0) or np.any(record['baseline_forcing'][..., 0, 1:] != 0):
        raise ValueError('Induction flux channels must vanish identically')
    if (record['values'][..., 4] <= 0).any() or (record['values'][..., 5] <= 0).any():
        raise ValueError('Frozen baseline density/temperature must be positive')
    with open(path, 'rb') as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    return FrozenTangent(record['fixed_coefficient_tangent'], record['baseline_forcing'], record['values'],
                         record['gradients'], digest, str(path))


def fixed_dofs_from_trajectories(paths, scale, boundary_labels, *, tolerance=1e-7, minimum_gap=100.):
    """Boundary DOFs whose normalized coefficient never changes in any supplied (training) trajectory.

    Interior DOFs always keep their physical rows, however little they move.
    """
    boundary = np.asarray(boundary_labels) != 0
    change = np.zeros((len(boundary), 7))
    for path in paths:
        with h5py.File(path) as stream:
            states = stream['native_state']
            first = states[0]
            for start in range(1, states.shape[0], 16):
                block = states[start:start+16]
                change = np.maximum(change, (np.abs(block-first)/scale).max(0))
    fixed = boundary[:, None] & (change < tolerance)
    free_boundary = boundary[:, None] & ~fixed
    largest_fixed = float(change[fixed].max()) if fixed.any() else 0.
    smallest_free = float(change[free_boundary].min()) if free_boundary.any() else np.inf
    if fixed.any() and largest_fixed > 0 and smallest_free < minimum_gap*largest_fixed:
        raise ValueError(f'Ambiguous boundary Dirichlet classification: {largest_fixed} vs {smallest_free}')
    report = dict(tolerance=tolerance, largest_fixed_change=largest_fixed, smallest_free_boundary_change=smallest_free,
                  smallest_interior_change=float(change[~boundary].min()),
                  fixed_per_field={name: int(fixed[:, i].sum()) for i, name in enumerate(FIELDS)},
                  trajectories=len(paths), scope='boundary DOFs only; interior rows are never Dirichlet')
    return fixed, report


def dof_nodes(geometry):
    """Vertex value-DOF and local type (0 value, 1 s-, 2 t-, 3 cross-derivative) of every native DOF."""
    n = len(geometry.dof_ids)
    node, kind = np.full(n, -1), np.full(n, -1)
    for k in range(geometry.dof_map.shape[1]):
        node[geometry.dof_map[:, k]] = geometry.dof_map[:, 4*(k//4)]
        kind[geometry.dof_map[:, k]] = k % 4
    if (node < 0).any():
        raise ValueError('Unmapped native DOF')
    return node, kind


@dataclass(frozen=True)
class BoundaryRelations:
    """JOREK Mach-1 wall rows for Vpar, identified from training data (physical units).

    kind 0 (node value):            Vpar_d = c0 + c1 sqrt(T_v)
    kind 1 (along-wall derivative):  Vpar_d = c0 T_d / sqrt(T_v) + c1 sqrt(T_v)
    with T_v the node's temperature value DOF and T_d the temperature DOF of the same type.
    """
    rows: np.ndarray
    kind: np.ndarray
    value_dofs: np.ndarray
    coefficients: np.ndarray

    def evaluate(self, temperature):
        """G(T) for [batch, ndof] physical temperature coefficients (NumPy or Torch)."""
        lib = torch if isinstance(temperature, torch.Tensor) else np
        c0, c1 = self.coefficients[:, 0], self.coefficients[:, 1]
        value_dofs, rows, kind = self.value_dofs, self.rows, self.kind
        if lib is torch:
            device = temperature.device
            c0, c1 = (torch.as_tensor(c, dtype=temperature.dtype, device=device) for c in (c0, c1))
            value_dofs, rows, kind = (torch.as_tensor(a, device=device) for a in (value_dofs, rows, kind))
        root = lib.sqrt(temperature[:, value_dofs])
        derivative = temperature[:, rows]/root
        return lib.where(kind == 0, c0 + c1*root, c0*derivative + c1*root)

    def gradient(self, temperature):
        """dG/dT_v and dG/dT_d at one [ndof] physical temperature state."""
        tv, td = temperature[self.value_dofs], temperature[self.rows]
        c0, c1 = self.coefficients[:, 0], self.coefficients[:, 1]
        root = np.sqrt(tv)
        value = np.where(self.kind == 0, c1/(2*root), -c0*td/(2*tv*root) + c1/(2*root))
        own = np.where(self.kind == 0, 0., c0/root)
        return value, own


def mach_fit(temperature, vpar, d, value_dof, kind):
    """Least-squares Mach-1 coefficients for one DOF; returns (coefficients, features fn)."""
    def features(T):
        root = np.sqrt(T[:, value_dof])
        return np.stack([np.ones_like(root), root], 1) if kind == 0 else np.stack([T[:, d]/root, root], 1)
    coefficients, *_ = np.linalg.lstsq(features(temperature), vpar[:, d], rcond=None)
    return coefficients, features


def identify_boundary_rows(geometry, frozen, fixed, identity_states, fit_samples, heldout_samples, *, F0=0.64,
                           code_interval, weak_tolerance=1e-2, fit_tolerance=1e-3):
    """Classify free Vpar wall rows whose native equation is not the volume weak form.

    identity_states: [(q_{k-1}, q_k, q_{k+1}, exact_label_k)] training states; a row is 'weak' if
    M(q_k)(q_{k+1}-q_{k-1})/2h matches the assembled native forcing within weak_tolerance in every state.
    Non-weak rows of value / along-wall-derivative type whose Mach-1 form fits the training trajectories
    (fit and held-out cases) within fit_tolerance become BoundaryRelations; other non-weak rows are held.
    fit_samples / heldout_samples: (T, Vpar) physical [S, ndof] arrays from disjoint training trajectories.
    """
    n = len(geometry.dof_ids)
    node, kind = dof_nodes(geometry)
    candidates = np.flatnonzero((geometry.boundary_labels != 0) & ~fixed[:, 6])
    assembler = WeakAssembler(geometry, channels=tuple(range(15)))
    worst = np.zeros(len(candidates))
    for previous, current, following, label in identity_states:
        local = current[geometry.dof_map]
        values = np.einsum('eql,elc->eqc', geometry.basis, local)
        gradients = np.einsum('eqlr,elc->eqcr', geometry.basis_grad, local)
        jets = np.concatenate((values, gradients[..., 0], gradients[..., 1]), -1)
        c = (frozen.forcing.reshape(*frozen.forcing.shape[:2], 15)
             + np.einsum('eqjo,eqj->eqo', frozen.tangent, jets-frozen.jets, optimize=True)).reshape(-1, 15)
        c[:, list(ACTIVE_CHANNELS)] += label
        forcing = assembler.numpy(c[None])[0][:, 4]
        mass = assemble_mass(geometry, FrozenTangent(frozen.tangent, frozen.forcing, values, gradients, '', ''), F0=F0)
        rate = (following-previous)/(2*code_interval)
        action = sum(mass[4, f] @ rate[:, f] for f in range(7) if (4, f) in mass)
        mismatch = np.abs(action-forcing)/(np.abs(action)+np.abs(forcing)+1e-300)
        worst = np.maximum(worst, mismatch[candidates])
    nonweak = candidates[worst >= weak_tolerance]
    relation_rows, relation_kind, relation_value, relation_coefficients, fits = [], [], [], [], []
    held = np.zeros((n, 7), dtype=bool)
    for d in nonweak:
        accepted = False
        if kind[d] in (0, 1, 2):
            form = 0 if kind[d] == 0 else 1
            coefficients, features = mach_fit(fit_samples[0], fit_samples[1], d, node[d], form)
            errors = [np.abs(features(T) @ coefficients - V[:, d]).max()/max(np.abs(V[:, d]).max(), 1e-300)
                      for T, V in (fit_samples, heldout_samples)]
            if max(errors) < fit_tolerance:
                accepted = True
                relation_rows.append(d); relation_kind.append(form); relation_value.append(node[d])
                relation_coefficients.append(coefficients); fits.append(errors)
        if not accepted:
            held[d, 6] = True
    relations = BoundaryRelations(np.asarray(relation_rows, dtype=np.int64), np.asarray(relation_kind, dtype=np.int64),
                                  np.asarray(relation_value, dtype=np.int64),
                                  np.asarray(relation_coefficients, dtype=np.float64).reshape(-1, 2))
    fits = np.asarray(fits).reshape(-1, 2)
    report = dict(free_boundary_vpar_rows=int(len(candidates)), weak_form_rows=int(len(candidates)-len(nonweak)),
                  mach_relation_rows=int(len(relations.rows)), held_rows=int(held.sum()),
                  mach_rows_by_kind={str(k): int((relations.kind == k).sum()) for k in (0, 1)},
                  held_rows_by_local_type={str(t): int((held[:, 6] & (kind == t)).sum()) for t in range(4)},
                  maximum_fit_error=float(fits[:, 0].max()) if len(fits) else None,
                  maximum_heldout_fit_error=float(fits[:, 1].max()) if len(fits) else None,
                  weak_tolerance=weak_tolerance, fit_tolerance=fit_tolerance, identity_states=len(identity_states))
    return held, relations, report


def _test_functions(geometry):
    """[E,Q,16,3]: value and physical R,Z gradients of every local basis function."""
    return np.concatenate((geometry.basis[..., None], geometry.basis_grad), axis=-1)


def _global(local, geometry):
    """Assemble [E,16,16] element matrices into an [ndof,ndof] CSR matrix."""
    rows, cols = np.broadcast_arrays(geometry.dof_map[:, :, None], geometry.dof_map[:, None, :])
    n = len(geometry.dof_ids)
    return sp.coo_matrix((local.ravel(), (rows.ravel(), cols.ravel())), shape=(n, n)).tocsr()


def assemble_tangent(geometry, frozen, chunk=1024):
    """Global weak L* blocks: {(block, field): [ndof, ndof]} in native code units."""
    test = _test_functions(geometry)
    weights = geometry.volume_weights
    jacobian = frozen.tangent.reshape(*frozen.tangent.shape[:2], 3, 7, 5, 3)  # [E,Q,d,f,b,i]
    blocks = {}
    for b in range(5):
        for f in range(7):
            coefficient = jacobian[:, :, :, f, b, :]
            if not np.any(coefficient):
                continue
            local = np.concatenate([np.einsum('eqki,eqdi,eqld,eq->ekl', test[s:s+chunk], coefficient[s:s+chunk],
                                              test[s:s+chunk], weights[s:s+chunk], optimize=True)
                                    for s in range(0, len(test), chunk)])
            blocks[b, f] = _global(local, geometry)
    return blocks


def assemble_mass(geometry, frozen, *, F0, rho_1=0.01, coefficients=(0.5, 0.5)):
    """Native temporal operator rows frozen at q*: {(block, field): [ndof, ndof]}."""
    b, g, w = geometry.basis, geometry.basis_grad, geometry.volume_weights
    radius = geometry.quadrature_rz[..., 0]
    rho, T, vpar = frozen.values[..., 4], frozen.values[..., 5], frozen.values[..., 6]
    corrected = corr_neg_dens1(rho, rho_1, coefficients)
    grad_psi = frozen.gradients[..., 0, :]
    mass = lambda weight: _global(np.einsum('eql,eqm,eq->elm', b, b, weight, optimize=True), geometry)
    return {
        (0, 0): mass(w/radius**2),
        (1, 1): _global(np.einsum('eqlr,eqmr,eq->elm', g, g, -radius**2*corrected*w, optimize=True), geometry),
        (2, 4): mass(w),
        (3, 4): mass(T*w),
        (3, 5): mass(corrected*w),
        (4, 0): _global(np.einsum('eql,eqr,eqmr,eq->elm', b, grad_psi, g, rho*vpar*w/radius**2, optimize=True), geometry),
        (4, 6): mass(corrected*F0**2*w/radius**2),
    }


def constraint_blocks(geometry):
    """Weak current (R^-2 weight) and vorticity definitions, identical to PhysicalGuard."""
    b, g = geometry.basis, geometry.basis_grad
    result = {}
    for name, weight in (('current', geometry.volume_weights/geometry.quadrature_rz[..., 0]**2),
                         ('vorticity', geometry.volume_weights)):
        result[name] = (_global(np.einsum('eqlr,eqmr,eq->elm', g, g, weight, optimize=True), geometry),
                        _global(np.einsum('eql,eqm,eq->elm', b, b, weight, optimize=True), geometry))
    return result


class WeakAssembler:
    """Fixed sparse weak assembly of pointwise channels; exact transpose for autograd."""
    def __init__(self, geometry, channels=ACTIVE_CHANNELS):
        e, q, local = geometry.basis.shape
        n = len(geometry.dof_ids)
        rows = np.broadcast_to(geometry.dof_map[:, None, :], (e, q, local)).ravel()
        cols = np.broadcast_to(np.arange(e*q).reshape(e, q, 1), (e, q, local)).ravel()
        w = geometry.volume_weights[..., None]
        self.test = [sp.coo_matrix(((values*w).ravel(), (rows, cols)), shape=(n, e*q)).tocsr()
                     for values in (geometry.basis, geometry.basis_grad[..., 0], geometry.basis_grad[..., 1])]
        self.channels = tuple(divmod(c, 3) for c in channels)
        self.ndof = n

    def apply(self, x, operation, transpose=False):
        if operation != 'assemble':
            raise ValueError('Unknown weak operation')
        if transpose:
            out = np.zeros((x.shape[0], self.test[0].shape[1], len(self.channels)))
            for c, (block, kind) in enumerate(self.channels):
                out[:, :, c] = (self.test[kind].T @ x[:, :, block].T).T
            return out
        out = np.zeros((x.shape[0], self.ndof, 5))
        for c, (block, kind) in enumerate(self.channels):
            out[:, :, block] += (self.test[kind] @ x[:, :, c].T).T
        return out

    def __call__(self, pointwise):
        from .euler import _FixedNative
        return _FixedNative.apply(pointwise, self, 'assemble')

    def numpy(self, pointwise):
        return self.apply(np.asarray(pointwise, dtype=np.float64), 'assemble')


METIS_LIBRARY = ('/pitagora/prod/spack/6.1/install/0.22/linux-rhel9-zen4/gcc-12.3.0/'
                 'metis-5.1.0-x3ar5apalmwiveo2ok4sz2fckxeblsr5/lib/libmetis.so')


def dof_graph(geometry):
    """CSR adjacency of native DOFs that share an element (no self loops)."""
    n = len(geometry.dof_ids)
    rows, cols = np.broadcast_arrays(geometry.dof_map[:, :, None], geometry.dof_map[:, None, :])
    graph = sp.coo_matrix((np.ones(rows.size, dtype=np.int8), (rows.ravel(), cols.ravel())), shape=(n, n)).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def nested_dissection(graph, library=METIS_LIBRARY):
    """METIS_NodeND fill-reducing order (perm[new] = old) for a symmetric CSR graph."""
    import ctypes
    metis = ctypes.CDLL(library)
    n = graph.shape[0]
    idx = ctypes.c_int32
    xadj = np.ascontiguousarray(graph.indptr, dtype=np.int32)
    adjncy = np.ascontiguousarray(graph.indices, dtype=np.int32)
    options = np.full(40, -1, dtype=np.int32)
    metis.METIS_SetDefaultOptions(options.ctypes.data_as(ctypes.POINTER(idx)))
    options[17] = 0  # METIS_OPTION_NUMBERING: C-style
    perm, iperm = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)
    nvtxs = idx(n)
    pointer = lambda a: a.ctypes.data_as(ctypes.POINTER(idx))
    status = metis.METIS_NodeND(ctypes.byref(nvtxs), pointer(xadj), pointer(adjncy), None, pointer(options),
                                pointer(perm), pointer(iperm))
    if status != 1 or not np.array_equal(np.sort(perm), np.arange(n)):
        raise RuntimeError(f'METIS_NodeND failed with status {status}')
    return perm.astype(np.int64)


class _Factor:
    """Row-scaled square system for one step fraction; LU reused for all solves.

    ordering='metis': nested dissection of the DOF graph with the seven field unknowns of
    each DOF kept adjacent, then SuperLU with that fixed column order and diagonal-preferring
    threshold pivoting. ordering='colamd': SuperLU's own column ordering (much more fill here).
    """
    def __init__(self, A, B, constant, forcing_rows, forcing_scale, h, refinement, *, permutation=None,
                 pivot_threshold=.01, relation_rows=None, relation_scale=None):
        started = time.monotonic()
        self.A, self.B, self.constant = A.tocsc(), B.tocsr(), constant
        self.forcing_rows, self.forcing_scale, self.h = forcing_rows, forcing_scale, h
        self.relation_rows = np.zeros(0, dtype=np.int64) if relation_rows is None else relation_rows
        self.relation_scale = np.zeros(0) if relation_scale is None else relation_scale
        self.permutation = permutation
        if permutation is None:
            self.lu = splu(self.A, permc_spec='COLAMD')
        else:
            permuted = self.A[permutation][:, permutation].tocsc()
            self.lu = splu(permuted, permc_spec='NATURAL', diag_pivot_thresh=pivot_threshold,
                           options=dict(SymmetricMode=True))
        self.refinement = refinement
        self.seconds = time.monotonic()-started
        self.fill = int(self.lu.L.nnz + self.lu.U.nnz)

    def _solve(self, rhs, trans):
        if self.permutation is None:
            return self.lu.solve(rhs, trans=trans)
        out = np.empty_like(rhs)
        out[self.permutation] = self.lu.solve(rhs[self.permutation], trans=trans)
        return out

    def solve(self, rhs, transpose=False, tolerance=1e-11):
        """LU solve plus at least `refinement` and at most 6 iterative-refinement steps, stopping at tolerance."""
        trans = 'T' if transpose else 'N'
        x = self._solve(rhs, trans)
        matrix = self.A.T if transpose else self.A
        scale = max(np.linalg.norm(rhs), 1e-300)
        for iteration in range(max(self.refinement, 6)):
            residual = rhs - matrix @ x
            if iteration >= self.refinement and np.linalg.norm(residual) <= tolerance*scale:
                break
            x = x + self._solve(residual, trans)
        return x

    def relative_residual(self, rhs, x, transpose=False):
        matrix = self.A.T if transpose else self.A
        return float(np.linalg.norm(matrix @ x - rhs)/max(np.linalg.norm(rhs), 1e-300))


class _ImplicitStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, forcing, explicit, system, fraction):
        factor = system.factor(fraction)
        ctx.factor, ctx.system = factor, system
        y = state.detach().cpu().double().numpy()
        n = forcing.detach().cpu().double().numpy()
        e = explicit.detach().cpu().double().numpy()
        out = np.empty_like(y)
        for i in range(len(y)):
            rhs = factor.B @ system.vector(y[i]) + factor.constant
            rhs[factor.forcing_rows] += factor.forcing_scale*n[i].T.ravel()[system.forcing_index]
            rhs[factor.relation_rows] += factor.relation_scale*e[i]
            out[i] = system.unvector(factor.solve(rhs))
        return torch.as_tensor(out, dtype=state.dtype, device=state.device)

    @staticmethod
    def backward(ctx, gradient):
        factor, system = ctx.factor, ctx.system
        g = gradient.detach().cpu().double().numpy()
        grad_state, grad_forcing = np.empty_like(g), np.zeros((len(g), system.ndof, 5))
        grad_explicit = np.zeros((len(g), len(factor.relation_rows)))
        for i in range(len(g)):
            adjoint = factor.solve(system.vector(g[i]), transpose=True)
            grad_state[i] = system.unvector(factor.B.T @ adjoint)
            flat = np.zeros(5*system.ndof)
            flat[system.forcing_index] = factor.forcing_scale*adjoint[factor.forcing_rows]
            grad_forcing[i] = flat.reshape(5, system.ndof).T
            grad_explicit[i] = factor.relation_scale*adjoint[factor.relation_rows]
        as_tensor = lambda a: torch.as_tensor(a, dtype=gradient.dtype, device=gradient.device)
        return as_tensor(grad_state), as_tensor(grad_forcing), as_tensor(grad_explicit), None, None


class ImplicitNativeSystem:
    """Normalized implicit step y -> y' for q = mean + scale*y (field-major unknowns)."""
    def __init__(self, geometry, frozen, mean, scale, fixed, *, code_interval, theta=1., F0=0.64,
                 rho_1=0.01, density_coefficients=(0.5, 0.5), refinement=1, ordering='metis', relations=None):
        started = time.monotonic()
        n = len(geometry.dof_ids)
        mean, scale, fixed = np.asarray(mean, float), np.asarray(scale, float), np.asarray(fixed, bool)
        if mean.shape != (n, 7) or scale.shape != (7,) or fixed.shape != (n, 7) or not (scale > 0).all():
            raise ValueError('Normalization or Dirichlet mask does not match the native geometry')
        if relations is None:
            relations = BoundaryRelations(*(np.zeros(0, dtype=np.int64) for _ in range(3)), np.zeros((0, 2)))
        if fixed[relations.rows, 6].any() or len(np.unique(relations.rows)) != len(relations.rows):
            raise ValueError('Mach-1 relation rows must be distinct, non-Dirichlet Vpar DOFs')
        if (mean[relations.value_dofs, 5] <= 0).any():
            raise ValueError('Mach-1 linearization needs positive mean wall temperature')
        self.relations = relations
        # Frozen linearization of the wall relations at the training mean temperature.
        self.relation_gradient = relations.gradient(mean[:, 5])
        relation_mask = np.zeros((n, 7), dtype=bool)
        relation_mask[relations.rows, 6] = True
        self.relation_mask = relation_mask
        if not 0.5 <= theta <= 1 or not np.isfinite(code_interval) or code_interval <= 0:
            raise ValueError('Require 1/2 <= theta <= 1 and positive code-time interval')
        if type(refinement) is not int or refinement < 0:
            raise ValueError('Refinement must be a nonnegative integer')
        self.geometry, self.ndof, self.theta, self.code_interval = geometry, n, float(theta), float(code_interval)
        self.mean, self.scale, self.fixed, self.refinement = mean, scale, fixed, refinement
        self.tangent_sha256 = frozen.sha256
        self.L = assemble_tangent(geometry, frozen)
        self.M = assemble_mass(geometry, frozen, F0=F0, rho_1=rho_1, coefficients=density_coefficients)
        self.constraints = constraint_blocks(geometry)
        pointwise = frozen.forcing.reshape(*frozen.forcing.shape[:2], 15) - np.einsum(
            'eqjo,eqj->eqo', frozen.tangent, frozen.jets, optimize=True)
        assembler = WeakAssembler(geometry, channels=tuple(range(15)))
        reference = assembler.numpy(pointwise.reshape(1, -1, 15))[0]  # F(q*) - L* q* in weak form
        # Constant part of h[F(q*) + L*(q - q*)] after substituting q = mean + scale*y.
        self.reference_forcing = np.stack([reference[:, b] + sum(
            self.L[b, f] @ mean[:, f] for f in range(7) if (b, f) in self.L) for b in range(5)], axis=1)
        weak = ~fixed & ~relation_mask
        self.forcing_index = np.concatenate([b*n + np.flatnonzero(weak[:, row]) for b, row in enumerate(BLOCK_ROWS)])
        self.forcing_global_rows = np.concatenate([row*n + np.flatnonzero(weak[:, row]) for row in BLOCK_ROWS])
        self.relation_global_rows = 6*n + relations.rows
        if ordering not in ('metis', 'colamd'):
            raise ValueError('ordering must be metis or colamd')
        self.ordering = ordering
        self.permutation = None
        if ordering == 'metis':
            dofs = nested_dissection(dof_graph(geometry))
            # Field-major unknown f*n+d placed at 7*position(d)+f: one dense 7-field block per DOF.
            self.permutation = (np.arange(7)[None, :]*n + dofs[:, None]).ravel()
        self._factors = {}
        self.setup_seconds = time.monotonic()-started

    def vector(self, y):
        return np.asarray(y).T.ravel()

    def unvector(self, v):
        return v.reshape(7, self.ndof).T

    def matrices(self, fraction):
        """Square [7n,7n] A,B and constant c for one step of `fraction` saved intervals."""
        n, h, theta = self.ndof, self.code_interval*fraction, self.theta
        S = sp.diags(np.repeat(self.scale, n))
        zero = sp.csr_matrix((n, n))
        A_rows, B_rows, c_rows = [], [], []
        for field in range(7):
            if field in BLOCK_ROWS:
                b = BLOCK_ROWS.index(field)
                M = [self.M.get((b, f), zero) for f in range(7)]
                L = [self.L.get((b, f), zero) for f in range(7)]
                A_rows.append([M[f] - theta*h*L[f] for f in range(7)])
                B_rows.append([M[f] + (1-theta)*h*L[f] for f in range(7)])
                c_rows.append(h*self.reference_forcing[:, b])
            else:
                stiff, mass = self.constraints['current' if field == 2 else 'vorticity']
                potential = 0 if field == 2 else 1
                row = [zero]*7
                row[potential], row[field] = stiff, mass
                A_rows.append(row)
                B_rows.append([zero]*7)
                c_rows.append(-(stiff @ self.mean[:, potential] + mass @ self.mean[:, field]))
        A = sp.bmat(A_rows, format='csr') @ S
        B = sp.bmat(B_rows, format='csr') @ S
        constant = np.concatenate(c_rows)
        free = ~self.fixed.T.ravel()
        weak = free & ~self.relation_mask.T.ravel()
        keep, pin = sp.diags(weak.astype(float)), sp.diags((~free).astype(float))
        # Mach-1 rows: q'_V - g_v q'_Tv - g_d q'_Td = E(q_n), linearized at the training mean.
        r = self.relations
        g_value, g_own = self.relation_gradient
        rows = np.repeat(self.relation_global_rows, 3)
        cols = np.stack((6*n + r.rows, 5*n + r.value_dofs, 5*n + r.rows), 1).ravel()
        values = np.stack((np.full(len(r.rows), self.scale[6]), -g_value*self.scale[5], -g_own*self.scale[5]), 1).ravel()
        relation = sp.coo_matrix((values, (rows, cols)), shape=(7*n, 7*n)).tocsr()
        A = (keep @ A + pin + relation).tocsr()
        B = (keep @ B + pin).tocsr()
        constant = constant*weak
        constant[self.relation_global_rows] = (-self.mean[r.rows, 6] + g_value*self.mean[r.value_dofs, 5]
                                               + g_own*self.mean[r.rows, 5])
        magnitude = np.asarray(abs(A).max(axis=1).todense()).ravel()
        if not (magnitude > 0).all():
            raise ValueError('Singular implicit row: empty equation')
        row_scale = 1/magnitude
        R = sp.diags(row_scale)
        return (R @ A).tocsr(), (R @ B).tocsr(), row_scale*constant, row_scale, h

    def factor(self, fraction):
        key = round(float(fraction), 12)
        if not 0 < key <= 1:
            raise ValueError('Step fraction must lie in (0, 1]')
        if key not in self._factors:
            A, B, constant, row_scale, h = self.matrices(key)
            self._factors[key] = _Factor(A, B, constant, self.forcing_global_rows,
                                         h*row_scale[self.forcing_global_rows], h, self.refinement,
                                         permutation=self.permutation, relation_rows=self.relation_global_rows,
                                         relation_scale=row_scale[self.relation_global_rows])
        return self._factors[key]

    def relation_explicit(self, state):
        """E(q_n) = G(T_n) - g_v T_n,v - g_d T_n,d: explicit nonlinear part of the wall rows (Torch)."""
        if not len(self.relations.rows):
            return state.new_zeros(state.shape[0], 0)
        temperature = state[..., 5]*self.scale[5] + torch.as_tensor(self.mean[:, 5], dtype=state.dtype, device=state.device)
        value_dofs = torch.as_tensor(self.relations.value_dofs, device=state.device)
        rows = torch.as_tensor(self.relations.rows, device=state.device)
        if (temperature[:, value_dofs] <= 0).any():
            raise FloatingPointError('Nonpositive wall temperature in Mach-1 relation')
        g_value, g_own = (torch.as_tensor(g, dtype=state.dtype, device=state.device) for g in self.relation_gradient)
        return self.relations.evaluate(temperature) - g_value*temperature[:, value_dofs] - g_own*temperature[:, rows]

    def step(self, state, forcing, fraction=1.):
        if state.ndim != 3 or state.shape[1:] != (self.ndof, 7):
            raise ValueError('Expected normalized [batch,ndof,7] state')
        if forcing is None:
            forcing = state.new_zeros(state.shape[0], self.ndof, 5)
        if forcing.shape != (state.shape[0], self.ndof, 5) or not torch.isfinite(forcing).all():
            raise ValueError('Expected finite assembled [batch,ndof,5] forcing')
        return _ImplicitStep.apply(state, forcing, self.relation_explicit(state), self, fraction)

    def residual(self, state, next_state, forcing=None, fraction=1.):
        """Row-scaled residual of the step equations for diagnostics and tests."""
        factor = self.factor(fraction)
        y, z = self.vector(state), self.vector(next_state)
        rhs = factor.B @ y + factor.constant
        if forcing is not None:
            rhs[factor.forcing_rows] += factor.forcing_scale*np.asarray(forcing).T.ravel()[self.forcing_index]
        rhs[factor.relation_rows] += factor.relation_scale*self.relation_explicit(torch.as_tensor(state)[None])[0].numpy()
        return factor.A @ z - rhs

    def report(self):
        return dict(ndof=self.ndof, unknowns=7*self.ndof, theta=self.theta, code_interval=self.code_interval,
                    ordering=self.ordering, mach_relation_rows=int(len(self.relations.rows)),
                    tangent_sha256=self.tangent_sha256, setup_seconds=self.setup_seconds,
                    fixed_per_field={name: int(self.fixed[:, i].sum()) for i, name in enumerate(FIELDS)},
                    factors={str(k): dict(seconds=f.seconds, lu_nonzeros=f.fill, matrix_nonzeros=int(f.A.nnz),
                                          h_code=f.h) for k, f in self._factors.items()})


class ImplicitOpsSplitModel(nn.Module):
    """Native implicit linear physics plus a learned explicit volume remainder N_theta."""
    family = 'opsplit'
    variant = 'implicit_native_tangent'

    def __init__(self, space, system, model_config, remainder_scale, *, physical_guard, physics_only=False,
                 zero_output=True):
        super().__init__()
        remainder_scale = torch.as_tensor(remainder_scale, dtype=torch.float64)
        if remainder_scale.shape != (len(ACTIVE_CHANNELS),) or not torch.all(torch.isfinite(remainder_scale) & (remainder_scale > 0)):
            raise ValueError('Require one positive training-label scale per active weak channel')
        self.space, self.system, self.physical_guard, self.physics_only = space, system, physical_guard, physics_only
        self.operator = FieldSplineOperator(7, len(ACTIVE_CHANNELS), space.geometry.bounds, model_config,
                                            condition_transport=True)
        if zero_output:
            with torch.no_grad():
                self.operator.backbone.project.weight.zero_()
                self.operator.backbone.project.bias.zero_()
        self.register_buffer('coordinates', torch.tensor(space.geometry.quadrature_rz.reshape(-1, 2), dtype=torch.float64))
        self.register_buffer('remainder_scale', remainder_scale)
        self.assembler = WeakAssembler(space.geometry)

    def remainder(self, state, parameters):
        """Pointwise [batch, E*Q, 13] weak remainder channels in native code units."""
        values = self.space.sample(state)
        return self.operator(values.float(), self.coordinates, parameters.float()).to(state.dtype)*self.remainder_scale

    def forcing(self, state, parameters):
        if self.physics_only:
            return None
        return self.assembler(self.remainder(state, parameters))

    def forward(self, state, parameters, *, substeps=1, interval=1.):
        if type(substeps) is not int or substeps < 1 or not 0 < interval <= 1:
            raise ValueError('Require positive substeps and 0 < frame interval <= 1')
        for _ in range(substeps):
            proposal = self.system.step(state, self.forcing(state, parameters), interval/substeps)
            state = self.physical_guard(state, proposal) if self.physical_guard is not None else proposal
        return state


def remainder_loss(prediction, target, weights, scale):
    """Volume-weighted MSE per channel in units of the training label scale, averaged over channels."""
    w = weights/weights.sum()
    return ((prediction-target).square()*w[None, :, None]/scale.square()).sum(1).mean()
