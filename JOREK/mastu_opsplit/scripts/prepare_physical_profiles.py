#!/usr/bin/env python3
"""Freeze the supported model303 profile branch from native input and source defaults.

This deliberately small parser accepts literal assignments used by the pinned
case, including D exponents, numeric arithmetic, indexed arrays and array
constructors. Unsupported expressions fail; no missing physics value is guessed.
No native simulation or training is performed.
"""
import argparse
import ast
import hashlib
import json
import operator
from pathlib import Path
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mastu_opsplit.implicit_opsplit import load_frozen_tangent
from mastu_opsplit.physical_rhs import NATIVE_PROFILE_KEYS, NativeProfiles
from mastu_opsplit.restart import read_restart

SOURCE = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/single_writer_build_20261102/source')
ARRAYS = dict(rho_coef=5, t_coef=5, ff_coef=9, corr_neg_temp_coef=2, corr_neg_dens_coef=2,
              d_perp=5, zk_perp=5, tgnum=7)
ZERO = '''tauic eta_num visco_num visco_par_num d_perp_num zk_perp_num edgeparticlesource
          particlesource_gauss heatsource_gauss v_0 v_1 implicit_heat_source'''.split()
FALSE = '''neo bootstrap use_pellet freeboundary freeboundary_equil refinement rmp_on
           keep_n0_const linear_run'''.split()
FILES = 'rho_file t_file ffprime_file rot_file d_perp_file zk_perp_file'.split()
OTHER = '''xpoint xcase keep_current_prof time_evol_scheme d_prof_neg_thresh zk_prof_neg_thresh
           rho_min_neg rho_min corr_neg_dens_coef'''.split()
NEEDED = NATIVE_PROFILE_KEYS | set(ZERO+FALSE+FILES+OTHER) | {'tgnum'}
ASSIGNMENT = re.compile(r'(?<![\w%])([a-z]\w*)\s*(?:\(([^()\n]*)\))?\s*=(?!=|>)', re.I)


def uncomment(line):
    quote = None
    for index, char in enumerate(line):
        if char in "\"'":
            quote = None if quote == char else char if quote is None else quote
        elif char == '!' and quote is None:
            return line[:index]
    return line


def assignments(text, source, *, namelist=False):
    """Record only explicit needed assignments; their source text stays in the manifest."""
    result = []
    for line_number, original in enumerate(text.splitlines(), 1):
        line = uncomment(original)
        matches = list(ASSIGNMENT.finditer(line))
        for index, match in enumerate(matches):
            key = match[1].lower()
            if key not in NEEDED:
                continue
            end = matches[index+1].start() if index+1 < len(matches) else len(line)
            expression = line[match.end():end].strip().rstrip(',;').strip()
            if not expression or '&' in expression:
                raise ValueError(f'Unsupported continued assignment {source}:{line_number}: {key}')
            result.append(dict(key=key, indices=match[2], expression=expression, source=str(source),
                               line=line_number, namelist=namelist))
    return result


def scalar(expression):
    value = expression.strip()
    if value.lower() in ('.true.', '.t.', '.false.', '.f.'):
        return value.lower() in ('.true.', '.t.')
    if value.startswith(("'", '"')):
        return ast.literal_eval(value)
    value = re.sub(r'(?<=[0-9.])[dD]([+-]?\d+)', r'e\1', value)
    tree = ast.parse(value, mode='eval')
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow}
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return evaluate(node.operand)*(1 if isinstance(node.op, ast.UAdd) else -1)
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError(f'Unsupported native literal expression: {expression}')
    result = evaluate(tree.body)
    if not np.isfinite(result):
        raise ValueError(f'Nonfinite native literal: {expression}')
    return result


def resolve(records):
    values, evidence = {}, {}
    for key in sorted(NEEDED):
        selected = [r for r in records if r['key'] == key]
        if not selected:
            raise ValueError(f'Missing explicit native default/input for {key}')
        if key not in ARRAYS:
            if selected[-1]['indices'] is not None:
                raise ValueError(f'Unexpected indexed scalar {key}')
            values[key] = scalar(selected[-1]['expression'])
            evidence[key] = selected[-1:]
            continue
        size = ARRAYS[key]
        value = [None]*size
        for record in selected:
            expression = record['expression']
            if expression.startswith('(/') and expression.endswith('/)'):
                expression = expression[2:-2]
            entries = [scalar(item) for item in expression.split(',')]
            spec = record['indices']
            if spec is None or spec.strip() == ':':
                indices = list(range(size))
                if record['namelist'] and spec is None:
                    indices = indices[:len(entries)]
            elif ':' in spec:
                start, end = (int(x) for x in spec.split(':'))
                indices = list(range(start-1, end))
            else:
                indices = [int(spec)-1]
            if not indices or min(indices) < 0 or max(indices) >= size:
                raise ValueError(f'Unsupported {key} array section {spec}')
            if len(entries) == 1 and not record['namelist']:
                entries *= len(indices)
            if len(entries) != len(indices):
                raise ValueError(f'Incomplete literal array assignment for {key}')
            for index, entry in zip(indices, entries):
                value[index] = entry
        if any(item is None for item in value):
            raise ValueError(f'Unresolved native array values for {key}')
        values[key], evidence[key] = value, selected
    return values, evidence


def audit_branches(values):
    for key in ZERO:
        if values[key] != 0:
            raise ValueError(f'Unsupported nonzero native branch: {key}={values[key]}')
    for key in FALSE:
        if values[key] is not False:
            raise ValueError(f'Unsupported active native branch: {key}')
    for key in FILES:
        if values[key] != 'none':
            raise ValueError(f'Unsupported numerical profile: {key}={values[key]}')
    if values['xpoint'] is not True or values['xcase'] != 3 or values['keep_current_prof'] is not True:
        raise ValueError('Require analytic double-null with the native current source enabled')
    if any(values['tgnum']) or values['time_evol_scheme'] not in ('Crank-Nicholson', 'Euler'):
        raise ValueError('Unsupported Taylor-Galerkin stabilization or native multistep history')
    if values['d_perp'][1] != 0 or values['zk_perp'][1] != 0:
        raise ValueError('Only constant perpendicular diffusivity profiles are implemented')
    if values['d_prof_neg_thresh'] > 0 or values['zk_prof_neg_thresh'] > 0:
        raise ValueError('Positive-state perpendicular transport overrides are unsupported')
    if values['rho_min_neg'] >= 0 or values['rho_1'] <= 0 or any(x <= 0 for x in values['corr_neg_dens_coef']):
        raise ValueError('Require the native exponential low-density mass correction')


def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def prepare(namelist, reference_restart, tangent, source_root):
    paths = {'namelist': Path(namelist), 'reference_restart': Path(reference_restart), 'frozen_tangent': Path(tangent)}
    for name, relative in [('defaults', 'models/preset_parameters.f90'),
                           ('initialization', 'models/initialise_and_broadcast_parameters.f90'),
                           ('model_initialization', 'models/model303/initialise_parameters.f90'),
                           ('plasma_functions', 'models/mod_plasma_functions.f90'),
                           ('profile_selection', 'models/read_num_profiles.f90')]:
        paths[name] = Path(source_root)/relative
    records = assignments(paths['defaults'].read_text(), paths['defaults'])
    records += assignments(paths['namelist'].read_text(), paths['namelist'], namelist=True)
    values, evidence = resolve(records)
    audit_branches(values)
    transformations = []
    if values['t_min_zkpar'] < -1e10:
        expected = re.compile(r'if\s*\(T_min_ZKpar\s*<\s*-1\.d10\)\s*T_min_ZKpar\s*=\s*T_min', re.I)
        if not expected.search(paths['initialization'].read_text()):
            raise ValueError('Native T_min_ZKpar sentinel initialization changed')
        transformations.append(dict(parameter='t_min_zkpar', before=values['t_min_zkpar'],
                                    after=values['t_min'], rule='native sentinel <-1e10 resolves to T_min'))
        values['t_min_zkpar'] = values['t_min']
    physics = {key: values[key] for key in sorted(NATIVE_PROFILE_KEYS)}
    physics['d_perp'], physics['zk_perp'] = values['d_perp'][0], values['zk_perp'][0]
    geometry, reference, time = read_restart(reference_restart)
    frozen = load_frozen_tangent(tangent, geometry)
    NativeProfiles(geometry, physics, None)  # Validate all active parameter shapes and supported ranges.
    local = reference[geometry.dof_map]
    actual = np.concatenate((np.einsum('eql,elc->eqc', geometry.basis, local),
                             np.einsum('eql,elc->eqc', geometry.basis_grad[..., 0], local),
                             np.einsum('eql,elc->eqc', geometry.basis_grad[..., 1], local)), -1)
    error = np.max(np.abs(actual-frozen.jets), axis=(0, 1))/(np.max(np.abs(frozen.jets), axis=(0, 1))+1e-30)
    if np.any(error > 1e-7):
        raise ValueError(f'Native reference differs from frozen jets: {error.max():.3g}')
    with h5py.File(reference_restart) as stream:
        native = {key: np.asarray(stream[key]).item() for key in
                  ('jorek_model', 'n_tor', 'n_coord_tor', 'n_order', 'n_degrees', 'n_var', 'n_dim', 'index_now', 't_now')}
        if np.asarray(stream['freeboundary']).item() not in (False, b'F'):
            raise ValueError('Reference restart enables unsupported free boundary')
        if not np.isclose(np.asarray(stream['F0']).item(), physics['f0'], rtol=0, atol=1e-15):
            raise ValueError('Reference F0 differs from native input')
    return dict(schema='mastu_native_profiles_v1', frozen_sha256=frozen.sha256,
                supported_branches='n0_model303_analytic_double_null_v1', physics=physics,
                native_parameters=values, parameter_evidence=evidence, transformations=transformations,
                reference=dict(**native, native_time=time, maximum_relative_jet_error=float(error.max()),
                               interpretation='actual warm native baseline; stationarity not assumed'),
                provenance={name: file_record(path) for name, path in paths.items()},
                scope='Configured coefficient only; exact volume RHS, frozen native coupled temporal operator. '
                      'No new simulation, training, future restart or fitted profile coefficient.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namelist', type=Path, required=True)
    parser.add_argument('--reference-restart', type=Path, required=True)
    parser.add_argument('--tangent', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.namelist, args.reference_restart, args.tangent, args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(file_record(args.output)), flush=True)


if __name__ == '__main__':
    main()
