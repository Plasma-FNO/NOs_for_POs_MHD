"""Source-backed manifest parsing and explicit unsupported-physics rejection."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('prepare_physical_profiles',
        Path(__file__).parents[1]/'scripts/prepare_physical_profiles.py')
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def test_literal_fortran_defaults_support_exact_exponents_arithmetic_and_booleans():
    assert prepare.scalar('5.d0 / 3.d0') == 5/3
    assert prepare.scalar('0.01d-2') == 1e-4
    assert prepare.scalar('-1.d12') == -1e12
    assert prepare.scalar('.t.') is True
    assert prepare.scalar('.false.') is False
    assert prepare.scalar("'none'") == 'none'
    for expression in ['unknown_parameter', 'float(1)', '__import__("os")']:
        with pytest.raises(ValueError, match='Unsupported native literal'):
            prepare.scalar(expression)


def test_native_array_defaults_are_overridden_without_guessing_missing_entries(monkeypatch):
    monkeypatch.setattr(prepare, 'NEEDED', {'ff_coef', 'gamma', 'xcase'})
    default = 'FF_coef = 0.d0; FF_coef(1) = -1.d0\ngamma=5.d0/3.d0\nxcase=LOWER_XPOINT'
    case = 'FF_coef(1:3)=-1.2, .2, 0.\n FF_coef(8)=.03d0 ! stop comment\nxcase=3'
    records = prepare.assignments(default, 'defaults') + prepare.assignments(case, 'input', namelist=True)
    values, evidence = prepare.resolve(records)
    assert values['ff_coef'] == [-1.2, .2, 0., 0., 0., 0., 0., .03, 0.]
    assert values['gamma'] == 5/3 and values['xcase'] == 3
    assert evidence['ff_coef'][-1]['source'] == 'input'
    assert evidence['ff_coef'][-1]['line'] == 2
    with pytest.raises(ValueError, match='Missing explicit'):
        prepare.resolve([])


def supported():
    values = {k: 0. for k in prepare.ZERO}
    values.update({k: False for k in prepare.FALSE})
    values.update({k: 'none' for k in prepare.FILES})
    values.update(xpoint=True, xcase=3, keep_current_prof=True, tgnum=[0.]*7,
                  time_evol_scheme='Crank-Nicholson', d_perp=[1e-5, 0, 0, 99, 99],
                  zk_perp=[1e-5, 0, 0, 99, 99], d_prof_neg_thresh=0., zk_prof_neg_thresh=0.,
                  rho_min_neg=-1e12, rho_1=.01, corr_neg_dens_coef=[.5, .5])
    return values


@pytest.mark.parametrize('key,value', [('freeboundary', True), ('particlesource_gauss', 1e-3),
                                      ('rho_file', 'custom.dat'), ('xcase', 1),
                                      ('d_perp', [1e-5, 1e-6, 0, 99, 99]),
                                      ('d_prof_neg_thresh', .001), ('keep_current_prof', False)])
def test_unsupported_native_branch_fails_before_manifest_write(key, value):
    values = supported()
    prepare.audit_branches(values)
    values[key] = value
    with pytest.raises(ValueError):
        prepare.audit_branches(values)
