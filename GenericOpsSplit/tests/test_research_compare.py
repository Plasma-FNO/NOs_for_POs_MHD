import copy
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('research_compare', Path(__file__).parents[1]/'research_compare.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def test_comparison_preserves_budgets_and_flags_incomplete_metrics(tmp_path):
    baseline = {'example': {'horizons': ['4'], 'ood': ['2.0'], 'arms': {
        'node': {'test': {'4': {'mean': .4, 'std': .01, 'n': 3}}},
        'implicit_known': {'test': {'4': {'mean': .2, 'std': .01, 'n': 3}}}}}}
    run = dict(problem='example', arm='candidate', seed=0, parameters=10, config=dict(iters=100, scheme='ars222'),
               train=dict(status='completed', best_iteration=100, seconds=1.),
               eval=dict(test={'4': dict(mean_clipped=.1)}, long=dict(bounded=1.),
                         ood={'2.0': {'4': dict(mean_clipped=.3)}}))
    for seed in (0, 1):
        path = tmp_path/'runs'/'example'/f'candidate_s{seed}'/'result.json'
        path.parent.mkdir(parents=True)
        run['seed'] = seed
        if seed == 1:
            run['config'].update(operator_diagnostics=False, skip_jacobian=False,
                                 max_step=False, zero_equilibrium=False, closure_in_range=False, lr_min=1e-5)
        path.write_text(json.dumps(run))
    longer = copy.deepcopy(run)
    longer['config']['iters'] = 200
    path = tmp_path/'runs'/'example'/'candidate_longer'/'result.json'
    path.parent.mkdir()
    path.write_text(json.dumps(longer))
    summary = compare.aggregate([tmp_path, tmp_path], baseline)
    rows = summary['example']['runs']
    assert len(rows) == 2
    shorter = next(row for row in rows if row['config']['iters'] == 100)
    assert shorter['seeds'] == [0, 1]
    assert shorter['historical_ratio'] == pytest.approx(.5)
    assert shorter['primary']['n'] == 2
    assert shorter['transfer'] == {}
    assert shorter['ks_spectrum'] is None
    assert 'zero-parameter' in compare.markdown(summary)
    capped = copy.deepcopy(run)
    capped['config']['max_step'] = True
    capped['seed'] = 2
    path = tmp_path/'runs'/'example'/'candidate_capped'/'result.json'
    path.parent.mkdir()
    path.write_text(json.dumps(capped))
    rows = compare.aggregate([tmp_path], baseline)['example']['runs']
    assert len(rows) == 3
    assert sum(row['config']['max_step'] for row in rows) == 1
    anchored = copy.deepcopy(run)
    anchored['config']['zero_equilibrium'] = True
    anchored['seed'] = 3
    path = tmp_path/'runs'/'example'/'candidate_zero'/'result.json'
    path.parent.mkdir()
    path.write_text(json.dumps(anchored))
    rows = compare.aggregate([tmp_path], baseline)['example']['runs']
    assert len(rows) == 4
    assert sum(row['config']['zero_equilibrium'] for row in rows) == 1
    gated = copy.deepcopy(run)
    gated['config']['closure_in_range'] = True
    gated['seed'] = 4
    path = tmp_path/'runs'/'example'/'candidate_gated'/'result.json'
    path.parent.mkdir()
    path.write_text(json.dumps(gated))
    rows = compare.aggregate([tmp_path], baseline)['example']['runs']
    assert len(rows) == 5
    assert sum(row['config']['closure_in_range'] for row in rows) == 1


def test_missing_or_nonfinite_metrics_are_not_silently_dropped():
    assert compare.statistics([1., None]) is None
    assert compare.statistics([1., float('inf')])['mean'] == float('inf')
