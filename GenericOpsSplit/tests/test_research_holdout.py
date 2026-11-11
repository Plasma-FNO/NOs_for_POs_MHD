import json

import pytest
import torch

from gos.research_holdout import benchmark, error_summary, freeze, generate, merge_reports, ood_plan, paired_bootstrap, read_frozen, step_counts
from gos import research_holdout


def test_paired_bootstrap_preserves_matching_trajectories():
    result = paired_bootstrap([[.1, .2, .3], [.2, .4, .6]], [[.2, .4, .6], [.4, .8, 1.2]], repeats=100)
    assert result['ratio'] == pytest.approx(.5)
    assert result['ci95'] == pytest.approx([.5, .5])
    assert result['candidate_seed_means'] == pytest.approx([.2, .4])
    assert paired_bootstrap([[.1, .2]], [[0., 0.]], repeats=100)['ratio'] is None
    with pytest.raises(ValueError, match='matched'):
        paired_bootstrap([[.1, .2]], [[.1, .2, .3]])


def test_freeze_detects_modified_checkpoints(tmp_path):
    directory = tmp_path/'candidate'
    directory.mkdir()
    (directory/'result.json').write_text(json.dumps(dict(problem='burgers', arm='candidate', seed=0, config={})))
    (directory/'model.pt').write_bytes(b'original checkpoint')
    manifest = tmp_path/'manifest.json'
    freeze([directory], manifest)
    assert read_frozen(manifest)['runs'][0]['arm'] == 'candidate'
    with pytest.raises(FileExistsError):
        freeze([directory], manifest)
    (directory/'model.pt').write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='Frozen artifact changed'):
        read_frozen(manifest)


def test_error_summary_retains_failures():
    summary = error_summary([.1, float('nan')])
    assert summary['mean'] == float('inf')
    assert summary['finite_fraction'] == .5
    assert summary['mean_clipped'] == pytest.approx(1.05)


def test_holdout_generation_requires_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='requires a GPU'):
        generate(tmp_path/'manifest.json', tmp_path, tmp_path/'holdout')
    assert not (tmp_path/'holdout').exists()


def test_freeze_rejects_mixed_training_budgets(tmp_path):
    directories = []
    for seed, iterations in ((0, 100), (1, 200)):
        directory = tmp_path/f'seed{seed}'
        directory.mkdir()
        (directory/'result.json').write_text(json.dumps(dict(
            problem='burgers', arm='candidate', seed=seed, config=dict(iters=iterations))))
        (directory/'model.pt').write_bytes(b'checkpoint')
        directories.append(directory)
    with pytest.raises(ValueError, match='Mixed scientific configurations'):
        freeze(directories, tmp_path/'manifest.json')


def test_generation_rejects_ood_seed_overlap(monkeypatch, tmp_path):
    monkeypatch.setattr(research_holdout, 'device_info', lambda: {})
    # A fresh test seed of 0 would reuse the old OOD generation seed of 100.
    with pytest.raises(ValueError, match='independent generation seed'):
        generate(tmp_path/'manifest.json', tmp_path, tmp_path/'holdout', seed=0)


@pytest.fixture
def problem_reports(tmp_path):
    manifest = dict(runs=[])
    reports = {}
    for problem, arms in [('burgers', ['candidate', 'classical']), ('kdv', ['candidate'])]:
        runs = []
        for arm in arms:
            item = dict(problem=problem, arm=arm, seed=0, config={}, checkpoint_sha256=problem+arm)
            manifest['runs'].append(item)
            errors = [.1, .2] if arm == 'candidate' else [.2, .4]
            runs.append(dict(item, holdout_sha256=problem+'data', generation_seeds=dict(test=7103),
                             primary_horizon=4, splits={'test': {'x1': {'4': dict(errors=errors)}}}))
        reports[problem] = dict(device=dict(requested='cuda'), manifest=manifest, runs=runs,
                                selection_problem=problem, complete=False, complete_for_problem=True)
    paths = []
    for problem, report in reports.items():
        path = tmp_path/f'{problem}.json'
        path.write_text(json.dumps(report))
        paths.append(path)
    return paths


def test_merge_problem_reports_recomputes_paired_comparisons(problem_reports, tmp_path):
    destination = tmp_path/'merged.json'
    report = merge_reports(problem_reports, destination)
    assert report['complete'] and report['total_frozen_runs'] == 3
    assert len(report['comparisons']) == 2
    assert sorted(item['aggregate']['ratio'] for item in report['comparisons']) == pytest.approx([.5, 2.])
    with pytest.raises(FileExistsError):
        merge_reports(problem_reports, destination)
    for path in problem_reports:
        part = json.loads(path.read_text())
        part['timing_enabled'] = True
        for run in part['runs']:
            run['timing'] = {'x1': dict(seconds_per_macrostep=.1)}
        path.write_text(json.dumps(part))
    timed = merge_reports(problem_reports, tmp_path/'timed.json')
    assert timed['timing_enabled']
    assert all(run['timing']['x1']['seconds_per_macrostep'] == .1 for run in timed['runs'])


@pytest.mark.parametrize('corruption,match', [
    ('missing', 'Missing frozen runs'), ('duplicate', 'Duplicate evaluated run'),
    ('manifest', 'Mismatched frozen manifests'), ('checkpoint', 'checkpoint hash'),
    ('holdout', 'holdout data hashes'), ('configuration', 'frozen configuration'),
    ('timing', 'timing flags')])
def test_merge_rejects_missing_duplicate_or_mismatched_results(problem_reports, tmp_path, corruption, match):
    paths = list(problem_reports)
    if corruption == 'missing':
        paths.pop()
    elif corruption == 'duplicate':
        paths.append(paths[0])
    else:
        path = paths[1] if corruption == 'manifest' else paths[0]
        report = json.loads(path.read_text())
        if corruption == 'manifest':
            report['manifest']['different_source'] = True
        elif corruption == 'checkpoint':
            report['runs'][0]['checkpoint_sha256'] = 'wrong'
        elif corruption == 'holdout':
            report['runs'][1]['holdout_sha256'] = 'wrong'
        elif corruption == 'timing':
            report['timing_enabled'] = True
            for run in report['runs']:
                run['timing'] = {'x1': dict(seconds_per_macrostep=.1)}
        else:
            report['runs'][0]['config'] = dict(iters=1)
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=match):
        merge_reports(paths, tmp_path/'merged.json')


def test_timing_counts_capped_stages_equilibrium_subtraction_and_jvps():
    from gos.models import StepModel
    from gos.problems import get_problem
    from gos.research_models import ETDReference, ResearchModel
    problem = get_problem('burgers')
    net = dict(modes=4, width=8, layers=2)
    model = ResearchModel(problem, 'ars222', net, 1., 'cpu', substeps=2,
                          max_step=True, zero_equilibrium=True)
    counts = step_counts(model, 4)
    assert counts['internal_substeps'] == 8 and counts['neural_calls_per_macrostep'] == 2
    model.closure_each_stage = True
    assert step_counts(model, 4)['neural_calls_per_macrostep'] == 32
    exact = ETDReference(problem, 1., 'cpu', substeps=2, max_step=True)
    assert step_counts(exact, 4)['internal_substeps'] == 8
    assert step_counts(exact, 4)['neural_calls_per_macrostep'] == 0
    tangent = StepModel(problem, 'implicit_free_tf', net, 1., 'cpu')
    assert step_counts(tangent, 1)['neural_calls_per_macrostep'] == 2
    assert step_counts(tangent, 1)['neural_jvps_per_macrostep'] == 1
    with pytest.raises(ValueError, match='requires CUDA'):
        benchmark(model, torch.zeros(2, 2, 1, 256), torch.ones(2))


def test_extra_ood_plan_preserves_original_cases_and_checks_every_seed():
    from gos.problems import get_problem
    problem = get_problem('burgers_fem')
    values, seeds = ood_plan(problem, 7103, [.003, .018])
    assert values == [.002, .024, .003, .018]
    assert seeds == dict(test=7103, ood={'0.002': 7203, '0.024': 7303, '0.003': 7403, '0.018': 7503})
    with pytest.raises(ValueError, match='all test and OOD'):
        ood_plan(problem, 2**64-250, [.003, .018])
    for extras in ([.006], [.004], [.002], [.003, .003], [float('nan')], [-.003]):
        with pytest.raises(ValueError):
            ood_plan(problem, 7103, extras)
    with pytest.raises(ValueError, match='varying-coefficient'):
        ood_plan(get_problem('ks'), 7103, [2.])


def test_extra_ood_requires_one_problem(monkeypatch, tmp_path):
    monkeypatch.setattr(research_holdout, 'device_info', lambda: {})
    with pytest.raises(ValueError, match='exactly one selected'):
        generate(tmp_path/'manifest.json', tmp_path, tmp_path/'holdout', extra_ood=[.003, .018])
