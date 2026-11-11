"""Qualification reporting tests; synthetic reports only, no training or GPU work."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mastu_opsplit.common import config_hash, sha256_file

spec = importlib.util.spec_from_file_location('physical_qualification',
    Path(__file__).parents[1]/'scripts/summarize_physical_qualification.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def save(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def campaign(tmp_path):
    manifest = tmp_path/'manifest.json'
    cases = [f'test{i}' for i in range(20)]
    save(manifest, dict(cases=[dict(case_id='train', split='train')] +
                           [dict(case_id=case, split='test') for case in cases]))
    profile = tmp_path/'profile.json'
    save(profile, {})
    mean, scale = np.zeros(7), np.ones(7)
    normalization = dict(mean_sha256=hashlib.sha256(mean.tobytes()).hexdigest(),
                         scale_sha256=hashlib.sha256(scale.tobytes()).hexdigest(), training_case_ids=['train'])
    settings = dict(schema='mastu_physical_qualification_v1', expected_seeds=[17,18,19],
        training_updates=2, evaluation_horizons=[16,32,96], evaluation_requested_substeps=[1,2,4],
        internal_substeps=2, expected_test_cases=20, physics=dict(run_dir='physics'),
        learned=[dict(seed=seed, run_dir=f'seed{seed}') for seed in (17,18,19)],
        bootstrap=dict(samples=10000, seed=4242, confidence=.95),
        primary=dict(horizon=96, requested_substeps=1, maximum_mean_ratio=.95),
        source_sha256={'module.py':'source'}, manifest_sha256=sha256_file(manifest))
    for seed in (None,17,18,19):
        arm = 'physics' if seed is None else 'implicit_rollout'
        run = tmp_path/('physics' if seed is None else f'seed{seed}')
        folder = run/arm
        folder.mkdir(parents=True)
        config = dict(manifest=str(manifest), steps=2, seed=seed or 17, diagnostic_only=False,
            pretrain_steps=0, evaluation_horizons=[16,32,96], evaluation_substeps=[1,2,4], device='cuda',
            physical_defect=dict(internal_substeps=2, scheme='be',
                                 profile_manifest=dict(path=str(profile), sha256=sha256_file(profile))))
        provenance = dict(config=config, arm=arm, normalization=normalization,
                          source_sha256=settings['source_sha256'], manifest_sha256=settings['manifest_sha256'])
        save(run/'provenance.json', provenance)
        save(run/'verification.json', dict(all_trajectory_and_initial_hashes_verified=True,
            all_coefficients_finite=True, normalization=normalization))
        np.savez(run/'normalization.npz', mean=mean, scale=scale, training_case_ids=['train'])
        save(run/'results.json', {})
        records = []
        for index, case in enumerate(cases):
            for substeps in (1,2,4):
                rows = []
                for horizon in (16,32,96):
                    mse = (1+index/20)*(1 if seed is None else {17:.88,18:.90,19:.92}[seed])
                    stages = horizon*substeps*2
                    row = dict(case_id=case, horizon=horizon, substeps=substeps, start_frame=0,
                        audit_passed=True, normalized_volume_mse=mse, raw_normalized_volume_mse=mse*1.1,
                        per_field_normalized_mse=dict.fromkeys(summary.FIELDS,mse),
                        raw_per_field_normalized_mse=dict.fromkeys(summary.FIELDS,mse*1.1),
                        dynamic_relative_rms=dict.fromkeys(summary.FIELDS,1.), elapsed_seconds=1.,
                        guard_statistics=dict(limited_field_steps=1, minimum_alpha=.9),
                        stage_statistics=dict(stages=stages, boundary_iterations=stages,
                            physical_rhs_evaluations=stages, learned_network_evaluations=0 if seed is None else stages*2,
                            maximum_guard_change=.1, maximum_held_dof_change=0.))
                    rows.append(row)
                save(folder/f'evaluation_{index:03d}_substeps{substeps}.json',
                     dict(records=rows, failure=None, provenance_hash=config_hash(provenance)))
                records.extend(rows)
        aggregates = [dict(horizon=h, substeps=s, trajectories=20, expected_trajectories=20,
            audit_failures=0, complete=True, mean_normalized_volume_mse=float(np.mean([
                row['normalized_volume_mse'] for row in records if row['horizon']==h and row['substeps']==s])))
            for h in (16,32,96) for s in (1,2,4)]
        metrics = dict(arm=arm, device='cuda', device_name='synthetic', evaluation=dict(
            aggregates=aggregates, numerical_failures=0, audits_passed=True))
        if seed is not None:
            metrics.update(selected_step=2, rollout_history=[dict(step=i, training_seconds=1.) for i in (1,2)])
            (folder/'rollout').mkdir()
            torch.save(dict(step=2, selection='final fixed-budget update', provenance_hash=config_hash(provenance)),
                       folder/'rollout/selected.pt')
        save(folder/'metrics.json', metrics)
    path = tmp_path/'campaign.json'
    save(path, settings)
    return path


def test_complete_shared_control_qualification(campaign):
    result = summary.summarize(campaign)
    assert result['qualification_success'] and result['qualification_complete']
    assert (result['expected_rollouts'],result['expected_horizon_records'],result['expected_aggregates']) == (240,720,36)
    assert result['primary']['mean_ratio'] == pytest.approx(.9)
    assert result['primary']['bootstrap_interval'] == pytest.approx([.9,.9])
    assert len(result['comparisons']) == 9
    assert result['comparisons'][0]['seed_ratio_std'] == pytest.approx(.02)
    assert result['comparisons'][0]['physics']['work']['learned_network_evaluations'] == 0


@pytest.mark.parametrize('problem', ['missing_seed','missing_file','failure','audit','nonfinite',
                                    'raw_nonfinite','stale','control_network','malformed','budget'])
def test_incomplete_or_invalid_never_averages_survivors(campaign, problem):
    folder = campaign.parent/'seed19/implicit_rollout'
    path = folder/'evaluation_000_substeps1.json'
    if problem == 'missing_seed':
        (folder.parent/'provenance.json').unlink()
    elif problem == 'missing_file':
        path.unlink()
    elif problem == 'budget':
        metrics = json.loads((folder/'metrics.json').read_text())
        metrics['selected_step'] = 1
        save(folder/'metrics.json', metrics)
    else:
        if problem == 'control_network':
            path = campaign.parent/'physics/physics/evaluation_000_substeps1.json'
        report = json.loads(path.read_text())
        if problem == 'failure':
            report['failure'] = 'root solve failed'
            report['records'] = report['records'][:1]
        if problem == 'audit': report['records'][0]['audit_passed'] = False
        if problem == 'nonfinite': report['records'][0]['normalized_volume_mse'] = float('nan')
        if problem == 'raw_nonfinite': report['records'][0]['raw_normalized_volume_mse'] = float('nan')
        if problem == 'stale': report['provenance_hash'] = 'stale'
        if problem == 'control_network': report['records'][0]['stage_statistics']['learned_network_evaluations'] = 2
        if problem == 'malformed': report['records'][0].pop('guard_statistics')
        save(path, report)
    result = summary.summarize(campaign)
    assert not result['qualification_complete'] and not result['qualification_success']
    assert result['comparisons'] == [] and result['primary'] is None
    if problem == 'failure': assert result['runs']['19']['failures']


def test_bootstrap_uses_common_trajectory_draws_for_all_seeds():
    physics = np.array([1.,4.,9.])
    learned = np.array([[.5,8.,4.],[2.,1.,18.],[1.,3.,10.]])
    draws = np.array([[0,0,1],[1,2,2],[2,0,1],[0,0,0]])
    result = summary.paired_bootstrap(learned,physics,draws,.95)
    samples = np.array([[learned[seed,draw].mean()/physics[draw].mean() for draw in draws] for seed in range(3)])
    np.testing.assert_allclose(result['interval'],np.quantile(samples.mean(axis=0),[.025,.975]))
    np.testing.assert_allclose(result['per_seed_intervals'],np.quantile(samples,[.025,.975],axis=1).T)


def test_zero_control_draw_is_retained_as_undefined():
    result = summary.paired_bootstrap(np.ones((3,2)),np.array([0.,1.]),np.array([[0,0],[0,1]]),.95)
    assert result['interval'] is None
    assert 'no draws were discarded' in result['reason']


def test_missing_declared_seed_rejected(campaign):
    settings = json.loads(campaign.read_text())
    settings['learned'].pop()
    save(campaign,settings)
    with pytest.raises(ValueError,match='every intended seed'):
        summary.summarize(campaign)


@pytest.mark.parametrize('targets, success', [([.88,.90,1.02], False), ([.96,.96,.96], False), ([.949,.949,.949], True)])
def test_material_gain_requires_every_seed_and_five_percent(campaign, targets, success):
    for seed, initial, target in zip((17,18,19),(.88,.90,.92),targets):
        folder = campaign.parent/f'seed{seed}/implicit_rollout'
        factor = target/initial
        for path in folder.glob('evaluation_*.json'):
            report = json.loads(path.read_text())
            for row in report['records']:
                row['normalized_volume_mse'] *= factor
                row['per_field_normalized_mse'] = {field:value*factor for field,value in row['per_field_normalized_mse'].items()}
            save(path,report)
        metrics = json.loads((folder/'metrics.json').read_text())
        for row in metrics['evaluation']['aggregates']:
            row['mean_normalized_volume_mse'] *= factor
        save(folder/'metrics.json',metrics)
    result = summary.summarize(campaign)
    assert result['qualification_complete']
    assert result['qualification_success'] is success
