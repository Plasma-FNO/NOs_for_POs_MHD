import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from mastu_opsplit.common import config_hash, sha256_file
from mastu_opsplit.runner import classify, restart_health, run_case, render_slurm


def prepared_fake(tmp_path, source, *, steps=0, final_time=0):
    config = {"case_id": "fake", "parent_family_id": "fake-parent", "evolution": {"nstep_n": steps},
              "execution": {"mpi_ranks": 1, "omp_threads": 1, "max_attempts": 1, "timeout_s": 10,
                            "launcher": [], "require_slurm": False},
              "acceptance": {"required_files": ["jorek_restart.h5"], "final_time_s": final_time},
              "slurm": {"account": None, "partition": None, "walltime": "00:01:00", "memory": "1G", "nodes": 1}}
    directory = tmp_path / "prepared"
    directory.mkdir()
    for name, text in {"jorek.exe": f"#!{sys.executable}\n" + source, "input": " &in1\n /\n", "eqdsk.dat": "fixture"}.items():
        (directory / name).write_text(text)
    (directory / "jorek.exe").chmod(0o755)
    manifest = {"config_hash": config_hash(config), "snapshot_hashes": {name: sha256_file(directory / name) for name in ("jorek.exe", "input", "eqdsk.dat")}}
    (directory / "case.json").write_text(json.dumps(manifest))
    return config, directory


def test_false_zero_gmres_and_incomplete_output(tmp_path):
    config, prepared = prepared_fake(tmp_path, "print('NO CONVERGENCE AFTER 100 ITERATIONS. ABORTING')\n")
    result = run_case(config, prepared, tmp_path / "abort")
    assert result["status"] == "solver_aborted"
    assert result["attempts"][0]["returncode"] == 0
    (prepared / "jorek.exe").write_text(f"#!{sys.executable}\nprint('finished')\n")
    manifest = json.loads((prepared / "case.json").read_text())
    manifest["snapshot_hashes"]["jorek.exe"] = sha256_file(prepared / "jorek.exe")
    (prepared / "case.json").write_text(json.dumps(manifest))
    assert run_case(config, prepared, tmp_path / "incomplete")["status"] == "incomplete_output"
    with pytest.raises(FileExistsError):
        run_case(config, prepared, tmp_path / "incomplete")


def test_dry_run_and_prepared_integrity(tmp_path):
    config, prepared = prepared_fake(tmp_path, "raise RuntimeError('must not run')")
    run_case(config, prepared, tmp_path / "dry", dry_run=True)
    assert not (tmp_path / "dry").exists()
    with pytest.raises(ValueError, match="account"):
        render_slurm(config, prepared, tmp_path / "out", tmp_path / "config.yaml")
    (prepared / "input").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        run_case(config, prepared, tmp_path / "out", dry_run=True)


def test_restart_final_time_and_finiteness(tmp_path):
    path = tmp_path / "restart.h5"
    with h5py.File(path, "w") as f:
        for key, value in {"index_now": 2, "n_tor": 1, "sqrt_mu0_rho0": 1e-6, "xtime": [1, 2],
                           "energies": np.ones((2, 2, 1))}.items():
            f[key] = value
    assert restart_health(path, expected_steps=2, final_time_s=2e-6)["steps"] == 2
    with pytest.raises(ValueError, match="final time"):
        restart_health(path, expected_steps=2, final_time_s=3e-6)
    with h5py.File(path, "r+") as f:
        f["xtime"][1] = 1
    with pytest.raises(ValueError, match="increase"):
        restart_health(path, expected_steps=2, final_time_s=2e-6)


def test_classification():
    assert classify(0, False, "noise") == "completed"
    assert classify(0, True, "noise") == "timeout"
    assert classify(0, False, "tools/RTree.h: Assertion failed") == "rtree_assertion"
    assert classify(0, False, "HDF5-DIAG: Error detected in HDF5 (1.14.3)") == "hdf5_error"


def test_hdf5_failure_with_zero_exit_is_not_completed(tmp_path):
    config, prepared = prepared_fake(tmp_path, "print('HDF5-DIAG: Error detected in HDF5 (1.14.3)')\n")
    result = run_case(config, prepared, tmp_path / "hdf5-failed")
    assert result["status"] == "hdf5_error"
    assert result["attempts"][0]["returncode"] == 0
    assert len(result["attempts"]) == 1


def test_equilibrium_comparison_checks_native_values(tmp_path):
    from mastu_opsplit.runner import equilibrium_regression
    actual, reference = tmp_path / "actual.h5", tmp_path / "reference.h5"
    for path in (actual, reference):
        with h5py.File(path, "w") as f:
            f["values"] = [1.,2.]
    assert equilibrium_regression(actual, reference)["maximum_absolute_difference"] == 0
    with h5py.File(actual, "r+") as f:
        f["values"][0] = 1.01
    with pytest.raises(ValueError, match="discrepancy"):
        equilibrium_regression(actual, reference)


def test_resource_caps_and_cleanup_margin(tmp_path):
    from mastu_opsplit.runner import run_plan
    config, prepared = prepared_fake(tmp_path, "print('not run')")
    def update_identity():
        manifest = json.loads((prepared / "case.json").read_text())
        manifest["config_hash"] = config_hash(config)
        (prepared / "case.json").write_text(json.dumps(manifest))
    config["execution"]["mpi_ranks"] = 5
    update_identity()
    with pytest.raises(ValueError, match="caps"):
        run_plan(config, prepared, tmp_path / "unused")
    config["execution"]["mpi_ranks"] = 1
    config["slurm"].update(account="reviewed", partition="debug", walltime="00:01:00")
    update_identity()
    with pytest.raises(ValueError, match="cleanup"):
        render_slurm(config, prepared, tmp_path / "unused", "case.yaml")
    config["slurm"]["walltime"] = "00:02:00"
    update_identity()
    assert "--time=00:02:00" in render_slurm(config, prepared, tmp_path / "unused", "case.yaml")


def refinement_case(tmp_path):
    config, prepared = prepared_fake(tmp_path, "print('not run')")
    budget = tmp_path/'budget.json'
    budget.write_text(json.dumps({'resource_profile': 'cpu_refinement_20261102',
        'slurm_account': 'fupa2_fauna', 'remaining_local_resource_hours': 10000,
        'monthly_remaining_local_resource_hours': 5000,
        'phase': {'conservative_local_resource_hour_envelope': 2048}}))
    config['resource_authorization'] = {'submission_allowed': True,
        'resource_profile': 'cpu_refinement_20261102', 'budget_record': str(budget),
        'budget_record_sha256': sha256_file(budget)}
    config['execution'].update(mpi_ranks=16, timeout_s=7140)
    config['slurm'].update(account='fupa2_fauna', partition='dcgp_fua_prod',
                           walltime='02:00:00', memory='128G')
    return config, prepared, budget


def sync_config_identity(config, prepared):
    path = prepared/'case.json'
    manifest = json.loads(path.read_text())
    manifest['config_hash'] = config_hash(config)
    path.write_text(json.dumps(manifest))


def test_budget_backed_refinement_profile(tmp_path):
    config, prepared, budget = refinement_case(tmp_path)
    sync_config_identity(config, prepared)
    script = render_slurm(config, prepared, tmp_path/'output', 'case.yaml')
    assert '--ntasks=16' in script and '--mem=128G' in script and '--time=02:00:00' in script
    budget.write_text(budget.read_text()+'\n')
    with pytest.raises(ValueError, match='budget record identity'):
        render_slurm(config, prepared, tmp_path/'output', 'case.yaml')
    config['resource_authorization']['budget_record_sha256'] = sha256_file(budget)
    record = json.loads(budget.read_text())
    record['monthly_remaining_local_resource_hours'] = 2047
    budget.write_text(json.dumps(record))
    config['resource_authorization']['budget_record_sha256'] = sha256_file(budget)
    sync_config_identity(config, prepared)
    with pytest.raises(ValueError, match='insufficient verified CPU budget'):
        render_slurm(config, prepared, tmp_path/'output', 'case.yaml')


@pytest.mark.parametrize('section,key,value,match', [
    ('execution', 'mpi_ranks', 17, 'caps'),
    ('execution', 'omp_threads', 2, 'caps'),
    ('execution', 'max_attempts', 2, 'caps'),
    ('slurm', 'walltime', '02:00:01', 'walltime cap'),
    ('slurm', 'memory', '129G', 'memory cap'),
    ('slurm', 'partition', 'dcgp_fua_dbg', 'account/partition'),
])
def test_refinement_profile_limits(tmp_path, section, key, value, match):
    config, prepared, _ = refinement_case(tmp_path)
    config[section][key] = value
    sync_config_identity(config, prepared)
    with pytest.raises(ValueError, match=match):
        render_slurm(config, prepared, tmp_path/'output', 'case.yaml')


def test_unapproved_refinement_profile_is_rejected(tmp_path):
    config, prepared, _ = refinement_case(tmp_path)
    config['resource_authorization']['submission_allowed'] = False
    sync_config_identity(config, prepared)
    with pytest.raises(ValueError, match='unapproved resource profile'):
        render_slurm(config, prepared, tmp_path/'output', 'case.yaml')
