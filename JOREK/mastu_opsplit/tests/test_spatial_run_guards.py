import json
from pathlib import Path

import pytest

from mastu_opsplit.common import sha256_file
from test_spatial_pair import report_module


@pytest.mark.parametrize('failure', ['failed_capacity_parity', 'changed_validated_binary', 'changed_probe_binary'])
def test_unvalidated_executable_never_launches_full_case(tmp_path, monkeypatch, failure):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'reports'))
    monkeypatch.setattr('sys.argv', ['worker.py', str(tmp_path), str(root), 'baseline48'])
    worker = report_module('cpu_spatial_next_worker_20261103.py')
    (tmp_path/'source').mkdir()
    binary = tmp_path/'source/jorek_model303'
    binary.write_bytes(b'fake executable must never be launched')
    digest = sha256_file(binary)
    (tmp_path/'probe3').mkdir()
    (tmp_path/'regression8').mkdir()
    (tmp_path/'probe3/result.json').write_text(json.dumps({
        'status':'completed','binary_sha256':'changed' if failure=='changed_probe_binary' else digest}))
    (tmp_path/'regression8/result.json').write_text(json.dumps({'status':'failed_validation'}))
    manifest = {'input_hashes':{},'stages':{'baseline48':{}}}
    if failure != 'failed_capacity_parity':
        manifest['validated_unchanged_binary_sha256'] = 'changed' if failure=='changed_validated_binary' else digest
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    def forbidden(*args, **kwargs):
        pytest.fail('native execution reached despite invalid proof')
    monkeypatch.setattr(worker.subprocess,'run',forbidden)
    with pytest.raises(ValueError):
        worker.main()


@pytest.mark.parametrize('setting', ['equil_accuracy', 'tstep_n', 'eta'])
def test_reference_matching_allows_mesh_counts_but_rejects_numerical_or_physical_changes(tmp_path, setting):
    launcher=report_module('cpu_spatial_next_launch_20261103.py')
    original='&in1\n n_flux=40\n n_ext_block(3)=76\n equil_accuracy=1e-10\n tstep_n=.0625\n eta=1e-5\n/\n'
    old,new=tmp_path/'old',tmp_path/'new'
    old.write_text(original)
    refined=original.replace('n_flux=40','n_flux=240').replace('n_ext_block(3)=76','n_ext_block(3)=152')
    refined=refined.replace('/\n',' n_tht_equidistant=.true.\n n_ext_equidistant(3)=.true.\n/\n')
    new.write_text(refined)
    assert launcher.physics_text(old)==launcher.physics_text(new)
    values={'equil_accuracy':'1e-10','tstep_n':'.0625','eta':'1e-5'}
    new.write_text(refined.replace(setting+'='+values[setting],setting+'=1e-6'))
    assert launcher.physics_text(old)!=launcher.physics_text(new)

