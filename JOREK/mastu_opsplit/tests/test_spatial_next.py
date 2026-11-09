import json
from pathlib import Path
import shutil

import h5py
import pytest

from test_restart import rectangle
from test_spatial_pair import report_module
from mastu_opsplit.common import sha256_file


@pytest.mark.parametrize('manifest_name', ['manifest.json', 'analysis_manifest.json'])
def test_small_full_field_error_cannot_hide_unconverged_vorticity_response(tmp_path, monkeypatch, manifest_name):
    analysis = report_module('cpu_spatial_next_analysis_20261103.py')
    study = tmp_path/'study'
    (study/'reports').mkdir(parents=True)
    (study/'src/mastu_opsplit').mkdir(parents=True)
    shutil.copyfile(analysis.ROOT/'src/mastu_opsplit/queries.py', study/'src/mastu_opsplit/queries.py')
    base = tmp_path/'pair';base.mkdir()
    references = {}
    final_time = 1.4059131112280784e-6
    for mesh in ['existing', 'refined']:
        for label in ['baseline48', 'perturbed48']:
            work = (tmp_path/'old' if mesh=='existing' else base)/label
            work.mkdir(parents=True)
            hashes = {}
            for time,name in [('initial','jorek000000.h5'),('final','jorek_restart.h5')]:
                path = work/name
                rectangle(path)
                with h5py.File(path,'r+') as f:
                    f['index_now'][...] = 48 if time=='final' else 0
                    f['t_now'][...] = final_time/1e-6 if time=='final' else 0.
                    values=f['values'][()]
                    if time=='final':
                        values[3,0,0]=100.
                        if label=='perturbed48':
                            values[3,0,0]+=1.1 if mesh=='existing' else 1.
                    if label=='perturbed48':
                        values[4:6]*=1.05
                    f['values'][...]=values
                hashes[time+'_sha256']=sha256_file(path)
            if mesh=='existing':
                references[label]={'work':str(work),**hashes}
            else:
                (work/'result.json').write_text(json.dumps({'status':'completed',
                    'initial':{'restart_sha256':hashes['initial_sha256']},
                    'final':{'restart_sha256':hashes['final_sha256']}}))
    (base/manifest_name).write_text(json.dumps({'references':references,'final_time_s':final_time,
                                                 'field_relative_target':.01,'integral_relative_target':.02}))
    monkeypatch.setattr(analysis,'ROOT',study)
    arguments=[analysis.__file__,str(base)]
    if manifest_name!='manifest.json':
        arguments.append(str(base/manifest_name))
    monkeypatch.setattr(analysis.sys,'argv',arguments)
    analysis.main()
    result=json.loads((study/'reports/cpu_spatial_next_convergence_20261103.json').read_text())
    assert result['full_fields_one_percent_target_met_on_covered_volume']
    assert result['paired_density_temperature_pressure_one_percent_target_met_on_covered_volume']
    assert not result['paired_all_fields_one_percent_target_met_on_covered_volume']
    assert not result['comparisons']['final']['paired_response']['w']['target_met']
    assert result['status']=='spatial_targets_not_met'
