import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import orchestrate_native_transfer as sweep
from comparison import write_json
from data import select
from summarize_native_transfer import summarize_report


def test_sweep_pairs_all_arms_and_nested_cohorts(tmp_path):
    plan=sweep.trials([0,1,2],[8,32,64])
    assert len(plan)==36 and len({t['directory'] for t in plan})==36
    manifest=dict(entries=[dict(case='em',split='train',id=str(i),family=str(i)) for i in range(76)])
    for seed in (0,1,2):
        subsets=[{e['id'] for e in select(manifest,'em','train',n,seed)} for n in (8,32,64)]
        assert subsets[0]<subsets[1]<subsets[2]
    for trial in plan:
        audit,train=sweep.commands(tmp_path,trial,sys.executable,250)
        for key in ('--manifest','--case','--pretrained','--train-trajectories','--seed','--substeps'):
            assert audit[audit.index(key)+1]==train[train.index(key)+1]
        assert train[train.index('--epochs')+1]=='250'
        assert '--checkpoint-steps' in train
        assert audit[audit.index('--workers')+1]=='1'


def test_array_script_audits_once_and_resumes_after_timed_exit(tmp_path,monkeypatch):
    trial=dict(seed=0,budget=8,arm='scratch',directory='trial')
    folder=tmp_path/'trial';calls=tmp_path/'calls'
    helper=tmp_path/'fake.py'
    helper.write_text('import pathlib,sys\n'
        f'p=pathlib.Path({str(calls)!r})\n'
        'with p.open("a") as f:f.write(" ".join(sys.argv[1:])+"\\n")\n'
        'if sys.argv[1]=="train":\n'
        f' d=pathlib.Path({str(folder/"training")!r});d.mkdir(exist_ok=True)\n'
        ' (d/"last.pt").touch()\n'
        ' if "--resume" not in sys.argv:raise SystemExit(75)\n')
    monkeypatch.setattr(sweep,'commands',lambda *a:([sys.executable,str(helper),'audit'],[sys.executable,str(helper),'train']))
    monkeypatch.setattr(sweep,'evaluation_commands',lambda *a:[[sys.executable,str(helper),'eval10'],[sys.executable,str(helper),'eval100']])
    script=sweep.render_trial(tmp_path,trial,sys.executable,250)
    binary=tmp_path/'bin';binary.mkdir();scontrol=binary/'scontrol'
    scontrol.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> '+str(tmp_path/'requeues')+'\n')
    scontrol.chmod(0o700)
    env=dict(os.environ,PATH=str(binary)+':'+os.environ['PATH'],SLURM_ARRAY_JOB_ID='123',SLURM_ARRAY_TASK_ID='7')
    for _ in range(4):subprocess.run(['bash',str(script)],env=env,check=True)
    assert calls.read_text().splitlines()==['audit','train','train --resume','eval10','eval100']
    assert (tmp_path/'requeues').read_text().splitlines()==['requeue 123_7']*3
    assert (folder/'training.done').is_file()


def test_summary_keeps_failed_and_incomplete_trajectories(tmp_path):
    path=tmp_path/'test.json'
    frame=dict(observed_relative_l2=[.2]*6,persistence_observed_relative_l2=[.5]*6,delta_psi_relative_l2=.1)
    good=dict(id='a',status='ok',frames=[frame])
    invalid=dict(id='b',status='invalid_release_state',frames=[])
    write_json(path,dict(trajectories=[good,invalid]))
    result=summarize_report(path,['a','b'],1)
    assert result['status']=='ok' and result['excluded_ids']==['b']
    assert result['shared_relative_l2']==pytest.approx(.2)
    write_json(path,dict(trajectories=[good,dict(id='b',status='failed',frames=[])]))
    assert summarize_report(path,['a','b'],1)['status']=='failed'
    write_json(path,dict(trajectories=[good]))
    assert summarize_report(path,['a','b'],1)['status']=='incomplete'


def test_evaluation_records_invalid_pressure_before_inference(tmp_path,monkeypatch):
    import evaluate_native_transfer as runner
    from test_workflow import make_h5
    entries=[]
    for name in ('a','b'):
        entry=make_h5(tmp_path/(name+'.h5'),case='em');entry['split']='test';entries.append(entry)
    manifest=tmp_path/'manifest.json';write_json(manifest,dict(entries=entries))
    def invalid(*args):
        q=torch.ones(3,4,12,12);q[1,1]=0
        return q,torch.ones(12),torch.ones(12)
    monkeypatch.setattr(runner,'load_trajectory',invalid)
    output=tmp_path/'test.json'
    monkeypatch.setattr(sys,'argv',['evaluate','--manifest',str(manifest),'--output',str(output),
                                 '--trajectories','0','--device','cpu','--horizon','2'])
    with pytest.raises(SystemExit) as exc:runner.main()
    assert exc.value.code==1
    rows=json.loads(output.read_text())['trajectories']
    assert len(rows)==2 and all(r['status']=='invalid_release_state' for r in rows)
