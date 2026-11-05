"""Refresh recorded campaign status from immutable run artifacts and Slurm."""
import collections,datetime,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];path=root/'comparison_status.json';s=json.loads(path.read_text())
s['recorded_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();ids=[]
for case,c in s['cases'].items():
 campaign=Path(c['native_campaign']);jobs=json.loads((campaign/'jobs.json').read_text())
 for j in jobs:c['jobs'][j['stage']]=j['job_id']
 for arm in ('ar','node','opsplit-scratch','opsplit-pretrained'):
  directory=(Path(s['campaign_root'])/case if arm in ('ar','node') else campaign)/arm
  history=directory/'history.jsonl';rows=[json.loads(l) for l in history.read_text().splitlines()] if history.exists() else []
  progress=dict(completed_epochs=len(rows),latest=rows[-1] if rows else None)
  metrics=directory/'test_metrics.json'
  if metrics.exists():
   d=json.loads(metrics.read_text());progress['test_complete']=d['complete']
   if d['complete']:progress['test_failures_by_horizon']={k:v['failed'] for k,v in d['horizons'].items()}
  log=directory.parent/f'{arm}-{c["jobs"][arm]}.log'
  if log.exists():
   for line in log.read_text().splitlines():
    if not line.startswith('{'):continue
    try:row=json.loads(line)
    except ValueError:continue
    if 'offset' in row:progress['latest_batch']=row
  if (directory/'failure.json').exists():progress['failure']=json.loads((directory/'failure.json').read_text())
  c['training_progress'][arm]=progress
 for arm in ('opsplit-scratch','opsplit-pretrained'):
  report=campaign/(arm+'-audit.json');d=json.loads(report.read_text()) if report.exists() else {}
  failures=d.get('failures',[])
  c['native_audits'][arm]=dict(report=str(report),expected_windows=c['valid_training_windows'],attempted_windows=d.get('windows',0),gradient_trajectories=d.get('gradient_trajectories',0),validation_trajectories=d.get('validation_trajectories',0),completed=d.get('completed',False),passed=d.get('passed',False),failure_count=len(failures),failure_types=dict(collections.Counter('CUDA OOM' if 'out of memory' in f['error'] else f['error'] for f in failures)),first_failures=failures[:3])
 ids.extend(c['jobs'].values())
s['scheduler_snapshot']=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(sorted(set(ids))),'--format=JobID,JobName%70,State,ExitCode,Elapsed'],text=True).strip().splitlines()
s['scheduler_queue']=subprocess.check_output(['squeue','-h','-j',','.join(sorted(set(ids))),'-o','%i|%j|%T|%M|%R'],text=True).strip().splitlines()
path.write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps({case:c['native_audits'] for case,c in s['cases'].items()},indent=2))
print('\n'.join(s['scheduler_queue']))
