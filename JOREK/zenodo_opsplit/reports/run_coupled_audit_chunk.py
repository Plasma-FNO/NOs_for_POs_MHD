"""Advance a held audit in an existing allocation; always release its queued job."""
import json,subprocess,sys,shlex
from pathlib import Path
case,arm,parent=sys.argv[1:4]
root=Path(sys.argv[4]) if len(sys.argv)>4 else Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-matched-coupled-20261104')/case
jobs=json.loads((root/'jobs.json').read_text());job=next(j for j in jobs if j['stage']==arm+'-audit')
info=subprocess.check_output(['scontrol','show','job',job['job_id'],'-o'],text=True)
assert 'JobState=PENDING' in info and 'JobName=jorek-compare-'+case+'-'+arm+'-audit ' in info
subprocess.run(['scontrol','hold',job['job_id']],check=True)
try:
 old=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-matched-20261104')/case
 report=root/(arm+'-audit.json')
 cmd=['/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python',str(root/'code/train_comparison.py'),'--data',str(old/'data'),'--arm',arm,'--output',str(report),'--audit','--audit-workers','4','--max-seconds','600']
 if arm.endswith('pretrained'):cmd+=['--pretrained',str(old/'terms/best.pt')]
 if report.exists():cmd+=['--resume']
 command=['srun','--overlap','--jobid='+parent,'--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--time=00:40:00','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1']+cmd
 print(shlex.join(command),flush=True)
 result=subprocess.run(command)
 print('Audit chunk exit',result.returncode,'(75 means resumable partial coverage)',flush=True)
 if result.returncode not in (0,75):raise SystemExit(result.returncode)
finally:
 subprocess.run(['scontrol','release',job['job_id']],check=True)
