import json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
r=Path('/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit/reports')
phase=json.loads((r/'cpu_refinement_continuation_20261102.json').read_text()); small=json.loads((r/'cpu_mesh22_continuation_20261102.json').read_text())
records=small['runs']+phase['builds']+phase['runs']
ids=[str(x['job_id']) for x in records]
cmd=['sacct','-n','-P','-X','-j',','.join(ids),'--format=JobIDRaw,State,ElapsedRaw,AllocCPUS,NNodes,ExitCode,AllocTRES,Partition']
x=subprocess.run(cmd,capture_output=True,text=True,check=True); rows={}
for line in x.stdout.splitlines():
 t=line.split('|')
 if len(t)<8 or t[0] not in ids:continue
 rows[t[0]]={'state':t[1],'elapsed_seconds':int(t[2]),'allocated_cpus':int(t[3]),'allocated_nodes':int(t[4]),'exit_code':t[5],'alloc_tres':t[6],'partition':t[7]}
for rec in records:
 job=str(rec['job_id'])
 if job in rows:rec['accounting']=rows[job]
 if 'bulk_output' in rec:
  result=Path(rec['bulk_output'])/'result.json'
  if result.is_file():
   rec['result']=json.loads(result.read_text()); rec['status']=rec['result']['status']
  elif (Path(rec['bulk_output'])/'returncode.txt').is_file():
   rc=int((Path(rec['bulk_output'])/'returncode.txt').read_text());rec['status']='diagnostic_completed' if rc==0 else 'diagnostic_failed'; rec['returncode']=rc
 elif job in rows:rec['status']=rows[job]['state'].lower()
 if job=='2768359':rec['failure_site']='segment_surface_length -> define_leg_grid; process SIGSEGV, zero evolution steps'
 if job=='2768404':rec['failure_site']='close_contour_with_wall -> get_symmetric_separatrix_contours; process SIGSEGV, zero evolution steps'
for name,data in [('cpu_mesh22_continuation_20261102.json',small),('cpu_refinement_continuation_20261102.json',phase)]: (r/name).write_text(json.dumps(data,indent=2)+'\n')
extra_cpu=sum(a['elapsed_seconds']*a['allocated_cpus']/3600 for a in rows.values());extra_node=sum(a['elapsed_seconds']*a['allocated_nodes']/3600 for a in rows.values())
usage={'as_of_utc':datetime.now(timezone.utc).isoformat(),'base_before_budget_refinement':{'allocated_cpu_core_hours':5.0072222222222225,'allocated_node_hours':1.64,'simulation_jobs':31,'build_jobs':7},'additional_unique_jobs':rows,'additional_allocated_cpu_core_hours':extra_cpu,'additional_allocated_node_hours':extra_node,'scope':'Unique scheduler allocations, including failed jobs; running-job elapsed usage is provisional. Memory-weighted final billing is separate. Unrelated jobs excluded.'}
(r/'cpu_budget_refinement_usage_20261102.json').write_text(json.dumps(usage,indent=2)+'\n')
p=r/'resource_ledger.json';ledger=json.loads(p.read_text());ledger.update(as_of_utc=usage['as_of_utc'],actual_hpc_cpu_core_hours=5.0072222222222225+extra_cpu,actual_hpc_node_hours=1.64+extra_node,simulation_jobs_submitted=31+len(small['runs'])+len(phase['runs']),scientific_jobs_submitted=31+len(small['runs'])+len(phase['runs']),cpu_build_jobs_submitted=7+len(phase['builds']),development_jobs_with_accounting=38+len(rows),usage_scope=usage['scope'])
ledger['cpu_refinement']={'budget_record':'reports/cpu_budget_authorization_20261102.json','phase_record':'reports/cpu_refinement_continuation_20261102.json','usage_record':'reports/cpu_budget_refinement_usage_20261102.json','reserved_cpu_core_hours':phase['reserved_cpu_core_hours'],'reserved_node_hours':phase['reserved_node_hours'],'scientific_training_enabled':False,'production_ensemble_enabled':False}
p.write_text(json.dumps(ledger,indent=2)+'\n')
print(json.dumps({'accounting':rows,'total_allocated_cpu_hours':ledger['actual_hpc_cpu_core_hours']},indent=2))
