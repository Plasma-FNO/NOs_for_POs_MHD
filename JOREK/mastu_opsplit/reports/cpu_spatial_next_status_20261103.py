"""Idempotent accounting and status for the current diagnosis/refinement phase."""
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    reports=Path(__file__).resolve().parent
    path=reports/'cpu_spatial_next_20261103.json'
    record=json.loads(path.read_text())
    diagnosis=json.loads((reports/'cpu_spatial_diagnosis_job_20261103.json').read_text())
    record['jobs']['diagnosis']={'job_id':diagnosis['job_id'],'kind':'analysis'}
    jobs={entry['job_id']:entry for entry in record['jobs'].values()}
    rows=subprocess.check_output(['sacct','-n','-P','-X','-j',','.join(jobs),
                                  '--format=JobIDRaw,State,ElapsedRaw,AllocCPUS,NNodes,ExitCode'],text=True)
    cpu,node,count=0.,0.,0
    for line in rows.splitlines():
        row=line.split('|')
        if row[0] not in jobs:
            continue
        entry=jobs[row[0]]
        entry['accounting']={'state':row[1],'elapsed_seconds':int(row[2]),'allocated_cpus':int(row[3]),
                             'nodes':int(row[4]),'exit_code':row[5]}
        cpu+=int(row[2])*int(row[3])/3600
        node+=int(row[2])*int(row[4])/3600
        count+=1
    for stage,entry in record['stages'].items():
        work=Path(entry['bulk_output'])
        if (work/'result.json').exists():
            entry['result']=json.loads((work/'result.json').read_text())
        steps=re.findall(r'After step\s+(\d+)',(work/'stdout.log').read_text()) if (work/'stdout.log').exists() else []
        entry['steps_logged']=int(steps[-1]) if steps else 0
    tolerance_record=reports/'cpu_equilibrium_tolerance_20261103.json'
    if tolerance_record.exists():
        tolerance=json.loads(tolerance_record.read_text())
        tolerance_result=Path(tolerance['bulk_output'])/'result.json'
        if tolerance_result.exists():
            record['equilibrium_tolerance_result']=json.loads(tolerance_result.read_text())
    report=reports/'cpu_spatial_next_convergence_20261103.json'
    if report.exists():
        record['comparison_report']=str(report)
        record['comparison_status']=json.loads(report.read_text())['status']
        record['status']='completed' if record['jobs'].get('analysis',{}).get('accounting',{}).get('state')=='COMPLETED' else 'analysis_finishing'
    elif any(entry.get('active',True) and entry.get('accounting',{}).get('state') in ['FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL'] for entry in jobs.values()):
        record['status']='requires_failure_review'
    elif record['jobs'].get('probe',{}).get('accounting',{}).get('state')=='COMPLETED' and 'baseline48' not in record['jobs']:
        record['status']='probe_validated'
    record['as_of_utc']=datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(record,indent=2)+'\n')
    ledger_path=reports/'resource_ledger.json'
    ledger=json.loads(ledger_path.read_text());before=record['ledger_before']
    simulations=sum(x['kind']=='simulation' for x in record['jobs'].values())
    ledger.update(as_of_utc=record['as_of_utc'],actual_hpc_cpu_core_hours=before['actual_hpc_cpu_core_hours']+cpu,
                  actual_hpc_node_hours=before['actual_hpc_node_hours']+node,
                  development_jobs_with_accounting=before['development_jobs_with_accounting']+count,
                  simulation_jobs_submitted=before['simulation_jobs_submitted']+simulations,
                  scientific_jobs_submitted=before['scientific_jobs_submitted']+simulations)
    ledger['spatial_next']={'record':path.name,'job_ids':list(jobs),'requested_cpu_core_hours_max':192,
                            'requested_node_hours_max':51,'scientific_gates_passed':[],
                            'usage_note':'Unique allocations include separate localization and composite capacity/probe; active elapsed usage provisional.'}
    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'jobs':{k:{'job_id':v['job_id'],**v.get('accounting',{})} for k,v in record['jobs'].items()},
                      'stages':{k:{'steps':v['steps_logged'],'result':v.get('result',{}).get('status')} for k,v in record['stages'].items()},
                      'phase_allocated_cpu_hours':cpu,'comparison_status':record.get('comparison_status')},indent=2))


if __name__=='__main__':
    main()
