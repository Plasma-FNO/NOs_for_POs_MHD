"""Freeze a capacity-safe directional probe using the unchanged validated solver."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import sha256_file


def main():
    strict=len(sys.argv)>1 and sys.argv[1]=='strict'
    reports=ROOT/'reports'
    record_path=reports/'cpu_spatial_next_20261103.json'
    record=json.loads(record_path.read_text())
    strict = strict or record.get('strict_equilibrium_tolerance_required',False)
    if record.get('native_wall_probe_prepared'):
        raise ValueError('directional probe already prepared; do not duplicate')
    history=reports/'cpu_spatial_before_native_wall_probe_20261103.json'
    if history.exists():
        raise ValueError('pre-probe history already saved; inspect preparation state')
    plan_record_path=Path(sys.argv[2]) if len(sys.argv)>2 else reports/'cpu_native_wall_mesh_20261103.json'
    plan_record=json.loads(plan_record_path.read_text())
    plan=Path(plan_record['bulk_output'])
    selection=json.loads((plan/'selection.json').read_text())
    if max(selection['nodes'],selection['elements'])>155000:
        raise ValueError('candidate exceeds frozen capacity margin')
    selected=plan/selection['selected']
    if json.loads((selected/'result.json').read_text())['status']!='completed':
        raise ValueError('positive finite geometry audit required')
    validated=json.loads((reports/'cpu_restart_capacity_recovery_20261103.json').read_text())
    source=Path(validated['source_directory'])
    executable=source/'jorek_model303'
    qualification_path=plan/'qualification.json'
    qualification=json.loads(qualification_path.read_text())
    validation_record=reports/'cpu_restart_capacity_recovery_20261103.json'
    if qualification['solver160_accepted']:
        source=plan/'source160'
        executable=plan/'solver160.exe'
        validated={**validated,'binary_sha256':qualification['solver160_sha256']}
        validation_record=qualification_path
    if sha256_file(executable)!=validated['binary_sha256']:
        raise ValueError('validated executable changed')
    row=next(x for x in subprocess.check_output(['saldo','-b','--dcgp','-n'],text=True).splitlines() if x.split()[0]=='FUPA2_FAUNA')
    cols=row.split()
    if min(int(cols[3])-int(cols[4]),int(cols[7])-int(cols[8]))<512:
        raise ValueError('insufficient verified credit for bounded directional probe')
    shutil.copyfile(record_path,history)
    record['preparation_history']=[*record.get('preparation_history',[]),
                                   {'path':str(history),'sha256':sha256_file(history)}]
    budget={'authorization':'Current explicit request to diagnose discrepancies and run next required spatial refinement',
            'scope':'One directional three-step native probe with the unchanged validated executable; previous conditional full-pair/analysis reservations are reused, not duplicated.',
            'allocations_max':1,'nodes':1,'mpi_ranks':4,'threads':1,'memory_GiB':256,'wall_seconds':7200,
            'process_timeout_seconds':6600,'attempts_max':1,'rank_CPU_hours_max':8,'node_hours_max':2,
            'conservative_resource_hours_max':512,'saldo_row':row,'equilibrium_tolerance':1e-10 if strict else 1e-6}
    budget_path=reports/'cpu_spatial_native_wall_probe_authorization_20261103.json'
    budget_path.write_text(json.dumps(budget,indent=2)+'\n')
    base=plan.parent/'spatial_native_wall_pair_20261103'
    base.mkdir()
    (base/'source').mkdir()
    shutil.copyfile(executable,base/'source/jorek_model303')
    (base/'source/jorek_model303').chmod(0o555)
    shutil.copyfile(plan/'jorek-env.sh',base/'jorek-env.sh')
    shutil.copyfile(reports/'cpu_spatial_next_worker_20261103.py',base/'worker.py')
    stages={}
    for label,steps,amplitude,timeout in [('probe3',3,0.,6600),('baseline48',48,0.,20400),('perturbed48',48,.05,20400)]:
        work=base/label;work.mkdir()
        overrides={'amix':.5,'nstep_n':steps,'nout':8,'opsplit_amplitude(1)':amplitude,'opsplit_amplitude(2)':amplitude}
        if strict:
            overrides['equil_accuracy']=1e-10
        (work/'input').write_text(namelist_overrides((selected/'input').read_text(),overrides))
        for name in ['eqdsk.dat','jorek_equil_rz.h5']:
            shutil.copyfile(selected/name,work/name)
        stages[label]={'steps':steps,'amplitudes':[amplitude,amplitude],'timeout_seconds':timeout,'bulk_output':str(work)}
    parent=plan.parent/'spatial_pair_20261103'
    reference_works={label:str(parent/label) for label in ['baseline48','perturbed48']}
    if strict:
        reference_works=record['analysis_reference_works']
    files=[base/'worker.py',base/'jorek-env.sh',base/'source/jorek_model303',selected/'jorek_restart.h5']
    files += [p for label in stages for p in (base/label).iterdir()]
    manifest={'stages':stages,'final_time_s':1.4059131112280784e-6,'tstep_code_units':.0625,
              'field_relative_target':.01,'integral_relative_target':.02,
              'required_field_groups':['baseline48_all_8','perturbed48_all_8','paired_response_all_8'],
              'equil_accuracy':1e-10 if strict else 1e-6,'equilibrium_relaxation_amix':.5,
              'validated_unchanged_binary_sha256':validated['binary_sha256'],
              'validated_binary_record_sha256':sha256_file(validation_record),
              'validated_binary_record':str(validation_record),
              'exact_integer_neighbor_optimization_qualified':qualification['solver160_accepted'],
              'planned_geometry_reference':str(selected/'jorek_restart.h5'),
              'planned_geometry_sha256':sha256_file(selected/'jorek_restart.h5'),
              'mesh_selection':selection,'planning_manifest_sha256':sha256_file(plan/'manifest.json'),
              'analysis_reference_works':reference_works,
              'references_policy':'Reference output identities are frozen separately in analysis_manifest.json after validated matched-tolerance reference runs complete.',
              'input_hashes':{str(p):sha256_file(p) for p in files}}
    (base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    script=reports/'cpu_spatial_native_wall_probe_20261103.sbatch'
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-directional-probe3
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=02:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {base}/worker.py {base} {ROOT} probe3
''')
    command=['sbatch','--parsable','--chdir',str(base),'--output',str(base/'probe-%j.out'),'--error',str(base/'probe-%j.err'),str(script)]
    subprocess.run(['bash','-n',str(script)],check=True)
    subprocess.run(['sbatch','--test-only',*command[1:]],check=True)
    job=subprocess.check_output(command,text=True).strip()
    record.update(bulk_output=str(base),source_directory=str(source),manifest_sha256=sha256_file(base/'manifest.json'),
                  stages=stages,status='directional_probe_submitted',native_wall_probe_prepared=True,
                  directional_budget_record=str(budget_path),analysis_reference_works=reference_works)
    record['jobs']['probe']={'job_id':job,'kind':'simulation','script':str(script),'script_sha256':sha256_file(script),
                             'budget_record':str(budget_path)}
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'probe_job':job,'selected_mesh':selection,'equilibrium_tolerance':manifest['equil_accuracy']}))


if __name__=='__main__':
    main()
