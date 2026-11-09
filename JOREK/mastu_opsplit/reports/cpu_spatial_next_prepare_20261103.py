"""Freeze the measured next spatial refinement and bounded resource request."""
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import sha256_file


def main():
    reports = ROOT/'reports'
    record_path = reports/'cpu_spatial_next_20261103.json'
    if record_path.exists():
        raise ValueError('next refinement already prepared; do not duplicate submissions')
    previous = json.loads((reports/'cpu_spatial_pair_20261103.json').read_text())
    parent = Path(previous['bulk_output'])
    diagnosis = json.loads((reports/'cpu_spatial_diagnosis_20261103.json').read_text())
    assert diagnosis['snapshots']['48']['comparisons']['paired_response']['w']['relative_l2_difference'] > .01
    source = Path(previous['source_directory'])
    if sha256_file(source/'jorek_model303') != previous['binary_sha256']:
        raise ValueError('validated parent executable changed')
    row = next(line for line in subprocess.check_output(['saldo','-b','--dcgp','-n'], text=True).splitlines()
               if line.split()[0]=='FUPA2_FAUNA')
    cols = row.split()
    remaining, monthly = int(cols[3])-int(cols[4]), int(cols[7])-int(cols[8])
    if min(remaining, monthly)<4096:
        raise ValueError('insufficient verified project credit')
    budget = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'authorization': 'Explicit current user request: diagnose remaining discrepancies and run next refinement toward 1% spatial convergence',
              'scope': 'One capacity-build/regression/probe allocation, two conditional matched-horizon simulations, one dependent analysis',
              'saldo_row': row, 'remaining_resource_hours': remaining, 'monthly_remaining_resource_hours': monthly,
              'account': 'fupa2_fauna', 'partition': 'dcgp_fua_prod',
              'probe': {'allocations_max':1,'nodes':1,'mpi_ranks':4,'threads':1,'memory_GiB':256,'wall_seconds':7200},
              'simulations': {'allocations_max':2,'concurrent_max':2,'nodes_each':1,'mpi_ranks_each':4,'threads':1,
                              'memory_GiB_each':256,'wall_seconds_each':21600,'process_timeout_seconds_each':20400},
              'analysis': {'allocations_max':1,'nodes':1,'cpus':1,'memory_GiB':64,'wall_seconds':3600},
              'attempts_per_allocation':1,'compiler_processes_max':2,'cpu_core_hours_max':57,'node_hours_max':15,
              'conservative_local_resource_hours_max':4096,
              'billing_note':'Memory-weighted site resource-hours differ from allocated rank CPU-hours.',
              'scientific_training_enabled':False,'production_ensemble_enabled':False}
    budget_path = reports/'cpu_spatial_next_authorization_20261103.json'
    budget_path.write_text(json.dumps(budget,indent=2)+'\n')
    base = parent.parent/'spatial_next_20261103'
    base.mkdir()
    shutil.copytree(source, base/'source')
    patch = ROOT/'patches/0014-cover-next-spatial-grid-capacity.patch'
    applied = subprocess.run(['patch','--batch','-p1','-i',str(patch)], cwd=base/'source',text=True,capture_output=True,check=True)
    shutil.copyfile(parent/'jorek-env.sh', base/'jorek-env.sh')
    shutil.copyfile(reports/'cpu_spatial_next_worker_20261103.py', base/'worker.py')
    shutil.copyfile(parent/'baseline48/jorek000008.h5',base/'reference8.h5')
    original = (parent/'baseline48/input').read_text()
    # Double all flux-aligned topology counts; leave preliminary R/Z and all spacing/physics controls fixed.
    counts = {'n_flux':80,'n_tht':440,'n_open':8,'n_outer':64,'n_inner':16,
              'n_leg':24,'n_leg_out':160,'n_private':16,'n_up_leg':24,'n_up_leg_out':160,'n_up_priv':16,
              **{f'n_ext_block({i})':152 if i==3 else 80 for i in range(1,12)}}
    stages = {}
    for label,steps,amplitude,timeout in [('regression8',8,0.,1800),('probe3',3,0.,4800),
                                           ('baseline48',48,0.,20400),('perturbed48',48,.05,20400)]:
        work=base/label;work.mkdir()
        overrides = {'nstep_n':steps,'nout':8 if steps==48 else 1,
                     'opsplit_amplitude(1)':amplitude,'opsplit_amplitude(2)':amplitude}
        if label!='regression8':
            overrides.update(counts)
        (work/'input').write_text(namelist_overrides(original,overrides))
        for name in ['eqdsk.dat','jorek_equil_rz.h5']:
            shutil.copyfile(parent/'baseline48'/name, work/name)
        stages[label]={'steps':steps,'amplitudes':[amplitude,amplitude],'timeout_seconds':timeout,'bulk_output':str(work)}
    references = {label:{'work':str(parent/label),
                         'initial_sha256':sha256_file(parent/label/'jorek000000.h5'),
                         'final_sha256':sha256_file(parent/label/'jorek_restart.h5')}
                  for label in ['baseline48','perturbed48']}
    files = [base/'worker.py',base/'reference8.h5',base/'jorek-env.sh',base/'source/models/mod_settings.f90']
    files += [p for label in stages for p in (base/label).iterdir()]
    manifest = {'created_utc':budget['created_utc'],'parent_manifest_sha256':sha256_file(parent/'manifest.json'),
                'diagnosis_report':str(reports/'cpu_spatial_diagnosis_20261103.json'),
                'diagnosis_sha256':sha256_file(reports/'cpu_spatial_diagnosis_20261103.json'),
                'patch_sha256':sha256_file(patch),'patch_application_stdout':applied.stdout,
                'stages':stages,'references':references,'mesh_overrides':counts,
                'final_time_s':previous['final_time_s'],'tstep_code_units':.0625,
                'field_relative_target':.01,'integral_relative_target':.02,
                'required_field_groups':['baseline48_all_8','perturbed48_all_8','paired_response_all_8'],
                'coverage_policy':'Report uncovered physical volume explicitly; no extrapolation; do not claim full-domain convergence with uncovered points.',
                'parity_protocol':json.loads((reports/'cpu_native_parity_protocol_20261102.json').read_text()),
                'input_hashes':{str(p.relative_to(base)):sha256_file(p) for p in files}}
    (base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    record={'status':'prepared','bulk_output':str(base),'source_directory':str(base/'source'),
            'budget_record':str(budget_path),'budget_sha256':sha256_file(budget_path),
            'manifest_sha256':sha256_file(base/'manifest.json'),'stages':stages,'jobs':{},
            'ledger_before':json.loads((reports/'resource_ledger.json').read_text()),'scientific_gates_passed':[]}
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    python='/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
    script=reports/'cpu_spatial_next_probe_20261103.sbatch'
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-next-spatial-probe
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
cd {base}/source
timeout --signal=TERM --kill-after=10s 600s make -j2 jorek_model303 > {base}/build.out 2> {base}/build.err
sha256sum jorek_model303 > {base}/binary.sha256
{python} {base}/worker.py {base} {ROOT} regression8
{python} {base}/worker.py {base} {ROOT} probe3
''')
    subprocess.run(['bash','-n',str(script)],check=True)
    command=['sbatch','--parsable','--chdir',str(base),'--output',str(base/'probe-%j.out'),
             '--error',str(base/'probe-%j.err'),str(script)]
    check=subprocess.run(['sbatch','--test-only',*command[1:]],text=True,capture_output=True,check=True)
    job=subprocess.check_output(command,text=True).strip()
    record['jobs']['probe']={'job_id':job,'script':str(script),'script_sha256':sha256_file(script),
                             'scheduler_preflight':check.stderr,'submission_command':command,'kind':'simulation'}
    record['status']='probe_submitted'
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'probe_job':job,'bulk_output':str(base),'planned_final_elements_approx':190000,
                      'budget_rank_CPU_hours_max':57,'target':.01}),flush=True)


if __name__=='__main__':
    main()
