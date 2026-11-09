"""Launch the full pair only after measured capacity parity and probe validation."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.common import sha256_file


def physics_text(path):
    """Compare complete native inputs after excluding only spatial count controls."""
    counts={'n_flux','n_tht','n_open','n_outer','n_inner','n_leg','n_leg_out',
            'n_private','n_up_leg','n_up_leg_out','n_up_priv','n_tht_equidistant'}
    lines=[]
    for line in Path(path).read_text().splitlines():
        left=line.split('=',1)[0].replace(' ','').replace('\t','').lower()
        if left in counts or left.startswith('n_ext_block(') or left.startswith('n_ext_equidistant('):
            continue
        if line.strip():
            lines.append(line.strip())
    return '\n'.join(lines)


def main():
    reports=ROOT/'reports'
    path=reports/'cpu_spatial_next_20261103.json'
    record=json.loads(path.read_text());base=Path(record['bulk_output'])
    if any(label in record['jobs'] for label in ['baseline48','perturbed48','analysis']):
        raise ValueError('full pair already submitted; preserve jobs')
    if sha256_file(base/'manifest.json')!=record['manifest_sha256']:
        raise ValueError('frozen manifest changed')
    manifest=json.loads((base/'manifest.json').read_text())
    proof_stages=['probe3'] if 'validated_unchanged_binary_sha256' in manifest else ['regression8','probe3']
    for stage in proof_stages:
        result=json.loads((base/stage/'result.json').read_text())
        if result['status']!='completed' or sha256_file(base/'source/jorek_model303')!=result['binary_sha256']:
            raise ValueError('validated matching binary required')
    job=record['jobs']['probe']['job_id']
    state=subprocess.check_output(['sacct','-n','-P','-X','-j',job,'--format=State,ExitCode'],text=True).strip()
    if state!='COMPLETED|0:0':
        raise ValueError('probe allocation not cleanly completed: '+state)
    # Recheck shared credit before reserving the larger pair.
    row=next(line for line in subprocess.check_output(['saldo','-b','--dcgp','-n'],text=True).splitlines() if line.split()[0]=='FUPA2_FAUNA')
    cols=row.split()
    if min(int(cols[3])-int(cols[4]),int(cols[7])-int(cols[8]))<4096:
        raise ValueError('insufficient current shared project credit')
    record['launch_credit_row']=row
    analysis_manifest=base/'manifest.json'
    if 'analysis_reference_works' in manifest:
        analysis_manifest=base/'analysis_manifest.json'
        if analysis_manifest.exists():
            raise ValueError('analysis references already frozen; inspect existing submission state')
        references={}
        for label,folder in manifest['analysis_reference_works'].items():
            work=Path(folder)
            result=json.loads((work/'result.json').read_text())
            if result['status']!='completed':
                raise ValueError('validated matched-tolerance reference required: '+label)
            if physics_text(work/'input')!=physics_text(base/label/'input'):
                raise ValueError('reference and refined physics/numerical settings differ beyond spatial counts: '+label)
            for name in ['eqdsk.dat','jorek_equil_rz.h5']:
                if sha256_file(work/name)!=sha256_file(base/label/name):
                    raise ValueError('reference equilibrium source changed: '+name)
            for time,name in [('initial','jorek000000.h5'),('final','jorek_restart.h5')]:
                if sha256_file(work/name)!=result[time]['restart_sha256']:
                    raise ValueError('reference identity changed: '+label+'/'+time)
            if result['final']['health']['steps']!=48 or abs(result['final']['health']['final_time_s']-manifest['final_time_s'])>1e-15:
                raise ValueError('reference time or steps differ')
            references[label]={'work':str(work),'initial_sha256':result['initial']['restart_sha256'],
                               'final_sha256':result['final']['restart_sha256'],'input_sha256':sha256_file(work/'input')}
        analysis_manifest.write_text(json.dumps({**manifest,'references':references,
            'simulation_manifest_sha256':sha256_file(base/'manifest.json')},indent=2)+'\n')
        record['analysis_manifest']=str(analysis_manifest)
        record['analysis_manifest_sha256']=sha256_file(analysis_manifest)
    python='/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
    def submit(label,cpus,memory,hours,body,dependency=None):
        script=reports/f'cpu_spatial_next_{label}_20261103.sbatch'
        script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-next-{label}
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks={cpus}
#SBATCH --cpus-per-task=1
#SBATCH --mem={memory}G
#SBATCH --time={hours:02d}:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
{body}
''')
        subprocess.run(['bash','-n',str(script)],check=True)
        command=['sbatch','--parsable','--chdir',str(base),'--output',str(base/f'{label}-%j.out'),
                 '--error',str(base/f'{label}-%j.err')]
        if dependency:
            command+=['--dependency','afterok:'+':'.join(dependency)]
        command.append(str(script))
        preflight=subprocess.run(['sbatch','--test-only',*command[1:]],text=True,capture_output=True,check=True)
        job=subprocess.check_output(command,text=True).strip()
        record['jobs'][label]={'job_id':job,'kind':'analysis' if label=='analysis' else 'simulation',
                              'script':str(script),'script_sha256':sha256_file(script),
                              'submission_command':command,'scheduler_preflight':preflight.stderr}
        path.write_text(json.dumps(record,indent=2)+'\n')
        return job
    ids=[submit(label,4,256,6,f'{python} {base}/worker.py {base} {ROOT} {label}') for label in ['baseline48','perturbed48']]
    analysis=reports/'cpu_spatial_next_analysis_20261103.py'
    hashes={str(p):sha256_file(p) for p in [analysis,reports/'cpu_spatial_damped_tests_20261103.xml',
            ROOT/'src/mastu_opsplit/queries.py',ROOT/'src/mastu_opsplit/restart.py',ROOT/'src/mastu_opsplit/numerical.py',base/'manifest.json',analysis_manifest]}
    checksum=base/'analysis_startup.sha256'
    checksum.write_text(''.join(value+'  '+name+'\n' for name,value in hashes.items()))
    record['analysis_input_hashes']=hashes
    submit('analysis',1,64,1,f'sha256sum --check {checksum} > {base}/analysis_startup_verification.log\n{python} {analysis} {base} {analysis_manifest}',ids)
    record['status']='pair_submitted'
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'simulation_jobs':ids,'analysis_job':record['jobs']['analysis']['job_id']}))


if __name__=='__main__':
    main()
