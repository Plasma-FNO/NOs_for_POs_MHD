"""Bounded model199 diagnostic pilot, run only inside an allocated Slurm job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
JR = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/src/jorek')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run inside Slurm; do not run simulations on the login node.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--quarter-step', action='store_true')
    args = parser.parse_args()
    out = ROOT / ('run_' + os.environ['SLURM_JOB_ID'])
    out.mkdir(exist_ok=False)
    source = ROOT / 'snapshot'
    specs = [('regression', 1, 3000., True, 1e-6),
             ('startup', 40, 3000., False, 1e-6),
             ('baseline', 30, 3000., True, 1e-6),
             ('half_dt', 60, 1500., True, 1e-6),
             ('eta_x2', 30, 3000., True, 2e-6)]
    if args.quarter_step:
        specs = [('quarter_dt', 120, 750., True, 1e-6)]
    cap = 300 if args.quarter_step else 170
    results = {'job_id': os.environ['SLURM_JOB_ID'], 'cases': {},
               'mpi_ranks': 2, 'omp_threads': 1, 'per_case_timeout_seconds': cap,
               'snapshot_hashes': {p.name: sha(p) for p in source.iterdir() if p.is_file()}}
    for name, steps, dt, restart, eta in specs:
        work = out / name
        work.mkdir()
        text = (source / 'input').read_text()
        for key, value in {'restart': '.t.' if restart else '.f.', 'nstep_n': steps,
                           'tstep_n': dt, 'nout': 1, 'eta': eta}.items():
            pattern = rf'(?mi)^(\s*{key}\s*=)[^!\n]*'
            text, count = re.subn(pattern, lambda m: m[1] + ' ' + str(value), text)
            if count != 1:
                raise ValueError((key, count))
        (work / 'input').write_text(text)
        if restart:
            shutil.copyfile(source / 'begin.h5', work / 'jorek_restart.h5')
            shutil.copyfile(source / 'begin.h5', work / 'initial.h5')
        start = time.monotonic()
        with (work / 'logfile').open('w') as log, (work / 'input').open() as inp:
            proc = subprocess.run(['timeout', '--kill-after=10', str(cap), 'mpirun', '--bind-to', 'none',
                                   '-n', '2', str(source / 'jorek_model199')],
                                  stdin=inp, stdout=log, stderr=subprocess.STDOUT, cwd=work)
        result = dict(returncode=proc.returncode, wall_seconds=time.monotonic()-start,
                      requested_steps=steps, dt_native=dt, restart=restart, eta=eta)
        log = (work / 'logfile').read_text(errors='replace')
        result['error_lines'] = [line for line in log.splitlines() if re.search(
            r'HDF5-DIAG|assertion.*fail|aborting|not converged|failed to converge|NaN|SIGSEGV', line, re.I)]
        path = work / 'jorek_restart.h5'
        if path.exists():
            with h5py.File(path) as f:
                final_step = int(f['index_now'][0]); final_time = float(f['t_now'][0])
                result.update(final_step=final_step, final_time_native=final_time,
                              finite=bool(np.isfinite(f['values'][()]).all()))
                result['completed_horizon'] = final_step == (30 if restart else 0)+steps and abs(final_time-((90000 if restart else 0)+steps*dt)) < 1e-6
                if name == 'regression':
                    with h5py.File(source / 'end.h5') as ref:
                        result['reference_max_abs_error'] = float(np.max(np.abs(f['values'][()]-ref['values'][()])))
                        result['reference_pass_1e_12'] = result['reference_max_abs_error'] <= 1e-12
        result['healthy_completion'] = (proc.returncode == 0 and not result['error_lines'] and result.get('finite', False) and result.get('completed_horizon', False))
        results['cases'][name] = result
        (out / 'run_report.json').write_text(json.dumps(results, indent=2)+'\n')
        print(name, json.dumps(result), flush=True)
        if name == 'regression' and not (result['healthy_completion'] and result.get('reference_pass_1e_12')):
            raise SystemExit('Regression failed; remaining pilot cases skipped.')


if __name__ == '__main__':
    main()
