"""Exercise production Fortran weights with JOREK's already-rescaled increments."""
from pathlib import Path
import os
import re
import shutil
import subprocess

import pytest


def test_native_variable_step_history(tmp_path):
    compiler = shutil.which('gfortran')
    if compiler is None:
        pytest.skip('Fortran compiler unavailable')
    root = Path(__file__).resolve().parents[1]
    patch = (root / 'scripts/prepare_full_split.py').read_text()
    declarations = patch.split("declarations = '''", 1)[1].split("'''", 1)[0]
    procedures = (root / 'native/full_split_procedures.f90').read_text()
    # Use the actual injected mass predictor, with only the fixture's array name changed.
    predictor = re.search(r'^\s*(os_mid=os_qv[^\n]+)', patch, re.M).group(1)
    predictor = predictor.replace('delta_g(mp,1:7,ms,mt)', 'native_delta')
    source = '''module phys_module
real*8 :: tstep,tstep_prev,time_evol_theta
end module
module split_fixture
implicit none
''' + declarations + '\ncontains\n' + procedures + '''
end module
program check_history
use split_fixture
use phys_module
implicit none
integer :: level,k,n
real*8 :: q,previous,old_n,h,r,n_now,mass,new_q,errors(3),os_qv(7),os_mid(7),native_delta(7)
allocate(opsplit_vertices(4,1),opsplit_coeff(361,16,1))
allocate(opsplit_previous(3,5,16,1),opsplit_current(3,5,16,1))
allocate(opsplit_seen(16,1),opsplit_seen_previous(16,1))
opsplit_current=0.d0;opsplit_seen=.true.
do level=1,3
  n=10*2**(level-1);q=2.d0;previous=q;old_n=0.d0
  opsplit_last_step=0;tstep_prev=1.d0
  do k=1,n
    h=.5d0/n*merge(2.d0/3.d0,4.d0/3.d0,mod(k,2)==1)
    tstep=h;r=h/tstep_prev
    call opsplit_begin_step(k,1)
    os_qv=q;native_delta=r*(q-previous)
''' + predictor + '''
    mass=os_mid(1)
    if(k>1) then
      if(abs(mass-(q+r*(q-previous)))>1.d-13) error stop 'Double-scaled mass predictor'
      if(abs(opsplit_history_weight*native_delta(1)-r*r/(1.d0+r)*(q-previous))>1.d-13) &
        error stop 'Double-scaled history action'
    endif
    n_now=-.6d0*(q-2.d0)
    new_q=q+(h*(-2.d0-.4d0*(q-2.d0)+n_now+opsplit_ratio*(n_now-old_n)) &
          +opsplit_history_weight*mass*native_delta(1))/(opsplit_mass_weight*mass+.4d0*h)
    previous=q;q=new_q;old_n=n_now;tstep_prev=h
  enddo
  errors(level)=abs(q-1.5d0)
enddo
if(any(errors(1:2)/errors(2:3)<3.5d0)) error stop 'Lost second-order convergence'
print *, 'PASS native rescaled history', errors
end program
'''
    path = tmp_path / 'history.f90'
    path.write_text(source)
    executable = tmp_path / 'history'
    subprocess.run([compiler, '-ffree-line-length-none', '-fcheck=all', str(path),
                    '-o', str(executable)], cwd=tmp_path, check=True, capture_output=True)
    result = subprocess.run([str(executable)], cwd=tmp_path, check=True, capture_output=True,
                            text=True, env={**os.environ, 'OPSSPLIT_MODE': 'numerical',
                                            'OPSSPLIT_SCHEME': 'sbdf2'})
    assert 'PASS native rescaled history' in result.stdout
