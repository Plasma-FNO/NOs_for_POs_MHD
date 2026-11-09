"""Ensure fallback activation distinguishes stalled saddle switching from convergence."""
from pathlib import Path
import subprocess
import numpy as np
import pytest


@pytest.fixture(scope='module')
def detects_cycle(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    work = tmp_path_factory.mktemp('xpoint_cycle')
    driver = '''program check
 implicit none
 real*8 :: points(2,2,4),updates(4)
 logical :: opsplit_xpoint_cycle
 read(*,*) points,updates
 print *,merge(1,0,opsplit_xpoint_cycle(points,updates,1.d0,1.d-10))
end program
'''
    (work/'check.f90').write_text((root/'native/opsplit_xpoint_cycle.f90').read_text()+driver)
    subprocess.run(['gfortran','-O0','-fcheck=all',str(work/'check.f90'),'-o',str(work/'check')],check=True,capture_output=True)
    def run(points,updates):
        values=np.r_[np.asarray(points).T.ravel(order='F'),updates]
        out=subprocess.check_output([str(work/'check')],input=' '.join(map(str,values))+'\n',text=True)
        return int(out.strip()) == 1
    return run


@pytest.fixture
def switching():
    a=np.array([.48,-1.2444,.48,1.2447])
    b=np.array([.48,-1.2447,.48,1.2444])
    return np.array([a,b,a,b])


def test_stagnant_saddle_switch_activates(detects_cycle,switching):
    assert detects_cycle(switching,[7.07e-7,7.08e-7,7.07e-7,7.08e-7])


def test_contracting_iteration_keeps_original_search(detects_cycle,switching):
    assert not detects_cycle(switching,[1e-6,1e-7,1e-8,1e-9])


def test_converged_iteration_keeps_original_search(detects_cycle,switching):
    assert not detects_cycle(switching,[1e-12]*4)


def test_stationary_root_keeps_original_search(detects_cycle,switching):
    switching[:]=switching[0]
    assert not detects_cycle(switching,[1e-6]*4)


def test_drifting_roots_do_not_trigger(detects_cycle,switching):
    switching[:,1]+=np.arange(4)*1e-3
    assert not detects_cycle(switching,[1e-6]*4)


def test_nonfinite_history_does_not_trigger(detects_cycle,switching):
    switching[0,0]=np.nan
    assert not detects_cycle(switching,[1e-6]*4)


def test_unrelated_remote_roots_do_not_trigger(detects_cycle,switching):
    switching[1::2,0]+=.1
    assert not detects_cycle(switching,[1e-6]*4)
