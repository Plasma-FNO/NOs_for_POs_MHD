"""Exercise the patched native wrapper with injected MUMPS failures."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module", params=[False, True], ids=["original", "refinement"])
def wrapper(tmp_path_factory, request):
    work = tmp_path_factory.mktemp("mumps-wrapper")
    root = Path(__file__).resolve().parents[1]
    source = Path("/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/cached_rz_recovery_20261103/source")
    compiler = shutil.which("gfortran")
    assert compiler, "Fortran compiler required for the solver regression"
    (work / "solvers").mkdir()
    shutil.copyfile(source / "solvers/mod_mumps.f90", work / "solvers/mod_mumps.f90")
    subprocess.run(["patch", "--batch", "-p1", "-i", str(root / "patches/0011-mumps-workspace-retry-and-errors.patch")],
                   cwd=work, check=True, capture_output=True)
    subprocess.run(["patch", "--batch", "-p1", "-i", str(root / "patches/0012-extend-mumps-workspace-retry-budget.patch")],
                   cwd=work, check=True, capture_output=True)
    if request.param:
        subprocess.run(["patch", "--batch", "-p1", "-i", str(root / "patches/0029-refine-mumps-solution-and-report-backward-error.patch")],
                       cwd=work, check=True, capture_output=True)
    (work / "dmumps_struc.h").write_text('''
type DMUMPS_STRUC
  integer :: comm, job, sym, par, n, nz, nz_loc, nrhs, lrhs
  integer :: icntl(60)=0, infog(80)=0
  real*8 :: cntl(15), rinfog(40)=0d0
  real*8, pointer :: a(:), a_loc(:), rhs(:)
  integer, pointer :: irn(:), irn_loc(:), jcn(:), jcn_loc(:)
end type
''')
    (work / "stubs.f90").write_text('''
module phys_module
  integer :: mumps_ordering=7
  logical :: use_BLR_compression=.false.
  real*8 :: epsilon_BLR=0d0
end module
module mod_integer_types
  integer, parameter :: int_all=8
end module
module data_structure
  type type_SP_MATRIX
    integer :: ng=1, nnz=1
    real*8, pointer :: val(:)=>null()
    integer, pointer :: irn(:)=>null(), jcn(:)=>null()
  end type
  type type_RHS
    real*8, pointer :: val(:)=>null()
    integer :: nrhs=1
  end type
end module
module mpi_mod
  integer, parameter :: MPI_COMM_WORLD=99, MPI_DOUBLE_PRECISION=8
contains
  subroutine MPI_COMM_RANK(comm, rank, ierr)
    integer :: comm, rank, ierr
    rank=0; ierr=0
  end subroutine
  subroutine MPI_ABORT(comm, code, ierr)
    integer :: comm, code, ierr
    if (comm /= MPI_COMM_WORLD) stop 3
    print *, 'ABORT_WORLD'
    stop 2
  end subroutine
  subroutine MPI_Bcast(rhs, n, dtype, root, comm, ierr)
    integer :: n, dtype, root, comm, ierr
    real*8 :: rhs(n)
    print *, 'BROADCAST'
    ierr=0
  end subroutine
end module
module injection
  integer :: fail_job, failures, failure_code, calls=0
end module
''')
    (work / "driver.f90").write_text('''
subroutine DMUMPS(p)
  use mod_mumps, only: type_MUMPS_SOLVER
  use injection
  implicit none
  include 'dmumps_struc.h'
  type(DMUMPS_STRUC) :: p
  p%infog=0
  if (p%job == fail_job) then
    calls=calls+1
    if (calls <= failures) p%infog(1)=failure_code
  endif
  print *, 'JOB', p%job, 'RELAX', p%icntl(14)
end subroutine
program check_wrapper
  use mod_mumps
  use data_structure
  use injection
  implicit none
  type(type_MUMPS_SOLVER) :: solver
  type(type_SP_MATRIX) :: matrix
  type(type_RHS) :: rhs
  character(len=32) :: arg
  integer :: i, mode
  call get_command_argument(1, arg); read(arg,*) fail_job
  call get_command_argument(2, arg); read(arg,*) failures
  call get_command_argument(3, arg); read(arg,*) failure_code
  call get_command_argument(4, arg); read(arg,*) mode
  allocate(matrix%val(1),matrix%irn(1),matrix%jcn(1),rhs%val(1))
  rhs%val=1d0
  solver%projection=mode==1
  call mumps_initialize(solver,1)
  call mumps_analyze(solver,matrix)
  call mumps_factorize(solver,matrix)
  if (solver%projection) then
    call mumps_solve_multiple(solver,rhs)
  else
    call mumps_solve(solver,rhs)
  endif
  print *, 'SUCCESS'
end program
''')
    executable = work / "check_wrapper"
    subprocess.run([compiler, "-cpp", "-DUSE_MUMPS", "-fcheck=all", "-ffree-line-length-none", "-I", str(work),
                    "stubs.f90", "solvers/mod_mumps.f90", "driver.f90", "-o", str(executable)],
                   cwd=work, check=True, capture_output=True)
    return executable


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("job,failures,code,ok,retries", [
    (2, 0, 0, True, 0),
    (2, 2, -9, True, 2),
    (2, 4, -9, True, 4),
    (2, 5, -9, True, 5),
    (2, 1, -8, True, 1),
    (2, 1, 1, True, 0),
    (2, 99, -9, False, 5),
    (2, 1, -10, False, 0),
    (-1, 1, -13, False, 0),
    (1, 1, -6, False, 0),
    (3, 1, -3, False, 0),
])
def test_mumps_failure_handling(wrapper, mode, job, failures, code, ok, retries):
    result = subprocess.run([str(wrapper), str(job), str(failures), str(code), str(mode)],
                            capture_output=True, text=True, timeout=10)
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    assert result.stdout.count("MUMPS workspace retry:") == retries
    assert ("SUCCESS" in result.stdout) == ok
    assert ("ABORT_WORLD" in result.stdout) != ok
    if not ok:
        assert "MUMPS failure during" in result.stderr
        assert "BROADCAST" not in result.stdout
        if job != 3:
            assert not any(line.split()[:2] == ["JOB", "3"] for line in result.stdout.splitlines())
    if retries:
        start = 50 if mode == 0 else 80
        for attempt in range(1, retries + 1):
            assert str(start * 2**attempt) in result.stdout
