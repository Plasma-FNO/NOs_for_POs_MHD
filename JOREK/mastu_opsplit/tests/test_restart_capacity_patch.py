"""Reproduce cached-grid growth past restart capacity with native allocations."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("format", ["binary", "hdf5"])
@pytest.mark.parametrize("patched", [False, True])
@pytest.mark.parametrize("saved,grown", [(2, 6), (8, 8), (10, 10)])
def test_restart_preserves_capacity_and_active_count(tmp_path, format, patched, saved, grown):
    root = Path(__file__).resolve().parents[1]
    source = Path("/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/mumps_workspace_recovery_20261103/source")
    target = tmp_path / "communication/mod_import_restart.f90"
    target.parent.mkdir()
    shutil.copyfile(source / "communication/mod_import_restart.f90", target)
    if patched:
        subprocess.run(["patch", "--batch", "-p1", "-i", str(root / "patches/0013-preserve-restart-node-capacity.patch")],
                       cwd=tmp_path, check=True, capture_output=True)
    text = target.read_text()
    if format == "binary":
        allocation = text.split("  read(21) node_list%n_dof\n", 1)[1].split("  do i=1,node_list%n_nodes", 1)[0]
    else:
        allocation = text.split("  ! initialise and allocate node_list\n", 1)[1].split("  aux_values_read", 1)[0]
    native = (source / "datatypes/data_structure.f90").read_text()
    routines = []
    for name in ["init_node", "init_node_list", "dealloc_node_list"]:
        start = native.index(f"  subroutine {name}(")
        end = native.index(f"  end subroutine {name}", start) + len(f"  end subroutine {name}")
        routines.append(native[start:end])
    program = '''module fixture
  integer, parameter :: n_nodes_max=8, n_tor=1, n_degrees=4, n_var=7
  type type_node
    real*8, allocatable :: values(:,:,:), deltas(:,:,:)
  end type
  type type_node_list
    integer :: n_nodes, n_dof
    type(type_node), allocatable :: node(:)
  end type
contains
''' + "\n".join(routines) + '''
end module
program grow_grid
  use fixture
  implicit none
  type(type_node_list) :: node_list
  type(type_node), allocatable :: newnodes(:)
  integer :: i, n_nodes_tmp, n_dof_tmp, saved, grown
  character(len=32) :: arg
  call get_command_argument(1,arg); read(arg,*) saved
  call get_command_argument(2,arg); read(arg,*) grown
  node_list%n_nodes=saved; node_list%n_dof=4*saved
  n_nodes_tmp=saved; n_dof_tmp=4*saved
''' + allocation + '''
  if (node_list%n_nodes /= saved .or. node_list%n_dof /= 4*saved) stop 1
  do i=1,size(node_list%node)
    if (.not. allocated(node_list%node(i)%values)) stop 2
    if (.not. allocated(node_list%node(i)%deltas)) stop 3
    if (any(node_list%node(i)%values /= 0d0)) stop 4
    if (any(node_list%node(i)%deltas /= 0d0)) stop 5
  enddo
  allocate(newnodes(grown))
  do i=1,grown
    call init_node(newnodes(i),n_var)
    newnodes(i)%values=i
    newnodes(i)%deltas=-i
    node_list%node(i)=newnodes(i)
  enddo
  deallocate(newnodes)
  node_list%n_nodes=grown
  do i=1,grown
    if (any(node_list%node(i)%values /= dble(i))) stop 6
    if (any(node_list%node(i)%deltas /= -dble(i))) stop 7
  enddo
  print *, 'SUCCESS'
end program
'''
    path = tmp_path / "capacity.f90"
    path.write_text(program)
    compiler = shutil.which("gfortran")
    assert compiler, "Fortran compiler required for the native allocation regression"
    executable = tmp_path / "capacity"
    subprocess.run([compiler, "-fcheck=all", "-g", str(path), "-o", str(executable)],
                   cwd=tmp_path, check=True, capture_output=True)
    result = subprocess.run([str(executable), str(saved), str(grown)], text=True, capture_output=True, timeout=10)
    if patched or saved >= grown:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "SUCCESS" in result.stdout
    else:
        assert result.returncode != 0
        assert "above upper bound" in result.stderr
