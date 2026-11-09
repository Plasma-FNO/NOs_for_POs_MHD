import pytest
from mastu_opsplit.case import namelist_overrides


def test_override_preserves_unrelated_comments_and_rejects_duplicates():
    text = " &in1\n ! preserve this comment\n nstep_n = 0 ! old\n /\n"
    changed = namelist_overrides(text, {"nstep_n": 3, "restart": False, "time_evol_scheme": "Crank-Nicholson"})
    assert "! preserve this comment" in changed
    assert "nstep_n = 3" in changed
    assert "restart = .false." in changed
    assert "time_evol_scheme = 'Crank-Nicholson'" in changed
    with pytest.raises(ValueError, match="duplicate"):
        namelist_overrides("&in1\nnstep_n=0\nnstep_n=1\n/", {"nstep_n": 2})


def test_bad_namelist_keys_and_values():
    for value in [float("nan"), {"a": 1}, "bad'quote"]:
        with pytest.raises(ValueError):
            namelist_overrides("&in1\n/", {"foo": value})
    with pytest.raises(ValueError):
        namelist_overrides("&in1\n/", {"foo;rm": 1})


def test_array_index_whitespace_cannot_hide_existing_or_duplicate_assignments():
    text = '&in1\n n_ext_block( 1) = 10\n widths ( 1 : 4 ) = 2\n/\n'
    updated = namelist_overrides(text, {'n_ext_block(1)': 20, 'widths(1:4)': 3})
    assert updated.count('n_ext_block') == 1 and 'n_ext_block(1) = 20' in updated
    assert updated.count('widths') == 1 and 'widths(1:4) = 3' in updated
    with pytest.raises(ValueError, match='duplicate'):
        namelist_overrides('&in1\n a(1)=1\n A ( 1 )=2\n/\n', {'a(1)': 3})


def test_native_two_index_anchor_assignment_is_replaced_once():
    text = '&in1\n R_block_points_left( 3, 1 ) = 1.0238\n/\n'
    updated = namelist_overrides(text, {'R_block_points_left(3,1)': 1.0208868868653562})
    assert updated.count('R_block_points_left') == 1
    assert 'R_block_points_left(3,1) = 1.0208868868653562' in updated
    with pytest.raises(ValueError, match='duplicate'):
        namelist_overrides('&in1\n a(1,2)=1\n A ( 1 , 2 )=2\n/\n', {'a(1,2)': 3})


def test_initializer_fortran_jets_match_finite_differences(tmp_path):
    import ctypes
    import shutil
    import subprocess
    from pathlib import Path
    import numpy as np

    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran unavailable for isolated initializer helper")
    patch = Path(__file__).resolve().parents[1] / "patches/0001-model303-axisymmetric-perturbation.patch"
    additions = "\n".join(line[1:] for line in patch.read_text().splitlines() if line.startswith("+") and not line.startswith("+++"))
    helper = additions[additions.index("subroutine opsplit_perturb_node("):additions.index("end subroutine opsplit_perturb_node") + len("end subroutine opsplit_perturb_node")]
    source = tmp_path / "helper.f90"
    source.write_text(helper)
    library = tmp_path / "helper.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-fcheck=all", str(source), "-o", str(library)], check=True)
    function = ctypes.CDLL(str(library)).opsplit_perturb_node_
    pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [pointer] * 11

    def evaluate(s, t):
        r = np.array([1+.2*s+.1*t+.03*s*t, .2+.03*t, .1+.03*s, .03], dtype=float)
        z = np.array([.02+.05*s+.3*t+.01*s*t, .05+.01*t, .3+.01*s, .01], dtype=float)
        psi = np.array([.8+.03*s-.02*t+.02*s*t, .03+.02*t, -.02+.02*s, .02], dtype=float)
        rho = np.array([2+.4*s+.2*t+.1*s*t, .4+.1*t, .2+.1*s, .1], dtype=float)
        temp = np.array([3+.2*s+.1*t+.05*s*t, .2+.05*t, .1+.05*s, .05], dtype=float)
        args = [r, z, psi, np.array([0.]), np.array([1.]), np.array([1.,0.]),
                np.array([.1,.2]), np.array([.7,.9]), np.array([.1,.2]), rho, temp]
        function(*(a.ctypes.data_as(pointer) for a in args))
        return np.stack([rho,temp])
    s, t, h = .02, -.01, 1e-4
    values = evaluate(s,t)
    ds = (evaluate(s+h,t)[:,0]-evaluate(s-h,t)[:,0])/(2*h)
    dt = (evaluate(s,t+h)[:,0]-evaluate(s,t-h)[:,0])/(2*h)
    dst = (evaluate(s+h,t+h)[:,0]-evaluate(s+h,t-h)[:,0]-evaluate(s-h,t+h)[:,0]+evaluate(s-h,t-h)[:,0])/(4*h*h)
    np.testing.assert_allclose(values[:,1], ds, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(values[:,2], dt, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(values[:,3], dst, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("value", ["1.0e20", "-1.0e12", "1D-5", "10", ".1", "1e+6"])
def test_numeric_strings_cannot_be_quoted_as_fortran_reals(value):
    with pytest.raises(ValueError, match="numeric-looking string"):
        namelist_overrides("&in1\n/", {"ZK_par_max": value})


def test_checked_in_case_defaults_have_native_numeric_types():
    from pathlib import Path
    from mastu_opsplit.common import load_config
    root = Path(__file__).resolve().parents[1]
    for name in ("case.yaml", "equilibrium_request.yaml"):
        config = load_config(root / "configs" / name)
        defaults = config["input_defaults"]
        assert [key for key, value in defaults.items() if isinstance(value, str)] == ["time_evol_scheme"]
        text = namelist_overrides("&in1\n/", defaults)
        assert "ZK_par_max = 1e+20" in text
        assert "T_min_neg = -1000000000000.0" in text
