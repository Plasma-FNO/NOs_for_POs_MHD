"""Check that overflow cannot replace the last valid one-page delivery."""
import pytest
import subprocess
import sys
import json

from build import HERE, build, page_count, load_evidence


def test_tex_guard_rejects_second_page_and_preserves_output(tmp_path):
    output = tmp_path / "delivered.pdf"
    output.write_bytes(b"previous accepted artifact")
    source = tmp_path / "overflow.tex"
    text = (HERE / "results_one_page.tex").read_text()
    source.write_text(text.replace(r"\end{document}", r"\clearpage Intentional overflow.\end{document}"))
    with pytest.raises(RuntimeError, match="ONE PAGE LIMIT"):
        build(source, output)
    assert output.read_bytes() == b"previous accepted artifact"


def test_pdf_check_rejects_two_pages_even_without_tex_guard(tmp_path):
    source = tmp_path / "unguarded.tex"
    source.write_text(r"\documentclass{article}\begin{document}One.\newpage Two.\end{document}")
    output = tmp_path / "new.pdf"
    with pytest.raises(RuntimeError, match="generated 2 pages"):
        build(source, output)
    assert not output.exists()


def test_delivered_artifact_has_exactly_one_page():
    assert page_count(HERE / "results_one_page.pdf") == 1


def test_page_is_current_jorek_with_unmeasured_comparisons():
    text = subprocess.check_output(["pdftotext", str(HERE / "results_one_page.pdf"), "-"], text=True)
    assert "MAST-U OpsSplit with L-SplineFNO" in text
    assert "not measured" in text
    assert str(load_evidence()["regression_status"]["job_id"]) in text
    assert "CylinderFlow" not in text
    assert "0.2185" not in text


def test_changed_reference_evidence_preserves_output(tmp_path, monkeypatch):
    import build as builder
    config = json.loads((HERE / "results.json").read_text())
    evidence = load_evidence()
    study = tmp_path / config["study_root"]
    for key in ("validation", "initializer_build", "regression_status", "gates"):
        path = study / config[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence[key]))
    snapshot = study / config["reference_snapshot"]
    snapshot.mkdir(parents=True)
    (snapshot / "equilibrium_reference.h5").write_bytes(b"changed reference")
    (snapshot / "case.json").write_text(json.dumps({"snapshot_hashes": {"equilibrium_reference.h5": "0"*64}}))
    output = tmp_path / "accepted.pdf"
    output.write_bytes(b"last accepted page")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="reference changed"):
        builder.build(HERE / "results_one_page.tex", output)
    assert output.read_bytes() == b"last accepted page"


@pytest.mark.parametrize("output", [HERE / "results_one_page.tex", HERE.parents[1] / "2602.23113v2.pdf"])
def test_cli_protects_sources(output):
    from build import sha256
    before = sha256(output)
    result = subprocess.run([sys.executable, str(HERE / "build.py"), "--output", str(output)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "original paper is protected" in result.stderr
    assert sha256(output) == before
