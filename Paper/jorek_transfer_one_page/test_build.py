"""Check scientific evidence and the hard two-page publication boundary."""
import json
import subprocess

import pytest

import build as builder


def test_full_study_and_delivered_page():
    evidence = builder.load_evidence()
    assert len(evidence['rows']) == 39
    assert evidence['counts']['es']['train'] == 400
    assert evidence['counts']['em']['test'] == 10
    assert builder.page_count(builder.HERE/'results_one_page.pdf') <= builder.PAGE_LIMIT == 2
    text = subprocess.check_output(['pdftotext', str(builder.HERE/'results_one_page.pdf'), '-'], text=True)
    assert 'Electrostatic-to-Electromagnetic Transfer' in text
    assert 'full-data, three-seed experiment' in text
    assert 'not enforced' in text
    assert '135,422' in text
    # The methods expansion: every arm is described, with provenance-derived settings.
    assert 'How each OpsSplit arm is conducted' in text
    for arm in ('Zero-shot', 'Frozen transfer', 'Fine-tuned', 'Scratch split', 'Monolithic', 'Stage 1: ES source'):
        assert arm in text
    assert '67,774' in text and '67,648' in text and '2,560' in text
    snapshot = builder.load_evidence()
    assert 'Train (min)' in text
    assert f"{sum(snapshot['source_train_seconds'])/3/60:.1f}" in text
    assert all(row['train_seconds'] > 0 for row in snapshot['rows'] if row['mode'] != 'zero-shot')


def test_overflow_preserves_existing_pdf(tmp_path):
    source = tmp_path/'overflow.tex'
    source.write_text((builder.HERE/'results_one_page.tex').read_text().replace(
        r'\end{document}', r'\clearpage Overflow.\clearpage Overflow.\clearpage Overflow.\end{document}'))
    output = tmp_path/'previous.pdf'
    output.write_bytes(b'previous accepted report')
    with pytest.raises(RuntimeError, match='PAGE LIMIT'):
        builder.build(source, output)
    assert output.read_bytes() == b'previous accepted report'


def test_pdf_page_check_works_without_tex_guard(tmp_path):
    source = tmp_path/'two.tex'
    source.write_text(r'\documentclass{article}\begin{document}One\newpage Two\newpage Three\end{document}')
    with pytest.raises(RuntimeError, match='2-PAGE LIMIT'):
        builder.build(source, tmp_path/'three.pdf')
    assert not (tmp_path/'three.pdf').exists()


def test_method_settings_must_agree_across_runs(tmp_path):
    snapshot = builder.load_evidence()
    row = next(row for row in snapshot['rows'] if row['mode'] == 'finetune' and row['seed'] == 1)
    row['lr'] = 1e-3
    with pytest.raises(ValueError, match='Inconsistent lr'):
        builder.make_values(tmp_path, snapshot)


def test_changed_evidence_is_rejected(tmp_path, monkeypatch):
    snapshot = json.loads((builder.HERE/'results.json').read_text())
    evidence = tmp_path/'evidence.json'
    evidence.write_text('changed')
    snapshot['evidence_sha256'] = {str(evidence): '0'*64}
    (tmp_path/'results.json').write_text(json.dumps(snapshot))
    monkeypatch.setattr(builder, 'HERE', tmp_path)
    with pytest.raises(ValueError, match='Evidence changed'):
        builder.load_evidence()


def test_missing_seed_cannot_be_averaged():
    snapshot = builder.load_evidence()
    snapshot['rows'] = [row for row in snapshot['rows'] if row['seed'] != 2]
    with pytest.raises(ValueError, match='same three seeds'):
        builder.curves(snapshot, 'scratch', 64)


def test_mastu_pdf_is_protected():
    target = builder.HERE.parent/'results_one_page/results_one_page.pdf'
    before = builder.sha256(target)
    with pytest.raises(ValueError, match='protected'):
        builder.build(output=target)
    assert builder.sha256(target) == before
