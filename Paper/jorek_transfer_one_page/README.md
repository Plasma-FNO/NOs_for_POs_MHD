# JOREK ES-to-EM results record (two pages)

Open `results_one_page.pdf`. The file name is kept from the original one-page version so existing links still work. This companion matches the MAST-U page's letter-paper, two-column layout and 10-point body typography. The MAST-U report remains separate.

Page 1 has the result, the two-panel figure, the dataset/protocol and the additive split. Section 3 explains how each OpsSplit arm is conducted: ES source pretraining, then zero-shot, frozen transfer, fine-tuned, scratch split and monolithic. Table 1 shows what each arm loads, trains and rescales. Section 3 also covers the ES Φ / EM u scaling mismatch in the transfer arms. Section 4 covers the shared training loop, checkpoint selection and the test metric. Sections 5–6 give the three-seed results and limitations. The method description follows `JOREK/zenodo_opsplit/{train,model,data}.py`. The current `model.py` differs from the hashed run copy. It still reproduces the logged parameter counts and the first test-rollout errors (to <0.1%, CPU vs. the original GPU run) for every arm.

The page uses **all three completed full-study seeds**, not the earlier two-holdout pilot. There are 400 ES training trajectories and ten common EM holdouts, with 8/32/64-trajectory target budgets. The results table reports per-trajectory/per-shared-field R-weighted relative L2 at frame 10, then mean ± sample standard deviation across seeds. The rollout panel shows seed means for the 64-case budget; zero-shot uses no target training. The same holdouts across seeds are not counted as 30 independent test cases. Large finite errors remain included.

`results_one_page.tex` is the editable source. `results.json` freezes 39 EM comparison records, their 100-frame aggregate curves, and hashes of the original metric/provenance files and data manifest. The builder verifies these fingerprints before rendering. Parameter counts, learning rates, rollout horizon and update count in the methods text and Table 1 are read from that snapshot. The build rejects them if they disagree across seeds or budgets. Table 1 also gives training time per arm: the wall time at the end of the last epoch in each run's `history.jsonl`, including per-epoch validation and excluding the test rollout. It is averaged over three seeds and three budgets; the fixed update count makes it budget-independent. The runs used one H100 GPU and 4 CPUs on `boost_fua_prod`. The ES source pretraining time that the transfer arms reuse is shown as `+2.2`. Those history files are hashed in the snapshot. It writes PDF/source/snapshot hashes to `results_one_page.build.json`. The ES/EM scaling ranges in Section 3 were read from the run checkpoints (`best.pt`) and are not part of the frozen snapshot. Current experiment files remain under `/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-transfer-20261104/full-study`.

Rebuild the frozen evidence:

```bash
cd /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/jorek-transfer-paper-mpl \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python \
  Paper/jorek_transfer_one_page/build.py
```

Use `--refresh` only to explicitly freeze a new complete three-seed snapshot. Missing runs, mismatched holdouts, changed evidence, or incomplete/nonfinite rollouts are rejected; revise the reporting contract to represent such outcomes before publishing them. The build does not submit jobs or alter training data.

Two pages is a hard limit (`PAGE_LIMIT` in `build.py`). LaTeX and an independent `pdfinfo` check enforce it; overfull boxes reject the build. The previous PDF is replaced only after these checks pass. Shorten content rather than shrinking body text. Table 1 is defined on page 1 so it floats to the top of page 2; moving it later in the source pushes it onto a third page. The original paper and MAST-U PDF are protected outputs.

Validation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/jorek-transfer-paper-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python -m pytest -q \
  Paper/jorek_transfer_one_page/test_build.py
```

The result is specific to this effective split and training configuration. Historical export units and timestamps remain unaudited, and differential constraints are not enforced. The page does not claim an identified physical operator or valid long-time electromagnetic dynamics.
