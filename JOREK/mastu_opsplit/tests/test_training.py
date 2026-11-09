import random

import numpy as np
import pytest
import torch

from mastu_opsplit.training import checkpoint, restore, smoke_train
from test_learning import configuration


def test_complete_resume_restores_optimizer_scheduler_rng_and_identity(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    saved_weight = model.weight.detach().clone()
    path = tmp_path / "checkpoint.pt"
    checkpoint(path, model, optimizer, scheduler, identity="frozen", step=3, split={"train": ["p"]})
    expected_rng = (random.random(), np.random.rand(), torch.rand(1))
    with torch.no_grad():
        model.weight.zero_()
    assert restore(path, model, optimizer, scheduler, identity="frozen", split={"train": ["p"]}) == 3
    assert torch.equal(model.weight, saved_weight)
    assert (random.random(), np.random.rand()) == expected_rng[:2]
    assert torch.equal(torch.rand(1), expected_rng[2])
    assert optimizer.state and scheduler.last_epoch == 1
    with pytest.raises(ValueError, match="mismatch"):
        restore(path, model, optimizer, scheduler, identity="changed", split={"train": ["p"]})
    with pytest.raises(ValueError, match="mismatch"):
        restore(path, model, optimizer, scheduler, identity="frozen", split={"train": ["other"]})
    with pytest.raises(FileExistsError):
        checkpoint(path, model, optimizer, scheduler, identity="frozen", step=4, split={"train": ["p"]})


@pytest.mark.parametrize("family", ["opsplit", "whole_rhs", "autoregressive"])
def test_tiny_training_decreases_loss_and_rejects_overwrite(tmp_path, family):
    config = {"steps": 10, "seed": 1, "learning_rate": .01, "family": family}
    report = smoke_train(config, configuration(), tmp_path / family)
    assert report["synthetic"] and not report["scientific_gate_evidence"]
    assert report["final_loss"] < report["initial_loss"]
    assert len(report["study_source_sha256"]) == 8
    assert all(len(digest) == 64 for digest in report["study_source_sha256"].values())
    assert (tmp_path / family / "checkpoint.pt").is_file()
    if family == "whole_rhs":
        resumed = smoke_train(config, configuration(), tmp_path / "resumed",
                              resume=tmp_path / family / "checkpoint-step-5.pt")
        assert resumed["resumed_at"] == 5
        original = torch.load(tmp_path / family / "checkpoint.pt", weights_only=False)
        restarted = torch.load(tmp_path / "resumed" / "checkpoint.pt", weights_only=False)
        assert all(torch.equal(value, restarted["model"][key]) for key, value in original["model"].items())
    with pytest.raises(FileExistsError):
        smoke_train(config, configuration(), tmp_path / family)


def test_native_term_loader_preserves_layout_weights_and_split(tmp_path):
    from mastu_opsplit.data import write_geometry, write_trajectory
    from mastu_opsplit.training import load_term_trajectory
    from test_data import trajectory_fixture
    geometry, arrays, metadata = trajectory_fixture()
    gp, tp = tmp_path / "geometry.h5", tmp_path / "trajectory.h5"
    write_geometry(gp, geometry)
    write_trajectory(tp, geometry, gp, arrays, metadata)
    batch = load_term_trajectory(tp, expected_split="train", diagnostics_only=True)
    assert batch["fields"].shape == (2, 4, 7)
    assert batch["labels"].shape == (2, 4, 2)
    assert torch.equal(batch["labels"], torch.from_numpy(arrays["pointwise_terms"].reshape(2, 4, 2)))
    assert torch.equal(batch["volume_weights"], torch.from_numpy(geometry.volume_weights.reshape(-1)))
    with pytest.raises(ValueError, match="split"):
        load_term_trajectory(tp, expected_split="validation")
    metadata.update(synthetic=False, acceptance_status="provisional_exchange")
    real = tmp_path / "provisional.h5"
    write_trajectory(real, geometry, gp, arrays, metadata)
    with pytest.raises(ValueError, match="rejects synthetic or provisional"):
        load_term_trajectory(real, expected_split="train")
    assert not load_term_trajectory(real, expected_split="train", diagnostics_only=True)["scientific_gate_evidence"]


def test_bounded_batches_reject_duplicates_and_preserve_term_objective(tmp_path):
    from mastu_opsplit.data import write_geometry, write_trajectory
    from mastu_opsplit.training import iter_term_batches, pretrain_term_epoch, term_supervised_loss
    from mastu_opsplit.learning import build_family
    from test_data import trajectory_fixture
    geometry, arrays, metadata = trajectory_fixture()
    gp, tp = tmp_path / "geometry.h5", tmp_path / "trajectory.h5"
    write_geometry(gp, geometry)
    write_trajectory(tp, geometry, gp, arrays, metadata)
    batches = list(iter_term_batches([tp], expected_split="train", batch_size=1, diagnostics_only=True))
    assert len(batches) == 2 and all(batch["fields"].shape[0] == 1 for batch in batches)
    with pytest.raises(ValueError, match="unique cases"):
        list(iter_term_batches([tp, tp], expected_split="train", batch_size=1, diagnostics_only=True))
    with pytest.raises(ValueError, match="rejects synthetic"):
        list(iter_term_batches([tp], expected_split="train", batch_size=1))
    model = build_family("opsplit", 7, geometry.bounds, configuration())
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    losses = pretrain_term_epoch(model, batches, optimizer, [1., 1.])
    assert len(losses) == 2 and np.isfinite(losses).all()
    prediction = torch.tensor([[[1., 2.], [3., 4.]]])
    assert torch.isclose(term_supervised_loss(prediction, torch.zeros_like(prediction), [1., 3.], [1., 2.]), torch.tensor(5.125))
