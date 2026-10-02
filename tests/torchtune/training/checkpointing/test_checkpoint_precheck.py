# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the training checkpoint structural pre-check in
:class:`torchtune.training.checkpointing.DistributedCheckpointer`.

The pre-check runs before any checkpoint data is loaded and refuses to load a
checkpoint that is missing tensors or contains tensors with mismatched shapes,
without mutating the in-memory model or optimizer state. Valid checkpoints
continue to load as before.
"""

import shutil

import pytest
import torch

from torchtune.training.checkpointing import DistributedCheckpointer
from torchtune.training.checkpointing._checkpointer import preflight_checkpoint
from torchtune.training.seed import set_seed


class SmallModel(torch.nn.Module):
    """A tiny CPU model with a few named parameters."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.fc2 = torch.nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _clone(obj):
    """Deep-copy tensors inside nested dicts/lists/tuples, leaving other values."""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone(v) for v in obj)
    return obj


def _assert_state_dicts_equal(sd1, sd2, path=""):
    """Recursively assert that two (optimizer) state dicts are equal."""
    assert set(sd1.keys()) == set(sd2.keys()), "key mismatch at {}".format(path)
    for key in sd1:
        v1, v2 = sd1[key], sd2[key]
        child_path = "{}.{}".format(path, key) if path else str(key)
        if isinstance(v1, torch.Tensor):
            assert torch.equal(v1, v2), "tensor mismatch at {}".format(child_path)
        elif isinstance(v1, dict):
            _assert_state_dicts_equal(v1, v2, child_path)
        elif isinstance(v1, (list, tuple)):
            assert len(v1) == len(v2), "length mismatch at {}".format(child_path)
            for i, (a, b) in enumerate(zip(v1, v2)):
                if isinstance(a, torch.Tensor):
                    assert torch.equal(a, b), "tensor mismatch at {}[{}]".format(
                        child_path, i
                    )
                else:
                    assert a == b, "value mismatch at {}[{}]".format(child_path, i)
        else:
            assert v1 == v2, "value mismatch at {}".format(child_path)


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


@pytest.fixture
def checkpoint_setup(tmp_path):
    """
    Build a small CPU model + AdamW optimizer, populate the optimizer state with
    a training step, and save three checkpoints:

    * epoch_1: a valid checkpoint (model + optimizer + recipe state)
    * epoch_2: a checkpoint missing one model tensor (fc1.weight deleted)
    * epoch_3: a checkpoint with one tensor of the wrong shape
      (fc2.weight changed from [2, 8] to [4, 8])
    """
    torch.manual_seed(0)
    model = SmallModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Populate optimizer state so the checkpoint includes exp_avg / exp_avg_sq
    x = torch.randn(3, 4)
    model(x).sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    ckpt_dir = tmp_path / "ckpt_dir"
    out_dir = tmp_path / "out_dir"
    ckpt_dir.mkdir()
    out_dir.mkdir()
    checkpointer = DistributedCheckpointer(
        checkpoint_dir=str(ckpt_dir), output_dir=str(out_dir)
    )

    # This state dict mirrors the live model/optimizer (shares storage), exactly
    # like the recipe flow that calls dcp.load into model.state_dict().
    expected_sd = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "seed": 123,
        "epochs_run": 1,
    }

    # Snapshot the live model/optimizer state to verify it is not mutated.
    model_snapshot = _clone(model.state_dict())
    optimizer_snapshot = _clone(optimizer.state_dict())

    # 1) valid checkpoint
    checkpointer.save_checkpoint(_clone(expected_sd), epoch=1)

    # 2) checkpoint with a deleted tensor
    missing_sd = _clone(expected_sd)
    del missing_sd["model"]["fc1.weight"]
    checkpointer.save_checkpoint(missing_sd, epoch=2)

    # 3) checkpoint with a shape mismatch
    shape_sd = _clone(expected_sd)
    shape_sd["model"]["fc2.weight"] = torch.randn(4, 8)
    checkpointer.save_checkpoint(shape_sd, epoch=3)

    return {
        "model": model,
        "optimizer": optimizer,
        "checkpointer": checkpointer,
        "expected_sd": expected_sd,
        "model_snapshot": model_snapshot,
        "optimizer_snapshot": optimizer_snapshot,
        "out_dir": out_dir,
    }


def test_preflight_valid_checkpoint_passes(checkpoint_setup):
    """A valid checkpoint passes the pre-check with no missing or mismatched tensors."""
    ctx = checkpoint_setup
    result = preflight_checkpoint(ctx["expected_sd"], ctx["out_dir"] / "epoch_1")

    assert result.ok
    assert result.missing_keys == []
    assert result.shape_mismatches == {}
    assert result.unexpected_keys == []


def test_preflight_missing_tensor_is_listed(checkpoint_setup):
    """A checkpoint missing a tensor is reported in missing_keys."""
    ctx = checkpoint_setup
    result = preflight_checkpoint(ctx["expected_sd"], ctx["out_dir"] / "epoch_2")

    assert not result.ok
    assert "model.fc1.weight" in result.missing_keys
    assert result.shape_mismatches == {}
    # The error message should name the missing tensor.
    assert "model.fc1.weight" in result.error_message()


def test_preflight_shape_mismatch_is_listed(checkpoint_setup):
    """A checkpoint with a wrong-shaped tensor is reported in shape_mismatches."""
    ctx = checkpoint_setup
    result = preflight_checkpoint(ctx["expected_sd"], ctx["out_dir"] / "epoch_3")

    assert not result.ok
    assert result.missing_keys == []
    assert "model.fc2.weight" in result.shape_mismatches
    expected_shape, found_shape = result.shape_mismatches["model.fc2.weight"]
    assert expected_shape == torch.Size([2, 8])
    assert found_shape == torch.Size([4, 8])
    # The error message should describe both shapes.
    message = result.error_message()
    assert "model.fc2.weight" in message
    assert "[2, 8]" in message
    assert "[4, 8]" in message


def test_load_checkpoint_precheck_failure_does_not_mutate_model_or_optimizer(
    checkpoint_setup,
):
    """
    Loading a tampered checkpoint raises before any data is loaded, so the
    in-memory model and optimizer state are left unchanged.
    """
    ctx = checkpoint_setup
    checkpointer = ctx["checkpointer"]
    expected_sd = ctx["expected_sd"]

    # epoch_3 (shape mismatch) is the most recent checkpoint.
    with pytest.raises(ValueError, match="Checkpoint pre-check failed"):
        checkpointer.load_checkpoint(expected_sd)

    # Model and optimizer must be unchanged.
    _assert_state_dicts_equal(ctx["model"].state_dict(), ctx["model_snapshot"])
    _assert_state_dicts_equal(ctx["optimizer"].state_dict(), ctx["optimizer_snapshot"])

    # Remove the shape-mismatched checkpoint; epoch_2 (missing tensor) is next.
    shutil.rmtree(ctx["out_dir"] / "epoch_3")
    with pytest.raises(ValueError, match="Checkpoint pre-check failed"):
        checkpointer.load_checkpoint(expected_sd)

    # Model and optimizer must still be unchanged.
    _assert_state_dicts_equal(ctx["model"].state_dict(), ctx["model_snapshot"])
    _assert_state_dicts_equal(ctx["optimizer"].state_dict(), ctx["optimizer_snapshot"])


def test_valid_checkpoint_still_loads_and_restores_state(checkpoint_setup):
    """
    After the tampered checkpoints are removed, the valid checkpoint loads and
    restores the expected model and optimizer state.
    """
    ctx = checkpoint_setup
    checkpointer = ctx["checkpointer"]
    expected_sd = ctx["expected_sd"]

    # Remove the tampered checkpoints so epoch_1 (valid) is the most recent.
    shutil.rmtree(ctx["out_dir"] / "epoch_2")
    shutil.rmtree(ctx["out_dir"] / "epoch_3")

    loaded = checkpointer.load_checkpoint(expected_sd)

    # The loaded state dict matches the valid checkpoint exactly.
    _assert_state_dicts_equal(loaded["model"], ctx["model_snapshot"])
    _assert_state_dicts_equal(loaded["optimizer"], ctx["optimizer_snapshot"])

    # And the live model/optimizer were restored to the saved values.
    _assert_state_dicts_equal(ctx["model"].state_dict(), ctx["model_snapshot"])
    _assert_state_dicts_equal(ctx["optimizer"].state_dict(), ctx["optimizer_snapshot"])


def test_preflight_does_not_mutate_input_state_dict(checkpoint_setup):
    """The pre-check itself must not mutate the input state dict."""
    ctx = checkpoint_setup
    before = _clone(ctx["expected_sd"])
    _ = preflight_checkpoint(ctx["expected_sd"], ctx["out_dir"] / "epoch_3")
    _assert_state_dicts_equal(ctx["expected_sd"], before)
