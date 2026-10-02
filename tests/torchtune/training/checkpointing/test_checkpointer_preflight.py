# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn

from torchtune import training
from torchtune.training.checkpointing import (
    CheckpointPreflightError,
    FullModelTorchTuneCheckpointer,
    preflight_state_dict,
)
from torchtune.training.checkpointing._utils import safe_torch_load

_MODEL_FNAME = "model-00001-of-00001.bin"
_RECIPE_CKPT = "epoch_1/recipe_state.pt"


class TinyModel(nn.Module):
    """Tiny CPU model used to exercise checkpoint preflight."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 16)
        self.layer2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


def _assert_state_dicts_equal(sd1, sd2):
    """Recursively assert two (nested) state dicts hold identical values."""
    assert set(sd1.keys()) == set(sd2.keys())
    for key in sd1:
        v1, v2 = sd1[key], sd2[key]
        if isinstance(v1, dict):
            _assert_state_dicts_equal(v1, v2)
        elif isinstance(v1, torch.Tensor):
            assert torch.equal(v1, v2), f"tensor mismatch for key {key}"
        else:
            assert v1 == v2, f"value mismatch for key {key}"


def _clone_state_dict(state_dict):
    """Deep-clone a (nested) state dict so later mutation is detectable."""
    cloned = {}
    for key, value in state_dict.items():
        if isinstance(value, dict):
            cloned[key] = _clone_state_dict(value)
        elif isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


class TestCheckpointPreflight:
    @pytest.fixture
    def assets(self, tmp_path):
        """
        Build a tiny CPU model + optimizer, take one optimizer step (so the
        optimizer holds state), and save an intermediate (epoch_1) checkpoint
        with FullModelTorchTuneCheckpointer. Returns everything needed to
        resume and preflight.
        """
        torch.manual_seed(42)
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(torch.randn(4, 8)).sum()
        loss.backward()
        optimizer.step()

        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        output_dir = tmp_path / "output"

        # initial "base" checkpoint file, required by the checkpointer
        torch.save(model.state_dict(), checkpoint_dir / _MODEL_FNAME)

        save_checkpointer = FullModelTorchTuneCheckpointer(
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_files=[_MODEL_FNAME],
            model_type="LLAMA3",
            output_dir=str(output_dir),
        )
        save_checkpointer.save_checkpoint(
            {
                training.MODEL_KEY: model.state_dict(),
                training.OPT_KEY: optimizer.state_dict(),
                training.EPOCHS_KEY: 1,
                training.TOTAL_EPOCHS_KEY: 2,
                training.SEED_KEY: 1234,
            },
            epoch=1,
            intermediate_checkpoint=True,
        )

        epoch_dir = output_dir / "epoch_1"
        assert (epoch_dir / _MODEL_FNAME).exists()
        assert (epoch_dir / "recipe_state.pt").exists()

        return {
            "model": model,
            "optimizer": optimizer,
            "checkpoint_dir": checkpoint_dir,
            "output_dir": output_dir,
            "model_ckpt_path": epoch_dir / _MODEL_FNAME,
            "recipe_ckpt_path": epoch_dir / "recipe_state.pt",
        }

    def _resume_checkpointer(self, assets):
        return FullModelTorchTuneCheckpointer(
            checkpoint_dir=str(assets["checkpoint_dir"]),
            # when resuming, checkpoint files are resolved relative to output_dir
            checkpoint_files=[f"epoch_1/{_MODEL_FNAME}"],
            model_type="LLAMA3",
            output_dir=str(assets["output_dir"]),
            should_load_recipe_state=True,
            recipe_checkpoint=_RECIPE_CKPT,
        )

    def test_preflight_passes_and_valid_checkpoint_loads(self, assets):
        """A valid (previously saved) checkpoint passes preflight and loads as usual."""
        model, optimizer = assets["model"], assets["optimizer"]
        checkpointer = self._resume_checkpointer(assets)

        # preflight against a fresh model/optimizer pair (optimizer state is
        # empty before the first step, so nothing can be "missing")
        fresh_model = TinyModel()
        fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
        reports = checkpointer.preflight_checkpoint(
            model_state_dict=fresh_model.state_dict(),
            optimizer_state_dict=fresh_optimizer.state_dict(),
        )
        assert reports[training.MODEL_KEY]["missing"] == []
        assert reports[training.MODEL_KEY]["shape_mismatches"] == []
        assert reports[training.OPT_KEY]["missing"] == []
        assert reports[training.OPT_KEY]["shape_mismatches"] == []

        # preflight also passes against the original (fully populated) optimizer
        reports = checkpointer.preflight_checkpoint(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
        )
        assert reports[training.OPT_KEY]["missing"] == []
        assert reports[training.OPT_KEY]["shape_mismatches"] == []

        # old valid checkpoint loads exactly as before
        state_dict = checkpointer.load_checkpoint()
        assert training.MODEL_KEY in state_dict
        assert training.OPT_KEY in state_dict
        assert state_dict[training.EPOCHS_KEY] == 1

        fresh_model.load_state_dict(state_dict[training.MODEL_KEY])
        fresh_optimizer.load_state_dict(state_dict[training.OPT_KEY])
        _assert_state_dicts_equal(fresh_model.state_dict(), model.state_dict())
        _assert_state_dicts_equal(
            fresh_optimizer.state_dict(), optimizer.state_dict()
        )

    def test_preflight_missing_tensor_leaves_model_and_optimizer_untouched(
        self, assets
    ):
        """Deleting a tensor from the checkpoint is reported; nothing is modified."""
        model, optimizer = assets["model"], assets["optimizer"]

        # corrupt the checkpoint: drop one tensor
        # (mmap=False so the file is not locked when we overwrite it on Windows)
        ckpt = safe_torch_load(assets["model_ckpt_path"], mmap=False)
        removed = ckpt.pop("layer2.weight")
        assert removed is not None
        torch.save(ckpt, assets["model_ckpt_path"])

        model_before = _clone_state_dict(model.state_dict())
        optimizer_before = _clone_state_dict(optimizer.state_dict())

        checkpointer = self._resume_checkpointer(assets)
        with pytest.raises(CheckpointPreflightError, match="layer2.weight"):
            checkpointer.preflight_checkpoint(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
            )

        # the failed preflight must not have touched model or optimizer
        _assert_state_dicts_equal(model.state_dict(), model_before)
        _assert_state_dicts_equal(optimizer.state_dict(), optimizer_before)

    def test_preflight_shape_mismatch_leaves_model_and_optimizer_untouched(
        self, assets
    ):
        """A tensor with a wrong shape is reported; nothing is modified."""
        model, optimizer = assets["model"], assets["optimizer"]

        # corrupt the checkpoint: same key, wrong shape
        # (mmap=False so the file is not locked when we overwrite it on Windows)
        ckpt = safe_torch_load(assets["model_ckpt_path"], mmap=False)
        ckpt["layer1.weight"] = torch.randn(4, 4)
        torch.save(ckpt, assets["model_ckpt_path"])

        model_before = _clone_state_dict(model.state_dict())
        optimizer_before = _clone_state_dict(optimizer.state_dict())

        checkpointer = self._resume_checkpointer(assets)
        with pytest.raises(CheckpointPreflightError, match="layer1.weight"):
            checkpointer.preflight_checkpoint(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
            )

        _assert_state_dicts_equal(model.state_dict(), model_before)
        _assert_state_dicts_equal(optimizer.state_dict(), optimizer_before)

    def test_preflight_detects_corrupted_optimizer_state(self, assets):
        """Shape corruption inside recipe_state.pt optimizer state is caught."""
        model, optimizer = assets["model"], assets["optimizer"]

        recipe_state = safe_torch_load(assets["recipe_ckpt_path"], mmap=False)
        opt_state = recipe_state[training.OPT_KEY]["state"]
        param_id = next(iter(opt_state))
        opt_state[param_id]["exp_avg"] = torch.randn(3, 3)
        torch.save(recipe_state, assets["recipe_ckpt_path"])

        model_before = _clone_state_dict(model.state_dict())
        optimizer_before = _clone_state_dict(optimizer.state_dict())

        checkpointer = self._resume_checkpointer(assets)
        with pytest.raises(CheckpointPreflightError, match="exp_avg"):
            checkpointer.preflight_checkpoint(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
            )

        _assert_state_dicts_equal(model.state_dict(), model_before)
        _assert_state_dicts_equal(optimizer.state_dict(), optimizer_before)

    def test_preflight_requires_a_reference(self, assets):
        checkpointer = self._resume_checkpointer(assets)
        with pytest.raises(ValueError, match="at least one of"):
            checkpointer.preflight_checkpoint()

    def test_preflight_optimizer_requires_recipe_state(self, assets, tmp_path):
        checkpointer = FullModelTorchTuneCheckpointer(
            checkpoint_dir=str(assets["checkpoint_dir"]),
            checkpoint_files=[_MODEL_FNAME],
            model_type="LLAMA3",
            output_dir=str(tmp_path / "other_output"),
            should_load_recipe_state=False,
        )
        with pytest.raises(ValueError, match="should_load_recipe_state"):
            checkpointer.preflight_checkpoint(
                optimizer_state_dict=assets["optimizer"].state_dict()
            )


class TestPreflightStateDict:
    def test_report_lists_missing_and_shape_mismatches(self):
        reference = {
            "a.weight": torch.randn(4, 4),
            "b.weight": torch.randn(8, 4),
            "c.bias": torch.randn(8),
        }
        checkpoint = {
            "a.weight": torch.randn(2, 2),  # shape mismatch
            "b.weight": torch.randn(8, 4),  # ok
            # "c.bias" missing
        }
        report = preflight_state_dict(checkpoint, reference, state_name="model")
        assert report["missing"] == ["c.bias"]
        assert len(report["shape_mismatches"]) == 1
        assert "a.weight" in report["shape_mismatches"][0]
        assert "(2, 2)" in report["shape_mismatches"][0]
        assert "(4, 4)" in report["shape_mismatches"][0]

    def test_nested_optimizer_state_dict_is_flattened(self):
        reference = {
            "state": {0: {"exp_avg": torch.randn(4, 4), "step": 1.0}},
            "param_groups": [{"lr": 1e-3}],
        }
        checkpoint = {
            "state": {0: {"exp_avg": torch.randn(2, 2), "step": 1.0}},
            "param_groups": [{"lr": 1e-3}],
        }
        report = preflight_state_dict(checkpoint, reference, state_name="optimizer")
        assert report["missing"] == []
        assert len(report["shape_mismatches"]) == 1
        assert "state.0.exp_avg" in report["shape_mismatches"][0]

    def test_matching_state_dicts_pass(self):
        reference = {"w": torch.randn(4, 4), "b": torch.randn(4)}
        checkpoint = {"w": torch.randn(4, 4), "b": torch.randn(4)}
        report = preflight_state_dict(checkpoint, reference)
        assert report["missing"] == []
        assert report["shape_mismatches"] == []
