# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import runpy
import shutil
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from torchtune import training
from torchtune.models.llama2 import llama2
from torchtune.training.checkpointing._utils import safe_torch_load

RECIPE_PATH = "recipes/lora_finetune_single_device.py"
ASSETS_PATH = Path(__file__).parent.parent / "assets"
TINY_CKPT_NAME = "tiny_ckpt.pt"


def _write_tiny_ckpt(ckpt_path: Path) -> None:
    """Writes a tiny, randomly initialized llama2 checkpoint to disk."""
    model = llama2(
        vocab_size=32_000,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        embed_dim=64,
        max_seq_len=1024,
        norm_eps=1e-5,
    )
    torch.save(model.state_dict(), ckpt_path)


class TestLoRAFinetuneSingleDeviceResumeConsistency:
    """
    CPU-only tests for the resume-time consistency check of
    ``lora_finetune_single_device``: the epoch count, the optimizer step count
    and the LR scheduler step count restored from a checkpoint must agree with
    each other. If they do not, the recipe must raise an error and must not
    continue training. Valid checkpoints - including older ones that do not
    record ``steps_run`` - must resume as before.
    """

    def _get_config(self, ckpt_dir: Path, output_dir: Path):
        return OmegaConf.create(
            {
                "output_dir": str(output_dir),
                "model": {
                    "_component_": "torchtune.models.llama2.lora_llama2",
                    "lora_attn_modules": ["q_proj", "v_proj"],
                    "apply_lora_to_mlp": False,
                    "apply_lora_to_output": False,
                    "lora_rank": 4,
                    "lora_alpha": 8,
                    "lora_dropout": 0.0,
                    "vocab_size": 32_000,
                    "num_layers": 2,
                    "num_heads": 4,
                    "num_kv_heads": 4,
                    "embed_dim": 64,
                    "max_seq_len": 1024,
                    "norm_eps": 1e-5,
                },
                "tokenizer": {
                    "_component_": "torchtune.models.llama2.llama2_tokenizer",
                    "path": str(ASSETS_PATH / "m.model"),
                    "max_seq_len": None,
                    "prompt_template": None,
                },
                "checkpointer": {
                    "_component_": "torchtune.training.FullModelTorchTuneCheckpointer",
                    "checkpoint_dir": str(ckpt_dir),
                    "checkpoint_files": [TINY_CKPT_NAME],
                    "recipe_checkpoint": None,
                    "output_dir": str(output_dir),
                    "model_type": "LLAMA2",
                },
                "resume_from_checkpoint": False,
                "dataset": {
                    "_component_": "torchtune.datasets.alpaca_dataset",
                    "source": "json",
                    "data_files": str(ASSETS_PATH / "alpaca_tiny.json"),
                    "split": "train",
                },
                "seed": 9,
                "shuffle": True,
                "batch_size": 4,
                "optimizer": {
                    "_component_": "torch.optim.AdamW",
                    "lr": 2e-5,
                    "weight_decay": 0.01,
                },
                "lr_scheduler": {
                    "_component_": "torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup",
                    "num_warmup_steps": 1,
                },
                "loss": {
                    "_component_": "torchtune.modules.loss.LinearCrossEntropyLoss"
                },
                "epochs": 2,
                "max_steps_per_epoch": 2,
                "gradient_accumulation_steps": 1,
                "clip_grad_norm": None,
                "compile": False,
                "metric_logger": {
                    "_component_": "torchtune.training.metric_logging.DiskLogger",
                    "log_dir": str(output_dir / "logs"),
                },
                "log_every_n_steps": 1,
                "log_peak_memory_stats": False,
                "log_level": "INFO",
                "device": "cpu",
                "dtype": "fp32",
                "enable_activation_checkpointing": False,
                "enable_activation_offloading": False,
            }
        )

    def _write_resume_config(
        self,
        tmpdir: Path,
        ckpt_dir: Path,
        epoch_dir: Path,
        output_dir: Path,
        name: str,
    ) -> Path:
        """
        Writes a config that resumes from the intermediate checkpoint in
        ``epoch_dir`` and returns the path to the config file.
        """
        # the base model checkpoint must be resolvable from the (copied)
        # checkpoint dir of the intermediate checkpoint
        shutil.copy(ckpt_dir / TINY_CKPT_NAME, epoch_dir / TINY_CKPT_NAME)
        cfg = self._get_config(ckpt_dir=epoch_dir, output_dir=output_dir)
        cfg.resume_from_checkpoint = True
        cfg.checkpointer.adapter_checkpoint = str(epoch_dir / "adapter_model.pt")
        cfg.checkpointer.recipe_checkpoint = str(epoch_dir / "recipe_state.pt")
        cfg_path = tmpdir / f"{name}.yaml"
        OmegaConf.save(cfg, cfg_path)
        return cfg_path

    def _run_recipe(self, cfg_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", [RECIPE_PATH, "--config", str(cfg_path)])
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(RECIPE_PATH, run_name="__main__")

    def _get_logged_lrs(self, output_dir: Path) -> list[float]:
        log_file = next((output_dir / "logs").glob("log_*.txt"))
        logs = log_file.read_text()
        return [float(x) for x in re.findall(r"lr:([0-9.eE+-]+)", logs)]

    def _tamper_recipe_state(self, recipe_state_path: Path, kind: str) -> None:
        recipe_state = torch.load(recipe_state_path, weights_only=False)
        if kind == "optimizer":
            for param_state in recipe_state[training.OPT_KEY]["state"].values():
                param_state["step"].fill_(7)
        elif kind == "epochs":
            recipe_state[training.EPOCHS_KEY] = 5
        elif kind == "steps":
            recipe_state[training.STEPS_KEY] = 9
        else:
            raise ValueError(f"Unknown tamper kind: {kind}")
        torch.save(recipe_state, recipe_state_path)

    def test_resume_step_consistency(self, tmpdir, monkeypatch):
        tmpdir = Path(str(tmpdir))
        ckpt_dir = tmpdir / "base_ckpt"
        ckpt_dir.mkdir()
        _write_tiny_ckpt(ckpt_dir / TINY_CKPT_NAME)

        # ---- Train a tiny LoRA recipe for two epochs on CPU ----
        train_out = tmpdir / "train_out"
        train_cfg_path = tmpdir / "train.yaml"
        OmegaConf.save(
            self._get_config(ckpt_dir=ckpt_dir, output_dir=train_out), train_cfg_path
        )
        self._run_recipe(train_cfg_path, monkeypatch)

        # An intermediate checkpoint is saved at the end of the first epoch
        epoch_0 = train_out / "epoch_0"
        recipe_state_path = epoch_0 / "recipe_state.pt"
        assert (epoch_0 / "adapter_model.pt").exists()
        assert recipe_state_path.exists()

        # The checkpoint records a consistent progress state: 0 completed
        # epochs, 2 optimizer steps and 2 scheduler steps (2 steps per epoch)
        recipe_state = torch.load(recipe_state_path, weights_only=False)
        assert recipe_state[training.EPOCHS_KEY] == 0
        assert recipe_state[training.STEPS_KEY] == 2
        opt_steps = {
            int(param_state["step"])
            for param_state in recipe_state[training.OPT_KEY]["state"].values()
        }
        assert opt_steps == {2}

        # ---- Resuming from the untampered checkpoint works and is deterministic ----
        resumed_adapters = []
        resumed_lrs = []
        for name in ("resume_a", "resume_b"):
            output_dir = tmpdir / name
            cfg_path = self._write_resume_config(
                tmpdir, ckpt_dir, epoch_0, output_dir, name
            )
            self._run_recipe(cfg_path, monkeypatch)
            resumed_adapters.append(
                safe_torch_load(output_dir / "epoch_1" / "adapter_model.pt")
            )
            resumed_lrs.append(self._get_logged_lrs(output_dir))

        # Both resume runs restore the same state and therefore produce
        # identical parameters
        assert resumed_adapters[0].keys() == resumed_adapters[1].keys()
        for key in resumed_adapters[0]:
            torch.testing.assert_close(
                resumed_adapters[0][key], resumed_adapters[1][key]
            )

        # The LR scheduler resumes at the restored step count instead of
        # restarting: with num_warmup_steps=1 and 4 total training steps, the
        # first logged lr of a resumed run is the cosine value at step 3
        # (lr * 0.5 * (1 + cos(pi * 2 / 3))), not the post-warmup peak lr
        expected_lr = 2e-5 * 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * 2 / 3)))
        assert resumed_lrs[0][0] == pytest.approx(expected_lr.item(), rel=1e-4)

        # ---- Older checkpoints without a recorded steps_run still resume ----
        legacy_epoch_0 = tmpdir / "legacy_epoch_0"
        shutil.copytree(epoch_0, legacy_epoch_0)
        legacy_state = torch.load(
            legacy_epoch_0 / "recipe_state.pt", weights_only=False
        )
        legacy_state[training.STEPS_KEY] = None
        torch.save(legacy_state, legacy_epoch_0 / "recipe_state.pt")
        cfg_path = self._write_resume_config(
            tmpdir, ckpt_dir, legacy_epoch_0, tmpdir / "legacy_out", "legacy"
        )
        self._run_recipe(cfg_path, monkeypatch)
        assert (tmpdir / "legacy_out" / "epoch_1" / "adapter_model.pt").exists()

        # ---- Tampering with any of the step counts fails loudly ----
        for kind in ("optimizer", "epochs", "steps"):
            tampered_epoch_0 = tmpdir / f"tampered_{kind}_epoch_0"
            shutil.copytree(epoch_0, tampered_epoch_0)
            self._tamper_recipe_state(tampered_epoch_0 / "recipe_state.pt", kind)
            output_dir = tmpdir / f"tampered_{kind}_out"
            cfg_path = self._write_resume_config(
                tmpdir, ckpt_dir, tampered_epoch_0, output_dir, f"tampered_{kind}"
            )
            monkeypatch.setattr(sys, "argv", [RECIPE_PATH, "--config", str(cfg_path)])
            with pytest.raises(RuntimeError, match="Inconsistent checkpoint"):
                runpy.run_path(RECIPE_PATH, run_name="__main__")
            # training must not continue: no checkpoints are written
            assert not list(output_dir.glob("epoch_*"))
