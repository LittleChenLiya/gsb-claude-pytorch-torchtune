# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import runpy
import shutil
import sys
from pathlib import Path
from typing import Optional

import pytest
import torch
from omegaconf import OmegaConf
from tests.common import TUNE_PATH
from tests.recipes.utils import (
    CKPT_COMPONENT_MAP,
    dummy_alpaca_dataset_config,
    lora_llama3_test_config,
    MODEL_TEST_CONFIGS,
    write_hf_ckpt_config,
)
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    gen_log_file_name,
    get_loss_values_from_metric_logger,
    gpu_test,
    TOKENIZER_PATHS,
)
from torchtune import config, training

from torchtune.training.checkpointing._utils import (
    ADAPTER_MODEL_FNAME,
    get_largest_iter_folder,
    safe_torch_load,
    SHARD_FNAME,
)


class TestLoRAFinetuneSingleDeviceRecipe:
    def _get_test_config_overrides(self, dtype_str: str = "fp32", epochs: int = 2):
        return [
            f"dtype={dtype_str}",
            "dataset.train_on_input=False",
            "seed=9",
            f"epochs={epochs}",
            "max_steps_per_epoch=2",
            "optimizer.lr=2e-5",
            "log_every_n_steps=1",
            "clip_grad_norm=100",
        ] + dummy_alpaca_dataset_config()

    def _fetch_expected_loss_values(self, model_type):
        loss_values_map = {
            "llama3": [11.9838, 11.9691, 11.9616, 11.9383],
        }
        return loss_values_map[model_type]

    def _fetch_qlora_expected_loss_values(self, dtype):
        if dtype == "bf16":
            return [11.9857, 11.9711, 11.9619, 11.9407]
        return [11.9857, 11.9712, 11.9613, 11.9408]

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_type, ckpt_type, micro_batch_size, gradient_accumulation_steps, compile",
        [
            ("llama3/8B_lora_single_device", "llama3", "tune", 2, 4, True),
            ("llama3/8B_lora_single_device", "llama3", "tune", 2, 4, False),
        ],
    )
    @gpu_test(gpu_count=1)
    def test_loss(
        self,
        compile,
        micro_batch_size,
        gradient_accumulation_steps,
        config,
        model_type,
        ckpt_type,
        tmpdir,
        monkeypatch,
    ):
        ckpt_component = CKPT_COMPONENT_MAP[ckpt_type]
        ckpt = model_type + "_" + ckpt_type
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_type])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        cmd = f"""
        tune run lora_finetune_single_device \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj'] \
            model.apply_lora_to_mlp=False \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}] \
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={log_file} \
            compile={compile} \
        """.split()

        model_config = MODEL_TEST_CONFIGS[model_type + "_lora"]

        cmd = cmd + self._get_test_config_overrides(dtype_str="fp32") + model_config
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Make sure to clear compile state in between tests
        if compile:
            torch._dynamo.reset()

        loss_values = get_loss_values_from_metric_logger(log_file)
        expected_loss_values = self._fetch_expected_loss_values(model_type)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "dtype, compile, micro_batch_size, gradient_accumulation_steps",
        [
            ("fp32", True, 8, 1),
            ("bf16", True, 2, 4),
            ("fp32", False, 4, 2),
            ("bf16", False, 8, 1),
        ],
    )
    @gpu_test(gpu_count=1)
    def test_loss_qlora(
        self,
        dtype,
        compile,
        micro_batch_size,
        gradient_accumulation_steps,
        tmpdir,
        monkeypatch,
    ):
        ckpt = "llama3_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        cmd = f"""
        tune run lora_finetune_single_device
            --config llama3/8B_qlora_single_device \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=True \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            metric_logger.filename={log_file} \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            compile={compile} \
            enable_activation_checkpointing=False \
            enable_activation_offloading=False \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3_qlora"]

        cmd = cmd + self._get_test_config_overrides(dtype_str=dtype) + model_config
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Make sure to clear compile state in between tests
        if compile:
            torch._dynamo.reset()

        loss_values = get_loss_values_from_metric_logger(log_file)
        expected_loss_values = self._fetch_qlora_expected_loss_values(dtype=dtype)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("save_adapter_weights_only", [False, True])
    @pytest.mark.integration_test
    @gpu_test(gpu_count=1)
    def test_training_state_on_resume(
        self, tmpdir, monkeypatch, save_adapter_weights_only
    ):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 2 epochs
            - Resume training after epoch 1
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        """

        ckpt = "llama3_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune run lora_finetune_single_device \
            --config llama3/8B_lora_single_device \
            batch_size=8 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=True \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            save_adapter_weights_only={save_adapter_weights_only} \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3_lora"]

        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Resume training
        epoch_folder = get_largest_iter_folder(tmpdir)
        epoch_folder_minus_one = f"epoch_{int(epoch_folder.split('_')[-1]) - 1}"
        cmd_2 = f"""
        tune run lora_finetune_single_device \
            --config llama3/8B_lora_single_device \
            batch_size=8 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=True \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir={os.path.join(tmpdir, epoch_folder_minus_one)} \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.adapter_checkpoint={os.path.join(epoch_folder_minus_one, f"{ADAPTER_MODEL_FNAME}.pt")}
            checkpointer.recipe_checkpoint={os.path.join(tmpdir, epoch_folder_minus_one, "recipe_state.pt")}
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            resume_from_checkpoint=True \
            metric_logger.filename={log_file} \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
        """.split()
        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Second epoch only
        expected_loss_values = self._fetch_expected_loss_values("llama3")
        loss_values = get_loss_values_from_metric_logger(log_file)

        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("save_adapter_weights_only", [False, True])
    @pytest.mark.integration_test
    def test_training_state_on_resume_with_async_checkpointing(
        self, tmpdir, monkeypatch, save_adapter_weights_only
    ):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 2 epochs
            - Resume training after epoch 1
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        """

        ckpt = "llama3_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune run lora_finetune_single_device \
            --config llama3/8B_lora_single_device \
            batch_size=8 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=True \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            save_adapter_weights_only={save_adapter_weights_only} \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
            enable_async_checkpointing=True \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3_lora"]

        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Resume training
        shutil.rmtree(tmpdir / "epoch_1")

        cmd_2 = f"""
        tune run lora_finetune_single_device \
            --config llama3/8B_lora_single_device \
            batch_size=8 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=True \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir={ckpt_dir} \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            resume_from_checkpoint=True \
            metric_logger.filename={log_file} \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
            enable_async_checkpointing=True \
        """.split()
        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Second epoch only
        expected_loss_values = self._fetch_expected_loss_values("llama3")
        loss_values = get_loss_values_from_metric_logger(log_file)

        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("use_dora", [False, True])
    @pytest.mark.integration_test
    @gpu_test(gpu_count=1)
    def test_save_and_load_merged_weights(self, tmpdir, monkeypatch, use_dora):
        ckpt = "llama3_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent

        cmd = f"""
        tune run lora_finetune_single_device \
            --config llama3/8B_lora_single_device \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=True \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
        """.split()

        if use_dora:
            model_config = MODEL_TEST_CONFIGS["llama3_dora"]
        else:
            model_config = MODEL_TEST_CONFIGS["llama3_lora"]

        cmd = cmd + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Next load both the merged weights in a Llama3 base model
        # and the base model weights + trained adapter weights in the LoRA Llama 2 model
        # The results of calling forward on dummy inputs should be the same.
        inputs = torch.randint(low=0, high=32_000, size=(2, 100))

        # Build LoRA model for loading base + adapter weights separately
        lora_model = config.instantiate(OmegaConf.from_dotlist(model_config).model)

        # Build base llama3 model for loading merged weights
        base_llama3_config = MODEL_TEST_CONFIGS["llama3"]
        llama3_model = config.instantiate(
            OmegaConf.from_dotlist(base_llama3_config).model
        )

        # Load base model and trained adapter weights into LoRA model and call fwd
        epoch_folder = get_largest_iter_folder(tmpdir)
        adpt_path = os.path.join(tmpdir, epoch_folder, f"{ADAPTER_MODEL_FNAME}.pt")
        lora_sd = safe_torch_load(adpt_path, weights_only=True)

        with open(ckpt_path, "rb") as f:
            base_model_sd = torch.load(f, weights_only=True)
        lora_model.load_state_dict(lora_sd, strict=False)
        lora_model.load_state_dict(base_model_sd, strict=False)
        baseline_out = lora_model(inputs)

        # Load merged final ckpt directly into llama3 and call fwd
        model_ckpt_fname = (
            SHARD_FNAME.format(cpt_idx="1".zfill(5), num_shards="1".zfill(5)) + ".bin"
        )
        model_path = os.path.join(tmpdir, epoch_folder, model_ckpt_fname)
        sd = safe_torch_load(model_path, weights_only=True)

        llama3_model.load_state_dict(sd)
        merged_ckpt_out = llama3_model(inputs)
        torch.testing.assert_close(baseline_out, merged_ckpt_out, rtol=1e-5, atol=1e-5)

    # ------------------------------------------------------------------
    # CPU resume step-consistency test
    # ------------------------------------------------------------------

    # Path to the tiny tiktoken tokenizer shipped with the test assets.
    _TOKENIZER_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets",
        "tiktoken_small.model",
    )

    @staticmethod
    def _load_recipe_module():
        """Load the LoRA recipe module directly.

        The ``recipes/`` directory is not a package, so the recipe module
        cannot be imported via a normal ``import`` statement. We load it
        from the file path instead.
        """
        import importlib.util

        recipe_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "recipes",
            "lora_finetune_single_device.py",
        )
        spec = importlib.util.spec_from_file_location(
            "lora_finetune_single_device", recipe_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _build_tiny_lora_cfg(
        self,
        *,
        output_dir,
        base_ckpt_path,
        epochs: int,
        resume: bool = False,
        adapter_checkpoint=None,
        recipe_checkpoint=None,
    ):
        """Build a DictConfig for the tiny CPU LoRA recipe."""
        from omegaconf import OmegaConf

        # Convert pytest LocalPath objects to strings
        output_dir = str(output_dir)
        base_ckpt_path = str(base_ckpt_path)
        ckpt_dir = str(Path(base_ckpt_path).parent)
        if adapter_checkpoint is not None:
            adapter_checkpoint = str(adapter_checkpoint)
        if recipe_checkpoint is not None:
            recipe_checkpoint = str(recipe_checkpoint)

        cfg = OmegaConf.load("recipes/configs/llama3/8B_lora_single_device.yaml")
        overrides = [
            "dtype=fp32",
            "dataset.train_on_input=False",
            "seed=9",
            f"epochs={epochs}",
            "max_steps_per_epoch=2",
            "batch_size=2",
            "gradient_accumulation_steps=1",
            "optimizer.lr=2e-5",
            "log_every_n_steps=1",
            f"output_dir={output_dir}",
            # tiny llama3 LoRA model
            "model._component_=torchtune.models.llama3.lora_llama3",
            "model.vocab_size=128256",
            "model.num_layers=2",
            "model.num_heads=8",
            "model.num_kv_heads=4",
            "model.embed_dim=64",
            "model.max_seq_len=1024",
            "model.lora_rank=8",
            "model.lora_alpha=16",
            "model.lora_dropout=0.0",
            "model.lora_attn_modules=[q_proj,v_proj]",
            "model.apply_lora_to_mlp=False",
            # tokenizer
            f"tokenizer.path={self._TOKENIZER_PATH}",
            "tokenizer.max_seq_len=256",
            "tokenizer.prompt_template=null",
            # dataset
            "dataset._component_=torchtune.datasets.alpaca_dataset",
            "dataset.source=json",
            "dataset.data_files=tests/assets/alpaca_tiny.json",
            "dataset.split=train",
            # logging
            f"metric_logger.filename={output_dir}/log.txt",
            "enable_activation_checkpointing=False",
            "enable_activation_offloading=False",
        ]
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

        # Replace the checkpointer config entirely with a fresh one for
        # FullModelTorchTuneCheckpointer (no config.json needed).
        checkpointer_overrides = {
            "_component_": "torchtune.training.FullModelTorchTuneCheckpointer",
            "checkpoint_dir": ckpt_dir,
            "checkpoint_files": [base_ckpt_path],
            "output_dir": output_dir,
            "model_type": "LLAMA3",
        }
        if resume:
            checkpointer_overrides["resume_from_checkpoint"] = True
            checkpointer_overrides["adapter_checkpoint"] = adapter_checkpoint
            checkpointer_overrides["recipe_checkpoint"] = recipe_checkpoint
            # The recipe reads the top-level resume_from_checkpoint flag,
            # not the checkpointer's flag.
            cfg.resume_from_checkpoint = True
        cfg.checkpointer = OmegaConf.create(checkpointer_overrides)

        return cfg

    def _create_tiny_lora_base_checkpoint(self, ckpt_path) -> None:
        """Create a tiny random base checkpoint for the llama3 LoRA model."""
        from torchtune.models.llama3 import lora_llama3

        with torch.device("cpu"):
            model = lora_llama3(
                lora_attn_modules=["q_proj", "v_proj"],
                apply_lora_to_mlp=False,
                apply_lora_to_output=False,
                vocab_size=128_256,
                num_layers=2,
                num_heads=8,
                num_kv_heads=4,
                embed_dim=64,
                max_seq_len=1024,
                lora_rank=8,
                lora_alpha=16,
                lora_dropout=0.0,
            )
        torch.save(model.state_dict(), str(ckpt_path))

    @pytest.mark.integration_test
    def test_lora_resume_step_consistency_cpu(self, tmpdir):
        """
        Verify that resuming a LoRA recipe validates the consistency of the
        epoch count, optimizer step count, and scheduler step count.

        - Train a tiny LoRA model on CPU for 2 epochs (reference run).
        - Resume from the epoch-1 checkpoint and verify the final adapter
          weights match the reference run.
        - Tamper with each of the step counts in the checkpoint and verify
          that setup raises ``RuntimeError`` and training does not start.
        """
        recipe_module = self._load_recipe_module()
        LoRAFinetuneRecipeSingleDevice = (
            recipe_module.LoRAFinetuneRecipeSingleDevice
        )

        # --- Setup --------------------------------------------------------
        ckpt_dir = tmpdir / "base_ckpt"
        ckpt_dir.mkdir()
        base_ckpt_path = ckpt_dir / "base_model.pt"
        self._create_tiny_lora_base_checkpoint(base_ckpt_path)

        # --- Reference run: 2 epochs straight -----------------------------
        ref_dir = tmpdir / "ref"
        ref_dir.mkdir()
        ref_cfg = self._build_tiny_lora_cfg(
            output_dir=ref_dir,
            base_ckpt_path=base_ckpt_path,
            epochs=2,
        )
        recipe = LoRAFinetuneRecipeSingleDevice(ref_cfg)
        recipe.setup(ref_cfg)
        recipe.train()

        # --- Resume run part 1: 2 epochs ----------------------------------
        res_dir = tmpdir / "res"
        res_dir.mkdir()
        res_cfg = self._build_tiny_lora_cfg(
            output_dir=res_dir,
            base_ckpt_path=base_ckpt_path,
            epochs=2,
        )
        recipe = LoRAFinetuneRecipeSingleDevice(res_cfg)
        recipe.setup(res_cfg)
        recipe.train()

        # --- Resume run part 2: resume from epoch_0 -----------------------
        epoch_folder = get_largest_iter_folder(res_dir)
        epoch_folder_minus_one = (
            f"epoch_{int(epoch_folder.split('_')[-1]) - 1}"
        )
        res_dir2 = tmpdir / "res2"
        res_dir2.mkdir()
        resume_cfg = self._build_tiny_lora_cfg(
            output_dir=res_dir2,
            base_ckpt_path=base_ckpt_path,
            epochs=2,
            resume=True,
            adapter_checkpoint=res_dir
            / epoch_folder_minus_one
            / f"{ADAPTER_MODEL_FNAME}.pt",
            recipe_checkpoint=res_dir
            / epoch_folder_minus_one
            / "recipe_state.pt",
        )
        recipe = LoRAFinetuneRecipeSingleDevice(resume_cfg)
        recipe.setup(resume_cfg)
        recipe.train()

        # --- Compare final adapter weights --------------------------------
        # Use mmap=False so the file handle is released before the final
        # resume run tries to save to the same path.
        ref_adapter = safe_torch_load(
            ref_dir / epoch_folder / f"{ADAPTER_MODEL_FNAME}.pt",
            weights_only=True,
            mmap=False,
        )
        res_adapter = safe_torch_load(
            res_dir2 / epoch_folder / f"{ADAPTER_MODEL_FNAME}.pt",
            weights_only=True,
            mmap=False,
        )
        for key in ref_adapter:
            torch.testing.assert_close(
                ref_adapter[key],
                res_adapter[key],
                rtol=1e-5,
                atol=1e-5,
            )

        # --- Tamper tests -------------------------------------------------
        recipe_state_path = (
            res_dir / epoch_folder_minus_one / "recipe_state.pt"
        )
        # Use mmap=False so the file handle is released before we overwrite it.
        original_state = safe_torch_load(
            recipe_state_path, weights_only=False, mmap=False
        )

        def _tamper_and_setup(tamper_fn):
            """Tamper the recipe state, attempt setup, and restore."""
            # Use deepcopy so nested dicts (e.g. optimizer state) are not
            # shared with ``original_state``.
            tampered = copy.deepcopy(original_state)
            tamper_fn(tampered)
            torch.save(tampered, str(recipe_state_path))
            try:
                tampered_cfg = self._build_tiny_lora_cfg(
                    output_dir=res_dir2,
                    base_ckpt_path=base_ckpt_path,
                    epochs=2,
                    resume=True,
                    adapter_checkpoint=res_dir
                    / epoch_folder_minus_one
                    / f"{ADAPTER_MODEL_FNAME}.pt",
                    recipe_checkpoint=recipe_state_path,
                )
                tampered_recipe = LoRAFinetuneRecipeSingleDevice(tampered_cfg)
                with pytest.raises(RuntimeError, match="inconsistent"):
                    tampered_recipe.setup(tampered_cfg)
            finally:
                torch.save(original_state, str(recipe_state_path))

        # Tamper the epoch count
        _tamper_and_setup(
            lambda s: s.update({training.EPOCHS_KEY: 99})
        )

        # Tamper the recorded step count
        _tamper_and_setup(
            lambda s: s.update({training.STEPS_KEY: 99})
        )

        # Tamper the optimizer state's step counter
        def _tamper_optimizer_step(s):
            for param_state in s[training.OPT_KEY]["state"].values():
                if "step" in param_state:
                    param_state["step"] = param_state["step"] + 100

        _tamper_and_setup(_tamper_optimizer_step)

        # --- Restore original and verify resume still works --------------
        torch.save(original_state, str(recipe_state_path))
        valid_cfg = self._build_tiny_lora_cfg(
            output_dir=res_dir2,
            base_ckpt_path=base_ckpt_path,
            epochs=2,
            resume=True,
            adapter_checkpoint=res_dir
            / epoch_folder_minus_one
            / f"{ADAPTER_MODEL_FNAME}.pt",
            recipe_checkpoint=recipe_state_path,
        )
        valid_recipe = LoRAFinetuneRecipeSingleDevice(valid_cfg)
        valid_recipe.setup(valid_cfg)
        valid_recipe.train()
