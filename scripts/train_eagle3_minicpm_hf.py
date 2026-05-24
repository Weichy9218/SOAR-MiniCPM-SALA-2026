#!/usr/bin/env python3
"""Run SpecForge EAGLE3 training with MiniCPM-SALA compatibility patches."""

from __future__ import annotations

import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def patch_fla_head_first_compat() -> None:
    try:
        import fla.ops.simple_gla as simple_gla
        import fla.ops.simple_gla.chunk as simple_gla_chunk
    except Exception:
        return

    original_chunk_simple_gla = simple_gla_chunk.chunk_simple_gla

    def chunk_simple_gla_without_head_first(*args, **kwargs):
        kwargs.pop("head_first", None)
        return original_chunk_simple_gla(*args, **kwargs)

    simple_gla.chunk_simple_gla = chunk_simple_gla_without_head_first
    simple_gla_chunk.chunk_simple_gla = chunk_simple_gla_without_head_first


def patch_minicpm_sglang_server_args() -> None:
    import inspect

    import specforge.modeling.target.sglang_backend.patch as sglang_patch
    import specforge.modeling.target.sglang_backend.model_runner as specforge_model_runner
    from sglang.srt.layers import dp_attention
    from specforge.modeling.target.eagle3_target_model import SGLangEagle3TargetModel

    if len(inspect.signature(dp_attention.compute_dp_attention_world_info).parameters) == 4:
        def compute_dp_attention_world_info_compat(
            enable_dp_attention, tp_rank, tp_size, dp_size, _attn_cp_size=1
        ):
            return dp_attention.compute_dp_attention_world_info(
                enable_dp_attention, tp_rank, tp_size, dp_size
            )

        sglang_patch.compute_dp_attention_world_info = compute_dp_attention_world_info_compat
        sglang_patch.initialize_dp_attention = dp_attention.initialize_dp_attention
        specforge_model_runner.initialize_dp_attention = dp_attention.initialize_dp_attention

    original_from_pretrained = SGLangEagle3TargetModel.from_pretrained.__func__

    @classmethod
    def from_pretrained_with_minicpm_args(
        cls,
        pretrained_model_name_or_path,
        torch_dtype=None,
        device=None,
        cache_dir=None,
        trust_remote_code=False,
        **kwargs,
    ):
        if os.environ.get("EAGLE3_SGLANG_MINICPM_ARGS", "1") == "1":
            kwargs["attention_backend"] = "minicpm_flashinfer"
            kwargs.setdefault("chunked_prefill_size", 8192)
            kwargs.setdefault("skip_server_warmup", True)
            kwargs.setdefault("disable_radix_cache", True)
            kwargs.setdefault("dense_as_sparse", True)
        return original_from_pretrained(
            cls,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    SGLangEagle3TargetModel.from_pretrained = from_pretrained_with_minicpm_args


def patch_minicpm_hf_loader() -> None:
    original_from_pretrained = AutoModelForCausalLM.from_pretrained

    def from_pretrained_with_minicpm_remote_config(pretrained_model_name_or_path, *args, **kwargs):
        trust_remote_code = kwargs.get("trust_remote_code", False)
        if trust_remote_code and os.path.isdir(pretrained_model_name_or_path):
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
            )
            if config.__class__.__name__ == "MiniCPMHybridConfig":
                config_class = get_class_from_dynamic_module(
                    "configuration_minicpm_sala.MiniCPMSALAConfig",
                    pretrained_model_name_or_path,
                )
                model_class = get_class_from_dynamic_module(
                    "modeling_minicpm_sala.MiniCPMSALAForCausalLM",
                    pretrained_model_name_or_path,
                )
                remote_config = config_class.from_pretrained(pretrained_model_name_or_path)
                if os.environ.get("EAGLE3_HF_DENSE_TARGET", "1") == "1":
                    # The local environment does not provide flash_attn for the HF
                    # sparse attention path. For smoke training at max_length<=256,
                    # dense SDPA matches the short-context behavior closely enough to
                    # produce a serving-loadable draft head without changing serving.
                    remote_config.sparse_config = None
                    remote_config._attn_implementation = "sdpa"
                kwargs = dict(kwargs)
                kwargs.pop("trust_remote_code", None)
                if "torch_dtype" in kwargs and "dtype" not in kwargs:
                    kwargs["dtype"] = kwargs.pop("torch_dtype")
                return model_class.from_pretrained(
                    pretrained_model_name_or_path,
                    *args,
                    config=remote_config,
                    **kwargs,
                )
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    AutoModelForCausalLM.from_pretrained = from_pretrained_with_minicpm_remote_config


def patch_hf_target_forward() -> None:
    from specforge.modeling.target.eagle3_target_model import HFEagle3TargetModel

    original_generate = HFEagle3TargetModel.generate_eagle3_data

    @torch.no_grad()
    def generate_eagle3_data_without_router_logits(self, input_ids, attention_mask, loss_mask):
        try:
            return original_generate(self, input_ids, attention_mask, loss_mask)
        except TypeError as exc:
            if "output_router_logits" not in str(exc):
                raise

        captured_states = {}
        handles = []

        def get_hook(layer_idx):
            def hook(module, _input, output):
                captured_states[layer_idx] = output[0] if isinstance(output, tuple) else output

            return hook

        layers = self._get_transformer_layers()
        target_indices = self.aux_hidden_states_layers
        for idx in target_indices:
            if 0 <= idx < len(layers):
                handles.append(layers[idx].register_forward_hook(get_hook(idx)))
            else:
                raise ValueError(f"Layer index {idx} out of bounds for model with {len(layers)} layers.")

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=False,
            )
            target = outputs.logits
        finally:
            for handle in handles:
                handle.remove()

        if len(captured_states) != 3:
            raise RuntimeError(f"Expected to capture 3 layers, but captured {len(captured_states)}")

        from specforge.modeling.target.eagle3_target_model import Eagle3TargetOutput
        from specforge.utils import padding

        hidden_states = torch.cat(
            [captured_states[idx] for idx in target_indices],
            dim=-1,
        )
        return Eagle3TargetOutput(
            hidden_states=hidden_states,
            target=padding(target, left=False),
            loss_mask=loss_mask[..., None].to(target.device),
            input_ids=padding(input_ids, left=False),
            attention_mask=attention_mask,
        )

    HFEagle3TargetModel.generate_eagle3_data = generate_eagle3_data_without_router_logits


def patch_specforge_single_rank_cleanup() -> None:
    import torch.distributed as dist
    import specforge.distributed as specforge_distributed
    import train_eagle3

    original_destroy_distributed = specforge_distributed.destroy_distributed

    def destroy_distributed_tolerating_duplicate_groups() -> None:
        try:
            original_destroy_distributed()
        except ValueError as exc:
            if "Invalid process group specified" not in str(exc):
                raise
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass

    specforge_distributed.destroy_distributed = destroy_distributed_tolerating_duplicate_groups
    train_eagle3.destroy_distributed = destroy_distributed_tolerating_duplicate_groups


def main() -> None:
    patch_fla_head_first_compat()
    patch_minicpm_hf_loader()
    patch_minicpm_sglang_server_args()
    patch_hf_target_forward()

    from train_eagle3 import main as specforge_main

    patch_specforge_single_rank_cleanup()

    specforge_main()


if __name__ == "__main__":
    main()
