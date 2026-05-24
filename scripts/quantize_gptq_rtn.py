"""
Local RTN-to-GPTQ helper for MiniCPM-SALA.

This follows the SOAR demo-quant format path and exists as a fallback smoke
quantizer. It is not a new kernel and should not be selected for final results
unless public accuracy confirms it is safe.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


LINEAR_SUFFIXES = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".o_gate.weight",
    ".z_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
    ".gate_up_proj.weight",
)


def quantize_weight_rtn(
    weight: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    sym: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_features, in_features = weight.shape
    pack_factor = 32 // bits
    maxq = 2**bits - 1

    if group_size <= 0:
        group_size = in_features
    num_groups = (in_features + group_size - 1) // group_size

    if in_features % group_size != 0:
        pad = group_size * num_groups - in_features
        weight = torch.nn.functional.pad(weight, (0, pad))
        in_features = weight.shape[1]

    w = weight.float()
    w_grouped = w.reshape(out_features, num_groups, group_size)
    if sym:
        max_abs = w_grouped.abs().amax(dim=2, keepdim=True)
        scales = max_abs.clamp(min=1e-10) / (2 ** (bits - 1) - 1)
        zeros = torch.full_like(scales, 2 ** (bits - 1))
    else:
        w_min = w_grouped.min(dim=2, keepdim=True).values
        w_max = w_grouped.max(dim=2, keepdim=True).values
        scales = (w_max - w_min).clamp(min=1e-10) / maxq
        zeros = -w_min / scales

    qw = torch.clamp(torch.round(w_grouped / scales + zeros), 0, maxq).to(torch.int32)
    qw = qw.reshape(out_features, in_features)

    qw_t = qw.t().contiguous()
    qweight = torch.zeros(in_features // pack_factor, out_features, dtype=torch.int32)
    for offset in range(pack_factor):
        qweight |= qw_t[offset::pack_factor, :] << (bits * offset)

    scales_out = scales.squeeze(2).t().contiguous().half()

    zeros_int = torch.clamp(torch.round(zeros.squeeze(2)), 0, maxq).to(torch.int32)
    zeros_t = zeros_int.t().contiguous()
    qzeros = torch.zeros(num_groups, out_features // pack_factor, dtype=torch.int32)
    for offset in range(pack_factor):
        qzeros |= zeros_t[:, offset::pack_factor] << (bits * offset)

    g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
    return qweight, scales_out, qzeros, g_idx


def copy_metadata_files(src: Path, dst: Path) -> None:
    for path in src.iterdir():
        if path.suffix in (".json", ".py", ".model", ".txt") or path.name == "tokenizer.json":
            shutil.copy2(path, dst / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="RTN W4A16 GPTQ-format quantization")
    parser.add_argument("--input", required=True, help="Original model directory")
    parser.add_argument("--output", required=True, help="Quantized model output directory")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument(
        "--sym",
        action="store_true",
        help="Write a symmetric GPTQ config; useful for GPTQ-Marlin probes.",
    )
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)

    copy_metadata_files(src, dst)

    with open(src / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    config["quantization_config"] = {
        "bits": args.bits,
        "group_size": args.group_size,
        "quant_method": "gptq",
        "desc_act": False,
        "sym": args.sym,
    }
    config["torch_dtype"] = "float16"
    with open(dst / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    with open(dst / "quantize_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "bits": args.bits,
                "group_size": args.group_size,
                "desc_act": False,
                "sym": args.sym,
                "lm_head": False,
                "dynamic": {},
            },
            handle,
            indent=2,
        )

    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Expected sharded safetensors index: {index_path}")

    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)

    shard_files = sorted(set(index["weight_map"].values()))
    new_weight_map: dict[str, str] = {}
    total_quantized = 0

    for shard_name in shard_files:
        print(f"[quantize] Processing {shard_name} ...", flush=True)
        shard = load_file(str(src / shard_name))
        new_tensors = {}

        for name, tensor in shard.items():
            if name.endswith(LINEAR_SUFFIXES) and tensor.ndim == 2:
                qweight, scales, qzeros, g_idx = quantize_weight_rtn(
                    tensor.to(torch.float16),
                    bits=args.bits,
                    group_size=args.group_size,
                    sym=args.sym,
                )
                base = name.removesuffix(".weight")
                for suffix, value in (
                    (".qweight", qweight),
                    (".scales", scales),
                    (".qzeros", qzeros),
                    (".g_idx", g_idx),
                ):
                    new_tensors[base + suffix] = value
                    new_weight_map[base + suffix] = shard_name
                total_quantized += 1
            else:
                new_tensors[name] = tensor.to(torch.float16) if tensor.is_floating_point() else tensor
                new_weight_map[name] = shard_name

        save_file(new_tensors, str(dst / shard_name))

    with open(dst / "model.safetensors.index.json", "w", encoding="utf-8") as handle:
        json.dump({"metadata": {"total_size": 0}, "weight_map": new_weight_map}, handle, indent=2)

    print(
        f"[quantize] Done: {total_quantized} linear weights -> W{args.bits}A16 "
        f"(group_size={args.group_size}, sym={args.sym})"
    )


if __name__ == "__main__":
    main()
