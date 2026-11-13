# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merge a PEFT-LoRA CFG SFT checkpoint into a bare OpenPI Pi0 checkpoint.

A CFG-RL SFT run (``model_type=cfg_model``, ``is_lora=true``) saves the
PEFT-wrapped OpenPI PyTorch state dict, so its keys look like::

    paligemma_with_expert.paligemma.base_model.model.<...>.q_proj.base_layer.weight
    paligemma_with_expert.paligemma.base_model.model.<...>.q_proj.lora_A.default.weight
    paligemma_with_expert.paligemma.base_model.model.<...>.q_proj.lora_B.default.weight

The inference server loads a bare Pi0 checkpoint and applies no adapters, and
``sft_to_openpi`` does not read that layout either. This script merges the
adapters into the base weights, drops the PEFT names, and reuses
``old_to_new_state_dict`` to write the bare Pi0 layout together with the
``config.json`` and ``<asset_id>/norm_stats.json`` the server needs::

    python -m inference.tools.merge_cfg_lora_to_openpi \\
        --ckpt <run>/cfg_sft/checkpoints/global_step_30000/actor/model_state_dict/full_weights.pt \\
        --base-model /path/to/pi05-pytorch-base \\
        --asset-id zerith \\
        --out <run>/deploy/global_step_30000_openpi

``scripts/run_merge_cfg_lora_to_openpi.sh`` wraps this call with the paths taken
from environment variables. Then point ``model_path`` and
``openpi_data.norm_stats_path`` in ``inference/configs/server_*.yaml`` at
``--out`` and keep ``is_lora: false``: the adapters are already folded into the
base weights.
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import torch
from safetensors.torch import load_file, save_file

from rlinf.utils.ckpt_convertor.openpi.openpi_pytorch_to_openpi import (
    old_to_new_state_dict,
)

logger = logging.getLogger(__name__)

LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"
BASE_LAYER_SUFFIX = ".base_layer.weight"

# get_model() wraps this subtree with get_peft_model() for cfg_model, so the
# adapters live under the PEFT wrapper names.
PEFT_PREFIX = "paligemma_with_expert.paligemma.base_model.model."
PLAIN_PREFIX = "paligemma_with_expert.paligemma."

# old_to_new_state_dict() ties the shared embedder to PaliGemma's lm_head.
EMBEDDER_KEY = "llm.embedder.embedding.weight"
BASE_EMBEDDER_KEY = "paligemma_with_expert.paligemma.lm_head.weight"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Merge a PEFT-LoRA CFG SFT checkpoint into a bare Pi0 one."
    )
    parser.add_argument(
        "--ckpt",
        type=pathlib.Path,
        required=True,
        help="RLinf SFT full_weights.pt with lora_A/lora_B keys",
    )
    parser.add_argument(
        "--base-model",
        type=pathlib.Path,
        required=True,
        help="pre-training checkpoint holding config.json and <asset_id>/norm_stats.json",
    )
    parser.add_argument(
        "--asset-id",
        required=True,
        help="dataset asset directory under --base-model that holds norm_stats.json",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        required=True,
        help="output directory for the bare Pi0 checkpoint",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRA scaling lora_alpha/r; the cfg build uses lora_alpha == r (default: 1.0)",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="storage dtype of the written weights (default: bf16)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing checkpoint in --out instead of failing",
    )
    return parser.parse_args(argv)


def _split_state_dict(
    sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split a PEFT state dict into A adapters, B adapters, and plain tensors."""
    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}
    plain: dict[str, torch.Tensor] = {}
    for key, tensor in sd.items():
        if key.endswith(LORA_A_SUFFIX):
            lora_a[key[: -len(LORA_A_SUFFIX)]] = tensor
        elif key.endswith(LORA_B_SUFFIX):
            lora_b[key[: -len(LORA_B_SUFFIX)]] = tensor
        else:
            plain[key] = tensor
    if lora_a.keys() != lora_b.keys():
        only_a = sorted(set(lora_a) - set(lora_b))
        only_b = sorted(set(lora_b) - set(lora_a))
        raise ValueError(
            f"every lora_A must have a matching lora_B; A-only={only_a} B-only={only_b}"
        )
    return lora_a, lora_b, plain


def _merge_lora(
    sd: dict[str, torch.Tensor], scale: float
) -> tuple[dict[str, torch.Tensor], int]:
    """Fold ``W += scale * B @ A`` into ``.base_layer.weight`` of every module."""
    lora_a, lora_b, merged = _split_state_dict(sd)
    if not lora_a:
        raise ValueError(
            "no lora_A/lora_B keys found; the checkpoint is already merged or was "
            "not trained with cfg_model is_lora=true"
        )

    for module, a in lora_a.items():
        base_key = module + BASE_LAYER_SUFFIX
        if base_key not in merged:
            raise KeyError(
                f"merged checkpoint expects {base_key} for the adapter of {module}, "
                "but the state dict has no such tensor"
            )
        base = merged[base_key]
        delta = lora_b[module].to(torch.float32) @ a.to(torch.float32)
        merged[base_key] = (base.to(torch.float32) + scale * delta).to(base.dtype)
        logger.info(
            "merged %s: base %s %s + %s @ %s",
            module,
            tuple(base.shape),
            base.dtype,
            tuple(lora_b[module].shape),
            tuple(a.shape),
        )
    return merged, len(lora_a)


def _rename_to_openpi_pytorch(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop the PEFT wrapper and ``.base_layer`` names."""
    renamed: dict[str, torch.Tensor] = {}
    for key, tensor in sd.items():
        new_key = key.replace(PEFT_PREFIX, PLAIN_PREFIX)
        new_key = new_key.replace(".base_layer.", ".")
        leftover = [
            marker
            for marker in ("lora_", "base_model", "base_layer", "default.")
            if marker in new_key
        ]
        if leftover:
            raise RuntimeError(f"unexpected {leftover} in renamed key {new_key!r}")
        if new_key in renamed:
            raise RuntimeError(f"duplicate key after rename: {new_key!r}")
        renamed[new_key] = tensor
    return renamed


def _write_checkpoint(
    state: dict[str, torch.Tensor], out: pathlib.Path, dtype: torch.dtype
) -> pathlib.Path:
    """Write the tensors as one safetensors file, leaving no partial file behind."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model.safetensors"
    payload = {
        key: tensor.detach().to(dtype).contiguous() for key, tensor in state.items()
    }
    partial = out / "model.safetensors.partial"
    save_file(payload, str(partial))
    partial.replace(path)
    return path


def _copy_assets(base_model: pathlib.Path, asset_id: str, out: pathlib.Path) -> None:
    """Copy the config and the asset's normalization stats next to the weights."""
    import shutil

    config = base_model / "config.json"
    norm_stats = base_model / asset_id / "norm_stats.json"
    for source in (config, norm_stats):
        if not source.is_file():
            raise FileNotFoundError(f"missing {source}")
    shutil.copy2(config, out / "config.json")
    (out / asset_id).mkdir(parents=True, exist_ok=True)
    shutil.copy2(norm_stats, out / asset_id / "norm_stats.json")


def _check_embedder_moved(
    payload: dict[str, torch.Tensor], base_model: pathlib.Path
) -> None:
    """Fail when the SFT weights equal the frozen base, which means nothing merged."""
    base_path = base_model / "model.safetensors"
    if not base_path.is_file():
        logger.warning("no %s; skipping the embedder sanity check", base_path)
        return
    base = load_file(str(base_path), device="cpu")
    if BASE_EMBEDDER_KEY not in base:
        logger.warning(
            "no %s in %s; skipping the embedder sanity check",
            BASE_EMBEDDER_KEY,
            base_path,
        )
        return
    delta = (
        payload[EMBEDDER_KEY].to(torch.float32)
        - base[BASE_EMBEDDER_KEY].to(torch.float32)
    ).abs()
    logger.info("sanity %s: max|merged-base|=%.6f", EMBEDDER_KEY, delta.max().item())
    if delta.max().item() == 0.0:
        raise RuntimeError(
            f"{EMBEDDER_KEY} matches {BASE_EMBEDDER_KEY} exactly; the merge did nothing"
        )


def main(argv: list[str] | None = None) -> None:
    """Merge the adapters and write the deployable checkpoint."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    out: pathlib.Path = args.out
    weights = out / "model.safetensors"
    if weights.exists() and not args.overwrite:
        raise FileExistsError(f"{weights} exists; pass --overwrite to replace it")

    sd = torch.load(str(args.ckpt), map_location="cpu", mmap=True, weights_only=False)
    logger.info("loaded %d tensors from %s", len(sd), args.ckpt)

    merged, count = _merge_lora(sd, args.lora_scale)
    logger.info("merged %d LoRA adapters (scaling=%s)", count, args.lora_scale)

    renamed = _rename_to_openpi_pytorch(merged)
    logger.info("renamed %d tensors to the OpenPI PyTorch layout", len(renamed))

    bare = old_to_new_state_dict(renamed)
    logger.info("converted to %d bare Pi0 tensors", len(bare))

    path = _write_checkpoint(
        bare, out, torch.bfloat16 if args.dtype == "bf16" else torch.float32
    )
    logger.info("wrote %s", path)

    _copy_assets(args.base_model, args.asset_id, out)
    logger.info("copied config.json and %s/norm_stats.json", args.asset_id)
    _check_embedder_moved(bare, args.base_model)


if __name__ == "__main__":
    main()
