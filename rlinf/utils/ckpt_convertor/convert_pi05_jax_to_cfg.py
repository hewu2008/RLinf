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

"""Convert a ZERITH pi0.5 JAX checkpoint into the layout ``cfg_model`` trains from.

A ZERITH run writes Orbax JAX params under ``<ckpt>/params``; the pi0.5 subtask
runs also train Gemma LoRA adapters and, with ``subtask_classification``, a
``subtask_classifier`` head. ``model_type: cfg_model`` loads a torch
``PI0Pytorch`` state dict (``paligemma_with_expert.*``, ``action_*_proj.*``,
``time_mlp_*.*``) plus ``subtask_classifier.*`` from
``<model_path>/*.safetensors``, and reads norm stats from
``<model_path>/<asset_id>/norm_stats.json``.

The JAX LoRA adapters are folded into the base weights, because RLinf injects
its own PEFT adapters on top of a dense base; keeping them out would silently
drop what the ZERITH run learned.

Usage::

    python -m rlinf.utils.ckpt_convertor.convert_pi05_jax_to_cfg \\
        --checkpoint-dir /path/to/15999 \\
        --output-dir /path/to/pi05_zerith_subtask_cls_rlinf

Then point the training config at ``--output-dir`` (``model_path``) and
``<output-dir>/<asset-id>/norm_stats.json`` (``openpi_data.norm_stats_path``).
"""

from __future__ import annotations

import pathlib
from typing import Literal

import numpy as np
import torch

from rlinf.utils.ckpt_convertor.openpi._core import (
    copy_norm_stats,
    write_config_json,
)
from rlinf.utils.logging import get_logger

logger = get_logger()

# LoRA keys look like ``llm/layers/attn/q_einsum/lora_a`` for the einsum
# projections and ``llm/layers/mlp/gating_einsum_lora_a`` for the feed-forward.
_LORA_A_SUFFIXES = ("/lora_a", "_lora_a")

_PI05_PROJECTION_KEYS = (
    "action_in_proj",
    "action_out_proj",
    "time_mlp_in",
    "time_mlp_out",
)


def lora_keys(key: str) -> tuple[str, str, str] | None:
    """Return ``(base_key, a_key, b_key)`` for a LoRA parameter key, else ``None``.

    Args:
        key: Flattened JAX parameter key (``/``-separated).

    Returns:
        The base weight the delta belongs to and the two adapter keys.
    """
    for suffix in _LORA_A_SUFFIXES:
        if not key.endswith(suffix):
            continue
        stem = key[: -len(suffix)]
        b_key = f"{stem}_lora_b" if suffix == "_lora_a" else f"{stem}/lora_b"
        base_key = stem if suffix == "_lora_a" else f"{stem}/w"
        return base_key, key, b_key
    return None


def merge_lora_into_base(
    flat_params: dict[str, np.ndarray], scaling: float
) -> tuple[dict[str, np.ndarray], int]:
    """Fold ``w += scaling * (lora_a @ lora_b)`` into the base weights.

    The JAX ``Einsum``/``FeedForward`` apply LoRA as ``x @ a @ b`` over the last
    two axes of the weight, so the merged weight keeps the base shape and the
    adapter keys disappear from the result.

    Args:
        flat_params: Flattened PaliGemma params (``/``-separated keys).
        scaling: ``alpha / rank`` of the LoRA configuration.

    Returns:
        ``(merged_params, adapters_merged)``.

    Raises:
        KeyError: An adapter has no base weight or no partner.
        ValueError: Adapter shapes do not multiply into the base shape.
    """
    merged = dict(flat_params)
    merged_count = 0
    for key in sorted(flat_params):
        triple = lora_keys(key)
        if triple is None:
            continue
        base_key, a_key, b_key = triple
        for required in (base_key, b_key):
            if required not in merged:
                raise KeyError(
                    f"LoRA parameter {key!r} has no {required!r} in the checkpoint."
                )
        base = np.asarray(merged[base_key], dtype=np.float32)
        a = np.asarray(merged.pop(a_key), dtype=np.float32)
        b = np.asarray(merged.pop(b_key), dtype=np.float32)
        if a.ndim != b.ndim or a.shape[-1] != b.shape[-2]:
            raise ValueError(
                f"LoRA pair for {base_key!r} multiplies into nothing usable: "
                f"a {a.shape} and b {b.shape} do not share the rank axis."
            )
        expected = a.shape[:-1] + b.shape[-1:]
        if base.shape != expected:
            raise ValueError(
                f"LoRA pair for {base_key!r} multiplies into {expected} but the "
                f"base weight is {base.shape}."
            )
        merged[base_key] = base + scaling * np.matmul(a, b)
        merged_count += 1
    return merged, merged_count


def _head_leaf(value) -> np.ndarray:
    """Return one JAX leaf as float32, unwrapping the nnx ``{'value': array}`` form."""
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    return np.asarray(value, dtype=np.float32)


def subtask_classifier_state(
    head_params: dict, *, num_classes: int = 0
) -> dict[str, np.ndarray]:
    """Map the JAX ``subtask_classifier`` tree onto the torch head's state keys.

    JAX ``LayerNorm`` stores ``scale``/``bias`` and ``Linear`` stores
    ``kernel``/``bias`` with an ``(in, out)`` kernel, while torch stores
    ``weight``/``bias`` with an ``(out, in)`` weight.

    Args:
        head_params: ``params["subtask_classifier"]`` as restored from Orbax.
        num_classes: Expected class count; ``0`` takes it from the head itself.

    Returns:
        ``subtask_classifier.*`` tensors plus the widths when the caller needs
        them (see :func:`convert`).

    Raises:
        KeyError: A required sub-module or leaf is missing.
        ValueError: The head width or class count does not match.
    """
    expected = {
        "norm": ("scale", "bias"),
        "hidden": ("kernel", "bias"),
        "out": ("kernel", "bias"),
    }
    for module, leaves in expected.items():
        if module not in head_params:
            raise KeyError(f"subtask_classifier has no {module!r} module.")
        missing = [leaf for leaf in leaves if leaf not in head_params[module]]
        if missing:
            raise KeyError(f"subtask_classifier.{module} is missing {missing}.")

    norm_scale = _head_leaf(head_params["norm"]["scale"])
    norm_bias = _head_leaf(head_params["norm"]["bias"])
    hidden_kernel = _head_leaf(head_params["hidden"]["kernel"])
    hidden_bias = _head_leaf(head_params["hidden"]["bias"])
    out_kernel = _head_leaf(head_params["out"]["kernel"])
    out_bias = _head_leaf(head_params["out"]["bias"])

    width = norm_scale.shape[0]
    if hidden_kernel.shape[0] != width:
        raise ValueError(
            f"subtask_classifier.hidden.kernel expects {hidden_kernel.shape[0]} "
            f"inputs but the LayerNorm has width {width}."
        )
    hidden_dim = hidden_kernel.shape[1]
    classes = out_kernel.shape[1]
    if out_kernel.shape[0] != hidden_dim or out_bias.shape[0] != classes:
        raise ValueError(
            "subtask_classifier.out does not match the hidden layer: "
            f"kernel {out_kernel.shape}, bias {out_bias.shape}, "
            f"hidden {hidden_dim}."
        )
    if num_classes and classes != num_classes:
        raise ValueError(
            f"the checkpoint head predicts {classes} classes but "
            f"--num-subtask-classes={num_classes}; rebuild the class table."
        )

    return {
        "width": width,
        "hidden_dim": hidden_dim,
        "num_classes": classes,
        "state": {
            "subtask_classifier.norm.weight": norm_scale,
            "subtask_classifier.norm.bias": norm_bias,
            "subtask_classifier.hidden.weight": hidden_kernel.T,
            "subtask_classifier.hidden.bias": hidden_bias,
            "subtask_classifier.out.weight": out_kernel.T,
            "subtask_classifier.out.bias": out_bias,
        },
    }


def find_norm_stats(checkpoint_dir: str | pathlib.Path) -> pathlib.Path | None:
    """Locate the ``norm_stats.json`` a JAX checkpoint saved with its assets.

    Args:
        checkpoint_dir: JAX checkpoint step directory.

    Returns:
        The first match under ``assets/`` (or the directory root), or ``None``.
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    candidates = sorted((checkpoint_dir / "assets").rglob("norm_stats.json"))
    direct = checkpoint_dir / "norm_stats.json"
    if direct.is_file():
        candidates.insert(0, direct)
    return candidates[0] if candidates else None


def _projection_state(params: dict, pi05: bool) -> dict[str, np.ndarray]:
    """Map the action/time projection matrices onto torch ``Linear`` weights."""
    keys = _PI05_PROJECTION_KEYS if pi05 else ("state_proj", *_PI05_PROJECTION_KEYS)
    state: dict[str, np.ndarray] = {}
    for key in keys:
        if key not in params:
            raise KeyError(f"checkpoint is missing the {key!r} projection.")
        kernel = params[key]["kernel"]
        bias = params[key]["bias"]
        if isinstance(kernel, dict):
            kernel = kernel["value"]
        if isinstance(bias, dict):
            bias = bias["value"]
        state[f"{key}.weight"] = np.asarray(kernel).T
        state[f"{key}.bias"] = np.asarray(bias)
    return state


def _tied_aliases(state: dict[str, torch.Tensor]) -> set[str]:
    """Return state-dict keys that alias an earlier key through shared storage."""
    seen: dict[int, str] = {}
    aliases: set[str] = set()
    for name, tensor in state.items():
        pointer = tensor.data_ptr()
        if pointer in seen:
            aliases.add(name)
        else:
            seen[pointer] = name
    return aliases


def _lora_scaling(variant: str) -> float:
    """Return ``alpha / rank`` of the JAX LoRA configuration of ``variant``."""
    import openpi.models.gemma as gemma

    configs = gemma.get_config(variant).lora_configs
    if not configs:
        raise ValueError(f"variant {variant!r} carries no LoRA configuration.")
    scaling = {name: cfg.scaling_value for name, cfg in configs.items()}
    if len(set(scaling.values())) != 1:
        raise ValueError(
            f"variant {variant!r} mixes LoRA scalings {scaling}; pass --lora-scaling."
        )
    return next(iter(scaling.values()))


def convert(
    checkpoint_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    *,
    config_name: str = "pi05_zerith",
    asset_id: str = "zerith",
    precision: Literal["bfloat16", "float32"] = "bfloat16",
    merge_lora: bool = True,
    lora_variant: str = "gemma_2b_lora",
    lora_scaling: float | None = None,
    num_subtask_classes: int = 0,
    subtasks_jsonl: str | pathlib.Path | None = None,
    overwrite: bool = False,
) -> pathlib.Path:
    """Convert one JAX checkpoint step into a ``cfg_model`` training checkpoint.

    Args:
        checkpoint_dir: JAX step directory holding ``params/`` (and optionally
            ``assets/`` with the norm stats).
        output_dir: Directory to write ``model.safetensors``, ``config.json`` and
            ``<asset_id>/norm_stats.json`` into.
        config_name: RLinf OpenPI config that defines the target architecture.
        asset_id: Asset id the training config resolves norm stats with.
        precision: Storage dtype of the written weights.
        merge_lora: Fold the JAX LoRA adapters into the base weights.
        lora_variant: JAX LoRA variant, used to derive ``alpha / rank``.
        lora_scaling: Explicit LoRA scaling, overriding ``lora_variant``.
        num_subtask_classes: Expected class count; ``0`` takes it from the head.
        subtasks_jsonl: Optional ``meta/subtasks.jsonl``; copied next to the norm
            stats and cross-checked against the head.
        overwrite: Replace an existing ``model.safetensors``.

    Returns:
        The output directory.
    """
    import openpi.models.gemma
    import openpi.models_pytorch.pi0_pytorch
    import safetensors.torch

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.modules.subtask import SubtaskClassifier
    from rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python import (
        paligemma_torch_config,
        slice_gemma_state_dict,
        slice_initial_orbax_checkpoint,
        slice_paligemma_state_dict,
    )

    checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser()
    output_dir = pathlib.Path(output_dir).expanduser()
    if not (checkpoint_dir / "params").is_dir():
        raise FileNotFoundError(
            f"{checkpoint_dir} has no params/ directory; pass the JAX step "
            "directory (e.g. .../15999), not its parent."
        )
    weights_path = output_dir / "model.safetensors"
    if weights_path.exists() and not overwrite:
        raise FileExistsError(f"{weights_path} exists; pass --overwrite.")

    model_config = get_openpi_config(config_name).model
    if not model_config.pi05:
        raise ValueError(
            f"config {config_name!r} is not pi0.5; this converter handles the "
            "pi0.5 subtask checkpoints."
        )

    logger.info("restoring JAX params from %s", checkpoint_dir / "params")
    initial = slice_initial_orbax_checkpoint(
        checkpoint_dir=str(checkpoint_dir), restore_precision="float32"
    )
    flat_params = initial["paligemma_params"]
    adapter_count = 0
    if merge_lora:
        scaling = (
            float(lora_scaling)
            if lora_scaling is not None
            else _lora_scaling(lora_variant)
        )
        flat_params, adapter_count = merge_lora_into_base(flat_params, scaling)
        logger.info(
            "merged %d LoRA adapter pairs into the base weights (scaling=%s)",
            adapter_count,
            scaling,
        )
        if adapter_count == 0:
            logger.warning(
                "merge_lora is on but the checkpoint has no lora_a/lora_b params"
            )
    else:
        logger.warning(
            "merge_lora is off: the JAX LoRA deltas are dropped and the converted "
            "weights will not reproduce the ZERITH policy"
        )

    paligemma_params, expert_params = slice_paligemma_state_dict(
        flat_params, paligemma_torch_config()
    )
    action_expert_config = openpi.models.gemma.get_config(
        model_config.action_expert_variant
    )
    gemma_params = slice_gemma_state_dict(
        expert_params,
        action_expert_config,
        num_expert=1,
        checkpoint_dir=str(checkpoint_dir),
        pi05=model_config.pi05,
    )
    projection_params = _projection_state(
        initial["projection_params"], model_config.pi05
    )

    head = subtask_classifier_state(
        initial["projection_params"]["subtask_classifier"],
        num_classes=num_subtask_classes,
    )
    expected_width = openpi.models.gemma.get_config(
        model_config.paligemma_variant
    ).width
    if head["width"] != expected_width:
        raise ValueError(
            f"the head reads {head['width']} inputs but {config_name!r} builds a "
            f"{expected_width}-wide prefix."
        )
    if head["num_classes"] != num_subtask_classes:
        logger.info("head predicts %d subtask classes", head["num_classes"])

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)
    model.subtask_classifier = SubtaskClassifier(
        head["width"], head["hidden_dim"], head["num_classes"]
    )
    state = {
        **paligemma_params,
        **gemma_params,
        **projection_params,
        **{key: torch.from_numpy(value) for key, value in head["state"].items()},
    }
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            "the converted state is missing model tensors: "
            f"{incompatible.missing_keys[:8]}"
        )
    if incompatible.unexpected_keys:
        logger.warning(
            "ignored %d unconverted JAX tensors, e.g. %s",
            len(incompatible.unexpected_keys),
            incompatible.unexpected_keys[:5],
        )

    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    model = model.to(dtype=dtype)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    aliases = _tied_aliases(model.state_dict())
    safetensors.torch.save_model(model, str(weights_path))
    saved = safetensors.torch.load_file(str(weights_path), device="cpu")
    missing = sorted(set(model.state_dict()) - set(saved))
    if set(missing) - aliases:
        raise RuntimeError(
            f"{weights_path} lost non-tied tensors: {sorted(set(missing) - aliases)[:8]}"
        )
    logger.info(
        "wrote %s: %d tensors (%s), %d tied aliases folded, head=%d classes",
        weights_path,
        len(saved),
        precision,
        len(set(missing) & aliases),
        head["num_classes"],
    )

    subtasks_path = None
    if subtasks_jsonl is not None:
        subtasks_path = pathlib.Path(subtasks_jsonl).expanduser()
        if not subtasks_path.is_file():
            raise FileNotFoundError(f"subtasks jsonl not found: {subtasks_path}")
        class_count = sum(
            1 for line in subtasks_path.read_text().splitlines() if line.strip()
        )
        if class_count != head["num_classes"]:
            raise ValueError(
                f"{subtasks_path} lists {class_count} classes but the head "
                f"predicts {head['num_classes']}."
            )
        staged = output_dir / asset_id / "subtasks.jsonl"
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(subtasks_path.read_text())
        logger.info("staged the class table at %s", staged)

    norm_stats = find_norm_stats(checkpoint_dir)
    if norm_stats is None:
        logger.warning(
            "no norm_stats.json under %s; set openpi_data.norm_stats_path "
            "manually or training will run unnormalized",
            checkpoint_dir,
        )
    else:
        dest = output_dir / asset_id / "norm_stats.json"
        copy_norm_stats(norm_stats, dest)
        logger.info("staged norm stats at %s (from %s)", dest, norm_stats)

    write_config_json(
        {
            "action_dim": model_config.action_dim,
            "action_horizon": model_config.action_horizon,
            "paligemma_variant": model_config.paligemma_variant,
            "action_expert_variant": model_config.action_expert_variant,
            "pi05": model_config.pi05,
            "dtype": precision,
            "source_checkpoint": str(checkpoint_dir),
            "lora_merged": adapter_count,
            "subtask_classification": True,
            "num_subtask_classes": head["num_classes"],
            "subtask_class_hidden_dim": head["hidden_dim"],
        },
        output_dir,
    )
    return output_dir


def main(
    checkpoint_dir: str,
    output_dir: str,
    config_name: str = "pi05_zerith",
    asset_id: str = "zerith",
    precision: Literal["bfloat16", "float32"] = "bfloat16",
    merge_lora: bool = True,
    lora_variant: str = "gemma_2b_lora",
    lora_scaling: float | None = None,
    num_subtask_classes: int = 0,
    subtasks_jsonl: str | None = None,
    overwrite: bool = False,
) -> None:
    """Convert the checkpoint and print the training-config lines to apply."""
    output = convert(
        checkpoint_dir,
        output_dir,
        config_name=config_name,
        asset_id=asset_id,
        precision=precision,
        merge_lora=merge_lora,
        lora_variant=lora_variant,
        lora_scaling=lora_scaling,
        num_subtask_classes=num_subtask_classes,
        subtasks_jsonl=subtasks_jsonl,
        overwrite=overwrite,
    )
    logger.info(
        "done. Set these in the training config:\n"
        "  actor.model.model_path: %s\n"
        "  actor.model.openpi_data.norm_stats_path: %s\n"
        "  actor.model.openpi.subtask_classification: true\n"
        "  actor.model.openpi.num_subtask_classes: <class count from the head>",
        output,
        output / asset_id / "norm_stats.json",
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
