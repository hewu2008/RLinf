#!/bin/bash
# Convert a ZERITH pi0.5 JAX checkpoint (Orbax params, optional Gemma LoRA and a
# pi0.5 subtask classification head) into the checkpoint layout the RECAP policy
# training loads (`model_type: cfg_model`, FSDPCfgWorker).
#
# The JAX LoRA adapters are folded into the base weights: RLinf injects its own
# PEFT adapters on top of a dense base, so leaving them out would drop what the
# ZERITH run learned. The subtask classification head is carried over as
# `subtask_classifier.*`, and the norm stats are staged under the asset id the
# training config resolves them with (`zerith` by default).
#
# Usage (from the repository root):
#   bash scripts/run_convert_zerith_jax_to_cfg.sh
#   bash scripts/run_convert_zerith_jax_to_cfg.sh <ckpt> <out> <asset-id> [extra args]
#   CKPT=/path/to/15999 OUT=/path/to/out bash scripts/run_convert_zerith_jax_to_cfg.sh --overwrite
#
# Arguments left over after the three paths are forwarded to the python entry
# point, e.g. --overwrite, --precision float32, --num-subtask-classes 13,
# --subtasks-jsonl <dataset>/meta/subtasks.jsonl.

set -euo pipefail

# Defaults follow the ZERITH subtask+classification run being warm-started.
DEFAULT_CKPT="/data/4T-1/hewu/alg-product/ZERITH_Pi_OPEN/openpi_checkpoints/pi05_zerith_subtask_cls/part_sorting_20261027_subtask_cls_headlr1e-3_16k_lr1e-5_bs8/15999"
DEFAULT_OUT="/data/4T-1/hewu/model_zoo/pi05_zerith_subtask_cls_rlinf"

CKPT=${CKPT:-${1:-${DEFAULT_CKPT}}}
OUT=${OUT:-${2:-${DEFAULT_OUT}}}
ASSET_ID=${ASSET_ID:-${3:-zerith}}
EXTRA_ARGS=("${@:4}")

python -m rlinf.utils.ckpt_convertor.convert_pi05_jax_to_cfg \
	--checkpoint-dir "${CKPT}" \
	--output-dir "${OUT}" \
	--asset-id "${ASSET_ID}" \
	"${EXTRA_ARGS[@]}"

cat <<EOF
[convert] Done. Set these in the training config (examples/offline_rl/config):
  actor.model.model_path: ${OUT}
  actor.model.openpi_data.norm_stats_path: ${OUT}/${ASSET_ID}/norm_stats.json
  actor.model.openpi.subtask_classification: true
Keep is_lora: true if you want the fine-tune to train fresh adapters on top of
the merged base; set it to false to train the backbone densely.
EOF
