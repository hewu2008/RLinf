#!/bin/bash
# Fold a PEFT-LoRA CFG SFT checkpoint into a bare OpenPI Pi0 checkpoint that
# inference/run_server.sh can load.
#
# The CFG-RL SFT run (is_lora: true) saves full_weights.pt with PEFT wrapper
# names and lora_A/lora_B keys; the server loads a bare Pi0 layout and applies
# no adapters, so the adapters have to be merged into the base weights first.
# See inference/merge_cfg_lora_to_openpi.py for what the merge does.
#
# Usage (from the repository root):
#   bash scripts/run_merge_cfg_lora_to_openpi.sh
#   bash scripts/run_merge_cfg_lora_to_openpi.sh <ckpt> <base-model> <asset-id> <out> [extra args]
#   CKPT=/path/full_weights.pt OUT=/path/deploy bash scripts/run_merge_cfg_lora_to_openpi.sh --overwrite
#
# Arguments left over after the four paths are forwarded to the python entry
# point, e.g. --overwrite, --dtype fp32, --lora-scale 0.5.

set -euo pipefail

# Defaults follow the CFG-RL run currently deployed; edit these two paths when
# merging another run.
DEFAULT_CKPT="logs/cfg_rl/cfg_rl_openpi-20261110-19:07:16/cfg_sft/checkpoints/global_step_30000/actor/model_state_dict/full_weights.pt"
DEFAULT_OUT="logs/cfg_rl/cfg_rl_openpi-20261110-19:07:16/deploy/global_step_30000_openpi"

CKPT=${CKPT:-${1:-${DEFAULT_CKPT}}}
BASE_MODEL=${BASE_MODEL:-${2:-/data/4T-1/hewu/model_zoo/pi05-pytorch-base}}
ASSET_ID=${ASSET_ID:-${3:-zerith}}
OUT=${OUT:-${4:-${DEFAULT_OUT}}}
EXTRA_ARGS=("${@:5}")

python -m inference.merge_cfg_lora_to_openpi \
	--ckpt "${CKPT}" \
	--base-model "${BASE_MODEL}" \
	--asset-id "${ASSET_ID}" \
	--out "${OUT}" \
	"${EXTRA_ARGS[@]}"

cat <<EOF
[merge] Done. Set these in inference/configs/server_pi05_zerith.yaml:
  model_path: ${OUT}
  openpi_data.norm_stats_path: ${OUT}/${ASSET_ID}/norm_stats.json
Keep is_lora: false: the adapters are already in the base weights.
EOF
