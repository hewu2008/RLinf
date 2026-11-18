#!/bin/bash
# Fold a PEFT-LoRA CFG SFT checkpoint into a bare OpenPI Pi0 checkpoint that
# scripts/run_inference_server.sh can load.
#
# The CFG-RL SFT run (is_lora: true) saves full_weights.pt with PEFT wrapper
# names and lora_A/lora_B keys; the server loads a bare Pi0 layout and applies
# no adapters, so the adapters have to be merged into the base weights first.
# See inference/tools/merge_cfg_lora_to_openpi.py for what the merge does.
#
# Usage (from the repository root):
#   bash scripts/run_merge_cfg_lora_to_openpi.sh
#   bash scripts/run_merge_cfg_lora_to_openpi.sh <ckpt-dir|full_weights.pt> [base-model] [asset-id] [out] [extra args]
#   CKPT=/path/full_weights.pt OUT=/path/deploy bash scripts/run_merge_cfg_lora_to_openpi.sh --overwrite
#
# <ckpt> may be either the checkpoint directory (``.../global_step_6000``) or
# its ``actor/model_state_dict/full_weights.pt``. When <out> is omitted it is
# derived as ``<run>/deploy/<step>_openpi``, so one run never writes into
# another run's deploy directory.
#
# Arguments left over after the four paths are forwarded to the python entry
# point, e.g. --overwrite, --dtype fp32, --lora-scale 0.5.

set -euo pipefail

# Defaults follow the cfg_rl run currently deployed; edit CKPT when merging
# another run.
DEFAULT_CKPT="logs/cfg_rl/cfg_rl_openpi-20261117-22:26:44/cfg_sft/checkpoints/global_step_6000"

CKPT=${CKPT:-${1:-${DEFAULT_CKPT}}}
BASE_MODEL=${BASE_MODEL:-${2:-/data/4T-1/hewu/model_zoo/pi05-pytorch-base}}
ASSET_ID=${ASSET_ID:-${3:-zerith}}
OUT=${OUT:-${4:-}}
EXTRA_ARGS=("${@:5}")
# Interpreter that has torch, safetensors, and the repository on PYTHONPATH.
PYTHON=${PYTHON:-python}

if [[ -d "${CKPT}" ]]; then
	CKPT="${CKPT%/}/actor/model_state_dict/full_weights.pt"
fi
[[ -f "${CKPT}" ]] || { echo "[merge] no checkpoint at ${CKPT}" >&2; exit 1; }

if [[ -z "${OUT}" ]]; then
	run_root=${CKPT%/cfg_sft/checkpoints/*}
	step_release=${CKPT#*/cfg_sft/checkpoints/}
	step=${step_release%%/*}
	if [[ "${run_root}" == "${CKPT}" || -z "${step}" ]]; then
		echo "[merge] cannot derive the deploy directory from ${CKPT}; pass OUT=" >&2
		exit 1
	fi
	OUT="${run_root}/deploy/${step}_openpi"
fi

"${PYTHON}" -m inference.tools.merge_cfg_lora_to_openpi \
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
