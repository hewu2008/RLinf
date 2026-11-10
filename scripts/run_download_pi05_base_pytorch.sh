#!/bin/bash
# Download the PyTorch pi0.5 base checkpoint (thomas0829/pi05-pytorch-base),
# which examples/offline_rl CFG training takes as actor.model.model_path.
#
# Usage:
#   bash scripts/run_download_pi05_base_pytorch.sh            # -> /data/4T-1/hewu/model_zoo
#   bash scripts/run_download_pi05_base_pytorch.sh /data/models
#   MODEL_DIR=/data/models bash scripts/run_download_pi05_base_pytorch.sh

set -euo pipefail

MODEL_DIR=${MODEL_DIR:-${1:-/data/4T-1/hewu/model_zoo}}

# Override the mirror with HF_ENDPOINT=https://huggingface.co if needed.
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

if ! command -v hf >/dev/null 2>&1; then
	python -m pip install huggingface-hub
fi

mkdir -p "${MODEL_DIR}"

download() {
	local repo_id=$1
	local target="${MODEL_DIR}/${repo_id##*/}"

	if [ -f "${target}/model.safetensors" ]; then
		echo "[download] ${repo_id} already exists at ${target}, skipping."
		return
	fi

	echo "[download] ${repo_id} -> ${target}"
	hf download "${repo_id}" --local-dir "${target}"
}

download thomas0829/pi05-pytorch-base

cat <<EOF
[download] Done. Point the CFG config at:
  actor.model.model_path: ${MODEL_DIR}/pi05-pytorch-base

The repo ships model.safetensors + config.json only. RLinf also looks for
norm stats under <model_path>/assets/<asset_id>/norm_stats.json, so copy the
robot's norm_stats.json there before training.
EOF
