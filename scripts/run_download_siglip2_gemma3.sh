#!/bin/bash
# Download the SigLIP2 and Gemma3-270M checkpoints used by the RECAP / STEAM
# value-model configs under examples/offline_rl.
#
# Usage:
#   bash scripts/download_siglip2_gemma3.sh            # -> /data/4T-1/hewu/model_zoo
#   bash scripts/download_siglip2_gemma3.sh /data/models
#   MODEL_DIR=/data/models bash scripts/download_siglip2_gemma3.sh

set -euo pipefail

MODEL_DIR=${MODEL_DIR:-${1:-/data/4T-1/hewu/model_zoo}}

# google/gemma-3-270m is a gated repository: accept its license on the Hugging
# Face page and export HF_TOKEN (or run `hf auth login`) before running this.
# Override the mirror with HF_ENDPOINT=https://huggingface.co if needed.
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

if ! command -v hf >/dev/null 2>&1; then
	python -m pip install huggingface-hub
fi

mkdir -p "${MODEL_DIR}"

download() {
	local repo_id=$1
	local target="${MODEL_DIR}/${repo_id##*/}"

	if [ -f "${target}/config.json" ]; then
		echo "[download] ${repo_id} already exists at ${target}, skipping."
		return
	fi

	echo "[download] ${repo_id} -> ${target}"
	hf download "${repo_id}" --local-dir "${target}"
}

download google/siglip2-so400m-patch14-224
download google/gemma-3-270m

cat <<EOF
[download] Done. Point the value-model configs at:
  siglip_path:    ${MODEL_DIR}/siglip2-so400m-patch14-224
  gemma3_path:    ${MODEL_DIR}/gemma-3-270m
  tokenizer_path: ${MODEL_DIR}/gemma-3-270m
EOF
