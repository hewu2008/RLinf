#!/bin/bash
# Start the remote policy inference websocket server (RLinf torch Pi0/Pi0.5).
#
# The server answers one action chunk per observation. RTC guidance is enabled
# by the config: a client that announces how much of the previous chunk it has
# executed, plus its predicted delay, gets the new chunk sampled against the
# unexecuted tail. Run it from the repository root.
#
# The config takes a torch checkpoint. A JAX checkpoint has to be converted once:
#
#   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
#     --input-model       <ckpt step dir> \
#     --input-norm-stats  <ckpt step dir>/assets/EmbodiedLab/<exp>/norm_stats.json \
#     --output-model      <out> \
#     --output-norm-stats <out>/zerith/norm_stats.json
#
# and then set model_path / openpi_data.norm_stats_path in the config.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Checkpoints and assets are local, so keep transformers offline.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CONFIG="${1:-inference/configs/server_pi05_zerith.yaml}"

mkdir -p logs

python -m inference.server.main --config "${CONFIG}" \
  2>&1 | tee logs/remote_infer_server.log
