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
#
# Usage (from the repository root):
#   bash inference/run_server.sh                     # the config named below
#   bash inference/run_server.sh <config.yaml> --no-rtc
#   PORT=60000 bash inference/run_server.sh          # bind another port
#
# The first argument is the config; every later argument reaches
# `inference.server.main` (--host, --port, --default-prompt, --no-rtc,
# --no-warmup). A leading option skips the config, so
# `bash inference/run_server.sh --no-rtc` keeps the config named below.
# HOST and PORT are shorthand for --host/--port, and
# scripts/run_open_loop_eval.sh reads the same two variables, so one
# `PORT=...` covers both sides.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Checkpoints and assets are local, so keep transformers offline.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
  CONFIG="$1"
  shift
else
  CONFIG=inference/configs/server_pi05_zerith.yaml
fi

args=(--config "${CONFIG}")
if [ -n "${HOST:-}" ]; then args+=(--host "${HOST}"); fi
if [ -n "${PORT:-}" ]; then args+=(--port "${PORT}"); fi

mkdir -p logs

python -m inference.server.main "${args[@]}" "$@" \
  2>&1 | tee logs/remote_infer_server.log
