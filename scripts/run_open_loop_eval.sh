#!/bin/bash
# Open-loop evaluation of the running policy server against a recorded LeRobot
# dataset: every request replays one dataset frame through the same websocket
# path the robot uses (JPEG camera frames + raw joint state) and compares the
# returned action chunk with the record. No checkpoint is loaded locally, so
# start inference/run_server.sh first.
#
# Usage (from the repository root):
#   bash scripts/run_open_loop_eval.sh
#   TRAJ_IDS="0 1" MAX_CHUNKS=240 bash scripts/run_open_loop_eval.sh
#   SAVE_PLOT_DIR= bash scripts/run_open_loop_eval.sh --raw-images
#
# Extra arguments are forwarded to the python entry point. Leave PROMPT empty to
# let the client use each sample's task, then the server default_prompt.

set -euo pipefail

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-55555}
DATA_PATH=${DATA_PATH:-/data/4T-2/dataset/hewu2008/part_sorting_20261027_subtask}
TRAJ_IDS=${TRAJ_IDS:-"0 1 2 3"}
MAX_CHUNKS=${MAX_CHUNKS:-240}
CHUNK_STRIDE=${CHUNK_STRIDE:-}
PROMPT=${PROMPT:-"Sort all objects into the correct bins using both arms"}
CAMERA_NAMES=${CAMERA_NAMES:-"cam_high cam_left_wrist cam_right_wrist"}
SAVE_PLOT_DIR=${SAVE_PLOT_DIR:-./open_loop_test}
LOG_FILE=${LOG_FILE:-logs/open_loop_eval.log}

# The dataset is local; keep transformers/lerobot from reaching for the hub.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

read -r -a traj_array <<<"${TRAJ_IDS}"
read -r -a camera_array <<<"${CAMERA_NAMES}"

args=(
	--host "${HOST}"
	--port "${PORT}"
	--data-path "${DATA_PATH}"
	--traj-ids "${traj_array[@]}"
	--max-chunks "${MAX_CHUNKS}"
	--camera-names "${camera_array[@]}"
)
if [ -n "${CHUNK_STRIDE}" ]; then args+=(--chunk-stride "${CHUNK_STRIDE}"); fi
if [ -n "${PROMPT}" ]; then args+=(--prompt "${PROMPT}"); fi
if [ -n "${SAVE_PLOT_DIR}" ]; then args+=(--save-plot-dir "${SAVE_PLOT_DIR}"); fi

mkdir -p logs
python -m inference.open_loop_eval "${args[@]}" "$@" 2>&1 | tee -a "${LOG_FILE}"
