#!/bin/bash
# Drive the H1 dual-arm robot from a running policy server.
#
# RTC is on by default: --min_exec_horizon is how many control steps of the
# current chunk are executed before asking for the next one, and the measured
# delay d feeds the server-side guidance (keep d <= s <= H - d). The client
# refuses to start against a server that does not advertise RTC support; pass
# --no-rtc to run whole chunks without guidance.
#
# This side needs only inference/ plus the H1 vendor SDK, so it runs from the
# robot's own venv: ROBOT_PYTHON=/path/to/python bash inference/run_client.sh
# Point H1_SDK_DIR (or --sdk_dir) at the directory holding lib_h1_sdk_python and
# camera_client; without either, client/lib is used.
# Client deps: pip install -r inference/requirements.txt
#
# prompt1: Pick and place the door handle into the left bin and the L-shaped bent pipe into the right bin
# hdf5: 002e81207a19441bae7f4653ad759d7c
#
# prompt2: Use the left arm to pick and place the door handle into the left bin. Use the right arm to pick and place the L-shaped bent pipe into the right bin. Put all other objects into the front bin.
# hdf5: c066a2e76fdd4f299af074f79cd8bc04

set -euo pipefail

cd "$(dirname "$0")/.."

"${ROBOT_PYTHON:-python}" -m inference.client.main \
  --host 172.31.200.200 \
  --port 55555 \
  --sdk_dir "${H1_SDK_DIR:-inference/client/lib}" \
  --prompt "Sort all objects into the correct bins using both arms" \
  --init_hdf5 /data/zerith_data/1_clear_the_bin_box_20261027/04742e2eb140434d87b5b8d213741c1e/episode.hdf5 \
  --control_freq 30 \
  --min_exec_horizon 18 \
  --delay_buffer_size 8 \
  --num_steps 864000
