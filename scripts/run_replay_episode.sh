#!/bin/bash
# Replay one recorded rollout episode on the H1 robot.
#
# EPISODE is either an episode.hdf5 or the episode directory written by
# scripts/run_inference_client.sh (rollouts/<episode_id>):
#   ROBOT_PYTHON=/path/to/python EPISODE=rollouts/<episode_id> \
#     bash scripts/run_replay_episode.sh
#
# RATE is the playback frequency in Hz (default 30, the recorded control rate)
# and SPEED scales it, so SPEED=2 replays at twice that rate. With --fixed_rate
# the recorded timestamps are ignored and the gaps left by steps without an
# action disappear. DEINIT=1 releases the robot when the replay ends.
#
# MOVE_TO_START is on by default: the arms are driven to the first recorded
# state, so the replay begins where the episode began. Set MOVE_TO_START=0 to
# start from wherever the robot is standing.
#
# The replay needs inference/ plus the H1 vendor SDK; no policy server runs and
# no cameras are opened. The SDK is found through $H1_SDK_DIR, falling back to
# inference/client/lib, so it is never passed on the command line.
#
# Extra arguments are forwarded to the replay module, e.g. --skip_homing.
#
# Client deps: pip install -r inference/requirements.txt

set -euo pipefail

cd "$(dirname "$0")/.."

skip_move_to_start=""
if [ "${MOVE_TO_START:-1}" = "0" ]; then
  skip_move_to_start="--no_move_to_start"
fi

"${ROBOT_PYTHON:-python}" -m inference.client.replay \
  --episode "${EPISODE:?set EPISODE to an episode.hdf5 path or its directory}" \
  --fixed_rate "${RATE:-30}" \
  --speed "${SPEED:-1.0}" \
  ${DEINIT:+--deinit} \
  ${skip_move_to_start} \
  "$@"
