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
# The motors are driven through sudo by default, resolving ROBOT_PYTHON to an
# absolute path first so the conda environment survives sudo's PATH reset. Set
# SUDO=0 to run the interpreter directly and list the variables the vendor SDK
# needs in SUDO_ENV, e.g. SUDO_ENV="LD_LIBRARY_PATH=$LD_LIBRARY_PATH".
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

PYTHON=${ROBOT_PYTHON:-python}

launcher=""
if [ "${SUDO:-1}" = "1" ]; then
  case "${PYTHON}" in
    /*) ;;
    *)
      resolved=$(command -v "${PYTHON}" || true)
      if [ -z "${resolved}" ]; then
        echo "cannot resolve '${PYTHON}' on PATH; set ROBOT_PYTHON to an absolute path" >&2
        exit 1
      fi
      PYTHON=${resolved}
      ;;
  esac
  launcher="sudo"
  if [ -n "${SUDO_ENV:-}" ]; then
    launcher="sudo env ${SUDO_ENV}"
  fi
fi

skip_move_to_start=""
if [ "${MOVE_TO_START:-1}" = "0" ]; then
  skip_move_to_start="--no_move_to_start"
fi

# shellcheck disable=SC2086  # launcher is a command prefix, empty when unused.
${launcher} "${PYTHON}" -m inference.client.replay \
  --episode "${EPISODE:?set EPISODE to an episode.hdf5 path or its directory}" \
  --fixed_rate "${RATE:-30}" \
  --speed "${SPEED:-1.0}" \
  ${DEINIT:+--deinit} \
  ${skip_move_to_start} \
  "$@"
