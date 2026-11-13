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
# robot's own venv, launched from the repository root:
#   ROBOT_PYTHON=/path/to/python bash scripts/run_inference_client.sh
# The SDK path is fixed in the client: h1_robot._ensure_sdk_on_path resolves it
# to inference/client/lib, so it is not passed on the command line.
# Client deps: pip install -r inference/requirements.txt
#
# The motors are driven through sudo by default. sudo resets PATH, so the
# interpreter is resolved to an absolute path first and that path is executed,
# keeping the conda environment's site-packages:
#   ROBOT_PYTHON=/path/to/python bash scripts/run_inference_client.sh
# ROBOT_PYTHON may also be a plain name on PATH, resolved here; set SUDO=0 to run
# the interpreter directly. sudo drops the exported variables, so list the ones
# the vendor SDK needs in SUDO_ENV, e.g. SUDO_ENV="LD_LIBRARY_PATH=$LD_LIBRARY_PATH".
#
# HOST and PORT say where the policy server listens; they must match how it was
# started (scripts/run_inference_server.sh reads the same two variables).
#
# Press ``r`` during the run to end the current episode: the arms are driven back
# to the start pose while that return move is recorded, then the client writes
# rollouts/<episode_id>/episode.hdf5 in the layout of the mocap dataset and drops
# into pdb so the scene can be reset; type ``c`` to start the next episode.
# Pass --no-record to run without writing files.
#
# prompt1: Pick and place the door handle into the left bin and the L-shaped bent pipe into the right bin
# hdf5: 002e81207a19441bae7f4653ad759d7c
#
# prompt2: Use the left arm to pick and place the door handle into the left bin. Use the right arm to pick and place the L-shaped bent pipe into the right bin. Put all other objects into the front bin.
# hdf5: c066a2e76fdd4f299af074f79cd8bc04

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

# shellcheck disable=SC2086  # launcher is a command prefix, empty when unused.
${launcher} "${PYTHON}" -m inference.client.main \
  --host "${HOST:-172.31.200.200}" \
  --port "${PORT:-55555}" \
  --prompt "Sort all objects into the correct bins using both arms" \
  --init_hdf5 /data/zerith_data/1_clear_the_bin_box_20261027/073230fe3e9b4f66bed95e2082a3a1e3/episode.hdf5 \
  --init_frame_idx 30 \
  --control_freq 30 \
  --no-rtc \
  --min_exec_horizon 18 \
  --delay_buffer_size 10 \
  --num_steps 864000 \
  --save_dir rollouts \
  --task_name clear_the_bin_box \
  --task_id 1
