#!/bin/bash
# Convert recorded H1 episodes (hdf5, one directory per episode) into a LeRobot
# dataset the pi0.5 / CFG-RL training and evaluation configs can load.
#
# The hdf5 layout is the one inference/client records: the 23-D joint vector
# split per body part under observation/state and action, plus JPEG colour
# streams under observation/images/rs/<camera>/color. Actions are taken from
# the recorded action group when an episode has one and otherwise from the next
# observed state, so a converted dataset keeps the commands the policy actually
# sent instead of a shifted copy of the observations.
#
# Usage (from the repository root):
#   bash scripts/run_convert_hdf5_to_lerobot.sh
#   bash scripts/run_convert_hdf5_to_lerobot.sh <raw-dir> <repo-id> [extra args]
#   RAW_DIR=/path/to/hdf5 REPO_ID=org/name bash scripts/run_convert_hdf5_to_lerobot.sh --overwrite
#
# Arguments left over after the two paths are forwarded to the python entry
# point, e.g. --overwrite, --drop-homing, --cameras cam_high cam_left_wrist,
# --image-size 640x480, --episodes 0 1 2, --compute-stats.
#
# The dataset is written to ${HF_LEROBOT_HOME}/<repo-id>. HF_LEROBOT_HOME is
# the directory a bare repo id resolves against in the training configs.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Defaults follow the part-sorting run this converter was written for.
DEFAULT_RAW_DIR="/data/4T-2/mocap_data/hdf5/part-sorting-20261027"
DEFAULT_REPO_ID="hewu2008/part_sorting_20261027"

HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/data/4T-2/dataset}
export HF_LEROBOT_HOME
PYTHON=${PYTHON:-python}

# Positional overrides come first and are paths, so anything after the first
# flag is forwarded to the python entry point instead of being read as a path.
positional=()
extra=()
for arg in "$@"; do
	if [[ ${#extra[@]} -eq 0 && $arg != -* && ${#positional[@]} -lt 2 ]]; then
		positional+=("$arg")
	else
		extra+=("$arg")
	fi
done

RAW_DIR=${RAW_DIR:-${positional[0]:-${DEFAULT_RAW_DIR}}}
REPO_ID=${REPO_ID:-${positional[1]:-${DEFAULT_REPO_ID}}}

"${PYTHON}" "${REPO_ROOT}/toolkits/lerobot/convert_hdf5_to_lerobot.py" \
	--raw-dir "${RAW_DIR}" \
	--repo-id "${REPO_ID}" \
	"${extra[@]}"

cat <<EOF
[convert] Done. Dataset: ${HF_LEROBOT_HOME}/${REPO_ID}
Set data.train_data_paths (and data.eval_data_paths) to that directory.
EOF
