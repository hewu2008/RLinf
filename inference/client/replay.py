# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay one recorded rollout episode on the robot.

The episode is an hdf5 file written by EpisodeRecorder (scripts/run_inference_client.sh
creates one per episode). Every recorded action is sent to the motors at its
recorded time, so a replay reproduces the original trajectory including the
return-home segment at the end. No policy server and no cameras are involved.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import time

import numpy as np

from inference.client.episode_recorder import FIELD_PATHS, merge_joint_batch
from inference.client.h1_robot import H1RobotEnv

logger = logging.getLogger(__name__)

EPISODE_FILE = "episode.hdf5"


@dataclasses.dataclass(frozen=True)
class RecordedEpisode:
    """The action stream of one recorded episode."""

    actions: np.ndarray
    """``(T, 23)`` commanded actions in the robot's state layout."""

    timestamps: np.ndarray
    """``(T,)`` epoch milliseconds, one per action."""

    control_frequency: float
    homing_start_frame: int | None
    task_name: str


def resolve_episode_path(value: str) -> pathlib.Path:
    """Accept either ``episode.hdf5`` itself or the directory that holds it."""
    path = pathlib.Path(value).expanduser()
    if path.is_dir():
        path = path / EPISODE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no episode file at {path}")
    return path


def load_episode(path: str | pathlib.Path) -> RecordedEpisode:
    """Read the actions and timestamps of one recorded episode."""
    import h5py

    with h5py.File(path, "r") as episode:
        fields = {
            name: episode[f"action/{relative}"][:]
            for name, relative in FIELD_PATHS.items()
        }
        attrs = episode.attrs
        return RecordedEpisode(
            actions=merge_joint_batch(fields),
            timestamps=np.asarray(episode["timestamp/t"][:], dtype=np.float64),
            control_frequency=float(attrs.get("control_frequency", 30)),
            homing_start_frame=(
                int(attrs["homing_start_frame"])
                if "homing_start_frame" in attrs
                else None
            ),
            task_name=str(attrs.get("task_name", "")),
        )


def select_frames(
    episode: RecordedEpisode,
    *,
    start: int = 0,
    end: int | None = None,
    skip_homing: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice the ``[start, end)`` frames the caller wants to replay."""
    stop = len(episode.actions) if end is None else min(end, len(episode.actions))
    if skip_homing and episode.homing_start_frame is not None:
        stop = min(stop, episode.homing_start_frame)
    if not 0 <= start < stop:
        raise ValueError(
            f"frame range [{start}, {stop}) is empty for an episode of "
            f"{len(episode.actions)} frames"
        )
    return episode.actions[start:stop], episode.timestamps[start:stop]


def replay_actions(
    env: H1RobotEnv,
    actions: np.ndarray,
    timestamps: np.ndarray,
    *,
    speed: float = 1.0,
    fixed_rate: float | None = None,
) -> None:
    """Send every action at its recorded time, scaled by ``speed``.

    ``fixed_rate`` ignores the recorded timestamps and plays back at a constant
    frequency instead, which hides gaps left by control steps that had no action.
    ``speed`` still applies, so the two combine as ``fixed_rate * speed``.
    """
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    if fixed_rate is not None and fixed_rate <= 0:
        raise ValueError(f"fixed rate must be positive, got {fixed_rate}")

    origin = time.monotonic()
    base = float(timestamps[0])
    total = len(actions)
    for index, action in enumerate(actions):
        if fixed_rate is None:
            target = (float(timestamps[index]) - base) / 1000.0 / speed
        else:
            target = index / (fixed_rate * speed)
        remaining = origin + target - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        env.set_joint_action(np.asarray(action)[: env.motor_count])
        if index % 30 == 0 or index == total - 1:
            logger.info("replayed %d/%d frames", index + 1, total)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode",
        required=True,
        help="episode.hdf5 to replay, or the directory that holds it",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="playback speed factor, >1 is faster"
    )
    parser.add_argument(
        "--fixed_rate",
        type=float,
        default=None,
        help="replay at this rate in Hz instead of the recorded timestamps",
    )
    parser.add_argument(
        "--start_frame", type=int, default=0, help="first frame to replay"
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=None,
        help="frame to stop at, exclusive (default: the end of the episode)",
    )
    parser.add_argument(
        "--skip_homing",
        action="store_true",
        help="stop before the recorded return-home segment",
    )
    parser.add_argument(
        "--no_move_to_start",
        dest="move_to_start",
        action="store_false",
        default=True,
        help="do not drive to the first recorded state before replaying",
    )
    parser.add_argument(
        "--deinit", action="store_true", help="release the robot when the replay ends"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Replay one episode until the last recorded action."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )

    path = resolve_episode_path(args.episode)
    episode = load_episode(path)
    actions, timestamps = select_frames(
        episode,
        start=args.start_frame,
        end=args.end_frame,
        skip_homing=args.skip_homing,
    )
    logger.info(
        "replaying %s: %d frames, task %r, recorded at %g Hz, %s",
        path,
        len(actions),
        episode.task_name,
        episode.control_frequency,
        f"fixed {args.fixed_rate:g} Hz"
        if args.fixed_rate is not None
        else f"recorded timestamps at {args.speed:g}x",
    )

    env = H1RobotEnv(camera_names=None)
    try:
        if args.move_to_start:
            logger.info("driving to the first recorded pose")
            env.move_to_target_joint(actions[0])
            time.sleep(2.0)  # Let the arms settle before the first action.
        replay_actions(
            env,
            actions,
            timestamps,
            speed=args.speed,
            fixed_rate=args.fixed_rate,
        )
    finally:
        if args.deinit:
            env.move_to_deinit_pose()

    logger.info("replay finished")


if __name__ == "__main__":
    main()
