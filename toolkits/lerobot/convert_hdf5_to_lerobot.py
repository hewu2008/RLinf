#!/usr/bin/env python3
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

"""Convert recorded H1 episodes (hdf5) into a LeRobot dataset.

Every episode is one hdf5 file in the layout ``inference/client`` records and
the mocap collection software uses: the 23-D joint vector split per body part
under ``observation/state`` and ``action``, plus one JPEG colour stream per
camera under ``observation/images/rs/<camera>/color``.

Actions come from the recorded ``action`` group when an episode has one, which
is what the policy actually commanded; episodes without it fall back to the
next observed state, i.e. ``action[t] = state[t + 1]``.

Usage (from the repository root):
    python toolkits/lerobot/convert_hdf5_to_lerobot.py \\
        --raw-dir /path/to/hdf5_root --repo-id <org>/<dataset-name>

The dataset is written to ``$HF_LEROBOT_HOME/<repo-id>`` unless
``--dataset-root`` is given, and a bare repo id resolves to the same directory
in the training configs.
"""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tqdm

JOINT_FIELDS = {
    # part: (dataset path relative to the group, width)
    "arm": ("arm/position", 14),
    "gripper": ("effector/position", 2),
    "waist": ("waist/position", 3),
    "head": ("head/position", 2),
    "base": ("base/velocity", 2),
}
"""Per-part joint datasets; the names match what ``inference/client`` writes."""

STATE_DIM = sum(width for _, width in JOINT_FIELDS.values())
"""Width of the merged joint vector (left arm, left gripper, right arm,
right gripper, waist, head, base)."""

DEFAULT_FPS = 30.0
DEFAULT_ROBOT_TYPE = "mobile_arx"


@dataclass
class Episode:
    """One episode, already merged, resized, and in the target dtypes."""

    states: np.ndarray
    """``(T, 23)`` observed joint vectors as float32."""

    actions: np.ndarray
    """``(T, 23)`` commanded joint vectors as float32."""

    images: dict[str, np.ndarray]
    """Camera name to ``(T, H, W, 3)`` uint8 RGB frames."""

    task: str
    action_source: str
    """``"recorded"`` or ``"next-state"``, for the run summary."""


def read_joint_group(episode: Any, group: str) -> np.ndarray:
    """Read one per-part joint group as a ``(T, 23)`` float64 array.

    The merge order matches ``inference.client.main.load_reference_state`` so a
    converted dataset and a recorded episode index the same joint.
    """
    parts = {}
    for name, (relative, width) in JOINT_FIELDS.items():
        path = f"{group}/{relative}"
        if path not in episode:
            raise KeyError(f"{path} is missing")
        series = np.asarray(episode[path][:], dtype=np.float64)
        if series.ndim != 2 or series.shape[1] != width:
            raise ValueError(f"{path} should be (T, {width}), got {series.shape}")
        parts[name] = series

    lengths = {len(series) for series in parts.values()}
    if len(lengths) != 1:
        raise ValueError(f"{group} parts disagree on frame count: {sorted(lengths)}")

    arm, gripper = parts["arm"], parts["gripper"]
    return np.concatenate(
        [
            arm[:, :7],
            gripper[:, :-1],
            arm[:, 7:],
            gripper[:, -1:],
            parts["waist"],
            parts["head"],
            parts["base"],
        ],
        axis=1,
    )


def camera_root(episode: Any) -> Any:
    """Return the h5 group that holds the per-camera image datasets."""
    for candidate in ("observation/images/rs", "observation/images"):
        if candidate in episode:
            return episode[candidate]
    raise KeyError(
        "no camera group; tried observation/images/rs and observation/images"
    )


def discover_cameras(episode: Any) -> list[str]:
    """List the colour cameras of one episode, skipping depth channels."""
    cameras = sorted(name for name in camera_root(episode) if "depth" not in name)
    if not cameras:
        raise KeyError("the episode has no colour camera")
    return cameras


def camera_dataset(episode: Any, camera: str) -> Any:
    """Return the colour image dataset of *camera*."""
    for candidate in (
        f"observation/images/{camera}/color",
        f"observation/images/rs/{camera}/color",
    ):
        if candidate in episode:
            return episode[candidate]
    raise KeyError(f"camera {camera!r} has no colour images")


def resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize one HWC frame to ``(width, height)``, a no-op when it matches."""
    import cv2

    width, height = size
    if frame.shape[:2] == (height, width):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def decode_frame(payload: np.ndarray, camera: str, index: int) -> np.ndarray:
    """Decode one stored JPEG to an RGB frame.

    The recorder JPEG-encodes the camera's native BGR frame, so decoding has to
    swap the channels back.
    """
    import cv2

    frame = cv2.imdecode(np.asarray(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"camera {camera!r} frame {index} is not a decodable image")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_camera_frames(
    episode: Any, camera: str, size: tuple[int, int], limit: int | None
) -> np.ndarray:
    """Load and resize the colour frames of one camera as ``(T, H, W, 3)``."""
    dataset = camera_dataset(episode, camera)
    if dataset.ndim == 4:  # uncompressed frames: (T, H, W, C)
        rows = dataset[:] if limit is None else dataset[:limit]
        return np.stack([resize_frame(row, size) for row in rows])
    payloads = dataset[:] if limit is None else dataset[:limit]
    frames = [
        resize_frame(decode_frame(payload, camera, index), size)
        for index, payload in enumerate(payloads)
    ]
    return np.stack(frames)


def first_frame_size(episode: Any, camera: str) -> tuple[int, int]:
    """Return the native ``(width, height)`` of a camera without reading a run."""
    dataset = camera_dataset(episode, camera)
    if dataset.ndim == 4:
        return int(dataset.shape[2]), int(dataset.shape[1])
    if "orig_shape" in dataset.attrs:
        height, width = np.asarray(dataset.attrs["orig_shape"])[:2]
        return int(width), int(height)
    frame = decode_frame(dataset[0], camera, 0)
    return int(frame.shape[1]), int(frame.shape[0])


def load_episode(
    path: Path,
    cameras: Sequence[str],
    image_size: tuple[int, int],
    *,
    drop_homing: bool,
    default_task: str,
    expected_fps: float,
) -> Episode:
    """Read one hdf5 episode into arrays ready to be written as frames."""
    import h5py

    with h5py.File(path, "r") as episode:
        states = read_joint_group(episode, "observation/state")
        if f"action/{JOINT_FIELDS['arm'][0]}" in episode:
            actions = read_joint_group(episode, "action")
            if actions.shape != states.shape:
                raise ValueError(
                    f"action {actions.shape} and state {states.shape} disagree on the "
                    f"number of frames in {path}"
                )
            action_source = "recorded"
        else:
            actions = np.concatenate([states[1:], states[-1:]], axis=0)
            action_source = "next-state"

        keep = len(states)
        if drop_homing and "homing_start_frame" in episode.attrs:
            keep = min(keep, int(episode.attrs["homing_start_frame"]))
            if keep <= 0:
                raise ValueError(
                    f"the automatic return-home segment starts at frame "
                    f"{int(episode.attrs['homing_start_frame'])}, so the episode has "
                    f"no usable frames"
                )

        task = str(episode.attrs.get("task_name", default_task))
        fps = float(episode.attrs.get("control_frequency", expected_fps))
        if abs(fps - expected_fps) > 0.5:
            raise ValueError(
                f"episode was recorded at {fps:g} Hz but the dataset is "
                f"{expected_fps:g} Hz"
            )

        images = {
            camera: load_camera_frames(
                episode, camera, image_size, None if keep == len(states) else keep
            )
            for camera in cameras
        }
        for camera, frames in images.items():
            if len(frames) != keep:
                raise ValueError(
                    f"camera {camera!r} has {len(frames)} frames but the state has "
                    f"{keep}"
                )

    return Episode(
        states=states[:keep].astype(np.float32),
        actions=actions[:keep].astype(np.float32),
        images=images,
        task=task,
        action_source=action_source,
    )


def probe_source(
    files: Sequence[Path],
    camera_override: Sequence[str] | None,
    size_override: tuple[int, int] | None,
) -> tuple[list[str], tuple[int, int], float]:
    """Derive the dataset schema from the first readable episode.

    Every episode in one dataset shares the camera set, frame size, and fps;
    taking them from the data avoids hard-coding a camera list or resolution
    that a different robot or recorder revision does not use.
    """
    import h5py

    for path in files:
        try:
            with h5py.File(path, "r") as episode:
                cameras = (
                    list(camera_override)
                    if camera_override
                    else discover_cameras(episode)
                )
                size = size_override or first_frame_size(episode, cameras[0])
                fps = float(episode.attrs.get("control_frequency", DEFAULT_FPS))
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            print(f"[WARN] Cannot use {path} as the schema source: {error}")
            continue
        return cameras, size, fps
    raise RuntimeError(
        f"none of the {len(files)} episodes under {files[0].parent} could be read"
    )


def build_features(
    cameras: Sequence[str], image_size: tuple[int, int]
) -> dict[str, dict[str, Any]]:
    """Describe the dataset columns for ``LeRobotDataset.create``."""
    width, height = image_size
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["states"],
        },
        "action": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["actions"],
        },
    }
    for camera in cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }
    return features


def import_lerobot_dataset() -> Any:
    """Import ``LeRobotDataset`` from either lerobot layout RLinf supports."""
    try:  # lerobot >= 0.2
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        try:  # lerobot < 0.2
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ModuleNotFoundError as error:
            raise SystemExit(
                "lerobot is not installed in this environment; install it before "
                "converting"
            ) from error
    return LeRobotDataset


def create_dataset(
    repo_id: str,
    root: Path,
    fps: float,
    robot_type: str,
    features: dict[str, dict[str, Any]],
    *,
    image_writer_processes: int,
    image_writer_threads: int,
) -> Any:
    """Create an empty dataset, with the image writer the release supports."""
    LeRobotDataset = import_lerobot_dataset()
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "root": root,
        "fps": int(round(fps)),
        "robot_type": robot_type,
        "features": features,
    }
    writer_in_create = (
        "image_writer_processes" in inspect.signature(LeRobotDataset.create).parameters
    )
    if writer_in_create:
        kwargs["image_writer_processes"] = image_writer_processes
        kwargs["image_writer_threads"] = image_writer_threads
    dataset = LeRobotDataset.create(**kwargs)
    if not writer_in_create:
        start = getattr(dataset, "start_image_writer", None)
        if start is None:
            raise SystemExit(
                f"{type(dataset).__name__} exposes neither the image writer "
                "arguments on create() nor start_image_writer()"
            )
        start(num_processes=image_writer_processes, num_threads=image_writer_threads)
    return dataset


def add_episode_frames(dataset: Any, episode: Episode) -> None:
    """Append every frame of *episode* and close the episode.

    lerobot 0.3.x takes the task as an ``add_frame`` argument and rejects it as
    a frame field; the older and newer releases read it from the frame. Which
    one is installed is decided from the signature, as in
    ``rlinf.data.storage.lerobot.compat``.
    """
    frame_takes_task = "task" in inspect.signature(type(dataset).add_frame).parameters
    for step in range(len(episode.states)):
        frame = {
            "observation.state": episode.states[step],
            "action": episode.actions[step],
        }
        for camera, frames in episode.images.items():
            frame[f"observation.images.{camera}"] = frames[step]
        if frame_takes_task:
            dataset.add_frame(frame, task=episode.task)
        else:
            dataset.add_frame({**frame, "task": episode.task})

    if "task" in inspect.signature(dataset.save_episode).parameters:
        dataset.save_episode(task=episode.task)
    else:
        dataset.save_episode()


def finalize_dataset(dataset: Any, *, compute_stats: bool) -> None:
    """Close the image writer and write the dataset metadata."""
    consolidate = getattr(dataset, "consolidate", None)
    if consolidate is None:
        # lerobot >= 0.3 accumulates stats per episode in save_episode().
        if compute_stats:
            print(
                "[WARN] --compute-stats is ignored: this lerobot computes "
                "statistics while saving each episode"
            )
        stop = getattr(dataset, "stop_image_writer", None)
        if stop is not None:
            stop()
        finalize = getattr(dataset, "finalize", None)
        if finalize is not None:
            finalize()
        return
    consolidate(run_compute_stats=compute_stats)


def write_error_log(
    path: Path | None, skipped: Sequence[tuple[int, Path, str]]
) -> Path:
    """Write one line per skipped episode and return the log path."""
    if path is None:
        path = Path("logs") / f"convert_skipped_{datetime.now():%Y%m%d_%H%M%S}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Skipped files during conversion\n")
        for index, file, message in skipped:
            handle.write(f"episode_{index}\t{file}\t{message}\n")
    return path


def find_hdf5_files(raw_dir: Path) -> list[Path]:
    """Collect all ``*.hdf5`` episode files under *raw_dir*, sorted."""
    files = sorted(raw_dir.rglob("*.hdf5"))
    print(f"[INFO] Found {len(files)} hdf5 files under {raw_dir}")
    return files


def resolve_dataset_root(override: Path | None) -> Path:
    """Resolve the LeRobot home the dataset is written below."""
    if override is not None:
        return override.expanduser()
    for variable in ("HF_LEROBOT_HOME", "LEROBOT_HOME"):
        value = os.getenv(variable)
        if value:
            return Path(value).expanduser()
    hf_home = os.getenv("HF_HOME")
    base = Path(hf_home).expanduser() if hf_home else Path.home() / ".cache/huggingface"
    return base / "lerobot"


def parse_image_size(value: str) -> tuple[int, int]:
    """Parse a ``WIDTHxHEIGHT`` override such as ``640x480``."""
    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected WIDTHxHEIGHT such as 640x480, got {value!r}"
        ) from error
    return width, height


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Convert recorded H1 hdf5 episodes into a LeRobot dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        "--raw_dir",
        dest="raw_dir",
        type=Path,
        required=True,
        help="Directory searched recursively for *.hdf5 episode files.",
    )
    parser.add_argument(
        "--repo-id",
        "--repo_id",
        dest="repo_id",
        required=True,
        help="Dataset name below the LeRobot home, e.g. org/part_sorting_20261027.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="LeRobot home; defaults to $HF_LEROBOT_HOME.",
    )
    parser.add_argument("--robot-type", default=DEFAULT_ROBOT_TYPE)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override the control frequency stored in the episodes.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Camera names in dataset order; defaults to the episode's own set.",
    )
    parser.add_argument(
        "--image-size",
        type=parse_image_size,
        default=None,
        help="Frame size as WIDTHxHEIGHT; defaults to the first camera's size.",
    )
    parser.add_argument(
        "--task",
        default="",
        help="Task instruction for episodes that carry no task_name attribute.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Episode indices into the sorted file list; defaults to all.",
    )
    parser.add_argument(
        "--drop-homing",
        action="store_true",
        help="Drop the automatic return-home tail an episode marks with "
        "homing_start_frame.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rewrite an existing dataset directory.",
    )
    parser.add_argument(
        "--compute-stats",
        action="store_true",
        help="Let LeRobot compute normalization statistics while closing the "
        "dataset; norm stats are otherwise produced by calculate_norm_stats.py.",
    )
    parser.add_argument("--image-writer-processes", type=int, default=10)
    parser.add_argument("--image-writer-threads", type=int, default=5)
    parser.add_argument(
        "--error-log",
        type=Path,
        default=None,
        help="Where to list skipped episodes; defaults to logs/convert_skipped_*.log.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the conversion and print where the dataset landed."""
    args = parse_args(argv)

    files = find_hdf5_files(args.raw_dir)
    if not files:
        raise SystemExit(f"no .hdf5 files found under {args.raw_dir}")

    indexes = list(range(len(files))) if args.episodes is None else args.episodes
    for index in indexes:
        if not 0 <= index < len(files):
            raise SystemExit(
                f"episode index {index} is outside the {len(files)} files found "
                f"under {args.raw_dir}"
            )

    root = resolve_dataset_root(args.dataset_root) / args.repo_id
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} already exists; pass --overwrite to replace it")
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)

    cameras, image_size, fps = probe_source(files, args.cameras, args.image_size)
    if args.fps is not None:
        fps = args.fps
    print(
        f"[INFO] Cameras: {', '.join(cameras)}; frames {image_size[0]}x"
        f"{image_size[1]}; fps {int(round(fps))}"
    )

    dataset = create_dataset(
        args.repo_id,
        root,
        fps,
        args.robot_type,
        build_features(cameras, image_size),
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    skipped: list[tuple[int, Path, str]] = []
    sources: Counter[str] = Counter()
    frames_written = 0
    for index in tqdm.tqdm(indexes, desc="episodes", unit="ep"):
        path = files[index]
        try:
            episode = load_episode(
                path,
                cameras,
                image_size,
                drop_homing=args.drop_homing,
                default_task=args.task,
                expected_fps=fps,
            )
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            skipped.append((index, path, f"{type(error).__name__}: {error}"))
            print(f"[WARN] Skip episode_{index} ({path}): {error}")
            continue
        add_episode_frames(dataset, episode)
        sources[episode.action_source] += 1
        frames_written += len(episode.states)

    finalize_dataset(dataset, compute_stats=args.compute_stats)

    converted = sum(sources.values())
    if converted == 0:
        # An empty dataset would still look loadable to a training config.
        shutil.rmtree(root, ignore_errors=True)
        error_log = write_error_log(args.error_log, skipped)
        print(f"[ERROR] Every episode was skipped; list at {error_log}")
        return 1
    print(
        f"[INFO] Wrote {converted} episodes / {frames_written} frames to {root} "
        f"(actions: {dict(sources)})"
    )

    if skipped:
        error_log = write_error_log(args.error_log, skipped)
        print(f"[WARN] Skipped {len(skipped)} episodes; list at {error_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
