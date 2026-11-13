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

"""Record one rollout episode in the layout of the mocap hdf5 dataset."""

from __future__ import annotations

import json
import logging
import pathlib
import time
import uuid
from typing import Mapping, Sequence

import numpy as np

from inference.client.h1_robot import (
    ARM_JOINT_SLICES,
    BASE_SLICE,
    GRIPPER_INDICES,
    HEAD_SLICE,
    STATE_DIM,
    WAIST_SLICE,
)

logger = logging.getLogger(__name__)

FIELD_PATHS = {
    "arm": "arm/position",
    "gripper": "effector/position",
    "waist": "waist/position",
    "head": "head/position",
    "base": "base/velocity",
}
"""Per-part dataset paths under ``observation/state`` and ``action``.

The dataset calls the gripper joints ``effector``; only that field name keeps the
old spelling because it is part of the file layout.
"""


def split_joint_batch(vectors: np.ndarray) -> dict[str, np.ndarray]:
    """Split ``(T, 23)`` states or actions into the dataset's per-part arrays."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != STATE_DIM:
        raise ValueError(f"expected a (T, {STATE_DIM}) array, got {vectors.shape}")
    return {
        "arm": np.concatenate([vectors[:, arm] for arm in ARM_JOINT_SLICES], axis=1),
        "gripper": vectors[:, list(GRIPPER_INDICES)],
        "waist": vectors[:, WAIST_SLICE],
        "head": vectors[:, HEAD_SLICE],
        "base": vectors[:, BASE_SLICE],
    }


class EpisodeRecorder:
    """Buffer one episode in memory and write it when the operator ends it.

    Camera frames are JPEG-encoded from the camera's native BGR frame, which is
    what the recorded dataset stores; the RGB-ordered JPEGs
    :func:`inference.protocol.encode_frame` puts on the wire are not reused
    because a reader that decodes these files would see swapped channels.
    """

    def __init__(
        self,
        *,
        camera_names: Sequence[str],
        task_name: str = "rollout",
        task_id: int = 0,
        control_frequency: float = 30.0,
        jpeg_quality: int = 60,
    ) -> None:
        if not camera_names:
            raise ValueError("recording needs at least one camera")
        self.camera_names = tuple(camera_names)
        self.task_name = str(task_name)
        self.task_id = int(task_id)
        self.control_frequency = float(control_frequency)
        self.jpeg_quality = int(jpeg_quality)
        self.reset()

    def reset(self) -> None:
        """Drop the buffered episode."""
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._frames: list[dict[str, bytes]] = []
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._homing_start: int | None = None

    def __len__(self) -> int:
        return len(self._states)

    def mark_homing(self) -> None:
        """Mark where the automatic return-home segment starts."""
        self._homing_start = len(self._states)

    def add_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        images: Mapping[str, np.ndarray],
        *,
        timestamp_ms: float | None = None,
    ) -> None:
        """Append one control step: the observed state, commanded action, and frames."""
        state = np.asarray(state, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        if state.shape != (STATE_DIM,) or action.shape != (STATE_DIM,):
            raise ValueError(
                f"expected ({STATE_DIM},) state and action, got {state.shape} and "
                f"{action.shape}"
            )
        frames = {}
        for camera in self.camera_names:
            image = images.get(camera)
            if image is None:
                raise KeyError(
                    f"camera {camera!r} has no frame to record; got {sorted(images)}"
                )
            frames[camera] = self._encode(camera, image)
        self._states.append(state)
        self._actions.append(action)
        self._timestamps.append(
            time.time() * 1000.0 if timestamp_ms is None else float(timestamp_ms)
        )
        self._frames.append(frames)

    def save(self, save_dir: str | pathlib.Path) -> pathlib.Path | None:
        """Write ``<save_dir>/<episode_id>/episode.hdf5`` and its metadata."""
        import h5py

        count = len(self._states)
        if count == 0:
            logger.warning("nothing recorded: no episode file written")
            return None

        episode_id = uuid.uuid4().hex
        episode_dir = pathlib.Path(save_dir).expanduser() / episode_id
        episode_dir.mkdir(parents=True, exist_ok=False)
        path = episode_dir / "episode.hdf5"

        states = np.stack(self._states)
        actions = np.stack(self._actions)
        timestamps = np.asarray(self._timestamps, dtype=np.float64)
        chunk_rows = min(10, count)

        with h5py.File(path, "w") as episode:
            episode.attrs["action_mode"] = "absolute"
            episode.attrs["completed_subtasks"] = 1
            episode.attrs["control_frequency"] = int(round(self.control_frequency))
            episode.attrs["episode_id"] = episode_id
            episode.attrs["task_id"] = self.task_id
            episode.attrs["task_name"] = self.task_name
            episode.attrs["total_frames"] = count
            episode.attrs["total_subtasks"] = 1
            episode.attrs["version"] = "1.0.0"
            if self._homing_start is not None:
                episode.attrs["homing_start_frame"] = int(self._homing_start)

            for group, vectors in (("observation/state", states), ("action", actions)):
                for field, series in split_joint_batch(vectors).items():
                    episode.create_dataset(
                        f"{group}/{FIELD_PATHS[field]}",
                        data=series,
                        chunks=(chunk_rows, series.shape[1]),
                    )

            for camera in self.camera_names:
                dataset = episode.create_dataset(
                    f"observation/images/rs/{camera}/color",
                    shape=(count,),
                    dtype=h5py.special_dtype(vlen=np.uint8),
                    chunks=(chunk_rows,),
                )
                for index, frames in enumerate(self._frames):
                    dataset[index] = np.frombuffer(frames[camera], dtype=np.uint8)
                dataset.attrs["encoded_format"] = "jpeg"
                dataset.attrs["jpeg_quality"] = self.jpeg_quality
                dataset.attrs["orig_dtype"] = "uint8"
                dataset.attrs["orig_shape"] = np.asarray(
                    self._shapes[camera], dtype=np.int64
                )

            stamps = episode.create_dataset(
                "timestamp/t", data=timestamps, chunks=(chunk_rows,)
            )
            stamps.attrs["rate_hz"] = self.control_frequency
            stamps.attrs["source"] = f"rs/{self.camera_names[0]}"

            episode.create_dataset(
                "subtask_transitions", data=np.asarray([count], dtype=np.int32)
            )

        _write_metadata(
            episode_dir / "episode_meta.json",
            episode_id=episode_id,
            task_id=self.task_id,
            timestamps=timestamps,
        )
        logger.info(
            "wrote %s: %d steps, %.0f MB", path, count, path.stat().st_size / 1e6
        )
        return path

    def _encode(self, camera: str, image: np.ndarray) -> bytes:
        import cv2

        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"expected an HxWx3 BGR frame for {camera!r}, got shape {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(
                f"expected a uint8 frame for {camera!r}, got {frame.dtype}"
            )
        encoded, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not encoded:
            raise RuntimeError(
                f"JPEG encoding failed for camera {camera!r} of shape {frame.shape}"
            )
        self._shapes.setdefault(camera, tuple(frame.shape))
        return buffer.tobytes()


def _write_metadata(
    path: pathlib.Path, *, episode_id: str, task_id: int, timestamps: np.ndarray
) -> None:
    metadata = {
        "episode_id": episode_id,
        "task_id": task_id,
        "robot_id": "",
        "collector_id": 0,
        "collector_name": "",
        "collection_ts": int(time.time() * 1000),
        "step_index": [
            {
                "step_number": 1,
                "start_ts": int(timestamps[0]),
                "end_ts": int(timestamps[-1]),
                "start_frame_id": 0,
                "end_frame_id": int(len(timestamps) - 1),
            }
        ],
    }
    path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")
