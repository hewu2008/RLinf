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

"""H1 Pro dual-arm robot access for the inference client.

The vendor SDK (``lib_h1_sdk_python`` plus ``camera_client``) ships as a
directory of shared objects and protobuf stubs rather than an installed package.
This module puts that directory on ``sys.path`` and imports the SDK lazily, so
the client still imports on a machine without it.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DEFAULT_GRPC_TARGET = "localhost:50051"

SDK_DIR_ENV_VAR = "H1_SDK_DIR"
CAMERA_SOURCE_PREFIXES = ("", "rs/", "rs.", "rs_")

ARM_MOTOR_COUNT = 8  # Seven joints and one gripper per arm.
WAIST_MOTOR_COUNT = 3  # Lift, pitch, yaw.
HEAD_MOTOR_COUNT = 2
STATE_DIM = (
    23  # The joint vector plus two base-velocity slots the robot does not command.
)

FRAME_POLL_INTERVAL_SECONDS = 0.01


def camera_source_candidates(camera_name: str) -> tuple[str, ...]:
    """Names the camera service may publish one logical camera under.

    Devices whose streams come from RealSense cameras are registered with an
    ``rs`` prefix, so try the plain name and every prefixed form.
    """
    return tuple(f"{prefix}{camera_name}" for prefix in CAMERA_SOURCE_PREFIXES)


def motor_position(state: Any, motor_index: int, motor_information_cls: type) -> float:
    """Read ``Position_Actual`` out of one ``getMotorState`` reply.

    The SDK answers with ``(ok, Motor_Information)``, where ``ok`` is the C++
    side's success flag; a failed read reports no position at all.
    """
    if isinstance(state, (list, tuple)) and len(state) == 2:
        ok, state = bool(state[0]), state[1]
        if not ok:
            raise RuntimeError(f"getMotorState({motor_index}) reported a failed read")
    if isinstance(state, motor_information_cls):
        return float(state.Position_Actual)
    raise TypeError(
        f"getMotorState({motor_index}) returned {type(state)!r}; "
        "expected Motor_Information"
    )


def _ensure_sdk_on_path(sdk_dir: str | None) -> None:
    """Make the vendor SDK importable.

    ``sdk_dir`` wins, then ``$H1_SDK_DIR``, then the ``client/lib`` directory the
    robot machine is expected to carry.
    """
    candidates = []
    if sdk_dir:
        candidates.append(Path(sdk_dir).expanduser())
    if os.environ.get(SDK_DIR_ENV_VAR):
        candidates.append(Path(os.environ[SDK_DIR_ENV_VAR]).expanduser())
    candidates.append(Path(__file__).resolve().parent / "lib")
    for root in candidates:
        if not root.is_dir():
            continue
        for path in (root / "proto", root):
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))


class CameraRecorder:
    """Keep the latest frame of every camera in a background thread.

    ``CameraClient.get_latest_frame`` returns ``(frame, timestamp)`` in the
    camera's native BGR order, which is what :func:`inference.protocol.encode_frame`
    expects. The camera service does not report which name each stream is
    published under, so one camera resolves its source on the first frame it
    receives and keeps using that name.
    """

    def __init__(
        self,
        camera_client_cls,
        grpc_target: str,
        camera_names: Sequence[str],
        *,
        poll_interval: float = FRAME_POLL_INTERVAL_SECONDS,
    ) -> None:
        self.camera_names = tuple(camera_names)
        self._poll_interval = float(poll_interval)
        self._frames: dict[str, np.ndarray] = {}
        self._sources: dict[str, str] = {}
        self._lock = threading.Lock()
        self._client = camera_client_cls(grpc_target=grpc_target)
        self._client.start()
        for camera_name in self.camera_names:
            threading.Thread(
                target=self._record,
                args=(camera_name,),
                daemon=True,
                name=f"camera-{camera_name}",
            ).start()

    def _record(self, camera_name: str) -> None:
        while True:
            frame = self._read_frame(camera_name)
            if frame is None:
                time.sleep(self._poll_interval)
                continue
            image, _ = frame
            with self._lock:
                self._frames[camera_name] = image

    def _read_frame(self, camera_name: str):
        """Poll one camera, trying the source aliases until one answers."""
        resolved = self._sources.get(camera_name)
        candidates = (resolved,) if resolved else camera_source_candidates(camera_name)
        for source in candidates:
            try:
                frame = self._client.get_latest_frame(source)
            except (
                Exception
            ) as error:  # The camera service drops frames while it reconnects.
                logger.warning("reading camera %s failed: %s", source, error)
                continue
            if frame is not None:
                self._sources[camera_name] = source
                return frame
        return None

    def latest(self) -> dict[str, np.ndarray]:
        """Snapshot of the newest frame per camera."""
        with self._lock:
            return dict(self._frames)


class H1RobotEnv:
    """Joint-position control and observations for the H1 Pro dual arm.

    The 23-D state layout is ``[left arm 0-6, left gripper 7, right arm 8-14,
    right gripper 15, waist 16-18, head yaw/pitch 19-20, base velocity 21-22]``;
    the base-velocity slots stay zero because the robot reports no wheel speed.
    """

    def __init__(
        self,
        *,
        camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
        sdk_dir: str | None = None,
        grpc_target: str = DEFAULT_GRPC_TARGET,
    ) -> None:
        _ensure_sdk_on_path(sdk_dir)
        from camera_client import CameraClient
        from lib_h1_sdk_python import (
            EtherCAT_Motor_Index,
            H1Robot,
            Motor_Control,
            Motor_Information,
            MotorControlMode,
        )

        self._motor_control_cls = Motor_Control
        self._motor_information_cls = Motor_Information
        index = EtherCAT_Motor_Index
        self._arm_motors = [
            getattr(index, f"MOTOR_LEFT_ARM_{joint + 1}")
            for joint in range(ARM_MOTOR_COUNT)
        ] + [
            getattr(index, f"MOTOR_RIGHT_ARM_{joint + 1}")
            for joint in range(ARM_MOTOR_COUNT)
        ]
        self._waist_motors = [
            index.MOTOR_LIFT,
            index.MOTOR_WAIST_DOWN,
            index.MOTOR_WAIST_UP,
        ]
        self._head_motors = [index.MOTOR_HEAD_DOWN, index.MOTOR_HEAD_UP]
        self._motors = self._arm_motors + self._waist_motors + self._head_motors
        joint_count = 2 * ARM_MOTOR_COUNT + WAIST_MOTOR_COUNT + HEAD_MOTOR_COUNT
        if len(self._motors) != joint_count:
            raise RuntimeError(
                f"the H1 SDK exposes {len(self._motors)} motor indices, "
                f"expected {joint_count}"
            )
        self._lift_index = self._motors.index(index.MOTOR_LIFT)

        self.camera_names = tuple(camera_names)
        self._robot = H1Robot()
        self._connect(MotorControlMode)
        self._cameras = CameraRecorder(CameraClient, grpc_target, self.camera_names)

    @property
    def motor_count(self) -> int:
        """Number of commanded joints (23-D state minus the two base slots)."""
        return len(self._motors)

    def set_joint_action(self, action: np.ndarray) -> None:
        """Command absolute joint positions."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.motor_count,):
            raise ValueError(
                f"expected {self.motor_count} joint targets, got shape {action.shape}"
            )
        for index, position in zip(self._motors, action, strict=True):
            self._send_motor(index, position)

    def get_joint_position(self) -> np.ndarray:
        """Read every commanded joint position."""
        positions = np.zeros(self.motor_count)
        for slot, index in enumerate(self._motors):
            positions[slot] = self._read_motor_position(index)
        return positions

    def get_observation(self) -> dict[str, Any]:
        """Return ``{"qpos": float64[23], "images": {camera: BGR frame}}``."""
        qpos = np.zeros(STATE_DIM)
        qpos[: self.motor_count] = self.get_joint_position()
        images = self._cameras.latest()
        missing = [name for name in self.camera_names if name not in images]
        if missing:
            raise RuntimeError(
                f"cameras {missing} have not produced a frame yet; "
                f"expected {list(self.camera_names)}"
            )
        return {"qpos": qpos, "images": images}

    def move_to_init_pose(self) -> None:
        """Drive every joint to zero, keeping the current lift height."""
        target = np.zeros(self.motor_count)
        current = self.get_joint_position()
        target[self._lift_index] = current[self._lift_index]
        self._interpolate_to(target, label="the initial pose")

    def move_to_target_joint(self, action: np.ndarray) -> None:
        """Interpolate to a joint target given as 23-D state or 21-D joints."""
        target = np.asarray(action, dtype=np.float64)
        if target.shape == (STATE_DIM,):
            target = target[: self.motor_count]
        if target.shape != (self.motor_count,):
            raise ValueError(
                f"expected {self.motor_count} joint targets or a {STATE_DIM}-D "
                f"state, got shape {np.asarray(action).shape}"
            )
        self._interpolate_to(target, label="the target pose")

    def move_to_deinit_pose(self) -> None:
        """Release the robot."""
        if not self._robot.robot_deinit():
            raise RuntimeError("the H1 robot rejected robot_deinit()")
        logger.info("robot deinitialized")

    def _connect(
        self,
        motor_control_mode,
        *,
        retries: int = 10,
        retry_interval: float = 1.0,
    ) -> None:
        for attempt in range(retries):
            if self._robot.isRobotConnected():
                break
            try:
                self._robot.robot_connect()
            except Exception as error:  # The vendor SDK raises plain exceptions.
                logger.warning(
                    "robot connect failed (%d/%d): %s", attempt + 1, retries, error
                )
                time.sleep(retry_interval)
        if not self._robot.isRobotConnected():
            raise RuntimeError(
                f"could not connect to the H1 robot after {retries} attempts"
            )
        if not self._robot.switchControlMode(
            motor_control_mode.GRAVITY_COMPENSATION_LEVEL
        ):
            raise RuntimeError(
                "the H1 robot rejected switching to gravity-compensation mode"
            )
        if not self._robot.robot_init():
            raise RuntimeError("the H1 robot rejected robot_init()")
        logger.info("robot connected, control mode %s", self._robot.getCurrentMode())

    def _send_motor(self, index: int, position: float) -> None:
        command = self._motor_control_cls()
        command.Position = float(position)
        self._robot.setMotorControl_low(index, command)

    def _read_motor_position(self, index: int) -> float:
        state = self._robot.getMotorState(index)
        return motor_position(state, index, self._motor_information_cls)

    def _interpolate_to(
        self,
        target: np.ndarray,
        *,
        label: str,
        move_time: float = 2.0,
        steps: int = 1000,
    ) -> None:
        current = self.get_joint_position()
        for step in range(steps):
            alpha = (step + 1) / steps
            self.set_joint_action((1.0 - alpha) * current + alpha * target)
            time.sleep(move_time / steps)
        logger.info("reached %s in %.1fs", label, move_time)
