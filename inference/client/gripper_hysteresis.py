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

"""Gripper command hysteresis applied to every action before it reaches the motors."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from inference.client.h1_robot import GRIPPER_INDICES

CLOSE_THRESHOLD = 0.45
OPEN_THRESHOLD = 0.2
CLOSED_VALUE = 1.3
"""Gripper thresholds and the held close command, in H1 gripper units.

The server decodes its 32-D model actions back to the same 23-D environment
space the robot expects, so these calibrated values carry over unchanged.
"""


class GripperHysteresis:
    """Latch a gripper closed until the prediction drops well below the close threshold.

    A prediction above ``close_threshold`` holds ``closed_value``; the gripper
    reopens only after ``open_steps`` consecutive predictions below
    ``open_threshold``. Without the gap and the debounce, a policy hovering at
    the threshold toggles the gripper between chunks, and the RTC replan that
    swaps chunks every ``min_exec_horizon`` steps makes that more likely, not
    less. Each channel keeps its own state.
    """

    def __init__(
        self,
        *,
        channels: Sequence[int] = GRIPPER_INDICES,
        close_threshold: float = CLOSE_THRESHOLD,
        open_threshold: float = OPEN_THRESHOLD,
        closed_value: float = CLOSED_VALUE,
        close_steps: int = 1,
        open_steps: int = 1,
    ) -> None:
        if close_threshold <= open_threshold:
            raise ValueError(
                f"close_threshold must exceed open_threshold, got {close_threshold} "
                f"and {open_threshold}"
            )
        if close_steps < 1 or open_steps < 1:
            raise ValueError(
                f"debounce steps must be at least 1, got close_steps={close_steps} "
                f"and open_steps={open_steps}"
            )
        self.channels = tuple(int(channel) for channel in channels)
        self.close_threshold = float(close_threshold)
        self.open_threshold = float(open_threshold)
        self.closed_value = float(closed_value)
        self.close_steps = int(close_steps)
        self.open_steps = int(open_steps)
        self.reset()

    def reset(self) -> None:
        """Forget the latched state, e.g. when a new episode starts."""
        self._closed = dict.fromkeys(self.channels, False)
        self._close_counts = dict.fromkeys(self.channels, 0)
        self._open_counts = dict.fromkeys(self.channels, 0)

    def apply(self, action: np.ndarray) -> np.ndarray:
        """Rewrite the gripper slots of one 23-D action and return it."""
        action = np.asarray(action)
        if action.ndim != 1 or action.shape[0] <= max(self.channels):
            raise ValueError(
                f"expected a {max(self.channels) + 1}-D action at least, "
                f"got shape {action.shape}"
            )
        for channel in self.channels:
            if self._closed[channel]:
                if action[channel] < self.open_threshold:
                    self._open_counts[channel] += 1
                else:
                    self._open_counts[channel] = 0

                if self._open_counts[channel] >= self.open_steps:
                    self._closed[channel] = False
                    self._open_counts[channel] = 0
                    self._close_counts[channel] = 0
                else:
                    action[channel] = self.closed_value
            else:
                if action[channel] > self.close_threshold:
                    self._close_counts[channel] += 1
                else:
                    self._close_counts[channel] = 0

                if self._close_counts[channel] >= self.close_steps:
                    self._closed[channel] = True
                    self._close_counts[channel] = 0
                    self._open_counts[channel] = 0
                    action[channel] = self.closed_value
        return action
