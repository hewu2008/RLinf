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

"""Fixed-frequency control loop of the robot-side client."""

from __future__ import annotations

import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)


def run_control_loop(
    step_fn: Callable[[int], bool | None],
    *,
    frequency: float,
    num_steps: int,
) -> None:
    """Drive ``step_fn(step)`` on an absolute deadline schedule.

    A truthy return means the step ran a blocking phase (e.g. driving the arms
    to a pose) that consumed the schedule; the deadline is then rebased instead
    of bursting through the missed periods.
    """
    if frequency <= 0:
        raise ValueError(f"control frequency must be positive, got {frequency}")
    period = 1.0 / float(frequency)
    next_deadline = time.monotonic() + period
    for step in range(int(num_steps)):
        if step_fn(step):
            next_deadline = time.monotonic() + period
            continue
        remaining = next_deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        else:
            logger.warning("control loop overrun: %.1f ms late", -remaining * 1000)
        next_deadline += period
