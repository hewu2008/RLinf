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

"""Client-side Real-Time Chunking (RTC) scheduling.

The overlap reference lives on the server (it needs the chunk in model action
space), so the client only reports how much of the running chunk has been
executed and how long inference is currently taking. The bookkeeping here is the
robot-side counterpart of ``rlinf/workers/env/rtc_env_worker.py``: a chunk
arriving after ``d`` control steps replaces the queue with its remaining
``H - d`` actions, and the next request goes out once the queue is dry or
``min_exec_horizon`` steps have been executed.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Mapping

import numpy as np

from inference.protocol import RTC_PROTOCOL_VERSION

logger = logging.getLogger(__name__)


class LatencyTracker:
    """Sliding-window of measured inference delays, in control steps."""

    def __init__(self, maxlen: int = 8) -> None:
        self._values: deque[int] = deque(maxlen=maxlen)

    def reset(self) -> None:
        """Clear all recorded delays."""
        self._values.clear()

    def add(self, latency: int) -> None:
        """Record one measured delay; negative values are ignored."""
        value = int(latency)
        if value >= 0:
            self._values.append(value)

    def max(self) -> int | None:
        """Window maximum, or None when empty."""
        return max(self._values) if self._values else None

    def __len__(self) -> int:
        return len(self._values)


class ActionQueue:
    """The chunk currently being executed, one action per control step."""

    def __init__(self) -> None:
        self._actions: np.ndarray | None = None
        self._index = 0
        self._offset = 0

    def put(self, actions: np.ndarray, delay: int) -> None:
        """Replace the queue with ``actions``, skipping the stale prefix.

        The first ``delay`` actions describe control steps that already elapsed
        while the chunk was being computed, so executing them would drive the
        robot backwards. ``offset`` remembers how many were dropped: the server
        needs the count from the start of the chunk, not from the skip point, to
        know which part of its previous chunk is still unexecuted.
        """
        actions = np.asarray(actions, dtype=np.float32)
        skipped = max(0, min(int(delay), len(actions)))
        self._actions = actions[skipped:]
        self._index = 0
        self._offset = skipped
        if len(self._actions) == 0:
            logger.warning(
                "new chunk skipped entirely: delay=%d horizon=%d", delay, len(actions)
            )

    def clear(self) -> None:
        """Drop the current chunk."""
        self._actions = None
        self._index = 0
        self._offset = 0

    def get(self) -> np.ndarray | None:
        """Return the next action as a copy, or None when the chunk is exhausted."""
        if self._actions is None or self._index >= len(self._actions):
            return None
        action = np.array(self._actions[self._index])
        self._index += 1
        return action

    def empty(self) -> bool:
        """True when no action remains."""
        return self._actions is None or self._index >= len(self._actions)

    @property
    def position(self) -> int:
        """Control steps of the current chunk already executed."""
        return self._offset + self._index


class RTCScheduler:
    """Decide when to request a chunk and which action to execute next."""

    def __init__(
        self,
        worker,
        *,
        action_horizon: int,
        min_exec_horizon: int = 15,
        delay_buffer_size: int = 8,
        initial_delay_steps: int = 1,
        rtc_enabled: bool = True,
    ) -> None:
        self.worker = worker
        self.action_horizon = int(action_horizon)
        self.min_exec_horizon = int(min_exec_horizon)
        self.initial_delay_steps = int(initial_delay_steps)
        self.rtc_enabled = bool(rtc_enabled)
        self.queue = ActionQueue()
        self.tracker = LatencyTracker(maxlen=delay_buffer_size)
        self._t = 0  # control-step counter
        self._steps_since_swap = 0
        self._last_merged_query_t = -1
        self._chunk_id = 0

    def reset(self) -> None:
        """Reset all per-episode state."""
        self.queue.clear()
        self.tracker.reset()
        self._t = 0
        self._steps_since_swap = 0
        self._last_merged_query_t = -1
        self._chunk_id = 0

    def submit_bootstrap(self, observation: Mapping[str, Any]) -> bool:
        """Request the first chunk of an episode without guidance."""
        return self._maybe_submit(observation, force=True)

    def get_action(self, observation: Mapping[str, Any]) -> np.ndarray | None:
        """Per-step driver: merge results, maybe replan, advance one action."""
        self._consume_results()
        self._maybe_submit(observation)
        action = self.queue.get()
        self._t += 1
        self._steps_since_swap += 1
        return action

    def observe_chunk(self, query_t: int, actions: np.ndarray) -> int | None:
        """Merge a chunk returned for submission step ``query_t``.

        Returns the measured delay in control steps, or None when the chunk is
        stale (out of order, or older than one horizon) and was dropped.
        """
        query_t = int(query_t)
        real_delay = self._t - query_t
        if query_t <= self._last_merged_query_t:
            logger.warning(
                "dropping an out-of-order chunk (query_t=%d <= %d)",
                query_t,
                self._last_merged_query_t,
            )
            return None
        if real_delay < 0 or real_delay > self.action_horizon:
            logger.warning(
                "dropping a stale chunk (query_t=%d, delay=%d > horizon=%d)",
                query_t,
                real_delay,
                self.action_horizon,
            )
            return None

        self.queue.put(actions, real_delay)
        self._last_merged_query_t = query_t
        self.tracker.add(real_delay)
        self._steps_since_swap = real_delay
        return real_delay

    def _consume_results(self) -> None:
        while True:
            result = self.worker.poll()
            if result is None:
                return
            self.observe_chunk(*result)

    def _trigger(self) -> bool:
        if not self.rtc_enabled:
            return self.queue.empty()
        return self.queue.empty() or self._steps_since_swap >= self.min_exec_horizon

    def _maybe_submit(
        self, observation: Mapping[str, Any], *, force: bool = False
    ) -> bool:
        if not force:
            if self.worker is None or self.worker.in_flight:
                return False
            if not self._trigger():
                return False

        predicted_delay = self.tracker.max()
        if predicted_delay is None:
            predicted_delay = self.initial_delay_steps
        self._chunk_id += 1

        request: dict[str, Any] = dict(observation)
        if self.rtc_enabled:
            request["rtc"] = {
                "version": RTC_PROTOCOL_VERSION,
                "request_type": "bootstrap" if force else "replan",
                "executed_horizon": self.queue.position,
                "predicted_delay_steps": int(predicted_delay),
                "chunk_id": self._chunk_id,
            }
        self.worker.submit(request, query_t=self._t)
        self._steps_since_swap = 0
        logger.info(
            "requested chunk %d at step %d (executed=%d predicted_delay=%d)",
            self._chunk_id,
            self._t,
            self.queue.position,
            predicted_delay,
        )
        return True
