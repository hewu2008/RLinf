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

"""Background inference thread for the robot-side client."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

_IDLE_POLL_INTERVAL = 0.001


class InferenceWorker(threading.Thread):
    """Run policy inference off the control thread.

    Observations go through a latest-wins slot: a submission overwrites one the
    worker has not picked up yet, which applies back-pressure when inference is
    slower than the request rate instead of queueing stale frames.
    """

    def __init__(self, client) -> None:
        super().__init__(daemon=True, name="inference-worker")
        self._client = client
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pending_obs = None  # (query_t, observation) awaiting inference
        self._result = None  # latest unconsumed (query_t, actions)
        self._last_result_time = None  # monotonic time of the last successful infer
        self._in_flight = False  # True while a submitted observation is being inferred
        self._consecutive_failures = 0
        self._infer_times = []
        self._max_infer_time = 0.0

    @property
    def in_flight(self) -> bool:
        """True while a submitted observation is being inferred.

        Set when a pending observation is picked up and cleared on both the
        success and the failure path, so a scheduler gating submissions on this
        flag cannot deadlock after a failed request.
        """
        with self._lock:
            return self._in_flight

    def submit(self, observation, query_t: int) -> None:
        """Hand an observation to the worker, replacing any unsent one."""
        with self._lock:
            self._pending_obs = (query_t, observation)

    def poll(self):
        """Return the latest unconsumed ``(query_t, actions)``, or None."""
        with self._lock:
            result = self._result
            self._result = None
            return result

    def drain(self) -> int:
        """Discard finished-but-unconsumed results and return how many were dropped."""
        dropped = 0
        while self.poll() is not None:
            dropped += 1
        return dropped

    def last_result_age(self):
        """Seconds since the last successful inference, or None if none yet."""
        with self._lock:
            if self._last_result_time is None:
                return None
            return time.monotonic() - self._last_result_time

    def wait_for_first_result(self, timeout: float) -> bool:
        """Block until the first response arrives, at most ``timeout`` seconds."""
        deadline = time.monotonic() + timeout
        while not self._stop_event.is_set():
            with self._lock:
                ready = self._last_result_time is not None
            if ready:
                return True
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return False

    def stats(self):
        """Return ``(calls, mean_seconds, max_seconds)``, or None before any call."""
        with self._lock:
            count = len(self._infer_times)
            mean_time = sum(self._infer_times) / count if count else 0.0
            max_time = self._max_infer_time
        if count == 0:
            return None
        return count, mean_time, max_time

    def stop(self, timeout: float = 2.0) -> None:
        """Ask the thread to exit and join it, if it was ever started."""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=timeout)

    def run(self) -> None:
        """Infer submitted observations until :meth:`stop`."""
        while not self._stop_event.is_set():
            with self._lock:
                pending = self._pending_obs
                self._pending_obs = None
                self._in_flight = pending is not None
            if pending is None:
                time.sleep(_IDLE_POLL_INTERVAL)
                continue

            query_t, observation = pending
            time0 = time.monotonic()
            try:
                response = self._client.infer(observation)
                actions = np.asarray(response["actions"], dtype=np.float32)
            except Exception as error:
                self._consecutive_failures += 1
                with self._lock:
                    self._in_flight = False
                logger.error(
                    "inference failed (%d consecutive failures): %s",
                    self._consecutive_failures,
                    error,
                )
                continue

            elapsed = time.monotonic() - time0
            self._consecutive_failures = 0
            with self._lock:
                self._result = (query_t, actions)
                self._last_result_time = time.monotonic()
                self._in_flight = False
                self._infer_times.append(elapsed)
                self._max_infer_time = max(self._max_infer_time, elapsed)
