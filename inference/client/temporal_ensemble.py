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

"""ACT-style temporal ensembling of overlapping action chunks.

Without RTC the client runs one chunk at a time and stalls whenever the server
is slower than the chunk is long. This scheduler instead asks for the next
chunk while the current one is still being executed and, at every control step,
averages the predictions that cover that step, weighting a chunk by
``exp(-coeff * age)`` where ``age`` is the number of control steps since it was
requested. The freshest plan dominates; older ones only smooth the seam between
chunks. This is the "temporal ensembling" of the ACT paper (Zhao et al., 2023).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping

import numpy as np

logger = logging.getLogger(__name__)


class TemporalEnsembleScheduler:
    """Overlap inference and blend every live chunk at each control step."""

    def __init__(
        self,
        worker,
        *,
        action_horizon: int,
        min_exec_horizon: int = 15,
        ensemble_coeff: float = 0.05,
    ) -> None:
        if ensemble_coeff < 0.0:
            raise ValueError(
                f"ensemble_coeff must be non-negative, got {ensemble_coeff}"
            )
        self.worker = worker
        self.action_horizon = int(action_horizon)
        self.min_exec_horizon = int(min_exec_horizon)
        self.ensemble_coeff = float(ensemble_coeff)
        self._chunks: list[tuple[int, np.ndarray]] = []
        self._t = 0  # control-step counter
        self._last_query_t: int | None = None
        self._last_merged_query_t = -1
        self._chunk_id = 0

    def reset(self) -> None:
        """Reset all per-episode state."""
        self._chunks.clear()
        self._t = 0
        self._last_query_t = None
        self._last_merged_query_t = -1
        self._chunk_id = 0

    def submit_bootstrap(self, observation: Mapping[str, Any]) -> bool:
        """Request the first chunk of an episode."""
        return self._maybe_submit(observation, force=True)

    def get_action(self, observation: Mapping[str, Any]) -> np.ndarray | None:
        """Per-step driver: merge results, maybe replan, blend one action."""
        self._consume_results()
        self._maybe_submit(observation)
        action = self._blend()
        self._t += 1
        return action

    def observe_chunk(self, query_t: int, actions: np.ndarray) -> int | None:
        """Register a chunk returned for submission step ``query_t``.

        Returns the chunk's age in control steps, or None when it is stale (out
        of order, or requested at least one horizon ago) and was dropped.
        """
        query_t = int(query_t)
        age = self._t - query_t
        if query_t <= self._last_merged_query_t:
            logger.warning(
                "dropping an out-of-order chunk (query_t=%d <= %d)",
                query_t,
                self._last_merged_query_t,
            )
            return None
        if age < 0 or age >= self.action_horizon:
            logger.warning(
                "dropping a stale chunk (query_t=%d, age=%d >= horizon=%d)",
                query_t,
                age,
                self.action_horizon,
            )
            return None

        actions = np.asarray(actions, dtype=np.float32)
        self._chunks.append((query_t, actions))
        self._last_merged_query_t = query_t
        return age

    def _consume_results(self) -> None:
        while True:
            result = self.worker.poll()
            if result is None:
                return
            self.observe_chunk(*result)

    def _covered(self, step: int) -> bool:
        """Whether some live chunk predicts an action for ``step``."""
        return any(
            query_t <= step < query_t + len(actions)
            for query_t, actions in self._chunks
        )

    def _due(self) -> bool:
        """Whether a new request should go out now."""
        if self._last_query_t is None:
            return True
        if self._t - self._last_query_t >= self.min_exec_horizon:
            return True
        return not self._covered(self._t)

    def _blend(self) -> np.ndarray | None:
        """Weighted average of the chunks covering the current step."""
        live: list[tuple[int, int, np.ndarray]] = []
        for query_t, actions in self._chunks:
            age = self._t - query_t
            if age < len(actions):
                live.append((age, query_t, actions))
        self._chunks = [(query_t, actions) for _, query_t, actions in live]
        if not live:
            return None

        # Weights are taken relative to the youngest chunk: a common factor
        # cancels in the normalisation, and the shift keeps a large coefficient
        # from underflowing every weight to zero.
        youngest = min(age for age, _, _ in live)
        contributions = []
        weights = []
        for age, _, actions in live:
            weight = math.exp(-self.ensemble_coeff * (age - youngest))
            contributions.append(weight * actions[age].astype(np.float64))
            weights.append(weight)
        return (np.sum(contributions, axis=0) / sum(weights)).astype(np.float32)

    def _maybe_submit(
        self, observation: Mapping[str, Any], *, force: bool = False
    ) -> bool:
        if not force:
            if self.worker is None or self.worker.in_flight:
                return False
            if not self._due():
                return False

        self._chunk_id += 1
        self._last_query_t = self._t
        self.worker.submit(observation, query_t=self._t)
        logger.info(
            "requested chunk %d at step %d (min_exec_horizon=%d)",
            self._chunk_id,
            self._t,
            self.min_exec_horizon,
        )
        return True
