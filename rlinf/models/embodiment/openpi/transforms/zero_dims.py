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
"""Zero selected action dimensions.

Same semantics as openpi's ``ZeroDims``, which the pinned openpi build does not
ship. ``actions`` is ``(H, D)``; dims index the action vector, so zeroing before
or after padding is equivalent for trailing dims that are already zero.
"""

import dataclasses

import numpy as np
from openpi import transforms as _transforms


@dataclasses.dataclass(frozen=True)
class ZeroDims(_transforms.DataTransformFn):
    """Set ``actions[..., dims]`` to zero, keeping the rest untouched."""

    dims: tuple[int, ...]

    def __call__(self, data: dict) -> dict:
        # Inference samples carry no actions; the same transform sits in the input
        # group before DeltaActions, so it must tolerate their absence.
        if "actions" not in data:
            return data
        actions = data["actions"]
        actions = actions.copy() if isinstance(actions, np.ndarray) else actions.clone()
        actions[..., list(self.dims)] = 0
        return {**data, "actions": actions}
