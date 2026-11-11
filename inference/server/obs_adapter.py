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

"""Convert one wire observation into the env-observation dict RLinf expects."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from inference.protocol import decode_frame

CAMERA_SLOTS = ("base", "left_wrist", "right_wrist")
"""The three views the policy is trained on, in ``camera_layout`` keys."""


def build_env_obs(
    observation: Mapping[str, Any],
    *,
    camera_layout: Mapping[str, str],
    default_prompt: str | None,
    action_dim: int,
) -> dict[str, Any]:
    """Build the RLinf env-observation dict for one wire observation.

    The base view becomes ``main_images`` and the two wrist views travel
    together in ``extra_view_images`` (index 0 left, index 1 right), the
    three-camera convention RLinf's dual-arm policies already use.
    """
    images = observation.get("images")
    if not isinstance(images, Mapping):
        raise KeyError(
            f"observation is missing an 'images' mapping, got {sorted(observation)}"
        )

    frames = {}
    for slot in CAMERA_SLOTS:
        camera = camera_layout.get(slot)
        if camera is None:
            raise KeyError(
                f"server camera_layout has no {slot!r} entry; got {dict(camera_layout)}"
            )
        if camera not in images:
            raise KeyError(
                f"camera {camera!r} for the {slot} view is missing from the "
                f"request; got {sorted(images)}"
            )
        frames[slot] = np.ascontiguousarray(decode_frame(images[camera]))

    shapes = {slot: frame.shape for slot, frame in frames.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(
            f"the three camera frames must share one resolution, got {shapes}"
        )

    state = np.asarray(observation["state"], dtype=np.float32)
    if state.shape != (action_dim,):
        raise ValueError(f"expected a ({action_dim},) state, got {state.shape}")

    prompt = observation.get("prompt") or default_prompt
    if not prompt:
        raise ValueError(
            "the request carries no prompt and the server has no default_prompt"
        )

    return {
        "main_images": frames["base"][None],
        "extra_view_images": np.stack([frames["left_wrist"], frames["right_wrist"]])[
            None
        ],
        "states": state[None],
        "task_descriptions": [str(prompt)],
    }
