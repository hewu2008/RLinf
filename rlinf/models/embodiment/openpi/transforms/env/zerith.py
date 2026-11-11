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

"""Zerith H1 Pro env-obs adapter: a base view plus two separate wrist views."""

from __future__ import annotations

from typing import Callable


def repack_env_obs(env_obs: dict, *, select_state: Callable) -> dict:
    """Map a Zerith env observation onto its three named camera keys.

    ``extra_view_images`` carries both wrist views in one entry (index 0 left,
    index 1 right), the same layout the GenieSim adapter reads. ``ZerithInputs``
    wants them as separate ``observation/left_wrist_image`` and
    ``observation/right_wrist_image`` keys, so they are split here.
    """
    extra_view_images = env_obs.get("extra_view_images")
    if extra_view_images is None:
        raise KeyError(
            "Zerith env observations must carry extra_view_images with the left "
            "and right wrist views; got keys "
            f"{sorted(env_obs)}"
        )
    if extra_view_images.shape[1] < 2:
        raise ValueError(
            "Zerith env observations need two wrist views in extra_view_images, "
            f"got shape {extra_view_images.shape}"
        )
    return {
        "observation/image": env_obs["main_images"],
        "observation/left_wrist_image": extra_view_images[:, 0],
        "observation/right_wrist_image": extra_view_images[:, 1],
        "prompt": env_obs["task_descriptions"],
        "observation/state": select_state(env_obs["states"]),
    }
