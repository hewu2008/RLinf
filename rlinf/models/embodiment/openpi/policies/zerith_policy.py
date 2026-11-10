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
"""Input transform for the Zerith H1 Pro dual-arm robot."""

import dataclasses

import einops
import numpy as np
import torch
from openpi import transforms
from openpi.models import model as _model


def make_zerith_example() -> dict:
    """Creates a random input example for the Zerith policy."""
    return {
        "observation/state": np.random.rand(23),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(
            256, size=(480, 640, 3), dtype=np.uint8
        ),
        "observation/right_wrist_image": np.random.randint(
            256, size=(480, 640, 3), dtype=np.uint8
        ),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ZerithInputs(transforms.DataTransformFn):
    """Map Zerith observations to the pi0 image and state layout.

    The dataset stores three camera views and a 23-dimensional joint vector:
    left arm 0-6, left gripper 7, right arm 8-14, right gripper 15, waist
    16-18, head yaw/pitch 19-20, base velocity 21-22. All three views exist,
    so every image mask is ``True``.
    """

    action_dim: int  # model action dim, 32 by default in Pi0Config
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = data["observation/state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = transforms.pad_to_dim(state, self.action_dim)

        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        elif self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(
                zip(
                    names,
                    (base_image, left_wrist_image, right_wrist_image),
                    strict=True,
                )
            ),
            "image_mask": dict(zip(names, (np.True_, np.True_, np.True_), strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)

        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = (
                prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
            )

        return inputs


@dataclasses.dataclass(frozen=True)
class ZerithOutputs(transforms.DataTransformFn):
    """Trim the padded action dim back to the 23 joints commanded by Zerith."""

    action_dim: int = 23

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
