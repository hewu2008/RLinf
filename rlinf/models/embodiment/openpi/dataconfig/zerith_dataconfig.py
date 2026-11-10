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
"""Data config for the Zerith H1 Pro dual-arm robot: three cameras, 23 joints."""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import zerith_policy
from rlinf.models.embodiment.openpi.transforms.zero_dims import ZeroDims

# 23-dim layout: left arm 0-6, left gripper 7, right arm 8-14, right gripper 15,
# waist 16-18, head yaw/pitch 19-20, base velocity 21-22. Arm and waist joints
# are better conditioned as deltas; the grippers are bimodal and stay absolute.
ZERITH_DELTA_ACTION_MASK = _transforms.make_bool_mask(7, -1, 7, -1, 5, -2)

# Base velocity: white noise in the dataset and the base is not commanded at
# deployment, so it is zeroed in both the delta targets and the outputs.
ZERITH_ZEROED_DIMS = (21, 22)


@dataclasses.dataclass(frozen=True)
class LeRobotZerithDataConfig(DataConfigFactory):
    """LeRobot transforms for Zerith H1 Pro datasets."""

    # Injected into the inputs when the dataset has no "prompt" column.
    default_prompt: str | None = None
    # pi0/pi05 train on delta actions; disabling keeps the raw absolute actions.
    extra_delta_transform: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.cam_high",
                        "observation/left_wrist_image": "observation.images.cam_left_wrist",
                        "observation/right_wrist_image": "observation.images.cam_right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                zerith_policy.ZerithInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[zerith_policy.ZerithOutputs()],
        )

        if self.extra_delta_transform:
            # Group.push() appends inputs but prepends outputs, so this runs as
            # inputs: ZerithInputs -> DeltaActions -> ZeroDims
            # outputs: ZeroDims -> AbsoluteActions -> ZerithOutputs
            data_transforms = data_transforms.push(
                inputs=[
                    _transforms.DeltaActions(ZERITH_DELTA_ACTION_MASK),
                    ZeroDims(ZERITH_ZEROED_DIMS),
                ],
                outputs=[
                    ZeroDims(ZERITH_ZEROED_DIMS),
                    _transforms.AbsoluteActions(ZERITH_DELTA_ACTION_MASK),
                ],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
