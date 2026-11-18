# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pi0 config for PyTorch, aligned with JAX models/pi0_config.py."""

from __future__ import annotations

import dataclasses

import torch

from rlinf.models.embodiment.openpi.modules import gemma, model, pointnet


@dataclasses.dataclass
class Pi0Config:
    dtype: str = "bfloat16"
    paligemma_variant: gemma.Variant = "gemma_2b"
    action_expert_variant: gemma.Variant = "gemma_300m"
    pointnet_variant: pointnet.Variant = "pcd"

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    pi05: bool = False
    discrete_state_input: bool | None = None
    pcd: bool = False

    # pi0.5 high-level subtask prediction: the VLM predicts the current subtask
    # as text and the action expert is conditioned on it. Training is a
    # token-level CE; serving can also read a fixed class back from the
    # classification head (see ``subtask_classification``).
    train_subtask_prediction: bool = False
    sample_subtask_prediction: bool = False
    # Weight of the subtask cross-entropy added to the flow-matching loss.
    subtask_loss_weight: float = 0.1
    # Slots reserved for the generated subtask; also caps the supervised length.
    max_subtask_len: int = 24
    # 0.0 selects the argmax token (greedy), > 0.0 samples with logits / T.
    subtask_temperature: float = 0.0
    # Replay each decode step from a captured CUDA graph (falls back to eager
    # when CUDA or capture is unavailable).
    subtask_decode_graph: bool = True
    # Sub-task stage at serving time: one prefix forward plus the classification
    # head (``classifier``) or autoregressive text (``generate``). ``None`` or
    # ``auto`` picks the fastest stage the checkpoint supports.
    subtask_mode: str | None = None
    # Fast subtask readout trained next to the token-level loss: the head maps
    # the prefix cue hidden state to a class id, which is then mapped back to
    # the class text and fed to the action expert.
    subtask_classification: bool = False
    # Number of subtask classes, i.e. rows of ``meta/subtasks.jsonl``.
    num_subtask_classes: int = 0
    subtask_class_hidden_dim: int = 512
    # Sticky serving class: one frame reads one class, so a single misread flips
    # the action condition for that chunk. A change needs
    # ``subtask_switch_patience`` consecutive requests whose top-1 probability
    # clears ``subtask_conf_threshold``, and ``subtask_force_switch`` bounds how
    # long a transition below the threshold is held back (0 disables the bound).
    # The defaults reproduce the plain per-frame argmax.
    subtask_conf_threshold: float = 0.0
    subtask_switch_patience: int = 1
    subtask_force_switch: int = 0

    def __post_init__(self):
        if self.pi05 and self.max_token_len == 48:
            object.__setattr__(self, "max_token_len", 200)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if (
            self.train_subtask_prediction
            or self.sample_subtask_prediction
            or self.subtask_classification
        ) and not self.pi05:
            raise ValueError(
                "subtask prediction requires a pi0.5 model; got pi05=False."
            )
        if self.subtask_mode not in (None, "auto", "classifier", "generate", "off"):
            raise ValueError(
                "subtask_mode must be one of None, 'auto', 'classifier', "
                f"'generate', 'off'; got {self.subtask_mode!r}."
            )
        if self.subtask_classification and self.num_subtask_classes <= 0:
            raise ValueError(
                "num_subtask_classes must be positive when "
                f"subtask_classification is on; got {self.num_subtask_classes}."
            )
        if not 0.0 <= self.subtask_conf_threshold <= 1.0:
            raise ValueError(
                "subtask_conf_threshold must be in [0, 1]; "
                f"got {self.subtask_conf_threshold}."
            )
        if self.subtask_switch_patience < 1:
            raise ValueError(
                "subtask_switch_patience must be at least 1; "
                f"got {self.subtask_switch_patience}."
            )
        if self.subtask_force_switch < 0:
            raise ValueError(
                f"subtask_force_switch must be >= 0; got {self.subtask_force_switch}."
            )

    def fake_obs(self, batch_size: int = 1) -> model.Observation:
        image = torch.ones(batch_size, *model.IMAGE_RESOLUTION, 3)
        image_mask = torch.ones(batch_size, dtype=torch.bool)
        return model.Observation(
            images={
                "base_0_rgb": image,
                "left_wrist_0_rgb": image,
                "right_wrist_0_rgb": image,
            },
            image_masks={
                "base_0_rgb": image_mask,
                "left_wrist_0_rgb": image_mask,
                "right_wrist_0_rgb": image_mask,
            },
            state=torch.ones(batch_size, self.action_dim),
            tokenized_prompt=torch.ones(
                batch_size, self.max_token_len, dtype=torch.long
            ),
            tokenized_prompt_mask=torch.ones(
                batch_size, self.max_token_len, dtype=torch.bool
            ),
            pcd_xyz=torch.ones(batch_size, 16, 2025, 3) if self.pcd else None,
        )

    def fake_act(self, batch_size: int = 1) -> torch.Tensor:
        return torch.ones(batch_size, self.action_horizon, self.action_dim)
