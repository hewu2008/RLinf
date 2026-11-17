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

"""pi0.5 subtask classification head.

A lightweight readout of the subtask decision: ``LayerNorm -> Linear -> GELU ->
Linear`` applied to the prefix hidden state at the ``Subtask: `` cue token. It
replaces autoregressive subtask decoding at serving time with one prefix
forward plus an MLP, and trains against a per-frame ``subtask_index`` label.

The cue position is the same token in both layouts: training encodes
``prefix | subtask | advantage | cue`` with the subtask block marked in
``token_ar_mask``, serving reserves masked slots for ``prefix | slots | cue``,
and the first AR position minus one is the tail of ``Subtask: `` in both.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubtaskClassifier(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear readout of the subtask cue token."""

    def __init__(self, width: int, hidden_dim: int, num_classes: int):
        """Build the head.

        Args:
            width: Width of the VLM prefix hidden state.
            hidden_dim: Width of the single hidden layer.
            num_classes: Number of subtask classes, i.e. rows of
                ``meta/subtasks.jsonl``.
        """
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        self.norm = nn.LayerNorm(width)
        self.hidden = nn.Linear(width, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, width)`` cue hidden states to ``(B, num_classes)`` logits."""
        return self.out(F.gelu(self.hidden(self.norm(x))))


def subtask_cue_hidden(
    prefix_out: torch.Tensor, token_ar_mask: torch.Tensor | None
) -> torch.Tensor:
    """Return the ``(B, width)`` prefix hidden state the subtask answer reads from.

    Args:
        prefix_out: ``(B, num_image + text_len, width)`` prefix hidden states,
            image tokens first.
        token_ar_mask: ``(B, text_len)`` autoregressive mask of the prompt, or
            ``None`` for a plain pi0.5 prompt.

    Returns:
        Hidden state of the token right before the first AR token. A sample
        without an AR block (no subtask annotation) falls back to its last
        prompt token, which keeps the shape uniform inside a batch.
    """
    if prefix_out.dim() != 3:
        raise ValueError(
            f"prefix_out must be (B, S, width), got {tuple(prefix_out.shape)}."
        )
    batch_size = prefix_out.shape[0]
    if token_ar_mask is None:
        return prefix_out[:, -1]

    ar_mask = token_ar_mask.to(torch.bool)
    text_len = ar_mask.shape[-1]
    num_image = prefix_out.shape[1] - text_len
    if num_image < 0:
        raise ValueError(
            f"prefix_out has {prefix_out.shape[1]} positions but token_ar_mask "
            f"covers {text_len}; the prefix must contain the prompt tokens."
        )
    has_ar = ar_mask.any(dim=-1)
    first_ar = ar_mask.to(torch.int32).argmax(dim=-1)
    cue = torch.where(has_ar, first_ar - 1, torch.full_like(first_ar, text_len - 1))
    cue = cue.clamp(0, text_len - 1)
    return prefix_out[
        torch.arange(batch_size, device=prefix_out.device), num_image + cue
    ]


def subtask_class_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked cross-entropy of the subtask class.

    Args:
        logits: ``(B, num_classes)`` classifier logits.
        labels: ``(B,)`` class ids; ``-1`` marks a frame without an annotation
            and is excluded from both the loss and the accuracy.

    Returns:
        ``(loss, accuracy, valid_count)``: the mean CE over the annotated
        frames of the batch, their argmax accuracy, and how many frames carry a
        label. All three are on the logits device; with no annotated frame the
        loss and accuracy are zero.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, num_classes), got {tuple(logits.shape)}.")
    labels = labels.to(device=logits.device).long().reshape(-1)
    if labels.shape[0] != logits.shape[0]:
        raise ValueError(
            f"labels has {labels.shape[0]} entries but logits has "
            f"{logits.shape[0]} rows."
        )

    valid = labels >= 0
    count = valid.sum()
    if int(count) == 0:
        zeros = logits.new_zeros(())
        return zeros, zeros, count.to(logits.dtype)

    targets = labels[valid]
    selected = logits[valid].float()
    ce = F.cross_entropy(selected, targets, reduction="none")
    accuracy = (selected.argmax(dim=-1) == targets).to(ce.dtype).mean()
    return ce.mean(), accuracy, count.to(ce.dtype)
