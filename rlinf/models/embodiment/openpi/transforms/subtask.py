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

"""pi0.5 subtask prediction transforms.

Ported from the ZERITH OpenPI implementation. The VLM predicts the current
subtask as text (``Task: .., State: ..;\\nSubtask: <text>;\\nAction: ``) and the
action expert is conditioned on it; the subtask tokens carry a token-level
cross-entropy while everything before them is prefix.

The tokenizer mirrors the upstream ``PaligemmaTokenizer`` layout but adds the
subtask stage, so it wraps the upstream sentencepiece processor instead of
downloading a second copy.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from typing import Any, Mapping

import numpy as np
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.transforms import DataDict, DataTransformFn

logger = logging.getLogger(__name__)

# PaliGemma sentencepiece EOS id: stops subtask generation and is stripped on decode.
PALIGEMMA_EOS_TOKEN = 1

# Text that follows the (predicted) subtask; the action expert reads both.
SUBTASK_ACTION_CUE = "\nAction: "


def load_subtasks(root: str | pathlib.Path) -> dict[int, str]:
    """Load ``<root>/meta/subtasks.jsonl`` into ``{subtask_index: text}``.

    Args:
        root: LeRobot dataset root that holds the ``meta`` directory.

    Returns:
        Mapping from subtask index to its text. Missing file means the dataset
        has no annotations and the mapping is empty.
    """
    path = pathlib.Path(root) / "meta" / "subtasks.jsonl"
    if not path.exists():
        return {}

    subtasks: dict[int, str] = {}
    for line in path.read_text().splitlines():
        if not (line := line.strip()):
            continue
        record = json.loads(line)
        subtasks[int(record["subtask_index"])] = str(record["subtask"])
    return subtasks


@dataclasses.dataclass(frozen=True)
class SubtaskFromIndex(DataTransformFn):
    """Map the per-frame LeRobot ``subtask_index`` column to its text.

    The ``subtask`` key is always set: an empty string marks a frame without an
    annotation (``-1`` or an unknown index), which keeps the subtask loss off
    for that frame while still letting the tokenizer tell a training frame from
    a serving request.
    """

    subtasks: Mapping[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        index = data.get("subtask_index")
        class_id = -1 if index is None else int(index)
        if class_id not in self.subtasks:
            class_id = -1
        return {**data, "subtask": self.subtasks.get(class_id, "")}


class SubtaskPaligemmaTokenizer:
    """PaliGemma tokenizer with the pi0.5 subtask stage.

    The prefix layout matches the upstream pi0.5 prompt (``Task: .., State: ..;``).
    Training encodes the subtask as the autoregressive answer, inference reserves
    masked slots for it, and :meth:`detokenize` decodes the generated ids.
    """

    def __init__(self, base_tokenizer: PaligemmaTokenizer, max_len: int | None = None):
        """Reuse the sentencepiece processor of an upstream tokenizer.

        Args:
            base_tokenizer: Upstream ``PaligemmaTokenizer``; only its loaded
                processor is reused so the model is downloaded once.
            max_len: Maximum token length, defaults to the base tokenizer's.
        """
        self._tokenizer = base_tokenizer._tokenizer
        self._max_len = int(max_len if max_len is not None else base_tokenizer._max_len)

    def tokenize(
        self, prompt: str, state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize the plain pi0.5 prompt without a subtask stage.

        Args:
            prompt: Task description.
            state: Normalized robot state; required when the state is part of
                the language prompt (pi0.5).

        Returns:
            ``(tokens, token_mask)`` padded to ``max_len``.
        """
        cleaned_text = self._clean_text(prompt)
        if state is not None:
            full_prompt = (
                f"Task: {cleaned_text}, State: {self._state_str(state)};\nAction: "
            )
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True)
            tokens = tokens + self._tokenizer.encode("\n")
        tokens = self._truncate(tokens, "Token")
        token_mask = [True] * len(tokens)
        return self._pad(tokens, token_mask, [0] * len(tokens), [False] * len(tokens))[
            :2
        ]

    def encode_subtask_prefix(
        self, prompt: str, state: np.ndarray | None = None
    ) -> list[int]:
        """Encode ``Task: .., State: ..;\nSubtask: `` for autoregressive generation.

        Args:
            prompt: Task description.
            state: Normalized robot state; required when the state is part of
                the language prompt (pi0.5).

        Returns:
            Token ids of the generation prefix.
        """
        return self._tokenizer.encode(self._subtask_prefix(prompt, state), add_bos=True)

    def tokenize_subtask_training(
        self,
        prompt: str,
        subtask: str,
        state: np.ndarray | None = None,
        *,
        max_subtask_len: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a frame with its ground-truth subtask.

        Args:
            prompt: Task description.
            subtask: Ground-truth subtask text for the frame.
            state: Normalized robot state; required when the state is part of
                the language prompt (pi0.5).
            max_subtask_len: Cap on the supervised subtask length.

        Returns:
            ``(tokens, token_mask, ar_mask, loss_mask)`` padded to ``max_len``.
            Only the subtask tokens (including their EOS) carry the loss.
        """
        prefix_tokens = self._tokenizer.encode(
            self._subtask_prefix(prompt, state), add_bos=True
        )
        subtask_tokens = self._truncate(
            self._tokenizer.encode(f"{self._clean_text(subtask)};", add_eos=True),
            "Subtask",
            limit=max_subtask_len,
        )
        if subtask_tokens[-1] != PALIGEMMA_EOS_TOKEN:
            # A truncated subtask keeps an EOS so generation still learns to stop.
            subtask_tokens[-1] = PALIGEMMA_EOS_TOKEN
        cue_tokens = self._tokenizer.encode(SUBTASK_ACTION_CUE)

        tokens = prefix_tokens + subtask_tokens + cue_tokens
        if len(tokens) > self._max_len:
            raise ValueError(
                f"Subtask prompt needs {len(tokens)} tokens but max_token_len is "
                f"{self._max_len}; raise max_token_len or lower max_subtask_len. "
                "Silently truncating would drop the subtask loss."
            )
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * (
            len(subtask_tokens) + len(cue_tokens)
        )
        loss_mask = (
            [False] * len(prefix_tokens)
            + [True] * len(subtask_tokens)
            + [False] * len(cue_tokens)
        )
        return self._pad(tokens, token_mask, ar_mask, loss_mask)

    def tokenize_subtask_inference(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        *,
        max_subtask_len: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a serving request, reserving slots for the generated subtask.

        The reserved slots sit between ``Subtask: `` and ``\\nAction: ``. While a
        slot is still masked it is invisible as an attention key, and once a
        token is written the cumsum ordering keeps it before the cue, exactly as
        in training.

        Args:
            prompt: Task description.
            state: Normalized robot state; required when the state is part of
                the language prompt (pi0.5).
            max_subtask_len: Number of reserved subtask slots.

        Returns:
            ``(tokens, token_mask, ar_mask)`` padded to ``max_len``.
        """
        prefix_tokens = self._tokenizer.encode(
            self._subtask_prefix(prompt, state), add_bos=True
        )
        cue_tokens = self._tokenizer.encode(SUBTASK_ACTION_CUE)

        if len(prefix_tokens) + max_subtask_len + len(cue_tokens) > self._max_len:
            logger.warning(
                "Subtask prompt (%d + %d + %d) exceeds max length (%d); "
                "raise max_token_len or lower max_subtask_len.",
                len(prefix_tokens),
                max_subtask_len,
                len(cue_tokens),
                self._max_len,
            )

        tokens = prefix_tokens + ([0] * max_subtask_len) + cue_tokens
        token_mask = (
            ([True] * len(prefix_tokens))
            + ([False] * max_subtask_len)
            + ([True] * len(cue_tokens))
        )
        ar_mask = (
            ([0] * len(prefix_tokens))
            + ([1] * max_subtask_len)
            + ([1] * len(cue_tokens))
        )
        tokens, token_mask, ar_mask, _ = self._pad(
            tokens, token_mask, ar_mask, [False] * len(tokens)
        )
        return tokens, token_mask, ar_mask

    def detokenize(self, tokens: np.ndarray, mask: np.ndarray | None = None) -> str:
        """Decode generated subtask ids, dropping padding and everything from EOS.

        Args:
            tokens: Generated token ids.
            mask: Optional validity mask; padding is dropped when given.

        Returns:
            Decoded subtask text without the trailing EOS.
        """
        token_list = np.asarray(tokens).astype(np.int32).tolist()
        if mask is not None:
            valid = np.asarray(mask).astype(bool).tolist()
            token_list = [
                token for token, keep in zip(token_list, valid, strict=True) if keep
            ]
        else:
            token_list = [token for token in token_list if token != 0]

        if PALIGEMMA_EOS_TOKEN in token_list:
            token_list = token_list[: token_list.index(PALIGEMMA_EOS_TOKEN)]
        return self._tokenizer.decode(token_list).strip()

    def _clean_text(self, text: str) -> str:
        return text.strip().replace("_", " ").replace("\n", " ")

    def _state_str(self, state: np.ndarray) -> str:
        discretized_state = (
            np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        )
        return " ".join(map(str, discretized_state))

    def _subtask_prefix(self, prompt: str, state: np.ndarray | None) -> str:
        cleaned_text = self._clean_text(prompt)
        if state is None:
            return f"Task: {cleaned_text};\nSubtask: "
        return f"Task: {cleaned_text}, State: {self._state_str(state)};\nSubtask: "

    def _truncate(
        self, tokens: list[int], name: str, *, limit: int | None = None
    ) -> list[int]:
        limit = self._max_len if limit is None else min(limit, self._max_len)
        if len(tokens) > limit:
            logger.warning(
                "%s length (%d) exceeds max length (%d), truncating.",
                name,
                len(tokens),
                limit,
            )
            tokens = tokens[:limit]
        return tokens

    def _pad(
        self,
        tokens: list[int],
        token_mask: list[bool],
        ar_mask: list[int],
        loss_mask: list[bool],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        def pad(values: list, fill, dtype) -> np.ndarray:
            values = list(values)[: self._max_len]
            return np.asarray(
                values + [fill] * (self._max_len - len(values)), dtype=dtype
            )

        return (
            pad(tokens, 0, np.int32),
            pad(token_mask, False, np.bool_),
            pad(ar_mask, 0, np.int32),
            pad(loss_mask, False, np.bool_),
        )


def _prompt_text(prompt: Any, guidance_type: str | None) -> str:
    if guidance_type is None:
        return prompt
    return f"{prompt}\nAdvantage: {guidance_type}"


@dataclasses.dataclass(frozen=True)
class TokenizePi05SubtaskInputs(DataTransformFn):
    """Tokenize pi0.5 inputs with the subtask stage.

    Three modes, told apart by the ``subtask`` key that the training pipeline
    always attaches:

    - non-empty label: encode the subtask as the autoregressive answer and emit
      the AR / loss masks for its cross-entropy;
    - empty label (no annotation for this frame): plain pi0.5 prompt, no loss;
    - no key at all (serving): reserve ``max_subtask_len`` slots for generation.
    """

    tokenizer: SubtaskPaligemmaTokenizer
    discrete_state_input: bool = False
    max_subtask_len: int = 20

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        subtask = data.pop("subtask", None)
        if subtask is not None and not isinstance(subtask, str):
            subtask = subtask.item() if hasattr(subtask, "item") else str(subtask)

        if subtask:
            tokens, token_mask, ar_mask, loss_mask = (
                self.tokenizer.tokenize_subtask_training(
                    prompt,
                    subtask.strip(),
                    state,
                    max_subtask_len=self.max_subtask_len,
                )
            )
            return {
                **data,
                "tokenized_prompt": tokens,
                "tokenized_prompt_mask": token_mask,
                "token_ar_mask": ar_mask,
                "token_loss_mask": loss_mask,
            }

        if subtask is not None:
            # Annotated dataset with an unannotated frame: same keys as the
            # supervised branch so annotated and plain frames can share a batch.
            tokens, token_mask = self.tokenizer.tokenize(prompt, state)
            return {
                **data,
                "tokenized_prompt": tokens,
                "tokenized_prompt_mask": token_mask,
                "token_ar_mask": np.zeros_like(tokens, dtype=np.int32),
                "token_loss_mask": np.zeros_like(tokens, dtype=np.bool_),
            }

        tokens, token_mask, ar_mask = self.tokenizer.tokenize_subtask_inference(
            prompt, state, max_subtask_len=self.max_subtask_len
        )
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
        }


@dataclasses.dataclass(frozen=True)
class TokenizePi05SubtaskGuidanceInputs(DataTransformFn):
    """Tokenize the plain and advantage-conditioned prompts for CFG training.

    The subtask is an input condition, not a guidance axis: the same
    ground-truth subtask is encoded in every prompt variant, while the advantage
    suffix selects the conditional branch. All variants are padded to
    ``max_len``, so the model can stack them as one batch.
    """

    tokenizer: SubtaskPaligemmaTokenizer
    discrete_state_input: bool = False
    max_subtask_len: int = 20

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        subtask = data.pop("subtask", None)
        if subtask is not None and not isinstance(subtask, str):
            subtask = subtask.item() if hasattr(subtask, "item") else str(subtask)

        if not subtask:
            # No annotation: keep plain pi0.5 prompts so the CFG branches still
            # have equal shape, and leave the subtask loss off.
            plain = self.tokenizer.tokenize(prompt, state)
            positive = self.tokenizer.tokenize(_prompt_text(prompt, "positive"), state)
            negative = self.tokenizer.tokenize(_prompt_text(prompt, "negative"), state)
            tokens, token_mask = plain
            return {
                **data,
                "tokenized_prompt": tokens,
                "tokenized_prompt_mask": token_mask,
                "tokenized_positive_guidance_prompt": positive[0],
                "tokenized_positive_guidance_prompt_mask": positive[1],
                "tokenized_negative_guidance_prompt": negative[0],
                "tokenized_negative_guidance_prompt_mask": negative[1],
                "token_ar_mask": np.zeros_like(tokens, dtype=np.int32),
                "token_loss_mask": np.zeros_like(tokens, dtype=np.bool_),
            }

        subtask = subtask.strip()
        plain = self.tokenizer.tokenize_subtask_training(
            prompt, subtask, state, max_subtask_len=self.max_subtask_len
        )
        positive = self.tokenizer.tokenize_subtask_training(
            _prompt_text(prompt, "positive"),
            subtask,
            state,
            max_subtask_len=self.max_subtask_len,
        )
        negative = self.tokenizer.tokenize_subtask_training(
            _prompt_text(prompt, "negative"),
            subtask,
            state,
            max_subtask_len=self.max_subtask_len,
        )
        tokens, token_mask, ar_mask, loss_mask = plain
        for name, variant in (("positive", positive), ("negative", negative)):
            if variant[0].shape != tokens.shape:
                raise ValueError(
                    f"{name} guidance prompt has shape {variant[0].shape}, "
                    f"expected {tokens.shape}."
                )
            if int(variant[3].sum()) != int(loss_mask.sum()):
                raise ValueError(
                    f"{name} guidance prompt has {int(variant[3].sum())} subtask "
                    f"loss tokens, expected {int(loss_mask.sum())}; the subtask "
                    "text must be identical across guidance variants."
                )

        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "tokenized_positive_guidance_prompt": positive[0],
            "tokenized_positive_guidance_prompt_mask": positive[1],
            "tokenized_negative_guidance_prompt": negative[0],
            "tokenized_negative_guidance_prompt_mask": negative[1],
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }
