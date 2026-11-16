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

"""CUDA-graph accelerated single-token decode for pi0.5 subtask generation."""

from __future__ import annotations

import dataclasses
import logging

import torch

from rlinf.models.embodiment.openpi.modules.utils import set_torch_compile

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _DecodeStep:
    graph: torch.cuda.CUDAGraph
    token: torch.Tensor
    input_kv: tuple
    output_kv: tuple
    logits: torch.Tensor


class SubtaskDecodeGraph:
    """Capture one CUDA graph per decode step and replay them in sequence.

    The eager decode re-launches a few hundred kernels per token; the measured
    cost is 15 ms per token on a 2B model, and a replay brings it to 6 ms. The
    graphs are captured lazily per ``past_len`` because the KV cache and mask
    grow with every generated token, and the cache of the previous step is
    copied into the next step's static input buffer.

    Creation is best-effort: on a CPU-only host, or when capture raises, the
    caller falls back to the eager loop.
    """

    def __init__(
        self,
        model,
        *,
        prefix_len: int,
        max_steps: int,
    ) -> None:
        self._model = model
        self._prefix_len = int(prefix_len)
        self._max_steps = int(max_steps)
        self._steps: dict[int, _DecodeStep] = {}
        self._disabled = not torch.cuda.is_available()

    @property
    def available(self) -> bool:
        """Whether graph capture succeeded so far."""
        return not self._disabled

    def generate(
        self,
        logits: torch.Tensor,
        kv_cache: tuple,
        *,
        eos_token: int,
    ) -> list[torch.Tensor]:
        """Decode until EOS or ``max_steps`` and return the emitted tokens.

        Args:
            logits: ``(1, vocab)`` logits of the first generated token.
            kv_cache: prefix KV cache from the prefix forward.
            eos_token: token id that stops generation.

        Returns:
            Generated token ids (including EOS when it was emitted). Empty when
            graphs are unavailable, so the caller can fall back to eager.
        """
        if self._disabled:
            return []

        generated: list[torch.Tensor] = []
        try:
            for step in range(1, self._max_steps + 1):
                token = logits.argmax(dim=-1, keepdim=True)
                generated.append(token)
                if int(token[0, 0]) == eos_token:
                    break
                entry = self._ensure_step(step, kv_cache)
                entry.token.copy_(token)
                for dst, src in zip(entry.input_kv, kv_cache, strict=True):
                    for dst_tensor, src_tensor in zip(dst, src, strict=True):
                        dst_tensor.copy_(src_tensor)
                entry.graph.replay()
                logits = entry.logits
                kv_cache = entry.output_kv
        except Exception:
            logger.exception(
                "SubtaskDecodeGraph: capture/replay failed; falling back to eager"
            )
            self._disabled = True
            return []
        return generated

    def _ensure_step(self, step: int, kv_cache: tuple) -> _DecodeStep:
        if (entry := self._steps.get(step)) is not None:
            return entry

        past = self._prefix_len + step - 1
        token = torch.zeros((1, 1), dtype=torch.long, device=kv_cache[0][0].device)
        positions = torch.full((1, 1), past, dtype=torch.long, device=token.device)
        mask = torch.ones((1, 1, past + 1), dtype=torch.bool, device=token.device)
        input_kv = tuple((k.clone(), v.clone()) for k, v in kv_cache)

        # The rope / gated-residual kernels are wrapped in torch.compile; a lazy
        # recompile inside the capture stream reads the CUDA RNG state, which
        # CUDA graphs reject. Capture the eager kernels instead.
        set_torch_compile(False)
        try:
            with torch.no_grad():
                for _ in range(2):
                    self._model.llm(
                        [self._model.llm.embed(token), None],
                        positions=positions,
                        mask=mask,
                        kv_cache=input_kv,
                        return_prefix=True,
                    )
                    self._model.llm.embedder.decode(
                        self._model.llm(
                            [self._model.llm.embed(token), None],
                            positions=positions,
                            mask=mask,
                            kv_cache=input_kv,
                            return_prefix=True,
                        )[0][0][:, -1]
                    )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs, output_kv = self._model.llm(
                    [self._model.llm.embed(token), None],
                    positions=positions,
                    mask=mask,
                    kv_cache=input_kv,
                    return_prefix=True,
                )
                logits = self._model.llm.embedder.decode(outputs[0][:, -1])
            torch.cuda.synchronize()
        finally:
            set_torch_compile(True)

        entry = _DecodeStep(
            graph=graph,
            token=token,
            input_kv=input_kv,
            output_kv=output_kv,
            logits=logits,
        )
        self._steps[step] = entry
        return entry
