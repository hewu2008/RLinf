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

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.models.embodiment.openpi.env_io import EnvIO
from rlinf.models.embodiment.openpi.modules.model import (
    Observation,
    preprocess_observation,
)
from rlinf.models.embodiment.openpi.pi0 import Pi0, euler_denoise_loop, make_attn_mask
from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi.rlt_config import OpenPiPytorchRLTConfig
from rlinf.models.embodiment.openpi.transforms.subtask import (
    PALIGEMMA_EOS_TOKEN,
    SubtaskPaligemmaTokenizer,
    TokenizePi05SubtaskInputs,
)
from rlinf.utils.logging import get_logger

CFG_GUIDANCE_TYPES = ("positive", "negative")
"""Advantage suffixes the CFG-RL training data used for the conditioned branch."""


def _resolve_subtask_mode(config: Pi0Config) -> str | None:
    """Pick the sub-task stage of this checkpoint.

    ``auto`` (and the unset default) prefers the classification head, because it
    is one prefix forward instead of one decode step per token, and falls back
    to autoregressive generation when only that was trained. An explicit mode
    still has to be supported by the checkpoint, so a typo cannot silently
    disable the stage.

    Args:
        config: Model config of the checkpoint.

    Returns:
        ``"classifier"``, ``"generate"``, or ``None`` for no sub-task stage.
    """
    mode = config.subtask_mode
    if mode == "off":
        return None
    if mode in (None, "auto"):
        if config.subtask_classification:
            return "classifier"
        return "generate" if config.sample_subtask_prediction else None
    if mode == "classifier" and not config.subtask_classification:
        raise ValueError(
            "subtask_mode='classifier' needs a checkpoint trained with "
            "subtask_classification=True; this model config has no head."
        )
    return mode


def advantage_prompts(descriptions: Sequence[str], guidance_type: str) -> list[str]:
    """Append the advantage suffix that conditions the CFG branch."""
    if guidance_type not in CFG_GUIDANCE_TYPES:
        raise ValueError(
            f"guidance_type must be one of {list(CFG_GUIDANCE_TYPES)}, "
            f"got {guidance_type!r}"
        )
    return [
        f"{description}\nAdvantage: {guidance_type}" for description in descriptions
    ]


class Pi0Eval(EnvIO, Pi0):
    """Inference-only: deterministic Euler sampling, optional CFG / RTC guidance."""

    def __init__(
        self,
        config: Pi0Config,
        *,
        num_steps: int = 10,
        action_env_dim: int | None = None,
        action_chunk: int | None = None,
        config_name: str = "",
        state_indices: Sequence[int] | None = None,
        rlt_cfg: OpenPiPytorchRLTConfig | None = None,
        rtc_enabled: bool = False,
        rtc_guidance_mode: str = "approx",
        rtc_guidance_clip: float = 5.0,
        rtc_guidance_steps: int | None = None,
        cfgrl_guidance_scale: float = 0.0,
        guidance_type: str = "positive",
        subtask_classes: Sequence[str] | None = None,
    ):
        super().__init__(
            config,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            action_chunk=action_chunk,
            config_name=config_name,
            state_indices=state_indices,
            rlt_cfg=rlt_cfg,
            subtask_classes=subtask_classes,
        )
        if not math.isfinite(cfgrl_guidance_scale) or cfgrl_guidance_scale < 0.0:
            raise ValueError(
                "cfgrl_guidance_scale must be finite and >= 0, "
                f"got {cfgrl_guidance_scale!r}"
            )
        advantage_prompts([], guidance_type)
        if rtc_guidance_mode not in ("approx", "exact"):
            raise ValueError(
                "rtc_guidance_mode must be 'approx' or 'exact', "
                f"got {rtc_guidance_mode!r}"
            )
        if rtc_guidance_steps is not None and rtc_guidance_steps < 0:
            raise ValueError(
                f"rtc_guidance_steps must be >= 0, got {rtc_guidance_steps}"
            )
        self.rtc_enabled = rtc_enabled
        self.rtc_guidance_mode = rtc_guidance_mode
        self.rtc_guidance_clip = rtc_guidance_clip
        self.rtc_guidance_steps = (
            None if rtc_guidance_steps is None else int(rtc_guidance_steps)
        )
        self.cfgrl_guidance_scale = float(cfgrl_guidance_scale)
        self.guidance_type = guidance_type
        # Sub-task stage: ``classifier`` reads a fixed class out of the prefix,
        # ``generate`` decodes the subtask text token by token, ``None`` skips
        # the stage. Both stages condition the action branch on the text.
        self.subtask_mode = _resolve_subtask_mode(config)
        # Both stay ``None`` (the eager loops) until enable_sampler_graph swaps
        # in captured CUDA graphs.
        self._rtc_denoise_fn = None
        self._rtc_action_anchor = None
        self._rtc_anchor_mismatch_logged = False
        # Subtask generation state: the tokenizer is installed with the input
        # pipeline, and each decode step gets its own captured graph keyed by
        # the prefix length.
        self._subtask_tokenizer: SubtaskPaligemmaTokenizer | None = None
        self._subtask_decode_graphs: dict[int, Any] = {}
        # (prefix_mask, kv_cache, prefix_len) of the current request, reused by
        # the action branch so the image/task prefix is encoded once.
        self._last_subtask_prefix = None
        get_logger().info(
            "Pi0Eval: cfgrl_guidance_scale=%.3f guidance_type=%s cond_branch=%s "
            "num_steps=%d rtc=%s mode=%s steps=%s clip=%.2f subtask=%s",
            self.cfgrl_guidance_scale,
            guidance_type,
            "advantage" if self.cfgrl_guidance_scale > 0.0 else "off",
            num_steps,
            rtc_enabled,
            rtc_guidance_mode,
            "all" if self.rtc_guidance_steps is None else self.rtc_guidance_steps,
            rtc_guidance_clip,
            self.subtask_mode or "off",
        )

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = False,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        rtc_context=None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del compute_values, kwargs
        if mode != "eval":
            raise NotImplementedError(
                f"{type(self).__name__} only supports predict_action_batch(mode='eval'); "
                "use a training task wrapper (rl / dagger / dsrl) for train rollouts."
            )
        observation = self.env_obs_to_observation(env_obs)
        subtask = None
        prompt = None
        plain_prefix = None
        cond_prefix = None
        if self.subtask_mode is not None:
            prompt = self._prompt_from_env_obs(env_obs)
            # Regenerate from the current observation on every request: the
            # subtask depends on what the cameras see, which changes per frame.
            if self.subtask_mode == "classifier":
                subtask, plain_prefix = self.classify_subtask(observation, prompt)
            else:
                subtask = self.generate_subtask(observation, prompt)
                plain_prefix = self._subtask_prefix_cache(observation, prompt, subtask)
            if plain_prefix is None:
                observation = self._with_subtask_tokens(observation, prompt, subtask)
                cond_observation = (
                    self._with_subtask_tokens(
                        observation, prompt, subtask, advantage=self.guidance_type
                    )
                    if self.cfgrl_guidance_scale > 0.0
                    else None
                )
            else:
                cond_observation = None
                if self.cfgrl_guidance_scale > 0.0:
                    # The advantage segment goes after the subtask, so the
                    # subtask tokens cannot attend to it; only the action branch
                    # sees it, and the prefix cache already encodes that split.
                    cond_prefix = self._subtask_prefix_cache(
                        observation, prompt, subtask, advantage=self.guidance_type
                    )
                    cond_observation = observation
        else:
            cond_observation = self._guidance_observation(env_obs)
        logger = get_logger()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Pi0Eval request: batch=%d cfg_scale=%.3f guidance_type=%s "
                "cond=%s rtc=%s",
                observation.state.shape[0],
                self.cfgrl_guidance_scale,
                self.guidance_type,
                "advantage" if cond_observation is not None else "off",
                "off"
                if rtc_context is None
                else f"exec={rtc_context.executed_horizon} "
                f"delay={rtc_context.delay_steps}",
            )
        if rtc_context is not None and self.rtc_enabled:
            actions, result = self._predict_eval_with_rtc(
                observation,
                rtc_context=rtc_context,
                cond_observation=cond_observation,
                noise=noise,
                rng=rng,
                prefix=plain_prefix,
                cond_prefix=cond_prefix,
            )
        else:
            actions, result = self._predict_eval(
                observation,
                noise=noise,
                rng=rng,
                cond_observation=cond_observation,
                cfg_scale=self.cfgrl_guidance_scale,
                prefix=plain_prefix,
                cond_prefix=cond_prefix,
            )
        if subtask is not None:
            result["subtask"] = subtask
        return actions, result

    def enable_sampler_graph(self) -> None:
        """Serve both denoise loops from captured CUDA graphs.

        A CUDA graph records the eager kernels, so the returned chunk stays
        bit-identical to the eager loop and only the launch overhead goes away.
        ``torch.compile`` was measured to drift by up to 1.7e-2 on a 50-step
        chunk, which rules it out for a deployment that pins the numbers.

        The plain bootstrap loop (:attr:`_plain_denoise_fn`) and the RTC replan
        loop (:attr:`_rtc_denoise_fn`) are captured separately because they run
        a different sequence of kernels. Shapes are pinned by the model config
        and the camera layout, so changing ``max_token_len``,
        ``num_action_chunks``, ``num_steps`` or the camera resolution
        re-captures; scalar guidance arguments such as ``rtc_guidance_mode``,
        ``rtc_guidance_steps`` and ``rtc_guidance_clip`` are folded into the
        captured kernels, so changing one re-captures as well. Capture happens
        on the first call of each path, which the server warmup covers.

        ``rtc_guidance_mode="exact"`` is captured too: a graph records the
        backward kernels of every guided step, and the replay was measured
        bit-identical to the eager loop on the deployed Pi0.5. What the graph
        cannot remove is the backward arithmetic, so the exact correction still
        costs one extra backward pass per guided step; ``rtc_guidance_steps``
        bounds how many steps pay it.
        """
        from rlinf.models.embodiment.openpi.sampling.rtc_guidance import (
            LoopGraphSampler,
            rtc_denoise_loop,
        )

        if not torch.cuda.is_available():
            get_logger().warning(
                "Pi0Eval: the RTC sampler needs CUDA to run from a captured "
                "graph; keeping the eager loop"
            )
            return
        self._plain_denoise_fn = LoopGraphSampler(euler_denoise_loop)
        self._rtc_denoise_fn = LoopGraphSampler(rtc_denoise_loop)
        get_logger().info(
            "Pi0Eval: plain and RTC samplers run from captured CUDA graphs "
            "(rtc mode=%s guidance_steps=%s); the first call of each captures it",
            self.rtc_guidance_mode,
            "all" if self.rtc_guidance_steps is None else self.rtc_guidance_steps,
        )

    def disable_sampler_graph(self) -> None:
        """Go back to the eager denoise loops."""
        self._plain_denoise_fn = None
        self._rtc_denoise_fn = None

    @property
    def sampler_graphed(self) -> bool:
        """Whether the eval paths replay a captured graph."""
        return self._plain_denoise_fn is not None or self._rtc_denoise_fn is not None

    def _guidance_observation(self, env_obs: dict[str, Any]) -> Observation | None:
        """Advantage-conditioned twin of the request, or ``None`` when CFG is off."""
        if self.cfgrl_guidance_scale == 0.0:
            return None
        descriptions = env_obs.get("task_descriptions")
        if not descriptions:
            raise KeyError(
                "cfgrl_guidance_scale > 0 needs env_obs['task_descriptions'], "
                f"got {sorted(env_obs)}"
            )
        cond_env_obs = dict(env_obs)
        cond_env_obs["task_descriptions"] = advantage_prompts(
            descriptions, self.guidance_type
        )
        return self.env_obs_to_observation(cond_env_obs)

    @staticmethod
    def _prompt_from_env_obs(env_obs: dict[str, Any]) -> str:
        """Return the single task description a serving request carries."""
        prompt = env_obs.get("task_descriptions", env_obs.get("prompt"))
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(
                "subtask prediction needs a task description in env_obs; "
                f"got {sorted(env_obs)}"
            )
        return prompt

    @torch.no_grad()
    def classify_subtask(
        self, observation: Observation, prompt: str
    ) -> tuple[str, tuple | None]:
        """Read the subtask class out of the prefix with the classification head.

        One prefix forward plus the MLP replaces the per-token decode. The
        prefix of the serving layout ends with ``Subtask: `` followed by the
        reserved (masked) slots, so the cue token the head reads sits at the
        same position it did in training.

        Args:
            observation: Serving observation built by the input transforms.
            prompt: Task description of the request.

        Returns:
            ``(subtask text, action-branch prefix)``; the prefix is ``None``
            when the generation-layout cache is unavailable, in which case the
            caller tokenizes the training layout itself.
        """
        if self.subtask_classifier is None:
            raise RuntimeError(
                "classify_subtask needs a checkpoint trained with "
                "subtask_classification=True."
            )
        if not self.subtask_classes:
            raise RuntimeError(
                "classify_subtask needs the class table (openpi.subtask_classes "
                "or openpi.subtask_classes_path) to map an id back to text."
            )
        if self._subtask_tokenizer is None:
            raise RuntimeError(
                "classify_subtask needs the subtask tokenizer; call "
                "setup_transforms with a TokenizePi05SubtaskInputs stage."
            )
        if observation.state.shape[0] != 1:
            raise ValueError(
                "subtask classification supports one request per call; got batch "
                f"{observation.state.shape[0]}."
            )

        prepared = preprocess_observation(observation, train=False)
        if prepared.token_ar_mask is None or prepared.tokenized_prompt is None:
            raise ValueError(
                "subtask classification needs a tokenized prompt with an AR mask; "
                "the input pipeline did not apply TokenizePi05SubtaskInputs."
            )
        ar_mask = prepared.token_ar_mask.to(torch.bool)[0]
        if not bool(ar_mask.any()):
            raise ValueError(
                "serving request has no reserved subtask slots (token_ar_mask is "
                "all False); the prompt was tokenized for training, not serving."
            )
        # First reserved slot, i.e. the real prompt length of this request.
        valid_len = int(ar_mask.to(torch.int32).argmax().item())

        prefix_out, prefix_mask, kv_cache = self.build_prefix_cache(prepared)
        logits = self.subtask_class_logits(prefix_out, prepared)
        probs = F.softmax(logits.float(), dim=-1)[0]
        class_id = int(logits[0].argmax().item())
        if not 0 <= class_id < len(self.subtask_classes):
            raise IndexError(
                f"the head predicted class {class_id} but the class table has "
                f"{len(self.subtask_classes)} entries; the checkpoint and the "
                "table disagree, rebuild them together."
            )
        subtask = self.subtask_classes[class_id]
        image_len = prefix_out.shape[1] - prepared.tokenized_prompt.shape[1]
        # Same cache hand-off the generation path uses, so the action branch
        # pays for the image/task prefix only once.
        self._last_subtask_prefix = (prefix_mask, kv_cache, valid_len, image_len)
        get_logger().info(
            "Pi0Eval subtask classifier: class=%d conf=%.3f text=%s",
            class_id,
            float(probs[class_id].item()),
            subtask,
        )
        return subtask, self._subtask_prefix_cache(observation, prompt, subtask)

    @torch.no_grad()
    def generate_subtask(self, observation: Observation, prompt: str) -> str:
        """Generate the subtask text from the prefix with greedy decoding.

        The prefix ends at ``Subtask: `` so the decoded tokens are exactly the
        subtask answer; the action branch is tokenized separately afterwards.
        """
        if self._subtask_tokenizer is None:
            raise RuntimeError(
                "generate_subtask needs the subtask tokenizer; call "
                "setup_transforms with a TokenizePi05SubtaskInputs stage."
            )
        if observation.state.shape[0] != 1:
            raise ValueError(
                "subtask generation supports one request per call; got batch "
                f"{observation.state.shape[0]}."
            )

        state = observation.state[0].detach().to(torch.float32).cpu().numpy()
        prefix_tokens = self._subtask_tokenizer.encode_subtask_prefix(prompt, state)
        device = observation.state.device
        # Pad the prefix to a fixed budget so the per-step decode graphs do not
        # depend on the digit count of the discretized state.
        valid_len = len(prefix_tokens)
        budget = max(valid_len, self.max_token_len - self.max_subtask_len - 8)
        tokens = torch.zeros((1, budget), dtype=torch.long, device=device)
        tokens[0, :valid_len] = torch.tensor(
            prefix_tokens, dtype=torch.long, device=device
        )
        token_mask = torch.zeros((1, budget), dtype=torch.bool, device=device)
        token_mask[0, :valid_len] = True
        # The subtask depends on the camera views: reuse the full prefix
        # embedding (images + language) instead of the language tokens alone.
        prepared = preprocess_observation(observation, train=False)
        prefix_obs = Observation(
            images=prepared.images,
            image_masks=prepared.image_masks,
            state=prepared.state,
            tokenized_prompt=tokens,
            tokenized_prompt_mask=token_mask,
            token_ar_mask=torch.zeros_like(token_mask),
            token_loss_mask=None,
            pcd_xyz=getattr(prepared, "pcd_xyz", None),
        )
        prefix_tokens_emb, prefix_mask, prefix_ar_mask = self.embed_prefix(prefix_obs)
        image_len = prefix_tokens_emb.shape[1] - tokens.shape[1]
        positions = torch.cumsum(prefix_mask.int(), dim=1) - 1
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        outputs, kv_cache = self.llm(
            [prefix_tokens_emb, None],
            positions=positions,
            mask=prefix_attn_mask,
        )
        logits = self.llm.embedder.decode(outputs[0][:, image_len + valid_len - 1])
        self._last_subtask_prefix = (prefix_mask, kv_cache, valid_len, image_len)

        # ``budget`` pads the prefix so the decode graphs keep one shape, but the
        # padding slots carry no prompt token: the first subtask token sits at
        # ``image_len + valid_len`` and the decode masks ``[valid_len, budget)``
        # out, matching the positions the model was trained on.
        cache_len = image_len + budget
        valid_prefix = image_len + valid_len
        generated = self._graph_decode(cache_len, valid_prefix, logits, kv_cache)
        if generated is None:
            generated = self._eager_decode(
                outputs[0][:, image_len + valid_len - 1],
                cache_len,
                valid_prefix,
                kv_cache,
                device,
            )

        subtask = self._subtask_tokenizer.detokenize(
            torch.cat(generated, dim=1)[0].cpu().numpy()
        )
        if not subtask:
            get_logger().warning(
                "Pi0Eval: subtask generation produced an empty string; the action "
                "branch falls back to the plain prompt"
            )
        return subtask

    def _graph_decode(self, cache_len, valid_prefix, logits, kv_cache):
        """Decode with captured step graphs, or return None to fall back."""
        if (
            not self.subtask_decode_graph
            or self.subtask_temperature > 0.0
            or not torch.cuda.is_available()
        ):
            return None
        from rlinf.models.embodiment.openpi.sampling.subtask_decode import (
            SubtaskDecodeGraph,
        )

        graph = self._subtask_decode_graphs.get(cache_len)
        if graph is None:
            graph = SubtaskDecodeGraph(
                self, prefix_len=cache_len, max_steps=self.max_subtask_len
            )
            self._subtask_decode_graphs[cache_len] = graph
        if not graph.available:
            return None
        generated = graph.generate(
            logits, kv_cache, eos_token=PALIGEMMA_EOS_TOKEN, valid_prefix=valid_prefix
        )
        return generated or None

    def _eager_decode(self, hidden, cache_len, valid_prefix, kv_cache, device):
        """Reference decode loop used when graphs are disabled or unavailable."""
        generated = []
        for step in range(self.max_subtask_len):
            logits = self.llm.embedder.decode(hidden)
            if self.subtask_temperature > 0.0:
                probs = F.softmax(logits.float() / self.subtask_temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            else:
                token = logits.argmax(dim=-1, keepdim=True)
            generated.append(token)
            if int(token[0, 0]) == PALIGEMMA_EOS_TOKEN:
                break
            # The cache covers the padded prefix, so the mask has to reach
            # ``cache_len + step + 1`` while the position keeps counting from
            # the last real prompt token.
            step_mask = torch.ones(
                (1, 1, cache_len + step + 1), dtype=torch.bool, device=device
            )
            step_mask[..., valid_prefix:cache_len] = False
            step_positions = torch.full(
                (1, 1), valid_prefix + step, dtype=torch.long, device=device
            )
            step_outputs, kv_cache = self.llm(
                [self.llm.embed(token), None],
                positions=step_positions,
                mask=step_mask,
                kv_cache=kv_cache,
                # The decode reads the prefix expert's output, so the last
                # layer must keep its final FFN/norm (return_prefix=False
                # drops expert 0 and returns None).
                return_prefix=True,
            )
            hidden = step_outputs[0][:, -1]
        return generated

    def _subtask_prefix_cache(
        self,
        observation: Observation,
        prompt: str,
        subtask: str,
        advantage: str | None = None,
    ):
        """Extend the generation prefix cache with the action-branch tokens.

        Returns ``(prefix_mask, kv_cache)`` for ``sample_actions``, or ``None``
        when the generation prefix is unavailable.
        """
        if self._last_subtask_prefix is None or self._subtask_tokenizer is None:
            return None
        _, kv_cache, valid_len, image_len = self._last_subtask_prefix
        state = observation.state[0].detach().to(torch.float32).cpu().numpy()
        tokens, token_mask, ar_mask, _ = (
            self._subtask_tokenizer.tokenize_subtask_training(
                prompt,
                subtask,
                state,
                max_subtask_len=self.max_subtask_len,
                advantage=advantage,
            )
        )
        device = observation.state.device
        new_tokens = torch.as_tensor(
            tokens[None, valid_len:], dtype=torch.long, device=device
        )
        new_mask = torch.as_tensor(
            token_mask[None, valid_len:], dtype=torch.bool, device=device
        )
        new_ar = torch.as_tensor(
            ar_mask[None, valid_len:], dtype=torch.bool, device=device
        )
        # The generation cache holds the image tokens followed by the padded
        # language prefix; keep exactly those tokens and continue from there.
        past_len = image_len + valid_len
        past_cache = tuple((k[:, :past_len], v[:, :past_len]) for k, v in kv_cache)
        _, full_mask, full_kv = self.extend_prefix_cache(
            new_tokens, past_cache, past_len, new_mask, new_ar
        )
        return full_mask, full_kv

    def _with_subtask_tokens(
        self,
        observation: Observation,
        prompt: str,
        subtask: str,
        advantage: str | None = None,
    ) -> Observation:
        """Replace the plain prompt with the subtask-conditioned training layout."""
        if self._subtask_tokenizer is None:
            raise RuntimeError("subtask tokenizer is not installed")
        if not subtask:
            return observation
        state = observation.state[0].detach().to(torch.float32).cpu().numpy()
        tokens, token_mask, ar_mask, _ = (
            self._subtask_tokenizer.tokenize_subtask_training(
                prompt,
                subtask,
                state,
                max_subtask_len=self.max_subtask_len,
                advantage=advantage,
            )
        )
        device = observation.state.device
        return dataclasses.replace(
            observation,
            tokenized_prompt=torch.as_tensor(tokens, dtype=torch.long, device=device)[
                None
            ],
            tokenized_prompt_mask=torch.as_tensor(
                token_mask, dtype=torch.bool, device=device
            )[None],
            token_ar_mask=torch.as_tensor(ar_mask, dtype=torch.bool, device=device)[
                None
            ],
            token_loss_mask=None,
        )

    def setup_transforms(
        self,
        transforms: Sequence,
        output_transforms: Sequence,
    ) -> None:
        """Install the pipelines and cache what the RTC re-anchor needs."""
        super().setup_transforms(transforms, output_transforms)
        self._subtask_tokenizer = None
        for transform in self._input_transform_fn.transforms:
            if isinstance(transform, TokenizePi05SubtaskInputs):
                self._subtask_tokenizer = transform.tokenizer
                break
        if self.subtask_mode is not None and self._subtask_tokenizer is None:
            raise ValueError(
                f"subtask_mode={self.subtask_mode!r} is enabled but the input "
                "pipeline has no TokenizePi05SubtaskInputs; build the model from a "
                "Zerith pi0.5 config with the subtask tokenizer enabled."
            )
        if self.subtask_mode == "classifier" and self.subtask_classifier is None:
            raise ValueError(
                "subtask_mode='classifier' is enabled but the model has no "
                "subtask_classifier; set openpi.subtask_classification=True to "
                "match the trained checkpoint."
            )
        from rlinf.models.embodiment.openpi.sampling.rtc_guidance import (
            action_anchor_from_transforms,
        )

        self._rtc_action_anchor = action_anchor_from_transforms(
            self._input_transform_fn, self._output_transform_fn
        )
        if self.rtc_enabled and self._rtc_action_anchor is None:
            get_logger().warning(
                "Pi0Eval: the action pipeline has no delta transform or norm "
                "stats; RTC leftovers stay anchored to the previous request"
            )

    def _reanchored_context(self, rtc_context, cur_state: torch.Tensor):
        """Move the leftover deltas from the previous request to this state."""
        from rlinf.models.embodiment.openpi.sampling.rtc_guidance import (
            reanchor_prev_actions,
        )

        if (
            rtc_context is None
            or rtc_context.prev_model_actions is None
            or rtc_context.prev_state is None
            or self._rtc_action_anchor is None
        ):
            return rtc_context
        if cur_state.shape[-1] < self._rtc_action_anchor.state_scale.shape[0]:
            # ``state_indices`` selects a subset of the trained state, so the
            # per-dimension stats no longer line up with the model's state.
            if not self._rtc_anchor_mismatch_logged:
                get_logger().warning(
                    "Pi0Eval: the model state has %d dims but the norm stats "
                    "cover %d; RTC leftovers stay anchored to the previous "
                    "request",
                    cur_state.shape[-1],
                    self._rtc_action_anchor.state_scale.shape[0],
                )
                self._rtc_anchor_mismatch_logged = True
            return rtc_context
        prev_actions = reanchor_prev_actions(
            rtc_context.prev_model_actions,
            rtc_context.prev_state,
            cur_state,
            self._rtc_action_anchor,
        )
        return dataclasses.replace(rtc_context, prev_model_actions=prev_actions)

    def _predict_eval_with_rtc(
        self,
        observation,
        *,
        rtc_context,
        cond_observation,
        noise: torch.Tensor | None,
        rng: torch.Generator | None,
        prefix=None,
        cond_prefix=None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        from rlinf.models.embodiment.openpi.sampling.rtc_guidance import (
            sample_actions_with_rtc_guidance,
        )

        rtc_context = self._reanchored_context(rtc_context, observation.state)
        model_actions = sample_actions_with_rtc_guidance(
            self,
            observation,
            rtc_context,
            num_steps=self.num_steps,
            noise=noise,
            rng=rng,
            guidance_clip=self.rtc_guidance_clip,
            cond_observation=cond_observation,
            cfg_scale=self.cfgrl_guidance_scale,
            guidance_mode=self.rtc_guidance_mode,
            guidance_steps=self.rtc_guidance_steps,
            denoise_fn=self._rtc_denoise_fn,
            prefix=prefix,
            cond_prefix=cond_prefix,
        )
        actions = self.decode_actions(model_actions, observation.state)
        B = actions.shape[0]
        return actions, {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(B, -1).contiguous(),
                "model_action": model_actions.reshape(B, -1).contiguous(),
            },
            "model_actions": model_actions,
        }

    @torch.no_grad()
    def extract_rlt_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract the frozen Stage1 features consumed by the Stage2 RLT head."""
        self._require_rlt()
        observation = self.env_obs_to_observation(env_obs)

        prepared_observation = preprocess_observation(observation, train=False)
        prefix_output, prefix_mask, kv_cache = self.build_prefix_cache(
            prepared_observation
        )
        rlt_prefix_output, rlt_prefix_mask = self._select_rlt_prefix_embeddings(
            prefix_output, prefix_mask, prepared_observation.tokenized_prompt
        )
        z_rl = self._encode_rlt_flat(rlt_prefix_output, rlt_prefix_mask).to(
            dtype=torch.float32
        )

        model_actions = self._sample_actions_from_prefix_cache(
            prepared_observation,
            prefix_mask,
            kv_cache,
        )
        ref_chunk = self.output_transform(
            {"actions": model_actions, "state": observation.state}
        )["actions"]

        raw_proprio = self._select_configured_state(env_obs["states"])
        if "maniskill" in self.config_name.lower():
            state_dim = (
                raw_proprio.shape[-1]
                if hasattr(raw_proprio, "shape")
                else np.asarray(raw_proprio).shape[-1]
            )
            proprio = observation.state[..., :state_dim]
        else:
            proprio = raw_proprio
        if not torch.is_tensor(proprio):
            proprio = torch.as_tensor(proprio)

        return {
            "z_rl": z_rl,
            "proprio": proprio.to(device=z_rl.device, dtype=torch.float32),
            "ref_chunk": ref_chunk.to(device=z_rl.device, dtype=torch.float32),
        }

    def _sample_actions_from_prefix_cache(
        self,
        observation,
        prefix_mask: torch.Tensor,
        kv_cache: tuple,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch_size = observation.state.shape[0]
        device = observation.state.device
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.action_horizon,
                self.action_dim,
                device=device,
                generator=rng,
            )

        x_t = noise
        dt = -1.0 / self.num_steps
        t = 1.0
        while t >= -dt / 2:
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            suffix_out = self.run_suffix(
                observation, x_t, t_tensor, kv_cache, prefix_mask
            )
            v_t = self.velocity_from_suffix(suffix_out)
            x_t = x_t + dt * v_t
            t += dt
        return x_t
