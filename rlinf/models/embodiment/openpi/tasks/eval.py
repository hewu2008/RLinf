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

import logging
import math
from typing import Any, Literal, Sequence

import numpy as np
import torch

from rlinf.models.embodiment.openpi.env_io import EnvIO
from rlinf.models.embodiment.openpi.modules.model import (
    Observation,
    preprocess_observation,
)
from rlinf.models.embodiment.openpi.pi0 import Pi0, euler_denoise_loop
from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi.rlt_config import OpenPiPytorchRLTConfig
from rlinf.utils.logging import get_logger

CFG_GUIDANCE_TYPES = ("positive", "negative")
"""Advantage suffixes the CFG-RL training data used for the conditioned branch."""


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
    ):
        super().__init__(
            config,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            action_chunk=action_chunk,
            config_name=config_name,
            state_indices=state_indices,
            rlt_cfg=rlt_cfg,
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
        # Both stay ``None`` (the eager loops) until enable_sampler_graph swaps
        # in captured CUDA graphs.
        self._rtc_denoise_fn = None
        get_logger().info(
            "Pi0Eval: cfgrl_guidance_scale=%.3f guidance_type=%s cond_branch=%s "
            "num_steps=%d rtc=%s mode=%s steps=%s clip=%.2f",
            self.cfgrl_guidance_scale,
            guidance_type,
            "advantage" if self.cfgrl_guidance_scale > 0.0 else "off",
            num_steps,
            rtc_enabled,
            rtc_guidance_mode,
            "all" if self.rtc_guidance_steps is None else self.rtc_guidance_steps,
            rtc_guidance_clip,
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
            return self._predict_eval_with_rtc(
                observation,
                rtc_context=rtc_context,
                cond_observation=cond_observation,
                noise=noise,
                rng=rng,
            )
        return self._predict_eval(
            observation,
            noise=noise,
            rng=rng,
            cond_observation=cond_observation,
            cfg_scale=self.cfgrl_guidance_scale,
        )

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
        re-captures. Capture happens on the first call of each path, which the
        server warmup covers.

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

    def _predict_eval_with_rtc(
        self,
        observation,
        *,
        rtc_context,
        cond_observation,
        noise: torch.Tensor | None,
        rng: torch.Generator | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        from rlinf.models.embodiment.openpi.sampling.rtc_guidance import (
            sample_actions_with_rtc_guidance,
        )

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
