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

"""Inference service behind the websocket server.

The service owns one RLinf OpenPI eval policy and the per-connection RTC
reference chunk, and is transport-agnostic so it can be driven by a test
directly.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from inference.protocol import (
    RTC_PROTOCOL_VERSION,
    build_metadata,
    validate_rtc_payload,
)
from inference.server.obs_adapter import CAMERA_SLOTS, build_env_obs

logger = logging.getLogger(__name__)

WARMUP_IMAGE_SHAPE = (480, 640, 3)

RTC_WARMUP_STEPS = 3
"""Warmup replans; the first replan captures the RTC graph."""


@dataclasses.dataclass
class RTCSession:
    """Per-connection RTC state.

    ``model_actions`` is the previous chunk in model action space (the padded
    and normalized 32-D layout), which is what ``RTCGuidanceContext`` reads; the
    decoded 23-D chunk sent over the wire cannot serve as the reference.
    ``state`` is the physical state of that request: the chunk's deltas are
    anchored to it, so the next replan re-anchors them to the current state.
    """

    model_actions: torch.Tensor | None = None
    state: torch.Tensor | None = None
    chunk_id: int = 0


class PolicyService:
    """Serve one trained policy over the wire protocol."""

    def __init__(self, cfg: DictConfig) -> None:
        self._action_dim = int(cfg.action_dim)
        self._camera_layout = dict(
            OmegaConf.to_container(cfg.server.camera_layout, resolve=True)
        )
        missing = [slot for slot in CAMERA_SLOTS if slot not in self._camera_layout]
        if missing:
            raise ValueError(
                f"server.camera_layout must name {list(CAMERA_SLOTS)}, "
                f"got {self._camera_layout}"
            )
        self._default_prompt = OmegaConf.select(cfg, "server.default_prompt")
        self._rtc_enabled = bool(OmegaConf.select(cfg, "openpi.rtc_enabled"))
        self._device = str(
            OmegaConf.select(
                cfg,
                "server.device",
                default="cuda" if torch.cuda.is_available() else "cpu",
            )
        )

        self._model = self._build_model(cfg)
        if bool(OmegaConf.select(cfg, "openpi.graph_sampler", default=False)):
            enable_graph = getattr(self._model, "enable_sampler_graph", None)
            if enable_graph is None:
                logger.warning(
                    "openpi.graph_sampler is set but %s has no captured "
                    "sampler; the RTC loop stays eager",
                    type(self._model).__name__,
                )
            elif torch.cuda.is_available():
                enable_graph()
            else:
                logger.warning(
                    "openpi.graph_sampler is set but no CUDA device is "
                    "available; the RTC sampler stays eager"
                )
        self._cfg_scale = float(getattr(self._model, "cfgrl_guidance_scale", 0.0))
        self._guidance_type = getattr(self._model, "guidance_type", "off")
        chunk = getattr(self._model, "action_chunk", None) or self._model.action_horizon
        self._metadata = build_metadata(
            action_horizon=int(chunk),
            action_dim=self._action_dim,
            default_prompt=self._default_prompt,
            rtc_guided=self._rtc_enabled,
        )
        logger.info(
            "policy ready on %s: horizon=%d action_dim=%d rtc=%s "
            "cfgrl_guidance_scale=%.3f guidance_type=%s cond_branch=%s",
            self._device,
            chunk,
            self._action_dim,
            "guided" if self._rtc_enabled else "off",
            self._cfg_scale,
            self._guidance_type,
            "advantage" if self._cfg_scale > 0.0 else "off",
        )

    @property
    def metadata(self) -> dict:
        """Handshake frame sent to every client."""
        return self._metadata

    def new_session(self) -> RTCSession:
        """Create the state of a fresh connection."""
        return RTCSession()

    def warmup(self, steps: int = 1) -> None:
        """Run synthetic observations so the first real request is not the first compile."""
        if steps <= 0:
            return
        observation = {
            "images": {
                camera: np.zeros(WARMUP_IMAGE_SHAPE, dtype=np.uint8)
                for camera in self._camera_layout.values()
            },
            "state": np.zeros(self._action_dim, dtype=np.float32),
            "prompt": self._default_prompt or "warmup",
        }
        try:
            self._run_warmup(observation, steps)
        except Exception:
            if not getattr(self._model, "sampler_graphed", False):
                raise
            logger.exception(
                "warmup failed with the captured sampler; "
                "falling back to the eager loops"
            )
            self._model.disable_sampler_graph()
            self._run_warmup(observation, steps)

    def _run_warmup(self, observation: Mapping[str, Any], steps: int) -> None:
        session = RTCSession()
        for step in range(steps):
            start = time.monotonic()
            self.infer(observation, session)
            logger.info(
                "warmup bootstrap %d/%d done in %.1fs",
                step + 1,
                steps,
                time.monotonic() - start,
            )
        if not self._rtc_enabled:
            return
        # The bootstrap above filled session.model_actions, so these requests
        # take the guided replan path the robot drives in steady state.
        replan = {
            **observation,
            "rtc": {
                "version": RTC_PROTOCOL_VERSION,
                "request_type": "replan",
                "executed_horizon": 1,
                "predicted_delay_steps": 0,
                "chunk_id": 0,
            },
        }
        for step in range(RTC_WARMUP_STEPS):
            start = time.monotonic()
            self.infer(replan, session)
            logger.info(
                "warmup replan %d/%d done in %.1fs",
                step + 1,
                RTC_WARMUP_STEPS,
                time.monotonic() - start,
            )

    def infer(
        self, observation: Mapping[str, Any], session: RTCSession
    ) -> dict[str, Any]:
        """Answer one observation with one action chunk."""
        start = time.monotonic()
        env_obs = build_env_obs(
            observation,
            camera_layout=self._camera_layout,
            default_prompt=self._default_prompt,
            action_dim=self._action_dim,
        )
        prepared = time.monotonic()

        rtc_context = None
        rtc_desc = "off"
        request = observation.get("rtc")
        chunk_id = session.chunk_id + 1
        if request is not None:
            payload = validate_rtc_payload(request)
            chunk_id = payload["chunk_id"] or chunk_id
            rtc_desc = str(payload["request_type"])
            if self._rtc_enabled and payload["request_type"] == "replan":
                if session.model_actions is not None:
                    rtc_context = self._build_rtc_context(session, payload)
                    rtc_desc = (
                        f"replan exec={payload['executed_horizon']} "
                        f"delay={payload['predicted_delay_steps']}"
                    )

        actions, result = self._model.predict_action_batch(
            env_obs=env_obs, mode="eval", rtc_context=rtc_context
        )
        denoised = time.monotonic()
        model_actions = result.get("model_actions")
        if model_actions is not None:
            session.model_actions = model_actions.detach().to("cpu").contiguous()
        session.state = torch.as_tensor(env_obs["states"], dtype=torch.float32).clone()
        session.chunk_id = chunk_id

        chunk = actions.detach().to("cpu", dtype=torch.float32).numpy()[0]
        decoded = time.monotonic()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "chunk %d timings: env_prep=%.3fs denoise=%.3fs decode=%.3fs",
                chunk_id,
                prepared - start,
                denoised - prepared,
                decoded - denoised,
            )
        # CFG reuses one extra forward per denoise step, so log every guided
        # request; plain requests only show up under DEBUG.
        log = logger.info if self._cfg_scale > 0.0 else logger.debug
        log(
            "chunk %d: cfg_scale=%.2f guidance_type=%s rtc=%s actions=%s infer=%.3fs",
            chunk_id,
            self._cfg_scale,
            self._guidance_type,
            rtc_desc,
            tuple(chunk.shape),
            time.monotonic() - start,
        )
        return {
            "actions": chunk,
            "chunk_id": chunk_id,
            "guidance_applied": rtc_context is not None,
            "infer_time": time.monotonic() - start,
        }

    def _build_model(self, cfg: DictConfig) -> torch.nn.Module:
        from rlinf.models import get_model

        model = get_model(cfg)
        if model is None:
            raise ValueError(
                f"rlinf.models.get_model returned None for model_type="
                f"{OmegaConf.select(cfg, 'model_type')!r}"
            )
        return model.to(self._device).eval()

    @staticmethod
    def _build_rtc_context(session: RTCSession, payload: Mapping[str, Any]):
        from rlinf.models.embodiment.openpi.sampling.rtc_guidance import (
            RTCGuidanceContext,
        )

        return RTCGuidanceContext(
            prev_model_actions=session.model_actions,
            executed_horizon=int(payload["executed_horizon"]),
            delay_steps=int(payload["predicted_delay_steps"]),
            prev_state=session.state,
        )
