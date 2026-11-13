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

"""RTC overlap guidance for the vendored Pi0 eval sampler.

``RTCGuidanceContext`` and ``build_rtc_target_and_mask`` are the shared
protocol used by both ``openpi`` and the leftover OpenPI PyTorch
sampler; the Euler loop below is the vendored-Pi0 implementation of
Algorithm 1 from "Real-Time Execution of Action Chunking Flow Policies".

Two guidance modes are available, both servable from a captured CUDA graph:

``approx``
    Identity Jacobian (``correction = err``), no backward pass. The residual is
    applied as is; this is not bit-identical to the loop from before the exact
    mode existed, which measured the residual against ``x_t_mean`` and did not
    scale it by ``dt``.
``exact``
    The PiGDM vector-Jacobian product ``(I - t d v / d x)^T err`` of Eq. (pigdm1),
    computed with autograd through the suffix forward. A graph records the
    backward kernels as well, and the replay matches the eager loop bit for bit
    (measured on the deployed Pi0.5); the cost is the backward arithmetic.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass

import torch

from rlinf.models.embodiment.openpi.modules.model import preprocess_observation
from rlinf.models.embodiment.openpi.sampling import rl_sampler


@dataclass
class RTCGuidanceContext:
    """RTC context passed from rollout runtime to the OpenPI sampler."""

    prev_model_actions: torch.Tensor | None = None  # [B, H, A_model]
    executed_horizon: int = 0
    delay_steps: int = 0
    # Physical state of the request that produced ``prev_model_actions``. Set
    # when the leftover must be re-anchored, see :func:`reanchor_prev_actions`.
    prev_state: torch.Tensor | None = None  # [B, S_env]

    def get_prev_remaining(self) -> torch.Tensor | None:
        if self.prev_model_actions is None:
            return None
        executed_horizon = int(max(self.executed_horizon, 0))
        if executed_horizon >= self.prev_model_actions.shape[1]:
            return None
        return self.prev_model_actions[:, executed_horizon:, :]


def build_rtc_target_and_mask(
    prev_remaining: torch.Tensor | None,
    horizon: int,
    action_dim: int,
    delay_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build overlap target and soft mask in model action space."""
    batch_size = 1 if prev_remaining is None else prev_remaining.shape[0]
    target = torch.zeros((batch_size, horizon, action_dim), device=device, dtype=dtype)
    mask = torch.zeros((batch_size, horizon, 1), device=device, dtype=dtype)

    if prev_remaining is None or prev_remaining.numel() == 0:
        return target, mask

    overlap = min(prev_remaining.shape[1], horizon)
    if overlap <= 0:
        return target, mask

    target[:, :overlap] = prev_remaining[:, :overlap].to(device=device, dtype=dtype)
    hard_end = min(max(int(delay_steps), 0), overlap)
    if hard_end > 0:
        # Actions that are likely to be executed before the new chunk arrives
        # are treated as hard constraints.
        mask[:, :hard_end, 0] = 1.0
    if hard_end < overlap:
        # Later overlap positions are softly guided so the new plan can still
        # adapt to the latest camera observation.
        i = torch.arange(hard_end, overlap, device=device, dtype=dtype)
        denom = max(float(overlap - hard_end + 1), 1.0)
        c_i = (overlap - i) / denom
        soft = c_i * (torch.expm1(c_i) / (math.e - 1.0))
        mask[:, hard_end:overlap, 0] = soft
    return target, mask


def rtc_guidance_scale(paper_tau: torch.Tensor, guidance_clip: float) -> torch.Tensor:
    """PiGDM guidance weight ``min(clip, (tau^2 + (1-tau)^2) / (tau (1-tau)))``.

    ``paper_tau`` is the denoising progress ``tau = 1 - t`` of the paper, so the
    sampler's ``t = 1 - paper_tau`` and both ends of the loop clamp at ``clip``.
    """
    complement = 1.0 - paper_tau
    numerator = paper_tau * paper_tau + complement * complement
    denominator = torch.clamp(paper_tau * complement, min=1e-6)
    return torch.clamp(numerator / denominator, max=float(guidance_clip))


@dataclass(frozen=True)
class ActionAnchor:
    """Per-dimension constants that move a leftover chunk to a new state anchor."""

    delta_mask: torch.Tensor  # [A_model] bool, True where the action is a delta
    action_scale: torch.Tensor  # [A_env] float, normalizer applied to deltas
    state_mean: torch.Tensor  # [A_env] float
    state_scale: torch.Tensor  # [A_env] float


def reanchor_prev_actions(
    prev_actions: torch.Tensor,
    prev_state: torch.Tensor,
    cur_state: torch.Tensor,
    anchor: ActionAnchor,
) -> torch.Tensor:
    """Re-anchor a leftover chunk's deltas from ``prev_state`` to ``cur_state``.

    ``DeltaActions`` encodes a whole chunk as ``actions - state_at_request``, and
    the encoder normalizes per dimension. A leftover produced at an earlier state
    therefore describes offsets from that state, while the deltas of a chunk
    sampled now are anchored to the current one; guiding towards it unchanged
    pulls the arm by ``cur_state - prev_state``. Adding that difference back, in
    normalized units and on the delta dims only, gives the target the input
    pipeline would have produced for the same absolute actions.

    ``prev_state`` is physical, ``cur_state`` is already normalized by the input
    pipeline. Dimensions past the environment state, or without a delta, keep
    their original value.
    """
    dims = min(prev_state.shape[-1], anchor.action_scale.shape[-1])
    state_scale = anchor.state_scale[:dims].to(cur_state)
    state_mean = anchor.state_mean[:dims].to(cur_state)
    action_scale = anchor.action_scale[:dims].to(cur_state)
    delta_mask = anchor.delta_mask[:dims].to(cur_state.device)
    cur_phys = cur_state[..., :dims] * state_scale + state_mean
    shift = (prev_state[..., :dims].to(cur_phys) - cur_phys) / action_scale
    shift = torch.where(delta_mask, shift, torch.zeros_like(shift))
    reanchored = prev_actions.clone()
    reanchored[..., :dims] += shift.unsqueeze(-2)
    return reanchored


def _iter_transforms(transform_fn):
    """Yield the leaves of a composed openpi transform pipeline, depth first."""
    children = getattr(transform_fn, "transforms", None)
    if children is None:
        yield transform_fn
        return
    for child in children:
        yield from _iter_transforms(child)


def action_anchor_from_transforms(input_fn, output_fn) -> ActionAnchor | None:
    """Extract the delta mask and normalizers from the openpi pipelines.

    Returns ``None`` when the pipeline trains absolute actions, has no
    ``DeltaActions``, or carries no norm stats; the caller then leaves the
    leftover anchored to the previous request, exactly as before.
    """
    from openpi import transforms as openpi_transforms

    delta = None
    unnormalize = None
    for transform in _iter_transforms(input_fn):
        if isinstance(transform, openpi_transforms.DeltaActions):
            delta = transform
    for transform in _iter_transforms(output_fn):
        if isinstance(transform, openpi_transforms.Unnormalize):
            unnormalize = transform
    if delta is None or delta.mask is None or unnormalize is None:
        return None
    stats = unnormalize.norm_stats
    if not stats or "actions" not in stats or "state" not in stats:
        return None

    def _scale_mean(key):
        entry = stats[key]
        if unnormalize.use_quantiles:
            low = torch.as_tensor(entry.q01).float()
            high = torch.as_tensor(entry.q99).float()
            scale = (high - low + 1e-6) / 2.0
            mean = low + scale
        else:
            scale = torch.as_tensor(entry.std).float() + 1e-6
            mean = torch.as_tensor(entry.mean).float()
        return scale, mean

    action_scale, _ = _scale_mean("actions")
    state_scale, state_mean = _scale_mean("state")
    action_dims = action_scale.shape[0]
    delta_mask = torch.zeros(action_dims, dtype=torch.bool)
    mask_dims = min(len(delta.mask), action_dims)
    delta_mask[:mask_dims] = torch.as_tensor(delta.mask, dtype=torch.bool)[:mask_dims]
    return ActionAnchor(
        delta_mask=delta_mask,
        action_scale=action_scale,
        state_mean=state_mean,
        state_scale=state_scale,
    )


def _branch_velocity(
    pi0_model,
    observation,
    x_batched: torch.Tensor,
    t_tensor: torch.Tensor,
    kv_cache: tuple,
    prefix_mask: torch.Tensor,
    batch_size: int,
    cfg_scale: float,
) -> torch.Tensor:
    """Suffix velocity of ``x_batched``, mixing the CFG branches when present."""
    suffix_act = pi0_model.run_suffix(
        observation, x_batched, t_tensor, kv_cache, prefix_mask
    )
    v_t = pi0_model.velocity_from_suffix(suffix_act).to(torch.float32)
    if x_batched.shape[0] == 2 * batch_size:
        v_t = (1.0 - cfg_scale) * v_t[:batch_size] + cfg_scale * v_t[batch_size:]
    return v_t


def rtc_denoise_loop(
    pi0_model,
    observation,
    cond_observation,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_steps: int,
    guidance_clip: float,
    cfg_scale: float,
    guidance_mode: str = "approx",
    guidance_steps: int | None = None,
) -> torch.Tensor:
    """Run the guided Euler loop and return the sampled chunk.

    Split out of :func:`sample_actions_with_rtc_guidance` so the loop can be
    replayed from a CUDA graph (:class:`LoopGraphSampler`): every input has a
    shape that depends only on the model config (batch, ``action_horizon``,
    ``action_dim``, prefix length), nothing synchronizes with the host, and
    nothing branches on a tensor value. Keep it that way - a host copy, a
    ``.item()`` or a tensor-dependent ``if`` invalidates the capture.
    ``guidance_mode="exact"`` keeps that contract: the autograd pass runs at
    capture time and its backward kernels are recorded like any other,
    replaying without the CPU-side graph walk.

    Args:
        pi0_model: the Pi0 / Pi0Eval model providing the prefix cache and suffix
            forward passes.
        observation: prepared observation of the plain branch.
        cond_observation: prepared advantage-conditioned observation, or
            ``None`` to sample the plain branch alone.
        timesteps: ``(num_steps + 1,)`` float tensor from
            ``rl_sampler.get_timesteps``.
        noise: ``(B, action_horizon, action_dim)`` initial sample.
        target: overlap target, all zeros when there is no previous chunk.
        mask: overlap mask, all zeros when there is no previous chunk.
        num_steps: number of Euler steps.
        guidance_clip: upper clamp of the overlap guidance scale (paper ``beta``).
        cfg_scale: classifier-free guidance weight, applied as
            ``(1 - cfg_scale) * v + cfg_scale * v_cond``.
        guidance_mode: ``"approx"`` corrects with the residual itself (identity
            Jacobian, no backward pass); ``"exact"`` corrects with the PiGDM
            vector-Jacobian product of Eq. (pigdm1), one backward per step.
        guidance_steps: number of leading denoise steps served with the exact
            correction; the rest fall back to ``"approx"``. ``None`` applies the
            selected mode to every step.

    Each step follows Algorithm 1: the overlap residual is measured against the
    one-step clean estimate ``x1 = x - t v`` and then pulled onto ``target`` with
    weight ``w``, ``x <- x - dt v + dt w correction``.

    With a conditioned branch the suffix passes run over a batch of ``2B``, so
    each step costs one pass instead of two (measured 1.7x per step, 17% of a
    whole request); the batch axis only ever swaps kernel shapes, the mixed
    velocity is still assembled per branch afterwards.
    """
    if guidance_mode not in ("approx", "exact"):
        raise ValueError(
            f"guidance_mode must be 'approx' or 'exact', got {guidance_mode!r}"
        )
    B = observation.state.shape[0]
    device = observation.state.device

    prefix_out, prefix_mask, kv_cache = pi0_model.build_prefix_cache(observation)
    del prefix_out
    branches = 1
    if cond_observation is not None:
        cond_prefix_out, cond_prefix_mask, cond_kv_cache = pi0_model.build_prefix_cache(
            cond_observation
        )
        del cond_prefix_out
        observation, prefix_mask, kv_cache = stack_cfg_branches(
            observation,
            cond_observation,
            prefix_mask,
            cond_prefix_mask,
            kv_cache,
            cond_kv_cache,
        )
        branches = 2

    exact_steps = num_steps if guidance_steps is None else int(guidance_steps)
    x_t = noise
    # ``torch.zeros`` / ``torch.full`` fill on the device; ``torch.tensor(...,
    # device=...)`` would copy host memory and abort a CUDA graph capture.
    paper_tau = torch.zeros((), device=device, dtype=noise.dtype)
    dt = torch.full((), 1.0 / num_steps, device=device, dtype=noise.dtype)
    idx_step = torch.empty(B, device=device, dtype=torch.long)

    for idx in range(num_steps):
        t_tensor = timesteps[idx].repeat(B * branches)
        t_step = timesteps[idx]
        if guidance_mode == "exact" and idx < exact_steps:
            # Eq. (pigdm1): differentiate ``x1 = x - t v(x)`` so the correction
            # is ``(I - t dv/dx)^T err``. Only x_t is a leaf, so the backward
            # pass covers the suffix activations alone.
            with torch.enable_grad():
                x_in = x_t.detach().requires_grad_(True)
                v_t = _branch_velocity(
                    pi0_model,
                    observation,
                    x_in.repeat(branches, 1, 1),
                    t_tensor,
                    kv_cache,
                    prefix_mask,
                    B,
                    cfg_scale,
                )
                x1_t = x_in - t_step * v_t
                err = (target - x1_t) * mask
                (correction,) = torch.autograd.grad(x1_t, x_in, grad_outputs=err)
            v_t = v_t.detach()
        else:
            v_t = _branch_velocity(
                pi0_model,
                observation,
                x_t.repeat(branches, 1, 1),
                t_tensor,
                kv_cache,
                prefix_mask,
                B,
                cfg_scale,
            )
            x1_t = x_t - t_step * v_t
            correction = (target - x1_t) * mask

        idx_step.fill_(idx)
        x_t_mean, _ = rl_sampler.sample_mean_var(
            x_t.to(torch.float32),
            v_t,
            idx_step,
            noise_method="flow_ode",
            noise_level=0.0,
            num_steps=num_steps,
            timesteps=timesteps,
        )
        # Aligning in the flow mean with ``dt`` matches ``x <- x + h (v + w g)``
        # in the paper's tau = 0 -> 1 parameterisation.
        x_t = x_t_mean + dt * rtc_guidance_scale(paper_tau, guidance_clip) * correction
        paper_tau = paper_tau + dt
    return x_t


def stack_cfg_branches(
    observation,
    cond_observation,
    prefix_mask: torch.Tensor,
    cond_prefix_mask: torch.Tensor,
    kv_cache: tuple,
    cond_kv_cache: tuple,
) -> tuple:
    """Stack the two CFG branches on the batch axis for a single suffix pass.

    Shared by both denoise loops. Both branches share ``x_t``, so one pass with
    batch ``2B`` computes exactly what two passes with batch ``B`` did, per
    sample: each sample keeps its own prefix KV and its own mask. What changes is
    the launch count and the ``M`` of every skinny GEMM, which is where a suffix
    step spends its time (measured 1.7x on a 50-token step, 17% of a request on
    the deployed Pi0.5). Returns the batched observation, prefix mask and cache,
    plain branch first.

    ``run_suffix`` reads only ``state`` from the observation -- the images and
    the prompt reach the model exclusively through the prefix cache -- so the
    branch-specific media stay on ``cond_observation``.
    """
    if observation.state.shape[1:] != cond_observation.state.shape[1:]:
        raise ValueError(
            "CFG branches must share the state layout to be batched, got "
            f"{tuple(observation.state.shape)} and "
            f"{tuple(cond_observation.state.shape)}"
        )
    if kv_cache[0][0].shape[1] != cond_kv_cache[0][0].shape[1]:
        raise ValueError(
            "CFG branches must have equally long prefixes to be batched, got "
            f"{kv_cache[0][0].shape[1]} and {cond_kv_cache[0][0].shape[1]} tokens"
        )

    batched_observation = dataclasses.replace(
        observation,
        state=torch.cat([observation.state, cond_observation.state], dim=0),
    )
    batched_mask = torch.cat([prefix_mask, cond_prefix_mask], dim=0)
    batched_cache = tuple(
        (torch.cat([k, cond_k], dim=0), torch.cat([v, cond_v], dim=0))
        for (k, v), (cond_k, cond_v) in zip(kv_cache, cond_kv_cache, strict=True)
    )
    return batched_observation, batched_mask, batched_cache


def _clone_tensors(node):
    """Clone every tensor of a nested structure, leaving everything else as is."""
    if torch.is_tensor(node):
        return node.clone()
    if dataclasses.is_dataclass(node):
        return dataclasses.replace(
            node,
            **{
                field.name: _clone_tensors(getattr(node, field.name))
                for field in dataclasses.fields(node)
            },
        )
    if isinstance(node, dict):
        return {key: _clone_tensors(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_clone_tensors(value) for value in node]
    if isinstance(node, tuple):
        return tuple(_clone_tensors(value) for value in node)
    return node


def _copy_tensors_into(dst, src) -> bool:
    """Copy ``src`` into the static ``dst`` in place; ``False`` on a layout change."""
    if torch.is_tensor(dst):
        if not torch.is_tensor(src) or dst.shape != src.shape or dst.dtype != src.dtype:
            return False
        dst.copy_(src)
        return True
    if dataclasses.is_dataclass(dst):
        if not dataclasses.is_dataclass(src) or type(dst) is not type(src):
            return False
        return all(
            _copy_tensors_into(getattr(dst, field.name), getattr(src, field.name))
            for field in dataclasses.fields(dst)
        )
    if isinstance(dst, dict):
        if not isinstance(src, dict) or dst.keys() != src.keys():
            return False
        return all(_copy_tensors_into(dst[key], src[key]) for key in dst)
    if isinstance(dst, (list, tuple)):
        if not isinstance(src, type(dst)) or len(dst) != len(src):
            return False
        return all(
            _copy_tensors_into(dst_item, src_item)
            for dst_item, src_item in zip(dst, src, strict=True)
        )
    if dst is None:
        return src is None
    return False


class LoopGraphSampler:
    """Serve a denoise loop from one captured CUDA graph.

    The graph records the eager kernels, so a replay returns the same action
    bits as the eager loop; what disappears is the per-step launch overhead,
    which dominates a batch-of-one chunk. The recorded sequence includes the
    backward kernels of an autograd-guided loop (``guidance_mode="exact"``),
    so that mode replays too instead of falling back to eager. Inputs are
    copied into static buffers that are kept alive for the lifetime of the
    sampler, so the caller may keep the returned chunk without observing a
    later request overwrite it.

    ``torch.compile`` would also remove the launches but reorders and refolds
    the arithmetic (measured up to 1.7e-2 absolute drift on a 50-step chunk),
    so it cannot serve a deployment that requires the eager numbers.

    The shapes are pinned by the model config and the camera layout. A call
    whose tensors differ from the captured ones re-captures, and the capture
    must run on CUDA.

    Any ``*args`` are the loop's tensor arguments; they are passed straight to
    ``denoise_fn`` after the model, and keyword arguments are fixed at capture
    time (they are Python scalars such as ``num_steps``).
    """

    def __init__(self, denoise_fn) -> None:
        self._denoise_fn = denoise_fn
        self._graph = None
        self._buffers = None
        self._output = None
        self._kwargs = None

    def __call__(self, pi0_model, *inputs, **kwargs) -> torch.Tensor:
        if self._buffers is None or not _copy_tensors_into(self._buffers, inputs):
            self._capture(pi0_model, inputs, kwargs)
        self._graph.replay()
        return self._output.clone()

    def _capture(self, pi0_model, inputs: tuple, kwargs: dict) -> None:
        self._kwargs = dict(kwargs)
        self._buffers = tuple(_clone_tensors(item) for item in inputs)
        run = functools.partial(
            self._denoise_fn, pi0_model, *self._buffers, **self._kwargs
        )
        for _ in range(2):
            run()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._output = run()
        torch.cuda.synchronize()
        self._graph = graph


@torch.no_grad()
def sample_actions_with_rtc_guidance(
    pi0_model,
    observation,
    rtc_context: RTCGuidanceContext,
    *,
    num_steps: int,
    noise: torch.Tensor | None = None,
    rng: torch.Generator | None = None,
    guidance_clip: float = 5.0,
    cond_observation=None,
    cfg_scale: float = 0.0,
    guidance_mode: str = "approx",
    guidance_steps: int | None = None,
    denoise_fn=None,
) -> torch.Tensor:
    """Euler ODE sampling with overlap guidance against the previous chunk.

    ``cond_observation`` enables classifier-free guidance: each step's velocity
    becomes ``(1 - cfg_scale) * v + cfg_scale * v_cond``, with the overlap term
    applied to the mixed velocity. Both branches are sampled by one suffix pass
    over a doubled batch (:func:`stack_cfg_branches`), which is where a suffix
    step spends its time; the per-sample math is unchanged.

    ``guidance_mode`` and ``guidance_steps`` select the overlap correction, see
    :func:`rtc_denoise_loop`.

    ``denoise_fn`` replaces :func:`rtc_denoise_loop`, which is how a caller
    serves a compiled loop; it receives the same positional arguments and the
    same keyword arguments.
    """
    observation = preprocess_observation(observation, train=False)
    if cond_observation is not None:
        cond_observation = preprocess_observation(cond_observation, train=False)

    B = observation.state.shape[0]
    device = observation.state.device
    if noise is None:
        noise = torch.randn(
            B,
            pi0_model.action_horizon,
            pi0_model.action_dim,
            device=device,
            dtype=torch.float32,
            generator=rng,
        )
    else:
        noise = noise.to(device=device, dtype=torch.float32)

    timesteps = rl_sampler.get_timesteps(num_steps, device).to(noise.dtype)

    prev_remaining = (
        rtc_context.get_prev_remaining() if rtc_context is not None else None
    )
    target, mask = build_rtc_target_and_mask(
        prev_remaining=prev_remaining,
        horizon=pi0_model.action_horizon,
        action_dim=pi0_model.action_dim,
        delay_steps=0 if rtc_context is None else rtc_context.delay_steps,
        device=device,
        dtype=noise.dtype,
    )
    if prev_remaining is None:
        # First chunk of an episode: the overlap term is zero, so the exact
        # backward passes would only reproduce the plain Euler loop.
        guidance_mode = "approx"

    run_denoise = rtc_denoise_loop if denoise_fn is None else denoise_fn
    return run_denoise(
        pi0_model,
        observation,
        cond_observation,
        timesteps,
        noise,
        target,
        mask,
        num_steps=num_steps,
        guidance_clip=guidance_clip,
        cfg_scale=cfg_scale,
        guidance_mode=guidance_mode,
        guidance_steps=guidance_steps,
    )
