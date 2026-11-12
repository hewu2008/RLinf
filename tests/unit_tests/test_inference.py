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

"""Remote inference wire protocol, RTC client scheduling, and transport."""

from __future__ import annotations

import numpy as np
import pytest

from inference import protocol
from inference.client.h1_robot import camera_source_candidates, motor_position
from inference.client.rtc_scheduler import RTCScheduler

ACTION_DIM = 23


class _FakeWorker:
    """Stand-in for :class:`inference.client.async_infer.InferenceWorker`."""

    def __init__(self) -> None:
        self.in_flight = False
        self.results: list[tuple[int, np.ndarray]] = []
        self.submissions: list[tuple[int, dict]] = []

    def submit(self, observation, query_t: int) -> None:
        self.submissions.append((query_t, dict(observation)))
        self.in_flight = True

    def poll(self):
        return self.results.pop(0) if self.results else None


def _chunk(horizon: int, *, start: float = 0.0) -> np.ndarray:
    values = np.arange(start, start + horizon, dtype=np.float32)
    return np.repeat(values[:, None], ACTION_DIM, axis=1)


def test_bootstrap_then_replan_skips_the_inflight_prefix():
    worker = _FakeWorker()
    scheduler = RTCScheduler(
        worker,
        action_horizon=10,
        min_exec_horizon=3,
        delay_buffer_size=4,
        initial_delay_steps=1,
    )

    assert scheduler.submit_bootstrap({"prompt": "p"})
    assert worker.submissions[0][1]["rtc"]["request_type"] == "bootstrap"
    assert worker.submissions[0][1]["rtc"]["predicted_delay_steps"] == 1

    # Three control steps pass before the first chunk arrives.
    for _ in range(3):
        assert scheduler.get_action({"prompt": "p"}) is None

    worker.results.append((0, _chunk(10)))
    worker.in_flight = False
    action = scheduler.get_action({"prompt": "p"})

    # The chunk was computed three steps ago, so its first three actions are stale.
    assert action[0] == pytest.approx(3.0)
    assert worker.submissions[-1][1]["rtc"]["request_type"] == "replan"
    assert worker.submissions[-1][1]["rtc"]["predicted_delay_steps"] == 3
    # The server needs the count from the start of the chunk it is guiding.
    assert worker.submissions[-1][1]["rtc"]["executed_horizon"] == 3


def test_queue_is_replaced_by_each_new_chunk():
    worker = _FakeWorker()
    scheduler = RTCScheduler(worker, action_horizon=10, min_exec_horizon=2)

    worker.results.append((0, _chunk(10)))
    actions = [scheduler.get_action({"prompt": "p"})[0] for _ in range(2)]
    assert actions == [pytest.approx(0.0), pytest.approx(1.0)]

    worker.in_flight = False
    worker.results.append((2, _chunk(10, start=100.0)))
    assert scheduler.get_action({"prompt": "p"})[0] == pytest.approx(100.0)


def test_observe_chunk_drops_out_of_order_and_stale_chunks():
    worker = _FakeWorker()
    scheduler = RTCScheduler(worker, action_horizon=4, min_exec_horizon=0)
    chunk = np.zeros((4, ACTION_DIM), dtype=np.float32)

    assert scheduler.observe_chunk(0, chunk) == 0
    assert scheduler.observe_chunk(0, chunk) is None

    for _ in range(10):
        scheduler.get_action({"prompt": "p"})
    assert scheduler.observe_chunk(0, chunk) is None


def test_empty_queue_yields_no_action_and_reset_clears_state():
    worker = _FakeWorker()
    scheduler = RTCScheduler(
        worker, action_horizon=5, min_exec_horizon=5, rtc_enabled=False
    )

    assert scheduler.get_action({"prompt": "p"}) is None
    assert "rtc" not in worker.submissions[0][1]

    scheduler.reset()
    assert scheduler.tracker.max() is None
    assert scheduler.queue.empty()


def test_rtc_payload_is_normalized_and_version_checked():
    payload = protocol.validate_rtc_payload(
        {
            "version": protocol.RTC_PROTOCOL_VERSION,
            "request_type": "replan",
            "executed_horizon": -3,
            "predicted_delay_steps": -1,
            "chunk_id": 7,
        }
    )
    assert payload["executed_horizon"] == 0
    assert payload["predicted_delay_steps"] == 0
    assert payload["chunk_id"] == 7

    with pytest.raises(ValueError, match="version"):
        protocol.validate_rtc_payload({"version": 99, "request_type": "replan"})
    with pytest.raises(ValueError, match="request_type"):
        protocol.validate_rtc_payload(
            {"version": protocol.RTC_PROTOCOL_VERSION, "request_type": "sideways"}
        )


def test_metadata_requires_rtc_when_the_client_enables_it():
    metadata = protocol.build_metadata(
        action_horizon=50, action_dim=ACTION_DIM, default_prompt="p", rtc_guided=True
    )
    assert protocol.validate_metadata(metadata, require_rtc=True) is metadata

    plain = protocol.build_metadata(
        action_horizon=50, action_dim=ACTION_DIM, default_prompt="p", rtc_guided=False
    )
    assert protocol.validate_metadata(plain, require_rtc=False) is plain
    with pytest.raises(RuntimeError, match="RTC"):
        protocol.validate_metadata(plain, require_rtc=True)

    with pytest.raises(RuntimeError, match="version mismatch"):
        protocol.validate_metadata(
            dict(metadata, protocol_version=protocol.PROTOCOL_VERSION + 1),
            require_rtc=False,
        )


def test_build_env_obs_orders_the_three_views_and_the_state():
    from inference.server.obs_adapter import build_env_obs

    layout = {"base": "cam_high", "left_wrist": "cam_l", "right_wrist": "cam_r"}
    frames = {
        "cam_high": np.full((4, 6, 3), 1, dtype=np.uint8),
        "cam_l": np.full((4, 6, 3), 2, dtype=np.uint8),
        "cam_r": np.full((4, 6, 3), 3, dtype=np.uint8),
    }

    env_obs = build_env_obs(
        {"images": frames, "state": np.ones(ACTION_DIM, np.float32)},
        camera_layout=layout,
        default_prompt="fallback",
        action_dim=ACTION_DIM,
    )

    assert env_obs["main_images"].shape == (1, 4, 6, 3)
    assert env_obs["main_images"][0, 0, 0, 0] == 1
    # extra_view_images[0] is the left wrist and [1] the right one.
    assert env_obs["extra_view_images"].shape == (1, 2, 4, 6, 3)
    assert env_obs["extra_view_images"][0, 0, 0, 0, 0] == 2
    assert env_obs["extra_view_images"][0, 1, 0, 0, 0] == 3
    assert env_obs["states"].shape == (1, ACTION_DIM)
    assert env_obs["task_descriptions"] == ["fallback"]

    with pytest.raises(KeyError, match="cam_r"):
        build_env_obs(
            {
                "images": {name: frames[name] for name in ("cam_high", "cam_l")},
                "state": np.ones(ACTION_DIM, np.float32),
            },
            camera_layout=layout,
            default_prompt="fallback",
            action_dim=ACTION_DIM,
        )
    with pytest.raises(ValueError, match="state"):
        build_env_obs(
            {"images": frames, "state": np.ones(4, np.float32)},
            camera_layout=layout,
            default_prompt="fallback",
            action_dim=ACTION_DIM,
        )


def test_open_loop_replays_only_frames_with_a_full_recorded_horizon():
    pytest.importorskip("websockets")
    from inference.open_loop_eval import DEFAULT_CAMERAS, evaluate_episode

    horizon = 4

    class _Dataset:
        """One episode of ``length`` frames and a fixed recorded chunk."""

        def __init__(self, length: int) -> None:
            self.episode_data_index = {"from": np.array([0]), "to": np.array([length])}
            self.requested: list[int] = []

        def __getitem__(self, index: int) -> dict:
            self.requested.append(index)
            sample = {
                "observation.state": np.zeros(ACTION_DIM, np.float32),
                "action": np.zeros((horizon, ACTION_DIM), np.float32),
                "task": "pick",
            }
            for camera in DEFAULT_CAMERAS:
                sample[f"observation.images.{camera}"] = np.zeros((4, 4, 3), np.uint8)
            return sample

    class _Client:
        def infer(self, observation) -> dict:
            return {"actions": np.zeros((horizon, ACTION_DIM), np.float32)}

    def _run(dataset: _Dataset, compare_horizon: int | None = None) -> dict:
        return evaluate_episode(
            _Client(),
            dataset,
            0,
            action_horizon=horizon,
            action_dim=ACTION_DIM,
            chunk_stride=compare_horizon or horizon,
            max_chunks=0,
            prompt=None,
            camera_names=DEFAULT_CAMERAS,
            jpeg_quality=90,
            raw_images=True,
            compare_horizon=compare_horizon,
        )

    dataset = _Dataset(length=10)
    result = _run(dataset)
    # Frames 8 and 9 would need recorded actions past the episode end.
    assert dataset.requested == [0, 4]
    assert result["gt"].shape == (2 * horizon, ACTION_DIM)

    short = _Dataset(length=2)
    _run(short)
    assert short.requested == [0]

    # A shorter window is compared (and stepped) like the JAX eval's
    # --action_horizon, while every request still returns the full chunk.
    window = _Dataset(length=10)
    result = _run(window, compare_horizon=2)
    assert window.requested == [0, 2, 4, 6, 8]
    assert result["gt"].shape == (5 * 2, ACTION_DIM)


def test_h1_client_resolves_camera_aliases_and_sdk_motor_replies():
    class _MotorInformation:
        """Stand-in for the SDK's ``Motor_Information``."""

        def __init__(self, position: float) -> None:
            self.Position_Actual = position

    assert camera_source_candidates("cam_high") == (
        "cam_high",
        "rs/cam_high",
        "rs.cam_high",
        "rs_cam_high",
    )

    info = _MotorInformation(0.25)
    assert motor_position((True, info), 7, _MotorInformation) == 0.25
    assert motor_position(info, 7, _MotorInformation) == 0.25
    with pytest.raises(RuntimeError, match="failed read"):
        motor_position((False, info), 7, _MotorInformation)
    with pytest.raises(TypeError, match="Motor_Information"):
        motor_position(True, 7, _MotorInformation)


def test_pack_unpack_round_trips_arrays_and_bytes():
    pytest.importorskip("msgpack")
    state = np.arange(ACTION_DIM, dtype=np.float32)
    request = {
        "images": {"cam_high": b"\x00\x01\x02"},
        "state": state,
        "rtc": {"version": protocol.RTC_PROTOCOL_VERSION, "executed_horizon": 3},
    }

    decoded = protocol.unpack(protocol.pack(request))

    np.testing.assert_array_equal(decoded["state"], state)
    assert decoded["images"]["cam_high"] == b"\x00\x01\x02"
    assert decoded["rtc"]["executed_horizon"] == 3


def test_websocket_server_serves_chunks_and_forwards_tracebacks():
    pytest.importorskip("msgpack")
    pytest.importorskip("websockets")
    from inference.client.policy_client import WebsocketPolicyClient
    from inference.server.websocket_server import WebsocketPolicyServer

    class _Service:
        metadata = protocol.build_metadata(
            action_horizon=5, action_dim=ACTION_DIM, default_prompt="p", rtc_guided=True
        )

        def __init__(self) -> None:
            self.fail = False
            self.sessions = 0

        def new_session(self) -> dict:
            self.sessions += 1
            return {}

        def infer(self, observation, session) -> dict:
            if self.fail:
                raise RuntimeError("boom")
            actions = np.full((5, ACTION_DIM), observation["state"][0], np.float32)
            return {
                "actions": actions,
                "chunk_id": 1,
                "guidance_applied": False,
                "infer_time": 0.0,
            }

    service = _Service()
    server = WebsocketPolicyServer(service, host="127.0.0.1", port=0)
    server.start_background()

    client = WebsocketPolicyClient(
        "127.0.0.1", server.bound_port, retry_interval=0.01, wait_timeout=5.0
    )
    try:
        assert client.validate(require_rtc=True)["action_horizon"] == 5
        response = client.infer(
            {"images": {}, "state": np.ones(ACTION_DIM, np.float32), "prompt": "p"}
        )
        assert response["actions"].shape == (5, ACTION_DIM)
        assert response["actions"][0, 0] == pytest.approx(1.0)

        service.fail = True
        with pytest.raises(RuntimeError, match="boom"):
            client.infer(
                {"images": {}, "state": np.ones(ACTION_DIM, np.float32), "prompt": "p"}
            )
    finally:
        client.close()
