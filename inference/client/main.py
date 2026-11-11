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

"""Drive the H1 dual-arm robot from a remote policy server.

The control loop stays at a fixed frequency: the inference worker refreshes the
action chunk in the background and the RTC scheduler decides when the next chunk
is requested, so inference latency is absorbed by executing the current chunk.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Sequence

import numpy as np

from inference.client.async_infer import InferenceWorker
from inference.client.control_loop import run_control_loop
from inference.client.h1_robot import H1RobotEnv
from inference.client.policy_client import WebsocketPolicyClient
from inference.client.rtc_scheduler import RTCScheduler
from inference.protocol import encode_frame

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Sort all objects into the correct bins using both arms"
FIRST_CHUNK_TIMEOUT_SECONDS = 30.0


def build_request(
    env,
    camera_names: Sequence[str],
    *,
    prompt: str,
    jpeg_quality: int,
) -> dict[str, Any]:
    """Read one robot observation and turn it into a wire request."""
    observation = env.get_observation()
    images = observation["images"]
    missing = [name for name in camera_names if name not in images]
    if missing:
        raise KeyError(
            f"cameras {missing} are configured for the policy but have no frame; "
            f"available: {sorted(images)}"
        )
    return {
        "images": {
            name: encode_frame(images[name], quality=jpeg_quality)
            for name in camera_names
        },
        "state": np.asarray(observation["qpos"], dtype=np.float32),
        "prompt": prompt,
    }


def load_reference_state(path: str, frame_index: int) -> np.ndarray:
    """Read one 23-D joint state out of a recorded episode.

    Used to place the robot in the pose the demonstration starts from before the
    policy takes over.
    """
    import h5py

    with h5py.File(path, "r") as episode:
        arm = episode["/observation/state/arm/position"][:]
        effector = episode["/observation/state/effector/position"][:]
        waist = episode["/observation/state/waist/position"][:]
        head = episode["/observation/state/head/position"][:]
        base = episode["/observation/state/base/velocity"][:]
    states = np.concatenate(
        [
            arm[:, :7],
            effector[:, :-1],
            arm[:, 7:],
            effector[:, -1:],
            waist,
            head,
            base,
        ],
        axis=1,
    )
    if not 0 <= frame_index < len(states):
        raise IndexError(
            f"init_frame_idx {frame_index} is outside {path} with {len(states)} frames"
        )
    return np.asarray(states[frame_index], dtype=np.float64)


def postprocess_action(action: np.ndarray) -> np.ndarray:
    """Single hook for robot-side action shaping (identity in this version)."""
    return action


def warm_up(
    client: WebsocketPolicyClient,
    env,
    camera_names: Sequence[str],
    *,
    prompt: str,
    jpeg_quality: int,
    steps: int,
) -> None:
    """Prime the server so the first real request is not its first inference."""
    for step in range(steps):
        start = time.monotonic()
        client.infer(
            build_request(env, camera_names, prompt=prompt, jpeg_quality=jpeg_quality)
        )
        logger.info(
            "warmup %d/%d done in %.2fs", step + 1, steps, time.monotonic() - start
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="policy server host")
    parser.add_argument("--port", type=int, default=55555, help="policy server port")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="language instruction")
    parser.add_argument(
        "--num_steps", type=int, default=100000, help="control steps to run"
    )
    parser.add_argument(
        "--control_freq", type=float, default=30.0, help="control loop frequency in Hz"
    )
    parser.add_argument(
        "--min_exec_horizon",
        type=int,
        default=15,
        help="control steps to execute before requesting the next chunk (RTC only)",
    )
    parser.add_argument(
        "--initial_delay_steps",
        type=int,
        default=1,
        help="delay assumed for the first replan, before any latency is measured",
    )
    parser.add_argument(
        "--delay_buffer_size",
        type=int,
        default=8,
        help="sliding window of measured delays used to predict the next one",
    )
    parser.add_argument("--warmup_steps", type=int, default=10, help="warmup requests")
    parser.add_argument(
        "--watchdog_timeout",
        type=float,
        default=2.0,
        help="seconds without an inference result before logging a stall",
    )
    parser.add_argument(
        "--first_chunk_timeout",
        type=float,
        default=FIRST_CHUNK_TIMEOUT_SECONDS,
        help="seconds to wait for the first action chunk",
    )
    parser.add_argument(
        "--jpeg_quality", type=int, default=90, help="JPEG quality of the camera frames"
    )
    parser.add_argument(
        "--no-rtc",
        dest="rtc",
        action="store_false",
        default=True,
        help="request whole chunks without RTC guidance",
    )
    parser.add_argument(
        "--init_hdf5", default=None, help="episode used for the start pose"
    )
    parser.add_argument(
        "--init_frame_idx",
        type=int,
        default=0,
        help="frame of --init_hdf5 to start from",
    )
    parser.add_argument(
        "--sdk_dir",
        default=None,
        help=(
            "directory holding lib_h1_sdk_python and camera_client "
            "(default: $H1_SDK_DIR, then client/lib)"
        ),
    )
    parser.add_argument(
        "--grpc_target", default="localhost:50051", help="camera service gRPC target"
    )
    parser.add_argument(
        "--camera_names",
        nargs="+",
        choices=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        default=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        help="camera names, in the order the policy expects",
    )
    parser.add_argument(
        "--deinit", action="store_true", help="release the robot when the loop finishes"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the robot under policy control until interrupted."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )

    env = H1RobotEnv(
        camera_names=args.camera_names,
        sdk_dir=args.sdk_dir,
        grpc_target=args.grpc_target,
    )
    env.move_to_init_pose()

    client = WebsocketPolicyClient(args.host, args.port)
    metadata = client.validate(require_rtc=args.rtc)
    action_dim = int(metadata["action_dim"])
    action_horizon = int(metadata["action_horizon"])
    logger.info("server metadata: %s", metadata)

    if args.init_hdf5:
        reference = load_reference_state(args.init_hdf5, args.init_frame_idx)
        env.move_to_target_joint(reference)
        time.sleep(2.0)  # Let the arms settle before the first observation.

    if args.warmup_steps > 0:
        warm_up(
            client,
            env,
            args.camera_names,
            prompt=args.prompt,
            jpeg_quality=args.jpeg_quality,
            steps=args.warmup_steps,
        )

    worker = InferenceWorker(client)
    scheduler = RTCScheduler(
        worker,
        action_horizon=action_horizon,
        min_exec_horizon=args.min_exec_horizon,
        delay_buffer_size=args.delay_buffer_size,
        initial_delay_steps=args.initial_delay_steps,
        rtc_enabled=args.rtc,
    )

    def control_step(step: int) -> bool:
        request = build_request(
            env, args.camera_names, prompt=args.prompt, jpeg_quality=args.jpeg_quality
        )
        state = np.asarray(request["state"])
        if state.shape[0] != action_dim:
            raise ValueError(
                f"the robot reports a {state.shape[0]}-D state but the server "
                f"expects {action_dim}"
            )

        action = scheduler.get_action(request)
        if action is None:
            logger.warning("step %d: no action queued, holding the current pose", step)
            env.set_joint_action(state[: env.motor_count])
            return False

        env.set_joint_action(postprocess_action(np.copy(action))[:-2])

        result_age = worker.last_result_age()
        if result_age is not None and result_age > args.watchdog_timeout:
            logger.error("inference stalled for %.1fs", result_age)
        return False

    try:
        worker.start()
        scheduler.submit_bootstrap(
            build_request(
                env,
                args.camera_names,
                prompt=args.prompt,
                jpeg_quality=args.jpeg_quality,
            )
        )
        if not worker.wait_for_first_result(args.first_chunk_timeout):
            raise RuntimeError(
                f"no action chunk after {args.first_chunk_timeout:.0f}s; "
                "check the server logs"
            )
        logger.info("first action chunk received")
        run_control_loop(
            control_step, frequency=args.control_freq, num_steps=args.num_steps
        )
    finally:
        worker.stop()
        stats = worker.stats()
        if stats is not None:
            count, mean_time, max_time = stats
            logger.info(
                "inference stats: %d calls, mean %.0f ms, max %.0f ms",
                count,
                mean_time * 1000,
                max_time * 1000,
            )
        client.close()
        if args.deinit:
            env.move_to_deinit_pose()

    logger.info("inference finished")


if __name__ == "__main__":
    main()
