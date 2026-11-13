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
import pdb
import time
from typing import Any, Mapping, Sequence

import numpy as np

from inference.client.async_infer import InferenceWorker
from inference.client.control_loop import run_control_loop
from inference.client.episode_recorder import EpisodeRecorder
from inference.client.gripper_hysteresis import GripperHysteresis
from inference.client.h1_robot import DEFAULT_CAMERA_NAMES, STATE_DIM, H1RobotEnv
from inference.client.keyboard import KeyPressListener
from inference.client.policy_client import WebsocketPolicyClient
from inference.client.rtc_scheduler import RTCScheduler
from inference.protocol import encode_frame

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Sort all objects into the correct bins using both arms"
FIRST_CHUNK_TIMEOUT_SECONDS = 30.0


def build_request(
    observation: Mapping[str, Any],
    camera_names: Sequence[str],
    *,
    prompt: str,
    jpeg_quality: int,
) -> dict[str, Any]:
    """Turn one robot observation into a wire request."""
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


def postprocess_action(action: np.ndarray, gripper: GripperHysteresis) -> np.ndarray:
    """Single hook for robot-side action shaping."""
    return gripper.apply(action)


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
            build_request(
                env.get_observation(),
                camera_names,
                prompt=prompt,
                jpeg_quality=jpeg_quality,
            )
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
        choices=list(DEFAULT_CAMERA_NAMES),
        default=list(DEFAULT_CAMERA_NAMES),
        help=(
            "cameras to send; each name must appear as a value of "
            "server.camera_layout, and the order does not matter because the "
            "server places every view by name"
        ),
    )
    parser.add_argument(
        "--deinit", action="store_true", help="release the robot when the loop finishes"
    )
    parser.add_argument(
        "--no-record",
        dest="record",
        action="store_false",
        default=True,
        help="do not write one hdf5 file per episode",
    )
    parser.add_argument(
        "--save_dir",
        default="rollouts",
        help="root directory of the recorded episodes (default: %(default)s)",
    )
    parser.add_argument(
        "--record_jpeg_quality",
        type=int,
        default=60,
        help="JPEG quality of the recorded frames, matching the dataset",
    )
    parser.add_argument(
        "--task_name",
        default="rollout",
        help="task name stored in the episode metadata",
    )
    parser.add_argument(
        "--task_id", type=int, default=0, help="task id stored in the episode metadata"
    )
    parser.add_argument(
        "--homing_seconds",
        type=float,
        default=2.0,
        help="duration of the return-home move that ends an episode",
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

    reference = None
    if args.init_hdf5:
        logger.info(
            "loading the start pose from %s frame %d",
            args.init_hdf5,
            args.init_frame_idx,
        )
        reference = load_reference_state(args.init_hdf5, args.init_frame_idx)
        logger.info("reference state: %s", np.round(reference, 4).tolist())
        env.move_to_target_joint(reference)
        time.sleep(2.0)  # Let the arms settle before the first observation.
        logger.info("reference pose reached, arms settled")

    if args.warmup_steps > 0:
        warm_up(
            client,
            env,
            args.camera_names,
            prompt=args.prompt,
            jpeg_quality=args.jpeg_quality,
            steps=args.warmup_steps,
        )

    # The robot is in its start pose and the server is warm, so the operator resets
    # the scene here: type ``c`` to hand control to the policy, ``q`` to abort.
    logger.info("pausing to reset the scene before the control loop")
    pdb.set_trace()

    worker = InferenceWorker(client)
    gripper = GripperHysteresis()
    scheduler = RTCScheduler(
        worker,
        action_horizon=action_horizon,
        min_exec_horizon=args.min_exec_horizon,
        delay_buffer_size=args.delay_buffer_size,
        initial_delay_steps=args.initial_delay_steps,
        rtc_enabled=args.rtc,
    )
    keys = KeyPressListener()
    recorder = None
    if args.record:
        if action_dim != STATE_DIM:
            raise ValueError(
                f"recording expects the {STATE_DIM}-D robot action space, the "
                f"server reports {action_dim}"
            )
        recorder = EpisodeRecorder(
            camera_names=args.camera_names,
            task_name=args.task_name,
            task_id=args.task_id,
            control_frequency=args.control_freq,
            jpeg_quality=args.record_jpeg_quality,
        )
    episode = {"index": 1, "steps": 0}

    def record_commanded(observation: Mapping[str, Any], joints: np.ndarray) -> None:
        """Record one step: the observation plus the joint targets sent to the motors."""
        if recorder is None:
            return
        action = np.zeros(STATE_DIM)
        action[: env.motor_count] = joints
        recorder.add_step(
            np.asarray(observation["qpos"]), action, observation["images"]
        )

    def home_target() -> np.ndarray:
        """Joint target of the return-home move, preferring the episode's start pose."""
        if reference is not None:
            logger.info(
                "returning to the reference pose from %s frame %d",
                args.init_hdf5,
                args.init_frame_idx,
            )
            return np.asarray(reference[: env.motor_count], dtype=np.float64)
        logger.warning(
            "no --init_hdf5 reference; returning to the zero pose with the current "
            "lift height"
        )
        return env.init_pose_target()

    def return_home(target: np.ndarray) -> None:
        """Drive the joints to ``target`` at the control rate, recording every step."""
        start = np.asarray(env.get_joint_position(), dtype=np.float64)
        steps = max(1, int(round(args.homing_seconds * args.control_freq)))
        period = 1.0 / args.control_freq
        if recorder is not None:
            recorder.mark_homing()
        logger.info("returning home over %d steps", steps)
        deadline = time.monotonic()
        for step in range(steps):
            alpha = (step + 1) / steps
            joints = (1.0 - alpha) * start + alpha * np.asarray(target, np.float64)
            env.set_joint_action(joints)
            if recorder is not None:
                record_commanded(env.get_observation(), joints)
            deadline += period
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                # A late step is dropped rather than replayed back to back: bursting
                # through the schedule would jump the joint targets and jerk the arms.
                logger.warning("return-home overrun: %.1f ms late", -remaining * 1000)
                deadline = time.monotonic()

    def start_episode() -> bool:
        """Request the first chunk of an episode."""
        scheduler.submit_bootstrap(
            build_request(
                env.get_observation(),
                args.camera_names,
                prompt=args.prompt,
                jpeg_quality=args.jpeg_quality,
            )
        )
        return worker.wait_for_first_result(args.first_chunk_timeout)

    def finish_episode() -> None:
        """Drive home while recording, save the episode, then hand the terminal over."""
        return_home(home_target())
        if recorder is not None:
            try:
                recorder.save(args.save_dir)
            except Exception as error:
                logger.error("saving the episode failed: %s", error)
            recorder.reset()
        keys.stop()
        logger.info(
            "episode %d ended after %d steps; reset the scene, then type ``c`` "
            "to start the next one",
            episode["index"],
            episode["steps"],
        )
        pdb.set_trace()
        keys.start()
        worker.drain()
        scheduler.reset()
        gripper.reset()
        episode["index"] += 1
        episode["steps"] = 0
        if not start_episode():
            raise RuntimeError(
                f"no action chunk for episode {episode['index']} after "
                f"{args.first_chunk_timeout:.0f}s; check the server logs"
            )

    def control_step(step: int) -> bool:
        observation = env.get_observation()
        request = build_request(
            observation,
            args.camera_names,
            prompt=args.prompt,
            jpeg_quality=args.jpeg_quality,
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
        else:
            commanded = postprocess_action(np.copy(action), gripper)[:-2]
            env.set_joint_action(commanded)
            record_commanded(observation, commanded)

            result_age = worker.last_result_age()
            if result_age is not None and result_age > args.watchdog_timeout:
                logger.error("inference stalled for %.1fs", result_age)

        episode["steps"] += 1
        if keys.poll() == "r":
            finish_episode()
            return True
        return False

    try:
        worker.start()
        keys.start()
        if not start_episode():
            raise RuntimeError(
                f"no action chunk after {args.first_chunk_timeout:.0f}s; "
                "check the server logs"
            )
        logger.info("first action chunk received")
        run_control_loop(
            control_step, frequency=args.control_freq, num_steps=args.num_steps
        )
    finally:
        keys.stop()
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
