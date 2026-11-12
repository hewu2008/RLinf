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

"""Open-loop evaluation of a running policy server on a recorded LeRobot dataset.

Every request replays one dataset frame through the same wire path the robot
uses (JPEG camera frames plus the raw joint state), and the returned action
chunk is compared with the recorded chunk that starts at that frame. Requests
carry no ``rtc`` block, so the server takes its plain path and no chunk
bookkeeping is involved.

Run it from the repository root while ``inference/run_server.sh`` is up::

    python -m inference.open_loop_eval \\
        --data-path /data/4T-2/dataset/hewu2008/part_sorting_20261027_subtask \\
        --traj-ids 0 1 2 3 \\
        --max-chunks 240

``scripts/run_open_loop_eval.sh`` wraps this call.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import time
from typing import Any, Mapping, Sequence

import numpy as np

from inference.client.policy_client import WebsocketPolicyClient
from inference.protocol import encode_frame

logger = logging.getLogger(__name__)

DEFAULT_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
IMAGE_KEY = "observation.images.{}"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="policy server host")
    parser.add_argument("--port", type=int, default=55555, help="policy server port")
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=60.0,
        help="seconds to wait for the server before the first request",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="local LeRobot dataset directory or Hugging Face repo id",
    )
    parser.add_argument(
        "--traj-ids",
        type=int,
        nargs="+",
        default=[0],
        help="episodes to replay (default: 0)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=10,
        help="chunks per episode; <=0 replays the whole episode (default: 10)",
    )
    parser.add_argument(
        "--chunk-stride",
        type=int,
        default=None,
        help="frames between two requests; defaults to --action-horizon",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help=(
            "steps of each returned chunk compared with the record, and the "
            "default request stride; defaults to the server action horizon"
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="task instruction; defaults to the sample task, then server default_prompt",
    )
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        help="dataset cameras to send, as observation.images.<name> (default: %(default)s)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality of the camera frames (default: 90)",
    )
    parser.add_argument(
        "--raw-images",
        action="store_true",
        help="send HWC uint8 arrays instead of JPEG frames",
    )
    parser.add_argument(
        "--save-plot-dir",
        default=None,
        help="write one ground-truth-vs-prediction plot per episode here",
    )
    return parser.parse_args(argv)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a dataset value (tensor or array) to numpy."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _frame_to_uint8(image: Any) -> np.ndarray:
    """Convert one dataset image into an HxWx3 uint8 RGB frame."""
    frame = _to_numpy(image)
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected a CHW or HWC image, got shape {frame.shape}")
    if frame.dtype == np.uint8:
        return np.ascontiguousarray(frame)
    scale = 255.0 if float(frame.max(initial=0.0)) <= 1.0 else 1.0
    return np.ascontiguousarray(np.clip(frame * scale, 0, 255).astype(np.uint8))


def build_observation(
    sample: Mapping[str, Any],
    *,
    camera_names: Sequence[str],
    prompt: str | None,
    jpeg_quality: int,
    raw_images: bool,
) -> dict[str, Any]:
    """Turn one dataset sample into the request payload the server expects.

    ``protocol.encode_frame`` takes the BGR frames the robot cameras deliver and
    writes a JPEG whose pixels are RGB, so dataset frames (already RGB) are
    swapped first. ``--raw-images`` skips the swap and the JPEG step.
    """
    images: dict[str, Any] = {}
    for camera in camera_names:
        key = IMAGE_KEY.format(camera)
        if key not in sample:
            raise KeyError(f"dataset sample has no {key!r}; got {sorted(sample)}")
        frame = _frame_to_uint8(sample[key])
        if raw_images:
            images[camera] = frame
        else:
            images[camera] = encode_frame(
                np.ascontiguousarray(frame[:, :, ::-1]), quality=jpeg_quality
            )
    return {
        "images": images,
        "state": _to_numpy(sample[STATE_KEY]).astype(np.float32),
        "prompt": prompt,
    }


def _episode_range(dataset: Any, episode: int) -> tuple[int, int]:
    """Return the ``[start, end)`` frame range of one episode."""
    index = getattr(dataset, "episode_data_index", None)
    if index is not None:
        return int(index["from"][episode]), int(index["to"][episode])
    # Dataset layout v3 dropped episode_data_index in favor of per-episode rows.
    episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
    if episodes is not None:
        row = episodes[episode]
        return int(row["dataset_from_index"]), int(row["dataset_to_index"])
    raise AttributeError(
        "cannot locate episodes in the dataset: neither episode_data_index nor "
        "meta.episodes is available"
    )


def evaluate_episode(
    client: WebsocketPolicyClient,
    dataset: Any,
    episode: int,
    *,
    action_horizon: int,
    action_dim: int,
    chunk_stride: int,
    max_chunks: int,
    prompt: str | None,
    camera_names: Sequence[str],
    jpeg_quality: int,
    raw_images: bool,
    compare_horizon: int | None = None,
) -> dict[str, Any]:
    """Replay one episode and compare every predicted chunk with the record."""
    start, end = _episode_range(dataset, episode)
    gt_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    latencies: list[float] = []

    # A recorded chunk that runs past the episode end is padded with the
    # boundary frame, so only frames with a full window are replayed.
    steps = int(compare_horizon or action_horizon)
    if not 0 < steps <= action_horizon:
        raise ValueError(
            f"compare_horizon must be in [1, {action_horizon}], got {steps}"
        )
    last_full = end - steps
    if last_full < start:
        logger.warning(
            "episode %d has %d frames, shorter than the %d-step horizon; its "
            "recorded chunk is padded and the error includes that tail",
            episode,
            end - start,
            steps,
        )
        request_frames = [start]
    else:
        request_frames = range(start, last_full + 1, chunk_stride)

    used_prompt = None
    for frame in request_frames:
        if max_chunks > 0 and len(gt_chunks) >= max_chunks:
            break
        sample = dataset[frame]
        used_prompt = prompt or sample.get("task") or used_prompt
        observation = build_observation(
            sample,
            camera_names=camera_names,
            prompt=used_prompt,
            jpeg_quality=jpeg_quality,
            raw_images=raw_images,
        )
        state = observation["state"]
        if state.shape != (action_dim,):
            raise ValueError(
                f"the dataset reports a {state.shape} state but the server "
                f"expects ({action_dim},); check --data-path and the checkpoint"
            )

        started = time.monotonic()
        response = client.infer(observation)
        latencies.append(time.monotonic() - started)

        prediction = np.asarray(response["actions"], dtype=np.float32)
        if prediction.ndim == 1:
            prediction = prediction[np.newaxis]
        recorded = _to_numpy(sample[ACTION_KEY]).astype(np.float32)
        if recorded.ndim == 1:
            recorded = recorded[np.newaxis]

        frames = min(len(recorded), len(prediction), steps)
        gt_chunks.append(recorded[:frames])
        pred_chunks.append(prediction[:frames])
        logger.info(
            "episode %d frame %d: infer %.3fs, chunk %s",
            episode,
            frame,
            latencies[-1],
            prediction.shape,
        )

    if not gt_chunks:
        return {
            "episode": episode,
            "mse": float("nan"),
            "mae": float("nan"),
            "latencies": [],
            "gt": np.zeros((0, action_dim), dtype=np.float32),
            "pred": np.zeros((0, action_dim), dtype=np.float32),
            "prompt": used_prompt,
        }

    gt = np.concatenate(gt_chunks, axis=0)
    pred = np.concatenate(pred_chunks, axis=0)
    if gt.shape != pred.shape:
        raise ValueError(
            f"episode {episode}: recorded actions are {gt.shape} but the server "
            f"returned {pred.shape}"
        )
    mse = float(np.mean((gt - pred) ** 2))
    mae = float(np.mean(np.abs(gt - pred)))
    logger.info("episode %d: action MSE %.6f, MAE %.6f", episode, mse, mae)
    return {
        "episode": episode,
        "mse": mse,
        "mae": mae,
        "latencies": latencies,
        "gt": gt,
        "pred": pred,
        "prompt": used_prompt,
    }


def plot_episode(
    gt: np.ndarray, pred: np.ndarray, episode: int, out_dir: pathlib.Path
) -> pathlib.Path | None:
    """Write one plot of recorded versus predicted actions per dimension."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping the plot")
        return None

    columns = 4
    rows = max((gt.shape[1] + columns - 1) // columns, 1)
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.2 * columns, 1.8 * rows), sharex=True, squeeze=False
    )
    for dim in range(gt.shape[1]):
        axis = axes[dim // columns][dim % columns]
        axis.plot(gt[:, dim], label="recorded", linewidth=1.0)
        axis.plot(pred[:, dim], label="predicted", linewidth=1.0)
        axis.set_title(f"dim {dim}", fontsize=8)
        axis.tick_params(labelsize=7)
    for empty in range(gt.shape[1], rows * columns):
        axes[empty // columns][empty % columns].axis("off")
    axes[0][0].legend(fontsize=7)
    figure.suptitle(f"episode {episode}: open-loop actions")
    figure.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode_{episode:04d}.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def load_dataset(data_path: str, action_horizon: int) -> tuple[Any, Any]:
    """Load a local LeRobot dataset with one action chunk per sample."""
    try:
        from lerobot.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import (  # noqa: F401
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )

    path = pathlib.Path(data_path).expanduser()
    if path.is_dir():
        metadata = LeRobotDatasetMetadata(path.name, root=path)
        dataset = LeRobotDataset(
            path.name,
            root=path,
            delta_timestamps={
                ACTION_KEY: [t / metadata.fps for t in range(action_horizon)]
            },
        )
    else:
        metadata = LeRobotDatasetMetadata(data_path)
        dataset = LeRobotDataset(
            data_path,
            delta_timestamps={
                ACTION_KEY: [t / metadata.fps for t in range(action_horizon)]
            },
        )
    logger.info(
        "dataset %s: %d frames, %d episodes, fps %d",
        data_path,
        len(dataset),
        dataset.num_episodes,
        metadata.fps,
    )
    return metadata, dataset


def main(argv: list[str] | None = None) -> None:
    """Replay the requested episodes and report the open-loop error."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )

    client = WebsocketPolicyClient(args.host, args.port, wait_timeout=args.wait_timeout)
    try:
        metadata = client.validate(require_rtc=False)
        action_horizon = int(metadata["action_horizon"])
        action_dim = int(metadata["action_dim"])
        compare_horizon = args.action_horizon or action_horizon
        if compare_horizon > action_horizon:
            raise ValueError(
                f"--action-horizon {compare_horizon} exceeds the server action "
                f"horizon {action_horizon}; the server cannot return that many steps"
            )
        logger.info("server metadata: %s", metadata)

        _, dataset = load_dataset(args.data_path, action_horizon)
        stride = args.chunk_stride or compare_horizon
        logger.info(
            "comparing %d of %d steps per chunk", compare_horizon, action_horizon
        )

        results = []
        for episode in args.traj_ids:
            if not 0 <= episode < dataset.num_episodes:
                logger.warning(
                    "episode %d is outside [0, %d); skipping",
                    episode,
                    dataset.num_episodes,
                )
                continue
            results.append(
                evaluate_episode(
                    client,
                    dataset,
                    episode,
                    action_horizon=action_horizon,
                    action_dim=action_dim,
                    chunk_stride=stride,
                    max_chunks=args.max_chunks,
                    prompt=args.prompt,
                    camera_names=args.camera_names,
                    jpeg_quality=args.jpeg_quality,
                    raw_images=args.raw_images,
                    compare_horizon=compare_horizon,
                )
            )

        completed = [result for result in results if result["latencies"]]
        if not completed:
            logger.error("no chunk was replayed; check --traj-ids and --data-path")
            return

        latencies = [value for result in completed for value in result["latencies"]]
        logger.info(
            "action MSE across %d episodes: %.6f",
            len(completed),
            float(np.mean([result["mse"] for result in completed])),
        )
        logger.info(
            "action MAE across %d episodes: %.6f",
            len(completed),
            float(np.mean([result["mae"] for result in completed])),
        )
        logger.info(
            "server latency over %d requests: mean %.3fs, p95 %.3fs, max %.3fs",
            len(latencies),
            float(np.mean(latencies)),
            float(np.percentile(latencies, 95)),
            float(np.max(latencies)),
        )
        logger.info("prompt used: %r", completed[0]["prompt"])

        if args.save_plot_dir:
            out_dir = pathlib.Path(args.save_plot_dir).expanduser()
            for result in completed:
                path = plot_episode(
                    result["gt"], result["pred"], result["episode"], out_dir
                )
                if path is not None:
                    logger.info("wrote %s", path)
    finally:
        client.close()


if __name__ == "__main__":
    main()
