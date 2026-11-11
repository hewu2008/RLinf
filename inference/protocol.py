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

"""Wire protocol shared by the policy server and the robot-side client.

One websocket connection carries one msgpack frame per message: the server sends
its metadata first and then answers every observation with one action chunk.

This module is imported on the robot machine, which has neither torch nor RLinf,
so it may only depend on numpy (msgpack and OpenCV are imported by the functions
that need them).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

PROTOCOL_VERSION = 1
"""Version of the frame layout itself (metadata, request, response)."""

RTC_PROTOCOL_VERSION = 1
"""Version of the RTC fields inside a request."""

REQUEST_TYPES = ("bootstrap", "replan")
"""``bootstrap`` asks for a chunk without guidance, ``replan`` with it."""

_NDARRAY_TAG = "__ndarray__"


def _encode_obj(obj: Any) -> Any:
    """msgpack ``default`` hook: numpy arrays and scalars."""
    if isinstance(obj, np.ndarray):
        return {
            _NDARRAY_TAG: True,
            "dtype": obj.dtype.str,
            "shape": list(obj.shape),
            "data": obj.tobytes(),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot serialize {type(obj)!r} for the inference protocol")


def _decode_obj(obj: dict) -> Any:
    """msgpack ``object_hook``: restore arrays written by :func:`_encode_obj`."""
    if not obj.get(_NDARRAY_TAG):
        return obj
    array = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
    return array.reshape(obj["shape"]).copy()


def pack(payload: Mapping[str, Any]) -> bytes:
    """Serialize one frame."""
    import msgpack

    return msgpack.packb(dict(payload), default=_encode_obj, use_bin_type=True)


def unpack(frame: bytes) -> dict:
    """Deserialize one frame."""
    import msgpack

    return msgpack.unpackb(frame, object_hook=_decode_obj, raw=False)


def encode_frame(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode one camera frame into JPEG bytes whose decoded pixels are RGB.

    ``cv2.imencode`` keeps the array's channel order, so the BGR-to-RGB swap
    happens here and :func:`decode_frame` returns RGB pixels directly. The
    OpenPI transforms expect RGB.
    """
    import cv2

    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 camera frame, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"expected a uint8 camera frame, got {frame.dtype}")
    encoded, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not encoded:
        raise RuntimeError(f"JPEG encoding failed for a frame of shape {frame.shape}")
    return buffer.tobytes()


def decode_frame(image: Any) -> np.ndarray:
    """Decode a wire image into an RGB HxWx3 uint8 array.

    Uncompressed arrays pass through, so a local caller can skip the JPEG step.
    """
    if isinstance(image, np.ndarray):
        return image
    if not isinstance(image, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"expected JPEG bytes or an array for a camera frame, got {type(image)!r}"
        )
    import cv2

    decoded = cv2.imdecode(
        np.frombuffer(bytes(image), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if decoded is None:
        raise ValueError("JPEG decoding failed: the frame is not a valid image")
    return decoded


def build_metadata(
    *,
    action_horizon: int,
    action_dim: int,
    default_prompt: str | None,
    rtc_guided: bool,
) -> dict:
    """Build the frame the server sends once a client connects."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "action_horizon": int(action_horizon),
        "action_dim": int(action_dim),
        "default_prompt": default_prompt,
        "rtc": {"version": RTC_PROTOCOL_VERSION, "guided": bool(rtc_guided)},
    }


def validate_metadata(metadata: Any, *, require_rtc: bool) -> dict:
    """Check the server metadata before a client starts driving the robot."""
    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"expected metadata mapping from the inference server, got {type(metadata)!r}"
        )
    version = metadata.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise RuntimeError(
            f"inference protocol version mismatch: server {version!r}, "
            f"client {PROTOCOL_VERSION}."
        )
    for field in ("action_horizon", "action_dim"):
        if not isinstance(metadata.get(field), int):
            raise RuntimeError(
                f"inference metadata is missing an integer {field!r}: {metadata}"
            )
    rtc = metadata.get("rtc")
    guided = isinstance(rtc, dict) and bool(rtc.get("guided"))
    if isinstance(rtc, dict) and rtc.get("version") != RTC_PROTOCOL_VERSION:
        raise RuntimeError(
            f"server RTC protocol version {rtc.get('version')!r} is incompatible "
            f"with client version {RTC_PROTOCOL_VERSION}."
        )
    if require_rtc and not guided:
        raise RuntimeError(
            "server does not advertise RTC guidance; start it with "
            "openpi.rtc_enabled=true or run the client with --no-rtc."
        )
    return metadata


def validate_rtc_payload(payload: Any) -> dict:
    """Normalize the ``rtc`` request block, rejecting incompatible clients."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"expected the rtc request block to be a mapping, got {payload!r}"
        )
    if payload.get("version") != RTC_PROTOCOL_VERSION:
        raise ValueError(
            f"RTC request version {payload.get('version')!r} is incompatible with "
            f"server version {RTC_PROTOCOL_VERSION}."
        )
    request_type = str(payload.get("request_type", "replan"))
    if request_type not in REQUEST_TYPES:
        raise ValueError(
            f"RTC request_type must be one of {REQUEST_TYPES}, got {request_type!r}"
        )
    return {
        "version": RTC_PROTOCOL_VERSION,
        "request_type": request_type,
        "executed_horizon": max(int(payload.get("executed_horizon", 0)), 0),
        "predicted_delay_steps": max(int(payload.get("predicted_delay_steps", 0)), 0),
        "chunk_id": int(payload.get("chunk_id", 0)),
    }
