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

"""Synchronous websocket client of the policy server."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Mapping

import websockets.sync.client

from inference import protocol

logger = logging.getLogger(__name__)

RETRY_INTERVAL_SECONDS = 5.0


class WebsocketPolicyClient:
    """Send observations to the policy server and return action chunks."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 55555,
        *,
        retry_interval: float = RETRY_INTERVAL_SECONDS,
        wait_timeout: float | None = None,
    ) -> None:
        self._uri = f"ws://{host}:{port}"
        self._retry_interval = float(retry_interval)
        self._wait_timeout = wait_timeout
        # The connection is a context manager in current websockets releases;
        # enter it for the lifetime of the client instead of nesting the whole
        # control loop inside a `with` block.
        self._stack = contextlib.ExitStack()
        self._ws, self._metadata = self._connect()

    @property
    def uri(self) -> str:
        """Address this client is connected to."""
        return self._uri

    @property
    def metadata(self) -> dict:
        """Metadata the server sent during the handshake."""
        return self._metadata

    def validate(self, *, require_rtc: bool) -> dict:
        """Reject an incompatible server before the robot moves."""
        return protocol.validate_metadata(self._metadata, require_rtc=require_rtc)

    def infer(self, observation: Mapping[str, Any]) -> dict:
        """Send one observation and return the action-chunk response."""
        self._ws.send(protocol.pack(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"inference server error:\n{response}")
        return protocol.unpack(response)

    def close(self) -> None:
        """Close the connection."""
        self._stack.close()

    def _connect(self) -> tuple[Any, dict]:
        logger.info("waiting for the policy server at %s", self._uri)
        deadline = (
            None
            if self._wait_timeout is None
            else time.monotonic() + self._wait_timeout
        )
        while True:
            try:
                connection = self._stack.enter_context(
                    websockets.sync.client.connect(
                        self._uri, compression=None, max_size=None
                    )
                )
            except (ConnectionRefusedError, OSError) as error:
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(
                        f"no policy server at {self._uri} within "
                        f"{self._wait_timeout:.0f}s: {error}"
                    ) from error
                logger.info("still waiting for the policy server (%s)", error)
                time.sleep(self._retry_interval)
                continue
            metadata = protocol.unpack(connection.recv())
            logger.info("connected to %s", self._uri)
            return connection, metadata
