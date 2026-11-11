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

"""Websocket transport for the policy server."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback

import websockets.asyncio.server
import websockets.frames

from inference import protocol

logger = logging.getLogger(__name__)

STARTUP_TIMEOUT_SECONDS = 10.0


class WebsocketPolicyServer:
    """Serve one inference service over websockets.

    The first frame is the service metadata (protocol version, chunk shape, RTC
    availability); every later frame answers exactly one observation. A handler
    exception is sent as a text frame and then closes the connection, so a
    client sees the traceback instead of timing out.
    """

    def __init__(self, service, *, host: str = "0.0.0.0", port: int = 55555) -> None:
        self._service = service
        self._host = host
        self._port = port
        self._started = threading.Event()
        self._sockets: tuple = ()

    @property
    def metadata(self) -> dict:
        """Handshake frame sent to every connection."""
        return self._service.metadata

    @property
    def bound_port(self) -> int:
        """Port the server is listening on (resolves ``port=0``)."""
        if not self._sockets:
            raise RuntimeError("the policy server has not started yet")
        return int(self._sockets[0].getsockname()[1])

    def serve_forever(self) -> None:
        """Blocking entry point."""
        asyncio.run(self.run())

    async def run(self) -> None:
        """Serve until the task is cancelled."""
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            self._sockets = tuple(server.sockets)
            self._started.set()
            logger.info("serving the policy on %s:%d", self._host, self.bound_port)
            await server.serve_forever()

    def start_background(self) -> threading.Thread:
        """Run the event loop in a daemon thread; returns once it is listening."""
        thread = threading.Thread(
            target=self.serve_forever, daemon=True, name="policy-server"
        )
        thread.start()
        if not self._started.wait(STARTUP_TIMEOUT_SECONDS):
            raise RuntimeError(
                f"the policy server did not start on {self._host}:{self._port} "
                f"within {STARTUP_TIMEOUT_SECONDS:.0f}s"
            )
        return thread

    async def _handler(
        self, websocket: websockets.asyncio.server.ServerConnection
    ) -> None:
        logger.info("connection from %s opened", websocket.remote_address)
        session = self._service.new_session()
        await websocket.send(protocol.pack(self.metadata))

        while True:
            try:
                frame = await websocket.recv()
                if isinstance(frame, str):
                    raise ValueError(
                        "expected a msgpack binary frame from the client, got "
                        f"text: {frame[:200]!r}"
                    )
                receive_start = time.monotonic()
                observation = protocol.unpack(frame)
                infer_start = time.monotonic()
                response = self._service.infer(observation, session)
                logger.info(
                    "receive %.4fs, infer %.4fs",
                    infer_start - receive_start,
                    time.monotonic() - infer_start,
                )
                await websocket.send(protocol.pack(response))
            except websockets.ConnectionClosed:
                logger.info("connection from %s closed", websocket.remote_address)
                break
            except Exception:
                logger.exception("inference failed for %s", websocket.remote_address)
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="inference failed; traceback sent in the previous frame",
                )
                raise
