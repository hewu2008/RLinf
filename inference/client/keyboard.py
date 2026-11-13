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

"""Single-key input for the robot-side control loop."""

from __future__ import annotations

import logging
import select
import sys
from typing import TextIO

logger = logging.getLogger(__name__)


class KeyPressListener:
    """Read single keystrokes from the terminal without blocking the caller.

    The control loop must not wait on stdin, so the terminal is put into cbreak
    mode (keystrokes arrive without a newline, Ctrl+C still raises) and polled
    with a zero timeout. With stdin redirected (``nohup``, a pipe) :meth:`start`
    leaves :attr:`enabled` false and :meth:`poll` returns None, so the run keeps
    going instead of failing on an unusable terminal.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdin
        self._fd: int | None = None
        self._saved: list | None = None
        self.enabled = False

    def start(self) -> None:
        """Enter cbreak mode; safe to call when already started."""
        if self.enabled:
            return
        try:
            import termios
            import tty

            if not self._stream.isatty():
                raise OSError("stdin is not a terminal")
            self._fd = self._stream.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except (ImportError, OSError, ValueError) as error:
            self._fd = None
            self._saved = None
            self.enabled = False
            logger.warning("single-key input is disabled: %s", error)
            return
        self.enabled = True

    def stop(self) -> None:
        """Restore the terminal mode saved by :meth:`start`."""
        if not self.enabled:
            return
        import termios

        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        finally:
            self._fd = None
            self._saved = None
            self.enabled = False

    def poll(self) -> str | None:
        """Return the pending key in lower case, or None when nothing is pressed."""
        if not self.enabled:
            return None
        readable, _, _ = select.select([self._fd], [], [], 0)
        if not readable:
            return None
        key = self._stream.read(1)
        return key.lower() if key else None
