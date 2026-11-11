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

"""Entry point of the remote inference server.

Usage (from the repository root)::

    python -m inference.server.main --config inference/configs/server_pi05_zerith.yaml
"""

from __future__ import annotations

import argparse
import logging

from omegaconf import OmegaConf

from inference.server.policy_service import PolicyService
from inference.server.websocket_server import WebsocketPolicyServer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Serve a trained policy over websockets."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="server YAML (see inference/configs/); every flag below overrides it",
    )
    parser.add_argument("--host", default=None, help="bind address")
    parser.add_argument("--port", type=int, default=None, help="bind port")
    parser.add_argument("--default-prompt", default=None, help="fallback task prompt")
    parser.add_argument(
        "--no-rtc",
        dest="rtc",
        action="store_false",
        default=None,
        help="serve plain chunks and advertise no RTC support",
    )
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        default=None,
        help="skip the synthetic warmup requests",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load the policy and serve it until interrupted."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )

    cfg = OmegaConf.load(args.config)
    if args.host is not None:
        cfg.server.host = args.host
    if args.port is not None:
        cfg.server.port = args.port
    if args.default_prompt is not None:
        cfg.server.default_prompt = args.default_prompt
    if args.rtc is not None:
        cfg.openpi.rtc_enabled = args.rtc
    if args.warmup is not None:
        cfg.server.warmup = args.warmup

    service = PolicyService(cfg)
    if bool(OmegaConf.select(cfg, "server.warmup", default=True)):
        service.warmup()

    server = WebsocketPolicyServer(
        service,
        host=str(OmegaConf.select(cfg, "server.host")),
        port=int(OmegaConf.select(cfg, "server.port")),
    )
    logging.info("model: %s", OmegaConf.select(cfg, "model_path"))
    server.serve_forever()


if __name__ == "__main__":
    main()
