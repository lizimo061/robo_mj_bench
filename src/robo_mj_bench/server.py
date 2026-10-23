"""Operator entry point. This server belongs on the trusted side of the boundary."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import socket
from . import __version__

from .service import EpisodeService
from .robot_assets import ROBOT_ID, provenance
from .tasks.registry import TASK_NAMES, create_simulation

MAX_BODY_BYTES = 8192


def _reject_constant(_value):
    raise ValueError("nonfinite JSON")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RobotGateway"
        sys_version = ""
        protocol_version = "HTTP/1.0"

        def setup(self):
            super().setup()
            self.connection.settimeout(5.0)

        def log_message(self, *_args):
            pass  # Default logs can contain credentials or attacker-controlled URLs.

        def _reply(self, status, body):
            encoded = json.dumps(body, allow_nan=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_error(self, code, message=None, explain=None):
            self._reply(code, {"error": "invalid_request"})

        def do_GET(self):
            self._reply(405, {"error": "method_not_allowed"})

        do_HEAD = do_GET
        do_PUT = do_GET
        do_DELETE = do_GET
        do_OPTIONS = do_GET

        def do_POST(self):
            auth = self.headers.get_all("Authorization", [])
            if len(auth) != 1 or not service.authorized(auth[0]):
                self._reply(401, {"error": "unauthorized"})
                return
            lengths = self.headers.get_all("Content-Length", [])
            if (len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit()
                    or len(lengths[0]) > 6 or self.headers.get("Transfer-Encoding") is not None):
                self._reply(400, {"error": "invalid_request"})
                return
            length = int(lengths[0])
            if length > MAX_BODY_BYTES:
                self._reply(413, {"error": "request_too_large"})
                return
            if self.headers.get("Content-Type") != "application/json":
                self._reply(415, {"error": "unsupported_media_type"})
                return
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("truncated body")
                payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant,
                                     object_pairs_hook=_unique_object)
            except (ValueError, UnicodeError, RecursionError, socket.timeout):
                self._reply(400, {"error": "invalid_request"})
                return
            status, body = service.handle(self.path, payload, auth[0])
            self._reply(status, body)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--seed", type=int, default=None, help="Operator-only reproducibility seed")
    parser.add_argument("--task", choices=TASK_NAMES, default="button")
    parser.add_argument("--run-dir", required=True, type=Path, help="New private directory, never agent-mounted")
    args = parser.parse_args()
    os.umask(0o077)
    args.run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    token = secrets.token_urlsafe(32)
    (args.run_dir / "agent-token").write_text(token + "\n", encoding="ascii")
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    simulation = create_simulation(args.task, seed)
    (args.run_dir / "scene.xml").write_text(simulation.xml, encoding="utf-8")
    (args.run_dir / "manifest.json").write_text(json.dumps({"seed": seed, "task": simulation.public_spec()["task_id"],
        "task_family": args.task,
        "benchmark_version": __version__,
        "robot": getattr(simulation, "robot_id", ROBOT_ID), "robot_source_commit": provenance()["commit"],
        "protocol": 2 if getattr(simulation, "bimanual", False) else 1,
        "mujoco_version": __import__("mujoco").__version__}, indent=2), encoding="utf-8")
    service = EpisodeService(simulation, token, args.run_dir)
    server = HTTPServer((args.host, args.port), handler_for(service))
    print(f"Robot gateway listening on {args.host}:{server.server_port}. Private run: {args.run_dir}", flush=True)
    print("Development server: run evaluated agents on a separate isolated host/VM.", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
