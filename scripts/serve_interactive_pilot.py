"""Serve blind-layout button episodes for an interactive camera/action pilot.

The policy gets an allowlisted public bundle and episode gateway credentials.
This is local development orchestration, not adversarial filesystem isolation.
No scripted vision detector or manipulation policy is included.
"""

import argparse
from http.server import HTTPServer
import json
import os
from pathlib import Path
import secrets
import select
import shutil
import time

from robo_mj_bench.robot_assets import public_robot_files
from robo_mj_bench.server import handler_for
from robo_mj_bench.service import EpisodeService
from robo_mj_bench.simulation import ButtonSimulation

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 20:
        parser.error("episodes must be between 1 and 20")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    args.public_dir.mkdir(parents=True, exist_ok=False)
    for name in ("robot_client.py", "robot_kinematics.py", "INSTRUCTIONS.md"):
        shutil.copyfile(ROOT / "agent" / name, args.public_dir / name)
    for name, source in public_robot_files():
        target = args.public_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    services, servers, manifest = [], [], []
    try:
        for index in range(args.episodes):
            simulation = ButtonSimulation(secrets.randbits(63))
            token = secrets.token_urlsafe(32)
            service = EpisodeService(simulation, token, args.output / f"episode-{index+1}")
            server = HTTPServer(("127.0.0.1", 0), handler_for(service))
            services.append(service)
            servers.append(server)
            manifest.append({"episode": index+1, "url": f"http://127.0.0.1:{server.server_port}", "token": token})
        (args.public_dir / "gateways.json").write_text(json.dumps(manifest))
        print(f"Ready: {args.episodes} randomized button episodes. Public files: {args.public_dir}", flush=True)
        deadline = time.monotonic() + 1800
        while not all(service.closed for service in services):
            if time.monotonic() >= deadline:
                break
            ready, _, _ = select.select(servers, [], [], .2)
            for server in ready:
                server.handle_request()
    finally:
        for server in servers:
            server.server_close()
        for service in services:
            service.close()
        reports = []
        for index, service in enumerate(services):
            result = service.audit_dir / "result.json"
            if result.exists():
                raw = json.loads(result.read_text())
                reports.append({"episode": index+1, **{key: raw[key] for key in
                    ("valid", "success", "score", "termination", "usage", "sim_time")}})
        summary = {"kind": "interactive_development_pilot", "task": "button",
                   "controller": "current chat model + public robot-only IK",
                   "independent_model_evaluation": False,
                   "note": "Chat has implementation context; local host is not an adversarial sandbox.",
                   "planned_episodes": args.episodes, "runs": reports}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
