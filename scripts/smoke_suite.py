"""Exercise each task over real HTTP with an idle, public-API-only client.

This checks scene/render/lifecycle/security integration, not agent solvability.
Every idle task must remain incomplete. Terminal scoring fixtures live in tests.
"""

import argparse
from http.server import HTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time

from PIL import Image, ImageDraw, ImageFont

from robo_mj_bench.robot_assets import ROBOT_ID, provenance
from robo_mj_bench.server import handler_for
from robo_mj_bench.service import EpisodeService
from robo_mj_bench.tasks.registry import TASK_NAMES, create_simulation, BIMANUAL_TASKS

ROOT = Path(__file__).resolve().parents[1]
PROBE = '''import base64,json
from pathlib import Path
from robot_client import RobotClient,RobotError
r=RobotClient()
goal=r.task()
assert set(goal) <= {"task_id","instruction","approximation","reference_image_base64"}
assert all(k in goal for k in ("task_id","instruction","approximation"))
state=r.robot_state()
dual="left" in state
if dual:
    assert set(state)=={"left","right","sim_time"}
    for side in ("left","right"):
        assert set(state[side])=={"joint_positions","joint_velocities","gripper_opening"}
        assert len(state[side]["joint_positions"])==len(state[side]["joint_velocities"])==6
    command={side:{"targets":state[side]["joint_positions"],"opening":state[side]["gripper_opening"]} for side in ("left","right")}
    r.move_bimanual(command["left"],command["right"],.2)
else:
    assert set(state)=={"joint_positions","joint_velocities","gripper_opening","sim_time"}
    assert len(state["joint_positions"])==len(state["joint_velocities"])==6
r.wait(.2)
for camera in (("front","overhead","left","wrist_left","wrist_right") if dual else ("front","overhead","left","wrist")):
    Path(camera+".png").write_bytes(r.observe(camera))
if "reference_image_base64" in goal:
    Path("reference.png").write_bytes(base64.b64decode(goal["reference_image_base64"]))
receipt=r.submit()
assert receipt=={"status":"submitted"}
try:
    r.robot_state()
except RobotError as e:
    assert "409" in str(e)
else:
    raise AssertionError("Episode was not terminal")
print(json.dumps({"task_id":goal["task_id"],"approximation":goal["approximation"],"receipt":receipt}))
'''


def run_task(name, seed, output):
    directory = output / name
    directory.mkdir(mode=0o700)
    simulation = create_simulation(name, seed)
    token = secrets.token_urlsafe(32)
    service = EpisodeService(simulation, token, directory / "private")
    server = HTTPServer(("127.0.0.1", 0), handler_for(service))
    server.timeout = .2
    process = None
    try:
        (directory / "private" / "scene.xml").write_text(simulation.xml)
        with tempfile.TemporaryDirectory(prefix="robo-mj-bench-suite-probe-") as scratch:
            shutil.copyfile(ROOT / "agent" / "robot_client.py", Path(scratch) / "robot_client.py")
            (Path(scratch) / "probe.py").write_text(PROBE)
            environment = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "ROBOT_TOKEN", "ROBOT_URL")}
            environment.update(ROBOT_URL=f"http://127.0.0.1:{server.server_port}", ROBOT_TOKEN=token)
            process = subprocess.Popen([sys.executable, "probe.py"], cwd=scratch, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = time.monotonic() + 60
            # Keep serving once submit closes the episode: the probe verifies 409.
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Idle integration timeout: {name}")
                server.handle_request()
            stdout, stderr = process.communicate(timeout=5)
            (directory / "private" / "client.stdout").write_text(stdout)
            (directory / "private" / "client.stderr").write_text(stderr)
            if process.returncode:
                raise RuntimeError(f"{name} public API probe failed: {stderr}")
            for filename in ("front.png", "overhead.png", "left.png", "wrist.png", "wrist_left.png", "wrist_right.png", "reference.png"):
                source = Path(scratch) / filename
                if source.exists():
                    shutil.copyfile(source, directory / filename)
        report = json.loads((directory / "private" / "result.json").read_text())
        assert report["valid"] and report["termination"] == "submitted", report
        assert not report["success"] and report["score"] < 100, "An idle scene must not start complete"
        assert report["robot"] == getattr(simulation, "robot_id", ROBOT_ID)
        row = {"task_family": name, "task_id": report["task"], "seed": seed,
               "robot": report["robot"],
               "approximation": simulation.public_spec()["approximation"],
               "idle_score": report["score"], "idle_success": report["success"],
               "api_and_render_ok": True, "sim_time": report["sim_time"]}
        print(json.dumps(row), flush=True)
        return row
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        server.server_close()
        service.close()


def overview(output, rows):
    canvas = Image.new("RGB", (1280, 280 * ((len(rows)+3)//4)), "#111824")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    for index, row in enumerate(rows):
        x, y = index % 4 * 320, index // 4 * 280
        with Image.open(output / row["task_family"] / "overhead.png") as image:
            canvas.paste(image.resize((320, 240)), (x, y+40))
        draw.text((x+8, y+5), row["task_family"], fill="white", font=font)
        draw.text((x+8, y+23), row["approximation"], fill="#bdcbdc", font=ImageFont.load_default(size=11))
    canvas.save(output / "overview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=list(TASK_NAMES))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [run_task(name, args.seed, args.output) for name in args.tasks]
    result = {"kind": "idle_http_scene_integration", "robots": sorted({row["robot"] for row in rows}),
              "robot_source_commit": provenance()["commit"], "runs": rows,
              "note": "No manipulation policy or LLM completion is tested by idle runs."}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    overview(args.output, rows)
