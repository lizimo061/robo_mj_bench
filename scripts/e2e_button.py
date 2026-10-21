"""Run RGB-only button policies over real HTTP and record trusted multi-camera MP4s.

Local integration diagnostics, not adversarial VM isolation or LLM performance.
The policy runs in a separate working directory/process with only public files;
the trusted process owns physics, recording, tokens, seeds, and result checks.
"""

import argparse
import hashlib
from http.server import HTTPServer
import io
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

from robo_mj_bench.protocol import TIMESTEP
from robo_mj_bench.server import handler_for
from robo_mj_bench.service import EpisodeService
from robo_mj_bench.simulation import ButtonSimulation
from robo_mj_bench.robot_assets import public_robot_files

ROOT = Path(__file__).resolve().parents[1]


class VideoRecorder:
    width, height = 1280, 1040

    def __init__(self, path, seed, fps=20):
        self.path, self.seed, self.fps = path, seed, fps
        self.frames = 0
        self.last_image = None
        self._log = path.with_suffix(".ffmpeg.log").open("w")
        encoder = shutil.which("ffmpeg")
        if encoder is None:
            raise RuntimeError("ffmpeg is required for MP4 recording")
        self._process = subprocess.Popen([
            encoder, "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log)
        self.font = ImageFont.load_default(size=18)
        self.title_font = ImageFont.load_default(size=23)

    def write(self, image):
        self._process.stdin.write(image.tobytes())
        self.last_image = image
        self.frames += 1

    def capture(self, simulation, action):
        canvas = Image.new("RGB", (self.width, self.height), (16, 22, 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 8), f"I2RT YAM  |  BUTTON TASK  |  seed {self.seed}  |  RGB-only HTTP diagnostic", font=self.title_font, fill="white")
        draw.text((16, 38), f"Simulation time {simulation.data.time:5.2f}s  |  {action}  |  playback 1x simulation time", font=self.font, fill=(180, 200, 220))
        for camera, x, y in (("front", 0, 64), ("overhead", 640, 64),
                             ("left", 0, 544), ("wrist", 640, 544)):
            with Image.open(io.BytesIO(simulation.observe(camera))) as image:
                canvas.paste(image.convert("RGB"), (x, y))
            draw.rectangle((x + 8, y + 8, x + 150, y + 34), fill=(16, 22, 32))
            draw.text((x + 15, y + 11), camera.upper(), font=self.font, fill="white")
        self.write(canvas)

    def hold(self, seconds, label=None):
        image = self.last_image.copy()
        if label:
            draw = ImageDraw.Draw(image)
            draw.rectangle((10, 965, self.width - 10, 1028), fill=(16, 22, 32))
            draw.text((25, 979), label, font=self.title_font, fill="white")
        for _ in range(round(seconds * self.fps)):
            self.write(image)

    def close(self):
        if self._process.stdin is not None:
            self._process.stdin.close()
        code = self._process.wait(timeout=60)
        self._log.close()
        if code:
            raise RuntimeError(f"Video encoder failed: {self.path.with_suffix('.ffmpeg.log')}")


class RecordedSimulation(ButtonSimulation):
    """Recording reads are operator-only and consume no agent sensor allowance."""

    def __init__(self, seed, recorder):
        super().__init__(seed)
        self.recorder = recorder
        self.action = "initial observation"
        self.ticks = 0
        self.record_interval = round(1 / (recorder.fps * TIMESTEP))

    def _step(self):
        super()._step()
        self.ticks += 1
        if self.ticks % self.record_interval == 0:
            self.recorder.capture(self, self.action)


class RecordedService(EpisodeService):
    def handle(self, path, payload, authorization):
        if self.authorized(authorization):
            self.simulation.action = path.removeprefix("/v1/")
        return super().handle(path, payload, authorization)


def run_episode(seed, output, timeout):
    directory = output / f"seed-{seed}"
    directory.mkdir(mode=0o700)
    private = directory / "private"
    video_path = directory / "button.mp4"
    recorder = VideoRecorder(video_path, seed)
    simulation = RecordedSimulation(seed, recorder)
    token = secrets.token_urlsafe(32)
    service = RecordedService(simulation, token, private)
    server = HTTPServer(("127.0.0.1", 0), handler_for(service))
    server.timeout = 0.2
    process = None
    started = time.monotonic()
    try:
        (private / "scene.xml").write_text(simulation.xml)
        recorder.capture(simulation, simulation.action)
        recorder.hold(1.0, "Initial scene | controller receives RGB images only")
        with tempfile.TemporaryDirectory(prefix="robo-mj-bench-button-policy-") as agent_dir:
            for name in ("robot_client.py", "robot_kinematics.py", "vision_demo.py", "INSTRUCTIONS.md"):
                shutil.copyfile(ROOT / "agent" / name, Path(agent_dir) / name)
            for name, source in public_robot_files():
                target = Path(agent_dir) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            environment = {key: value for key, value in os.environ.items()
                           if key not in ("PYTHONPATH", "ROBOT_URL", "ROBOT_TOKEN")}
            environment["ROBOT_URL"] = f"http://127.0.0.1:{server.server_port}"
            environment["ROBOT_TOKEN"] = token
            process = subprocess.Popen([sys.executable, "vision_demo.py"], cwd=agent_dir,
                                       env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = time.monotonic() + timeout
            while not service.closed and process.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError("End-to-end policy exceeded wall-clock limit")
                server.handle_request()
            stdout, stderr = process.communicate(timeout=15)
            (private / "policy.stdout").write_text(stdout)
            (private / "policy.stderr").write_text(stderr)
        if not service.closed:
            raise RuntimeError("Policy exited without terminating its episode")
        report = json.loads((private / "result.json").read_text())
        if report["success"] and process.returncode != 0:
            raise RuntimeError("Policy crashed despite a submitted success; integration failed")
        events = [json.loads(line) for line in (private / "events.jsonl").read_text().splitlines()]
        responses = [event for event in events if event["event"] == "response"]
        paths = [event["path"] for event in events if event["event"] == "request"]
        if not responses or any(event["status"] != 200 for event in responses):
            raise RuntimeError("Policy received unsuccessful HTTP responses")
        if "/v1/observe" not in paths or "/v1/move_joints" not in paths:
            raise RuntimeError("Policy did not exercise the camera-to-action path")
        simulation.action = "terminal result (operator evaluation)"
        recorder.capture(simulation, simulation.action)
        label = "PASS" if report["valid"] and report["success"] else "FAIL"
        recorder.hold(2.0, f"{label} | score {report['score']} | {report['termination']} | RGB-only diagnostic, not an LLM score")
        recorder.last_image.save(directory / "poster.jpg", quality=90)
        row = {key: report[key] for key in ("seed", "robot", "robot_source_commit", "valid", "success", "score", "termination",
               "sim_time", "peak_button_force_n", "max_button_travel_m", "scene_sha256", "usage")}
        row.update({"video": f"seed-{seed}/button.mp4", "frames": recorder.frames,
                    "fps": recorder.fps, "video_seconds": recorder.frames / recorder.fps,
                    "policy_exit_code": process.returncode, "http_responses": len(responses),
                    "wall_seconds": time.monotonic() - started})
        return row
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        server.server_close()
        try:
            service.close()
        finally:
            recorder.close()
            if video_path.exists():
                digest = hashlib.sha256(video_path.read_bytes()).hexdigest()
                (directory / "video.sha256").write_text(digest + "  button.mp4\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505])
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in args.seeds:
        print(f"Starting RGB-only HTTP episode, seed={seed}", flush=True)
        row = run_episode(seed, args.output, args.timeout)
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {"kind": "rgb_only_http_integration", "controller": "scripted vision diagnostic; no LLM",
               "episodes": len(rows), "successes": sum(row["valid"] and row["success"] for row in rows),
               "runs": rows}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Completed {summary['successes']}/{summary['episodes']} successful episodes. {args.output}", flush=True)
    if summary["successes"] != summary["episodes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
