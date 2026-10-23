"""Exercise the pilot's real HTTP lifecycle without a model or renderer."""

import json
from pathlib import Path
import subprocess
import sys
import time

from agent.robot_client import RobotClient


def test_public_manifest_hides_ground_truth_and_idle_submission_is_failure(tmp_path):
    root = Path(__file__).resolve().parents[1]
    private, public = tmp_path / "private", tmp_path / "public"
    process = subprocess.Popen([sys.executable, str(root / "scripts/serve_interactive_pilot.py"),
                                "--output", str(private), "--public-dir", str(public), "--episodes", "1"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 15
        while not (public / "gateways.json").exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError("Pilot failed to start")
            time.sleep(.02)
        manifest = json.loads((public / "gateways.json").read_text())
        assert len(manifest) == 1 and set(manifest[0]) == {"episode", "url", "token"}
        assert len(manifest[0]["token"]) >= 16
        assert not (public / "vision_demo.py").exists()
        assert not (public / "scene.xml").exists()
        assert not (public / "result.json").exists()
        entry = manifest[0]
        robot = RobotClient(entry["url"], entry["token"])
        assert robot.task()["task_id"] == "press_button_yam_v1"
        assert set(robot.robot_state()) == {"joint_positions", "joint_velocities", "gripper_opening", "sim_time"}
        assert robot.submit() == {"status": "submitted"}
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        report = json.loads((private / "summary.json").read_text())
        assert not report["independent_model_evaluation"]
        assert report["planned_episodes"] == 1
        assert report["runs"][0]["valid"] and not report["runs"][0]["success"]
        assert report["runs"][0]["score"] == 0
        assert entry["token"] not in stdout
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
