"""Verify recorded videos and replay audited actions without a renderer."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from robo_mj_bench.simulation import ButtonSimulation


def verify(root):
    summary = json.loads((root / "summary.json").read_text())
    checks = []
    for row in summary["runs"]:
        directory = root / f"seed-{row['seed']}"
        video = root / row["video"]
        metadata = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(video),
        ], text=True))["streams"][0]
        assert metadata["codec_name"] == "h264", metadata
        assert (metadata["width"], metadata["height"]) == (1280, 1040), metadata
        assert metadata["r_frame_rate"] == f"{row['fps']}/1", metadata
        assert int(metadata["nb_frames"]) == row["frames"], metadata
        assert abs(float(metadata["duration"]) - row["video_seconds"]) < 0.001, metadata
        expected = (directory / "video.sha256").read_text().split()[0]
        assert hashlib.sha256(video.read_bytes()).hexdigest() == expected
        # Decode the entire file so broken middle frames cannot pass a header check.
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video),
                        "-f", "null", "-"], check=True, capture_output=True)
        report = json.loads((directory / "private" / "result.json").read_text())
        events = [json.loads(line) for line in (directory / "private" / "events.jsonl").read_text().splitlines()]
        simulation = ButtonSimulation(row["seed"])
        try:
            for event in events:
                if event["event"] != "request":
                    continue
                path, payload = event["path"], event["payload"]
                if path == "/v1/move_joints":
                    simulation.move_joints(payload["targets"], payload["duration"])
                elif path == "/v1/set_gripper":
                    simulation.set_gripper(payload["opening"], payload["duration"])
                elif path == "/v1/wait":
                    simulation.wait(payload["duration"])
            replay = simulation.finish()
        finally:
            simulation.close()
        np.testing.assert_array_equal(replay["final_qpos"], report["final_qpos"])
        for field in ("success", "score", "sim_time", "peak_button_force_n", "max_button_travel_m", "scene_sha256"):
            assert replay[field] == report[field], (row["seed"], field, replay[field], report[field])
        assert row["success"] == report["success"] and row["valid"] == report["valid"]
        assert row["score"] == report["score"]
        checks.append({"seed": row["seed"], "decoded_frames": row["frames"],
                       "sha256": expected, "replay_exact": True, "passed": report["success"]})
        print(json.dumps(checks[-1]), flush=True)
    result = {"verified_episodes": len(checks), "all_exact_replays": True, "videos": checks}
    (root / "verification.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    verify(args.directory)
