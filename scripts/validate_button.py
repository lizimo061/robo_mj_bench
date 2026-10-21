"""TRUSTED OPERATOR ONLY: privileged reference controller checks task solvability.

This is not an evaluated policy or LLM baseline. It deliberately reads the private
button location, then drives only actuator commands through the normal gateway.
"""

import argparse
import json
import sys
from pathlib import Path
import secrets

from robo_mj_bench.service import EpisodeService
from robo_mj_bench.simulation import ButtonSimulation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from robot_kinematics import RobotKinematics


def reference_pose(x, y, fingertip_z):
    """Numerical IK in the public upstream YAM model; no task state in the solver."""
    return RobotKinematics().downward_pose(x, y, fingertip_z)


def validate(seed, run_dir, render=False):
    sim = ButtonSimulation(seed)
    token = secrets.token_urlsafe(32)
    service = EpisodeService(sim, token, run_dir)
    auth = "Bearer " + token

    def call(path, payload):
        status, body = service.handle("/v1/" + path, payload, auth)
        if status != 200:
            raise RuntimeError(f"Reference command failed: {path}: {status} {body}")
        return body

    try:
        if render:
            call("observe", {"camera_id": "front"})
            call("observe", {"camera_id": "overhead"})
        call("set_gripper", {"opening": 0.012, "duration": 1.0})
        x, y = sim.button_xy  # Privileged operator read, NEVER copy this into agent/.
        kinematics = RobotKinematics()
        trajectory = ((0.12, 3.0), (0.10, 1.0), (0.08, 1.0), (0.06, 1.0),
                      (0.052, 0.5), (0.050, 0.5), (0.048, 0.5),
                      (0.046, 0.5), (0.044, 0.5), (0.042, 0.5))
        for height, duration in trajectory:
            call("move_joints", {"targets": kinematics.downward_pose(x, y, height), "duration": duration})
            call("wait", {"duration": 0.2})
            if sim._pressed:  # Operator-only scorer access; this is not a model baseline.
                break
        if render:
            for camera in ("front", "left", "overhead", "wrist"):
                call("observe", {"camera_id": camera})
        call("submit", {})
        return json.loads((Path(run_dir) / "result.json").read_text())
    finally:
        service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = [validate(seed, args.output / f"seed-{seed}", args.render) for seed in args.seeds]
    summary = {"kind": "privileged_solvability_check", "episodes": len(reports),
               "successes": sum(report["success"] for report in reports),
               "details": [{key: report[key] for key in ("seed", "success", "peak_button_force_n", "max_button_travel_m")} for report in reports]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["successes"] != summary["episodes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
