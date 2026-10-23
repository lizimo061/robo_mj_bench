"""Local IK using the PUBLIC, robot-only YAM model. No task scene or state access."""

from pathlib import Path

import mujoco
import numpy as np


def public_model_path():
    exported = Path(__file__).resolve().parent / "robot" / "yam_linear_4310.xml"
    if exported.exists():
        return exported
    # Development checkout only. This directory contains immutable public robot
    # assets, never a runtime-composed task, evaluator, seed, or episode state.
    return Path(__file__).resolve().parents[1] / "src/robo_mj_bench/models/i2rt_yam/yam_linear_4310.xml"


class RobotKinematics:
    def __init__(self, arm=None):
        if arm not in (None, "left", "right"):
            raise ValueError("arm must be left, right, or None")
        # Fixed public mounts match the dual-YAM scene; no episode data involved.
        self.base = np.array([0, -.32 if arm == "left" else .32, 0]) if arm else np.zeros(3)
        yaw = (.65 if arm == "left" else -.65) if arm else 0
        self.base_rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                       [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        self.model = mujoco.MjModel.from_xml_path(str(public_model_path()))
        self.data = mujoco.MjData(self.model)
        self.joints = [self.model.joint(f"joint{i}").id for i in range(1, 7)]
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        self.site = self.model.site("grasp_site").id
        self.limits = self.model.jnt_range[self.joints]
        self.last = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])

    def downward_pose(self, x, y, height):
        """Solve a downward grasp-site pose using robot geometry and caller targets."""
        target = self.base_rotation.T @ (np.array([x, y, height]) - self.base)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        starts = (self.last, np.array([0, 1.4, 1.8, 0, 0.5, 0]),
                  np.array([0, 2.0, 1.8, 0, -0.5, 0]))
        for start in starts:
            q = start.copy()
            for _ in range(250):
                self.data.qpos[self.qadr] = q
                mujoco.mj_forward(self.model, self.data)
                position = target - self.data.site_xpos[self.site]
                direction = self.data.site_xmat[self.site].reshape(3, 3)[:, 2]
                rotation = np.cross(direction, [0.0, 0.0, -1.0])
                angle = np.arccos(np.clip(-direction[2], -1.0, 1.0))
                norm = np.linalg.norm(rotation)
                rotation = rotation * (angle / norm) if norm > 1e-9 else np.array([angle, 0.0, 0.0])
                if np.linalg.norm(position) < 2e-5 and np.linalg.norm(rotation) < 2e-4:
                    self.last = q.copy()
                    return q.tolist()
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
                # Rotation about the downward tool axis is free; a button does
                # not require a particular gripper yaw. Solve five constraints.
                jacobian = np.vstack((jacp[:, self.vadr], 0.2 * jacr[:2, self.vadr]))
                error = np.concatenate((position, 0.2 * rotation[:2]))
                update = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-5 * np.eye(5), error)
                update *= min(1.0, 0.15 / max(np.linalg.norm(update), 1e-12))
                q = np.clip(q + update, self.limits[:, 0] + 1e-6, self.limits[:, 1] - 1e-6)
        raise RuntimeError("IK did not converge for the requested pose in the public YAM model")
