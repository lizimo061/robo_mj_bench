"""Trusted MuJoCo state, physical controller, renderer, and button evaluator."""

import hashlib
import io
import random

import mujoco
import numpy as np
from PIL import Image
from . import __version__

from .protocol import (
    BadRequest, SafetyStop, CAMERA_IDS, HOME_JOINTS, MAX_GRIPPER_SPEED,
    MAX_JOINT_SPEED, TIMESTEP,
)
from .scene import button_scene
from .robot_assets import ROBOT_ID, model_assets, provenance


class ButtonSimulation:
    """One episode. Only the operator can construct/reset/read privileged state."""

    def __init__(self, seed: int):
        self.seed = seed
        rng = random.Random(seed)
        self.button_xy = (rng.uniform(0.34, 0.42), rng.uniform(-0.07, 0.07))
        self._initialize_robot(seed, button_scene(*self.button_xy))
        self._button_qadr = self.model.jnt_qposadr[self.model.joint("button_travel").id]
        self._cap_id = self.model.geom("button_cap_geom").id
        self._indicator_id = self.model.geom("indicator").id
        self._pressed = False
        self._press_seconds = 0.0
        self._max_travel = 0.0
        self._peak_button_force = 0.0

    def _initialize_robot(self, seed, xml, extra_assets=None):
        self.seed, self.xml = seed, xml
        assets = model_assets()
        for name, content in (extra_assets or {}).items():
            if name in assets:
                raise ValueError("Task cannot replace a public robot asset")
            assets[name] = content
        self.model = mujoco.MjModel.from_xml_string(self.xml, assets=assets)
        self.data = mujoco.MjData(self.model)
        self._joint_ids = [self.model.joint(f"joint{i}").id for i in range(1, 7)]
        self._qadr = self.model.jnt_qposadr[self._joint_ids]
        self._vadr = self.model.jnt_dofadr[self._joint_ids]
        self._finger_ids = [self.model.joint(name).id for name in ("left_finger", "right_finger")]
        self._finger_qadr = self.model.jnt_qposadr[self._finger_ids]
        finger_bodies = {self.model.body(name).id for name in ("left_finger", "right_finger")}
        def is_finger(body):
            while body:
                if body in finger_bodies:
                    return True
                body = self.model.body_parentid[body]
            return False
        self._finger_geoms = {i for i in range(self.model.ngeom) if is_finger(self.model.geom_bodyid[i])}
        self.data.qpos[self._qadr] = HOME_JOINTS
        self.data.qpos[self._finger_qadr] = 0.02
        self.data.ctrl[:6] = HOME_JOINTS
        self.data.ctrl[6] = 0.02
        mujoco.mj_forward(self.model, self.data)
        self._renderer = None
        self._invalid_physics = False
        self._closed = False

    def public_spec(self):
        return {"task_id": "press_button_yam_v1", "instruction": "Press the red button with the gripper, then submit.",
                "approximation": "rigid_body", "reference_image_png": None}

    def robot_state(self):
        return {
            "joint_positions": self.data.qpos[self._qadr].tolist(),
            "joint_velocities": self.data.qvel[self._vadr].tolist(),
            "gripper_opening": float(sum(self.data.qpos[self._finger_qadr])),
            "sim_time": round(float(self.data.time), 6),
        }

    def observe(self, camera_id):
        if camera_id not in ("front", "left", "overhead", "wrist"):
            raise BadRequest("invalid camera")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="wrist_cam" if camera_id == "wrist" else camera_id)
        task = getattr(self, "task", None)
        if task is not None:
            task.decorate_scene(self._renderer.scene, self)
        rgb = self._renderer.render()
        image = Image.fromarray(rgb)
        output = io.BytesIO()
        image.save(output, format="PNG")  # New image: no source metadata or labels.
        return output.getvalue()

    def _physics_step(self):
        mujoco.mj_step(self.model, self.data)
        if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or any(w.number for w in self.data.warning)):
            self._invalid_physics = True
            raise RuntimeError("invalid physics")

    def _step(self):
        self._physics_step()
        travel = float(self.data.qpos[self._button_qadr])
        self._max_travel = max(self._max_travel, travel)
        finger_force = 0.0
        all_button_force = 0.0
        force = np.zeros(6)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if self._cap_id not in (contact.geom1, contact.geom2):
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            normal_force = max(0.0, float(force[0]))
            all_button_force += normal_force
            other = contact.geom2 if contact.geom1 == self._cap_id else contact.geom1
            if other in self._finger_geoms:
                finger_force += normal_force
        self._peak_button_force = max(self._peak_button_force, all_button_force)
        if travel > 0.013 or self._peak_button_force > 30.0:
            self._closed = True
            raise SafetyStop()
        if travel >= 0.006 and finger_force >= 0.5:
            self._press_seconds += TIMESTEP
        else:
            self._press_seconds = 0.0
        if self._press_seconds >= 0.08 - 1e-9:
            self._pressed = True
            self.model.geom_rgba[self._indicator_id] = (0.05, 1.0, 0.1, 1.0)

    def _advance(self, duration, target=None):
        if self._closed:
            raise RuntimeError("episode closed")
        steps = round(duration / TIMESTEP)
        initial = self.data.ctrl.copy()
        for step in range(steps):
            if target is not None:
                self.data.ctrl[:] = initial + (target - initial) * ((step + 1) / steps)
            self._step()

    def move_joints(self, targets, duration):
        target = self.data.ctrl.copy()
        target[:6] = targets
        if np.max(np.abs(target[:6] - self.data.ctrl[:6])) > MAX_JOINT_SPEED * duration + 1e-9:
            raise BadRequest("motion exceeds target speed limit")
        self._advance(duration, target)

    def set_gripper(self, opening, duration):
        if abs(opening - 2 * self.data.ctrl[6]) > MAX_GRIPPER_SPEED * duration + 1e-9:
            raise BadRequest("motion exceeds target speed limit")
        target = self.data.ctrl.copy()
        target[6] = opening / 2
        self._advance(duration, target)

    def wait(self, duration):
        self._advance(duration)

    def finish(self):
        """Private, terminal report. A press is an event; release is not required."""
        self._closed = True
        # Joint limits are compliant in MuJoCo; do not reward overtravel exploits.
        safe = (not self._invalid_physics and self._peak_button_force <= 30.0
                and self._max_travel <= 0.013)
        success = self._pressed and safe
        return {
            "task": "press_button_yam_v1",
            "benchmark_version": __version__,
            "robot": ROBOT_ID,
            "robot_source_commit": provenance()["commit"],
            "success": success,
            "score": 100 if success else 0,
            "pressed": self._pressed,
            "safe": safe,
            "max_button_travel_m": self._max_travel,
            "peak_button_force_n": self._peak_button_force,
            "sim_time": float(self.data.time),
            "seed": self.seed,
            "button_xy": self.button_xy,
            "final_qpos": self.data.qpos.tolist(),
            "scene_sha256": hashlib.sha256(self.xml.encode()).hexdigest(),
            "mujoco_version": mujoco.__version__,
        }

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
