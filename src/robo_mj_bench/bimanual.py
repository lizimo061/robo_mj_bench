"""Two standard YAM arms; only synchronized bounded actuator commands are public."""

import mujoco
import numpy as np

from .protocol import BadRequest, SafetyStop, HOME_JOINTS, MAX_JOINT_SPEED, MAX_GRIPPER_SPEED
from .robot_assets import model_assets
from .scene import dual_yam_scene
from .task_simulation import TaskSimulation


class BimanualSimulation(TaskSimulation):
    scene_factory = staticmethod(dual_yam_scene)
    robot_id = "dual_i2rt_yam_linear_4310"
    bimanual = True

    def _initialize_robot(self, seed, xml, extra_assets=None):
        self.seed, self.xml = seed, xml
        assets = model_assets()
        if set(assets).intersection(extra_assets or {}):
            raise ValueError("Task cannot replace robot assets")
        assets.update(extra_assets or {})
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)
        self.arms = {}
        for side in ("left", "right"):
            joints = [self.model.joint(f"{side}_joint{i}").id for i in range(1, 7)]
            fingers = [self.model.joint(f"{side}_{name}_finger").id for name in ("left", "right")]
            qadr, vadr = self.model.jnt_qposadr[joints], self.model.jnt_dofadr[joints]
            fadr = self.model.jnt_qposadr[fingers]
            actuators = [self.model.actuator(f"{side}_joint{i}").id for i in range(1, 7)]
            grip = self.model.actuator(f"{side}_gripper").id
            home = np.array(HOME_JOINTS)
            if side == "right":
                home[0] = -home[0]
            self.data.qpos[qadr] = home
            self.data.qpos[fadr] = .02
            self.data.ctrl[actuators] = home
            self.data.ctrl[grip] = .02
            self.arms[side] = (qadr, vadr, fadr, actuators, grip)
        self._renderer = None
        self._peak_inter_arm_force_n = 0.
        self._invalid_physics = self._closed = False
        mujoco.mj_forward(self.model, self.data)
        self.robot_bodies = {}
        for side in self.arms:
            root = self.model.body(side + "_arm").id
            bodies = set()
            for body in range(1, self.model.nbody):
                ancestor = body
                while ancestor:
                    if ancestor == root:
                        bodies.add(body)
                        break
                    ancestor = self.model.body_parentid[ancestor]
            self.robot_bodies[side] = bodies

    def robot_state(self):
        state = {"sim_time": round(float(self.data.time), 6)}
        for side, (qadr, vadr, fadr, _, _) in self.arms.items():
            state[side] = {"joint_positions": self.data.qpos[qadr].tolist(),
                           "joint_velocities": self.data.qvel[vadr].tolist(),
                           "gripper_opening": float(sum(self.data.qpos[fadr]))}
        return state

    def move_bimanual(self, left, right, duration):
        target = self.data.ctrl.copy()
        # Validate BOTH sides before any physics or controller state mutation.
        for side, command in (("left", left), ("right", right)):
            _, _, _, actuators, grip = self.arms[side]
            if np.max(np.abs(np.array(command["targets"]) - target[actuators])) > MAX_JOINT_SPEED * duration + 1e-9:
                raise BadRequest("motion exceeds target speed limit")
            if abs(command["opening"] - 2 * target[grip]) > MAX_GRIPPER_SPEED * duration + 1e-9:
                raise BadRequest("motion exceeds gripper speed limit")
            target[actuators] = command["targets"]
            target[grip] = command["opening"] / 2
        self._advance(duration, target)

    def move_joints(self, targets, duration):
        raise BadRequest("Use move_bimanual for dual-arm episodes")

    def set_gripper(self, opening, duration):
        raise BadRequest("Use move_bimanual for dual-arm episodes")

    def observe(self, camera_id):
        if camera_id not in ("front", "left", "overhead", "wrist_left", "wrist_right"):
            raise BadRequest("invalid camera")
        # The shared encoder accepts only fixed camera aliases.
        import io
        from PIL import Image
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = {"wrist_left": "left_wrist_cam", "wrist_right": "right_wrist_cam"}.get(camera_id, camera_id)
        self._renderer.update_scene(self.data, camera=camera)
        self.task.decorate_scene(self._renderer.scene, self)
        output = io.BytesIO()
        Image.fromarray(self._renderer.render()).save(output, format="PNG")
        return output.getvalue()

    def _physics_step(self):
        super()._physics_step()
        force = np.zeros(6)
        total = 0.0
        for index, contact in enumerate(self.data.contact[:self.data.ncon]):
            a, b = (int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if ((a in self.robot_bodies["left"] and b in self.robot_bodies["right"])
                    or (b in self.robot_bodies["left"] and a in self.robot_bodies["right"])):
                mujoco.mj_contactForce(self.model, self.data, index, force)
                total += max(0.0, float(force[0]))
        self._peak_inter_arm_force_n = max(self._peak_inter_arm_force_n, total)
        if total > 35:
            self._closed = self._safety_stopped = True
            raise SafetyStop()

    def finish(self):
        report = super().finish()
        report["peak_inter_arm_force_n"] = self._peak_inter_arm_force_n
        report["safety_scope"] = "physics_validity_and_35N_inter_arm_contact_limit; no general damage model"
        return report
