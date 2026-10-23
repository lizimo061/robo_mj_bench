"""Private contact-history evaluators for three cooperative manipulation tasks.

No public API exposes the automata, object geometry, poses, or scores.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ..protocol import TIMESTEP
from .base import Task, body_is_slow, box_corners


def geom(parent, name, size, pos="0 0 0", **extra):
    return ET.SubElement(parent, "geom", name=name, type="box", size=size,
                         pos=pos, friction="1 .005 .0005", **extra)


def free_body(world, name, position):
    body = ET.SubElement(world, "body", name=name, pos=" ".join(map(str, position)))
    ET.SubElement(body, "freejoint", name=name + "_free")
    return body


class CooperativeTask(Task):
    approximation = "rigid_body_contact_history"

    def bind(self, model, data):
        super().bind(model, data)
        self.pad_ids = {side: [model.geom(f"{side}_{finger}_pad").id
                              for finger in ("left", "right")]
                        for side in ("left", "right")}

    def gripping(self, side, geom_ids):
        """Both opposing physical pads, each carrying >=0.05 N solved contact."""
        normals = {}
        force = np.zeros(6)
        for index, contact in enumerate(self.data.contact[:self.data.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            pad = a if a in self.pad_ids[side] and b in geom_ids else (
                b if b in self.pad_ids[side] and a in geom_ids else None)
            if pad is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, force)
            if force[0] >= .05:
                normals[pad] = np.array(contact.frame[:3]) * (1 if pad == a else -1)
        pads = self.pad_ids[side]
        return (all(pad in normals for pad in pads)
                and float(normals[pads[0]] @ normals[pads[1]]) < -.5)

    def robot_contact(self, simulation, bodies):
        robot = simulation.robot_bodies["left"] | simulation.robot_bodies["right"]
        for contact in self.data.contact[:self.data.ncon]:
            a, b = (int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if (a in bodies and b in robot) or (b in bodies and a in robot):
                return True
        return False

    def unsupported(self, simulation, bodies):
        allowed = bodies | simulation.robot_bodies["left"] | simulation.robot_bodies["right"]
        for contact in self.data.contact[:self.data.ncon]:
            if contact.dist > 0:
                continue
            a, b = (int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if (a in bodies and b not in allowed) or (b in bodies and a not in allowed):
                return False
        return True

    def released_on_goal(self, simulation, geom_id, body_id, center, half, top):
        corners = box_corners(self.model, self.data, geom_id)
        inside = bool(np.all(np.abs(corners[:, :2] - center) < half))
        low = float(corners[:, 2].min())
        return (inside and top - .002 < low < top + .003
                and self.data.geom_xmat[geom_id].reshape(3, 3)[2, 2] > .98
                and body_is_slow(self.model, self.data, body_id)
                and not self.robot_contact(simulation, {body_id}))


class HandoverTask(CooperativeTask):
    task_id = "bimanual_handover"
    instruction = ("Lift the orange baton with the LEFT arm. Pass it directly in the air "
                   "to the RIGHT arm with both grippers overlapping their hold for at least 0.10 s. "
                   "The right arm must hold it alone in the air for 0.15 s, then place it flat "
                   "on the green pad and release it. Do not set it down during the transfer.")

    def build(self, world, root, rng):
        self.goal = np.array([rng.uniform(.40, .45), rng.uniform(.12, .16)])
        geom(world, "handover_goal", ".12 .075 .003", f"{self.goal[0]} {self.goal[1]} .003",
             rgba=".12 .65 .25 1")
        body = free_body(world, "baton", [rng.uniform(.25, .29), -.14, .013])
        geom(body, "baton_geom", ".09 .018 .012", mass=".09", rgba=".95 .4 .08 1")
        self.stage, self.dwell, self.failed = 0, 0., False

    def bind(self, model, data):
        super().bind(model, data)
        self.object_body = model.body("baton").id
        self.object_geom = model.geom("baton_geom").id

    def after_step(self, simulation):
        left = self.gripping("left", {self.object_geom})
        right = self.gripping("right", {self.object_geom})
        airborne = (self.data.xpos[self.object_body, 2] > .10
                    and self.unsupported(simulation, {self.object_body}))
        if 0 < self.stage < 3 and not airborne:
            self.failed = True
        if (self.stage == 1 and not left) or (self.stage == 2 and not right):
            self.failed = True
        condition = ((left and not right and airborne), (left and right and airborne),
                     (right and not left and airborne))
        thresholds = (.10, .10, .15)
        if self.stage < 3 and not self.failed:
            self.dwell = self.dwell + TIMESTEP if condition[self.stage] else 0.
            if self.dwell >= thresholds[self.stage] - 1e-9:
                self.stage += 1
                self.dwell = 0.

    def evaluate(self, simulation, answer=None):
        placed = self.released_on_goal(simulation, self.object_geom, self.object_body,
                                       self.goal, np.array([.12, .075]), .006)
        success = self.stage == 3 and placed and not self.failed
        return {"success": success, "score": 100 if success else (0 if self.failed else self.stage * 25),
                "metrics": {"transfer_stage": self.stage, "transfer_failed": self.failed, "placed": placed}}


class TrayCarryTask(CooperativeTask):
    task_id = "bimanual_tray"
    instruction = ("Grasp the two blue tray handles, LEFT arm on the negative-y handle and RIGHT "
                   "arm on the positive-y handle. Lift the tray level with both arms for at least "
                   "0.20 s above 9 cm near its starting position, keep both grips until the whole "
                   "tray is above the green pad, and set it down flat "
                   "and released. Keep every ball inside throughout the carry.")

    def build(self, world, root, rng):
        self.goal = np.array([rng.uniform(.32, .35), rng.uniform(.05, .08)])
        geom(world, "tray_goal", ".115 .195 .0005", f"{self.goal[0]} {self.goal[1]} .0005",
             contype="0", conaffinity="0", rgba=".1 .65 .25 1")
        x, y = rng.uniform(.21, .23), -.14
        tray = free_body(world, "tray", [x, y, .009])
        geom(tray, "tray_floor", ".075 .115 .008", mass=".16", rgba=".65 .65 .7 1")
        for name, size, pos in (("a", ".006 .12 .02", "-.081 0 .02"),
                                ("b", ".006 .12 .02", ".081 0 .02"),
                                ("c", ".075 .006 .02", "0 -.121 .02"),
                                ("d", ".075 .006 .02", "0 .121 .02")):
            geom(tray, "tray_wall_" + name, size, pos, mass=".012", rgba=".65 .65 .7 1")
        for side, sign in (("left", -1), ("right", 1)):
            geom(tray, side + "_handle", ".02 .025 .012", f"0 {sign*.15} .012",
                 mass=".015", rgba=".08 .3 .9 1")
        for index, (dx, dy) in enumerate(((-.03, -.05), (.03, -.05), (-.03, .05), (.03, .05))):
            ball = free_body(world, f"cargo{index}", [x + dx, y + dy, .03])
            ET.SubElement(ball, "geom", name=f"cargo_geom{index}", type="sphere", size=".012",
                          mass=".012", rgba=".95 .6 .05 1", friction=".6 .005 .0005")
        self.dwell, self.lifted, self.spilled, self.carry_failed = 0., False, False, False

    def bind(self, model, data):
        super().bind(model, data)
        self.tray = model.body("tray").id
        self.floor = model.geom("tray_floor").id
        self.handles = {side: model.geom(side + "_handle").id for side in ("left", "right")}
        self.cargo = [model.body(f"cargo{i}").id for i in range(4)]
        self.origin = self.data.xpos[self.tray, :2].copy()

    def goal_contains_tray(self):
        geoms = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == self.tray]
        return all(np.all(np.abs(box_corners(self.model, self.data, gid)[:, :2] - self.goal)
                          < [.115, .195]) for gid in geoms)

    def cargo_inside(self):
        rotation = self.data.xmat[self.tray].reshape(3, 3)
        points = (self.data.xpos[self.cargo] - self.data.xpos[self.tray]) @ rotation
        return bool(np.all(np.abs(points[:, :2]) < [.0635, .1035])
                    and np.all(points[:, 2] > .017) and np.all(points[:, 2] < .06))

    def after_step(self, simulation):
        if not self.cargo_inside():
            self.spilled = True
        both_grips = all(self.gripping(side, {gid}) for side, gid in self.handles.items())
        unsupported = self.unsupported(simulation, {self.tray, *self.cargo})
        level = self.data.xmat[self.tray].reshape(3, 3)[2, 2] > .98
        coordinated = (both_grips
                       and self.data.xpos[self.tray, 2] > .09
                       and level and unsupported and not self.spilled
                       and np.linalg.norm(self.data.xpos[self.tray, :2] - self.origin) < .06)
        self.dwell = self.dwell + TIMESTEP if coordinated else 0.
        self.lifted |= self.dwell >= .2 - 1e-9
        if self.lifted and not self.goal_contains_tray():
            if not (both_grips and unsupported and level and self.data.xpos[self.tray, 2] > .06):
                self.carry_failed = True

    def evaluate(self, simulation, answer=None):
        # Include handles and walls in containment, not just the tray center.
        contained = self.goal_contains_tray()
        placed = contained and self.released_on_goal(simulation, self.floor, self.tray, self.goal,
                                                     np.array([.115, .195]), 0.)
        released = not self.robot_contact(simulation, {self.tray, *self.cargo})
        slow = all(body_is_slow(self.model, self.data, body) for body in self.cargo)
        success = self.lifted and not self.spilled and not self.carry_failed and self.cargo_inside() and placed and released and slow
        return {"success": success, "score": 100 if success else (40 if self.lifted and not self.spilled and not self.carry_failed else 0),
                "metrics": {"cooperative_lift": bool(self.lifted), "spilled": self.spilled,
                            "carry_failed": self.carry_failed, "cargo_inside": self.cargo_inside(), "placed": bool(placed)}}


class DrawerInsertTask(CooperativeTask):
    task_id = "bimanual_drawer"
    instruction = ("Use the LEFT gripper to grasp the blue handle and hold the spring-loaded drawer "
                   "at least 10 cm open. While holding it, use the RIGHT gripper to insert the orange "
                   "block fully inside. Release the block and keep the left arm holding the drawer "
                   "open and steady through submission. Do not drop the block into a closed drawer.")

    def build(self, world, root, rng):
        x, y = rng.uniform(.26, .29), rng.uniform(.16, .18)
        case = ET.SubElement(world, "body", name="cabinet", pos=f"{x} {y} 0")
        for name, size, pos in (("floor", ".10 .09 .008", "0 0 .008"),
                                ("left", ".008 .09 .065", "-.10 0 .065"),
                                ("right", ".008 .09 .065", ".10 0 .065"),
                                ("back", ".10 .008 .065", "0 .09 .065"),
                                ("roof", ".10 .09 .008", "0 0 .13")):
            geom(case, "cabinet_" + name, size, pos, rgba=".3 .32 .36 1")
        drawer = ET.SubElement(case, "body", name="drawer", pos="0 0 .027")
        ET.SubElement(drawer, "joint", name="drawer_slide", type="slide", axis="0 -1 0",
                      range="0 .16", stiffness="12", damping="2", springref="0", armature=".01")
        geom(drawer, "drawer_floor", ".081 .072 .006", mass=".16", rgba=".65 .6 .5 1")
        for name, size, pos in (("left", ".005 .072 .022", "-.086 0 .022"),
                                ("right", ".005 .072 .022", ".086 0 .022"),
                                ("back", ".081 .005 .022", "0 .077 .022"),
                                ("front", ".081 .005 .022", "0 -.077 .022")):
            geom(drawer, "drawer_" + name, size, pos, mass=".012", rgba=".65 .6 .5 1")
        geom(drawer, "drawer_handle", ".025 .018 .012", "0 -.105 .025", mass=".02", rgba=".05 .3 .9 1")
        block = free_body(world, "insert_block", [x + .02, y + .17, .018])
        geom(block, "insert_geom", ".018 .018 .018", mass=".05", rgba=".95 .4 .08 1")
        self.dwell, self.inserted = 0., False
        self.was_inside = self.entered_cooperatively = False

    def bind(self, model, data):
        super().bind(model, data)
        self.drawer = model.body("drawer").id
        self.block = model.body("insert_block").id
        self.block_geom = model.geom("insert_geom").id
        self.handle = model.geom("drawer_handle").id
        self.slide = model.joint("drawer_slide").id

    def held_open(self):
        return bool(.10 <= self.data.qpos[self.model.jnt_qposadr[self.slide]] <= .162
                    and self.gripping("left", {self.handle}))

    def block_inside(self):
        corners = (box_corners(self.model, self.data, self.block_geom) - self.data.xpos[self.drawer])
        corners = corners @ self.data.xmat[self.drawer].reshape(3, 3)
        return bool(np.all(np.abs(corners[:, :2]) < [.080, .071])
                    and corners[:, 2].min() > .004 and corners[:, 2].max() < .065)

    def after_step(self, simulation):
        inside = self.block_inside()
        coordinated = self.held_open() and self.gripping("right", {self.block_geom})
        if not inside:
            self.entered_cooperatively = False
        elif not self.was_inside:
            self.entered_cooperatively = coordinated
        self.was_inside = inside
        condition = coordinated and inside and self.entered_cooperatively
        self.dwell = self.dwell + TIMESTEP if condition else 0.
        self.inserted |= self.dwell >= .10 - 1e-9

    def evaluate(self, simulation, answer=None):
        held = self.held_open()
        inside = self.block_inside()
        released = not self.robot_contact(simulation, {self.block})
        steady = (body_is_slow(self.model, self.data, self.block)
                  and abs(self.data.qvel[self.model.jnt_dofadr[self.slide]]) < .02)
        success = self.inserted and held and inside and released and steady
        return {"success": success, "score": 100 if success else (40 if self.inserted else 0),
                "metrics": {"coordinated_insertion": bool(self.inserted), "held_open": held,
                            "inside": inside, "released": released, "steady": bool(steady)}}
