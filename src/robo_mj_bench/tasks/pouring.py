"""Physical granular transfer proxy. All scoring state stays on the server."""

import math
import xml.etree.ElementTree as ET

import numpy as np

from .base import Task, body_is_slow, touching_robot


class PouringTask(Task):
    task_id = "pouring"
    approximation = "granular_transfer_proxy"
    instruction = (
        "Pour the beads from the blue source cup into the orange receiving cup. "
        "Transfer at least 90% without spilling more than one bead. Leave both cups "
        "upright on the table and release all objects before submitting. "
        "This task models granular transfer, not liquid pouring."
    )
    bead_count = 21
    bead_radius = 0.005
    bead_mass = 0.001
    source_radius = 0.029
    target_radius = 0.041
    floor_height = 0.006
    rim_height = 0.080

    def __init__(self):
        super().__init__()
        self._spilled = set()

    def _cup(self, world, name, xy, radius, color):
        body = ET.SubElement(world, "body", name=name, pos=f"{xy[0]} {xy[1]} 0")
        ET.SubElement(body, "freejoint", name=f"{name}_joint")
        ET.SubElement(body, "geom", name=f"{name}_base", type="box",
                      pos="0 0 0.003", size=f"{radius + .004} {radius + .004} .003", mass="0.10", rgba=color,
                      friction="0.8 0.01 0.001", solref="0.006 1", solimp="0.99 0.999 0.0001")
        # Polygon walls are actual colliders; contents leave only through the open rim.
        segments = 16
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            mid_radius = radius + .002
            ET.SubElement(body, "geom", name=f"{name}_wall_{i}", type="box",
                          pos=f"{mid_radius*math.cos(angle)} {mid_radius*math.sin(angle)} .043",
                          euler=f"0 0 {angle}",
                          size=f".002 {mid_radius*math.tan(math.pi/segments)+.001} .037",
                          mass="0.004", rgba=color, friction="0.6 0.005 0.001")
        return body

    def build(self, world, root, rng):
        # Task generator is trusted; poses never become observation metadata.
        source_xy = (rng.uniform(.29, .33), rng.uniform(-.15, -.11))
        target_xy = (rng.uniform(.41, .45), rng.uniform(.09, .13))
        self._cup(world, "pour_source", source_xy, self.source_radius, "0.08 0.25 0.9 1")
        self._cup(world, "pour_target", target_xy, self.target_radius, "0.95 0.4 0.05 1")
        points = [(0., 0.)] + [(.015*math.cos(i*math.pi/3), .015*math.sin(i*math.pi/3)) for i in range(6)]
        for index in range(self.bead_count):
            dx, dy = points[index % 7]
            z = self.floor_height + self.bead_radius + .0002 + (index // 7)*.0102
            bead = ET.SubElement(world, "body", name=f"pour_bead_{index}",
                                 pos=f"{source_xy[0]+dx} {source_xy[1]+dy} {z}")
            ET.SubElement(bead, "freejoint", name=f"pour_bead_joint_{index}")
            ET.SubElement(bead, "geom", type="sphere", size=str(self.bead_radius),
                          mass=str(self.bead_mass), rgba="0.9 0.85 0.2 1", condim="6", friction="0.35 0.005 0.001",
                          solref="0.008 1", solimp="0.95 0.99 0.001")

    def bind(self, model, data):
        super().bind(model, data)
        self._source = model.body("pour_source").id
        self._target = model.body("pour_target").id
        self._beads = [model.body(f"pour_bead_{i}").id for i in range(self.bead_count)]
        self._spilled = set()

    def _inside(self, cup, bead, radius):
        rotation = self.data.xmat[cup].reshape(3, 3)
        local = rotation.T @ (self.data.xpos[bead] - self.data.xpos[cup])
        # Account for complete bead volume and small compliant-contact penetration.
        return bool(np.linalg.norm(local[:2]) <= radius-self.bead_radius+.0005
                    and self.floor_height+self.bead_radius-.001 <= local[2]
                    <= self.rim_height-self.bead_radius+.0005)

    def _released_cup(self, cup):
        rotation = self.data.xmat[cup].reshape(3, 3)
        # Base must rest at table level; floating or robot-supported receivers fail.
        return (rotation[2, 2] >= .995 and abs(self.data.xpos[cup, 2]) < .0015
                and body_is_slow(self.model, self.data, cup))

    def after_step(self, simulation):
        for bead in self._beads:
            if (self.data.xpos[bead, 2] < .008
                    and not self._inside(self._source, bead, self.source_radius)
                    and not self._inside(self._target, bead, self.target_radius)):
                self._spilled.add(bead)

    def evaluate(self, simulation, answer=None):
        finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
                      and np.isfinite(self.data.xpos).all() and np.isfinite(self.data.cvel).all())
        target = [bead for bead in self._beads if finite and self._inside(self._target, bead, self.target_radius)]
        source = [bead for bead in self._beads if finite and self._inside(self._source, bead, self.source_radius)]
        ambiguous = set(target) & set(source)
        source = [bead for bead in source if bead not in ambiguous]
        outside = set(self._beads) - set(source) - set(target)
        spilled = self._spilled | outside
        released = finite and not touching_robot(self.model, self.data, [self._source, self._target, *self._beads])
        cups_stable = finite and self._released_cup(self._source) and self._released_cup(self._target)
        beads_stable = finite and all(np.linalg.norm(self.data.cvel[b, 3:]) < .02
                                     and np.linalg.norm(self.data.cvel[b, :3])*self.bead_radius < .02
                                     for b in self._beads)
        cups_separate = finite and np.linalg.norm(self.data.xpos[self._source, :2] - self.data.xpos[self._target, :2]) >= self.source_radius + self.target_radius + .007
        valid = bool(finite and released and cups_stable and beads_stable and cups_separate and not ambiguous)
        fraction = len(target) / self.bead_count
        success = bool(valid and fraction >= .9 and len(spilled) <= 1)
        # Full score is reserved for the full public criterion.
        score = 100. if success else (min(99., 100.*fraction) if valid else 0.)
        return {"success": success, "score": score, "metrics": {
            "approximation": self.approximation, "beads_total": self.bead_count,
            "beads_in_target": len(target), "beads_in_source": len(source),
            "beads_outside_cups": len(outside), "beads_spilled": len(spilled),
            "total_mass_kg": self.bead_count*self.bead_mass,
            "released": bool(released), "cups_stable": bool(cups_stable),
            "beads_stable": bool(beads_stable), "finite": finite,
            "cups_separate": bool(cups_separate), "ambiguous_beads": len(ambiguous),
        }}
