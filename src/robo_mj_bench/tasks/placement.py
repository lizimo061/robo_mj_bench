"""Place a graspable compound crocodile completely inside a painted rectangle."""
import math
import xml.etree.ElementTree as ET

import numpy as np

from .base import Task, body_is_slow, box_corners, touching_robot


def _text(values):
    return " ".join(str(float(v)) for v in values)


def _hull(points):
    points = sorted(set(map(tuple, points)))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def _area(points):
    if len(points) < 3:
        return 0.0
    return abs(float(np.sum(points[:, 0] * np.roll(points[:, 1], -1) -
                            points[:, 1] * np.roll(points[:, 0], -1)))) / 2


def _clip_rectangle(polygon, half_size):
    """Sutherland–Hodgman clipping in target-local coordinates."""
    output = list(polygon)
    for axis in (0, 1):
        for sign in (-1, 1):
            source, output = output, []
            if not source:
                break
            previous = source[-1]
            previous_distance = half_size[axis] - sign * previous[axis]
            for current in source:
                distance = half_size[axis] - sign * current[axis]
                if (distance >= 0) != (previous_distance >= 0):
                    fraction = previous_distance / (previous_distance - distance)
                    output.append(previous + fraction * (current - previous))
                if distance >= 0:
                    output.append(current)
                previous, previous_distance = current, distance
    return np.asarray(output)


class PlacementTask(Task):
    task_id = "placement"
    instruction = ("Place the entire green crocodile toy flat inside the blue rectangle on the table. "
                   "Its head, feet, and tail must all be inside the inner edges. Release the toy "
                   "and withdraw the robot so it rests freely on the table.")

    def build(self, world, root, rng):
        self.target_center = np.array([rng.uniform(.34, .46), rng.uniform(.10, .15)])
        self.target_angle = rng.uniform(-.30, .30)
        self.target_half_size = np.array([.12, .075])
        c, s = math.cos(self.target_angle), math.sin(self.target_angle)
        self.target_rotation = np.array([[c, -s], [s, c]])
        target = ET.SubElement(world, "body", name="placement_target",
                               pos=_text([*self.target_center, 0]), euler=f"0 0 {self.target_angle}")
        # Purely visual paint: it cannot physically catch or support the toy.
        for axis in (0, 1):
            for sign in (-1, 1):
                pos = [0, 0, .0001]
                pos[axis] = sign * (self.target_half_size[axis] + .001)
                size = [*self.target_half_size, .0001]
                size[axis] = .001
                ET.SubElement(target, "geom", type="box", pos=_text(pos), size=_text(size),
                              rgba=".1 .25 .95 1", contype="0", conaffinity="0")
        body = ET.SubElement(world, "body", name="placement_crocodile",
                             pos=f"{rng.uniform(.30,.40)} {rng.uniform(-.18,-.14)} .013",
                             euler=f"0 0 {rng.uniform(-.35,.35)}")
        ET.SubElement(body, "freejoint", name="placement_freejoint")
        # Compound rigid geometry includes an asymmetric snout, tapering tail and feet.
        parts = [("body", (0, 0), (.043, .021), 0),
                 ("head", (.062, .003), (.022, .017), 0),
                 ("snout", (.088, .003), (.012, .010), 0),
                 ("tail", (-.066, -.006), (.027, .009), .22),
                 ("tail_tip", (-.100, -.015), (.014, .005), .35)]
        parts += [(f"foot_{i}", (x, y), (.013, .008), angle)
                  for i, (x, y, angle) in enumerate([
                      (.026, .025, .35), (.026, -.025, -.35),
                      (-.028, .025, -.35), (-.028, -.025, .35)])]
        self.part_names = []
        for name, pos, size, angle in parts:
            geom_name = f"croc_{name}"
            self.part_names.append(geom_name)
            ET.SubElement(body, "geom", name=geom_name, type="box", pos=_text([*pos, 0]),
                          size=_text([*size, .012]), euler=f"0 0 {angle}",
                          mass=str(.12 / len(parts)), rgba=".12 .53 .17 1", friction=".9 .01 .001")
        # Non-colliding eyes are inside the footprint already occupied by the head.
        for y in (-.008, .014):
            ET.SubElement(body, "geom", type="sphere", size=".003", pos=f".060 {y} .013",
                          rgba=".02 .02 .02 1", contype="0", conaffinity="0", mass="0")

    def bind(self, model, data):
        super().bind(model, data)
        self.body_id = model.body("placement_crocodile").id
        self.geom_ids = [model.geom(name).id for name in self.part_names]
        self.table_geom = model.geom("table").id

    def evaluate(self, simulation, answer=None):
        corners = [box_corners(self.model, self.data, geom) for geom in self.geom_ids]
        local = [(points[:, :2] - self.target_center) @ self.target_rotation for points in corners]
        contained = all(bool(np.all(np.abs(points) <= self.target_half_size + 1e-7)) for points in local)
        area = overlap = 0.0
        for points in local:
            polygon = _hull(points)
            area += _area(polygon)
            overlap += _area(_clip_rectangle(polygon, self.target_half_size))
        fraction = min(1.0, max(0.0, overlap / area)) if area else 0.0
        bottom = min(float(np.min(points[:, 2])) for points in corners)
        upright = bool(self.data.xmat[self.body_id].reshape(3, 3)[2, 2] > .995)
        contact_table = any((contact.geom1 == self.table_geom and contact.geom2 in self.geom_ids) or
                            (contact.geom2 == self.table_geom and contact.geom1 in self.geom_ids)
                            for contact in self.data.contact[:self.data.ncon])
        held = touching_robot(self.model, self.data, [self.body_id])
        resting = bool(-.002 < bottom < .003 and upright and contact_table and not held and
                       body_is_slow(self.model, self.data, self.body_id))
        success = bool(contained and resting)
        return {"success": success, "score": 100.0 if success else (99.0 * fraction if resting else 0.0),
                "metrics": {"contained": contained, "containment_fraction": fraction,
                            "resting": resting, "robot_contact": held, "table_contact": bool(contact_table),
                            "bottom_height": bottom}}
