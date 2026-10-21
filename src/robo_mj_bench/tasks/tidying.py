"""Pack four blocks in a physical box and close its removable lid."""
import xml.etree.ElementTree as ET

import numpy as np

from .base import Task, body_is_slow, box_corners, touching_robot


def _vec(values):
    return " ".join(str(float(v)) for v in values)


class TidyingTask(Task):
    task_id = "tidying"
    instruction = ("Put all four colored blocks completely inside the open box, then place the "
                   "removable lid flat and flush on its rim. Release every object and withdraw the robot.")

    def build(self, world, root, rng):
        self.center = np.array([.43 + rng.uniform(-.01, .01), .065 + rng.uniform(-.008, .008)])
        self.inner = np.array([.055, .060])
        self.floor_top, self.rim_top = .008, .058
        box = ET.SubElement(world, "body", name="tidy_box", pos=_vec([*self.center, 0]))
        def geom(body, name, size, pos, color, mass=None):
            attrs = dict(name=name, type="box", size=_vec(size), pos=_vec(pos),
                         rgba=color, friction="1 .01 .001", solref=".006 1")
            if mass is not None:
                attrs["mass"] = str(mass)
            ET.SubElement(body, "geom", **attrs)
        geom(box, "tidy_floor", [.065, .070, .004], [0, 0, .004], ".35 .30 .25 1")
        for sign in (-1, 1):
            geom(box, f"tidy_wall_x{sign}", [.005, .070, .025], [sign * .060, 0, .033], ".55 .45 .32 1")
            geom(box, f"tidy_wall_y{sign}", [.055, .005, .025], [0, sign * .065, .033], ".55 .45 .32 1")
        colors = [".9 .15 .12 1", ".1 .35 .9 1", ".15 .75 .2 1", ".9 .7 .1 1"]
        self.block_names = [f"tidy_block_{i}" for i in range(4)]
        positions = [(x, y) for x in (.265, .325) for y in (-.105, .035)]
        rng.shuffle(positions)
        for name, (x, y), color in zip(self.block_names, positions, colors):
            body = ET.SubElement(world, "body", name=name, pos=_vec([x, y, .0155]), euler=f"0 0 {rng.uniform(-.3,.3)}")
            ET.SubElement(body, "freejoint", name=f"{name}_joint")
            geom(body, f"{name}_geom", [.015, .015, .015], [0, 0, 0], color, .045)
        lid = ET.SubElement(world, "body", name="tidy_lid", pos=".44 -.15 .0045")
        ET.SubElement(lid, "freejoint", name="tidy_lid_joint")
        geom(lid, "tidy_lid_geom", [.065, .070, .004], [0, 0, 0], ".28 .55 .65 1", .09)

    def bind(self, model, data):
        super().bind(model, data)
        self.block_ids = [model.body(n).id for n in self.block_names]
        self.block_geoms = [model.geom(f"{n}_geom").id for n in self.block_names]
        self.lid_id = model.body("tidy_lid").id
        self.lid_geom = model.geom("tidy_lid_geom").id

    def evaluate(self, simulation, answer=None):
        packed = []
        for body, geom in zip(self.block_ids, self.block_geoms):
            corners = box_corners(self.model, self.data, geom)
            xy = corners[:, :2] - self.center
            contained = (np.all(np.abs(xy) <= self.inner + 1e-6) and
                         np.min(corners[:, 2]) >= self.floor_top - .001 and
                         np.max(corners[:, 2]) <= self.rim_top + 1e-6)
            packed.append(bool(contained and body_is_slow(self.model, self.data, body) and
                               not touching_robot(self.model, self.data, [body])))
        lid_pos = self.data.xpos[self.lid_id]
        rot = self.data.xmat[self.lid_id].reshape(3, 3)
        corners = box_corners(self.model, self.data, self.lid_geom)
        # The rectangular lid allows a half turn but not a quarter turn. All
        # corners must lie at rim height, preventing tilted/perched success.
        lid_aligned = bool(np.all(np.abs(lid_pos[:2] - self.center) < .002) and
                           rot[2, 2] > .999 and abs(rot[1, 0]) < .025 and
                           np.all(np.abs(corners[:, 2] - np.where(
                               corners[:, 2] < lid_pos[2], self.rim_top,
                               self.rim_top + .008)) < .0015) and
                           body_is_slow(self.model, self.data, self.lid_id) and
                           not touching_robot(self.model, self.data, [self.lid_id]))
        count = sum(packed)
        lid_credit = count == 4 and lid_aligned
        return {"success": bool(lid_credit), "score": 20 * count + (20 if lid_credit else 0),
                "metrics": {"packed": packed, "packed_count": count, "lid_aligned": lid_aligned}}
