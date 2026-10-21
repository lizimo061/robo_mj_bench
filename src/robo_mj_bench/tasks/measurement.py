"""Physical box metrology; dimensions and error vectors stay evaluator-side."""

import io
import math
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

from .base import Task


def _values(values):
    return " ".join(str(float(v)) for v in values)


def _ruler_png():
    """A public 0--200 mm scale, independent of every episode's hidden box."""
    image = Image.new("RGB", (2048, 256), "#fff5cb")
    draw = ImageDraw.Draw(image)
    # The two endpoints of the image correspond exactly to 0 and 200 mm.
    for mm in range(201):
        x = round(mm * (image.width - 1) / 200)
        length = 88 if mm % 10 == 0 else (55 if mm % 5 == 0 else 30)
        draw.line((x, 0, x, length), fill="black", width=2)
        if mm % 20 == 0:
            label = str(mm)
            # Default Pillow font is available without system-font dependencies.
            bounds = draw.textbbox((0, 0), label, font_size=64)
            width = bounds[2] - bounds[0]
            draw.text((min(max(x - width / 2, 0), image.width - width), 97), label,
                      fill="black", font_size=64)
    draw.text((70, 180), "mm | 10 mm = 1 cm | SI scale", fill="black", font_size=40)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class MeasurementTask(Task):
    task_id = "measurement"
    instruction = (
        "Measure the orange box using the visible rulers (millimetres; 10 mm = 1 cm). "
        "Submit answer={\"dimensions_m\":[length,width,height]} in metres. "
        "Length is the initially +x-aligned horizontal box edge (longer horizontal edge); "
        "width is its initially +y-aligned edge (shorter horizontal edge); "
        "height is its initially vertical edge, even if you rotate the box. "
        "The table rulers run along +x and +y; the upright ruler runs upward from table level. "
        "Each dimension must be within 2 mm (inclusive). You may move the box or the robot "
        "to inspect it. Submit an answer to receive credit."
    )
    accepts_answer = True
    # This is an intrinsic-geometry answer task, not a terminal placement predicate.
    # TaskSimulation still rejects unsafe physics and safety stops.
    requires_stability = False
    tolerance_m = .002

    def build(self, world, root, rng):
        self.dimensions_m = np.array([
            rng.uniform(.075, .115), rng.uniform(.035, .065), rng.uniform(.025, .095)
        ])
        center = [rng.uniform(.335, .365), rng.uniform(-.025, .025), self.dimensions_m[2] / 2 + .001]
        box = ET.SubElement(world, "body", name="measured_box", pos=_values(center))
        ET.SubElement(box, "freejoint", name="measured_box_joint")
        ET.SubElement(box, "geom", name="measured_box_geom", type="box",
                      size=_values(self.dimensions_m / 2), mass=".09",
                      rgba=".95 .39 .07 1", friction=".9 .01 .001", solref=".006 1")
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        self.assets["ruler_scale.png"] = _ruler_png()
        ET.SubElement(asset, "texture", name="ruler_scale", type="2d", file="ruler_scale.png")
        ET.SubElement(asset, "material", name="ruler_surface", texture="ruler_scale",
                      texrepeat="1 1", texuniform="false", reflectance="0", specular="0", shininess="0")
        # A ruler's local x axis carries the scale; local -y is the tick edge.
        # Collision backs make the fixed measuring tools physical. Texture planes are
        # purely visual, coincident with the exposed faces of those backs.
        self._ruler(world, "table_x", [.36, -.12, .002], "0 0 0")
        self._ruler(world, "table_y", [.475, 0, .002], "0 0 1.5707963267948966")
        # x(local) -> z(world), so the lower endpoint is exactly at table z=0.
        self._ruler(world, "upright", [.485, .115, .1], quat="0 .7071067811865476 0 .7071067811865476")

    @staticmethod
    def _ruler(world, name, center, euler=None, quat=None):
        rotation = {"quat": quat} if quat else {"euler": euler}
        body = ET.SubElement(world, "body", name=f"ruler_{name}", pos=_values(center), **rotation)
        ET.SubElement(body, "geom", name=f"ruler_{name}_back", type="box", size=".1 .012 .001",
                      rgba=".95 .89 .65 1", friction=".8 .01 .001")
        ET.SubElement(body, "geom", name=f"ruler_{name}_scale", type="plane", size=".1 .012 .001",
                      pos="0 0 .0011", material="ruler_surface", contype="0", conaffinity="0")
        # Raised black metric ticks remain real visible geometry even at low texture
        # resolution. They have no collision response and cannot affect measurements.
        for mm in range(201):
            length = .008 if mm % 10 == 0 else (.005 if mm % 5 == 0 else .0025)
            ET.SubElement(body, "geom", type="box", size=f".00010 {length / 2} .00002",
                          pos=f"{-.1 + mm * .001} {-.012 + length / 2} .00115",
                          rgba=".04 .04 .04 1", contype="0", conaffinity="0")

    def bind(self, model, data):
        super().bind(model, data)
        self.box_body_id = model.body("measured_box").id

    @staticmethod
    def _valid_answer(answer):
        if type(answer) is not dict or set(answer) != {"dimensions_m"}:
            return False
        values = answer["dimensions_m"]
        return (type(values) is list and len(values) == 3 and
                all(type(value) in (int, float) and math.isfinite(value) and 0 < value <= 2
                    for value in values))

    def evaluate(self, simulation, answer=None):
        if not self._valid_answer(answer):
            return {"success": False, "score": 0, "metrics": {"answer_valid": False}}
        errors = np.abs(np.array(answer["dimensions_m"], dtype=float) - self.dimensions_m)
        # Tiny arithmetic allowance makes the published inclusive boundary reliable.
        correct = errors <= self.tolerance_m + 1e-12
        return {"success": bool(np.all(correct)), "score": 100 * int(np.sum(correct)) / 3,
                "metrics": {"answer_valid": True, "dimensions_m": self.dimensions_m.tolist(),
                            "absolute_errors_m": errors.tolist(), "correct_dimensions": correct.tolist()}}
