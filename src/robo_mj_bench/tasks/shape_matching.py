"""Three physical shape sockets. Geometry and all scoring stay trusted-side."""
import io
import math
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

from .base import Task, body_is_slow, touching_robot


def _text(values):
    return " ".join(str(float(v)) for v in values)


def _polygon(kind):
    if kind == "square":
        return np.array([[-.018, -.018], [.018, -.018], [.018, .018], [-.018, .018]])
    count = 3 if kind == "triangle" else 48
    radius = .025 if kind == "triangle" else .020
    return np.array([[radius * math.cos(2 * math.pi * i / count), radius * math.sin(2 * math.pi * i / count)] for i in range(count)])


def _prism(asset, name, polygon, bottom, top):
    vertices = [[x, y, z] for z in (bottom, top) for x, y in polygon]
    ET.SubElement(asset, "mesh", name=name, vertex=_text(np.array(vertices).ravel()))


class ShapeMatchingTask(Task):
    task_id = "shape_matching"
    instruction = ("Insert the three colored shapes into their matching sockets. The public target image "
                   "shows the desired socket pattern from above (top is +y, right is +x). "
                   "Seat each piece flat on the socket floor, release it, and withdraw the robot.")

    def build(self, world, root, rng):
        ET.SubElement(root.find("option"), "flag", multiccd="enable")
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        self.kinds = ["square", "circle", "triangle"]
        self.order = self.kinds.copy()
        rng.shuffle(self.order)
        self.targets = {}
        colors = {"square": (.85, .15, .10, 1), "circle": (.10, .35, .90, 1), "triangle": (.1, .7, .25, 1)}
        self.colors = colors
        for slot, kind in enumerate(self.order):
            center = np.array([.45 + rng.uniform(-.006, .006), (slot - 1) * .11])
            angle = rng.uniform(-.25, .25)
            self.targets[kind] = (center, angle)
            polygon = _polygon(kind) * 1.035
            outer = polygon * 1.65
            socket = ET.SubElement(world, "body", name=f"socket_{kind}", pos=_text([*center, 0]), euler=f"0 0 {angle}")
            # Bottom is real collision geometry; ring prisms leave an open cavity above it.
            ET.SubElement(socket, "geom", type="box", size=".045 .045 .004", pos="0 0 .004", rgba=".24 .24 .27 1", friction="1 .01 .001", solref=".006 1")
            for i in range(len(polygon)):
                j = (i + 1) % len(polygon)
                _prism(asset, f"rim_{kind}_{i}", [polygon[i], outer[i], outer[j], polygon[j]], .008, .034)
                ET.SubElement(socket, "geom", type="mesh", mesh=f"rim_{kind}_{i}", rgba=_text(colors[kind]), friction=".8 .01 .001")
        positions = [-.13, 0, .13]
        rng.shuffle(positions)
        for kind, y in zip(self.kinds, positions):
            body = ET.SubElement(world, "body", name=f"piece_{kind}", pos=f"{.29 + rng.uniform(-.01,.01)} {y} .013", euler=f"0 0 {rng.uniform(-math.pi,math.pi)}")
            ET.SubElement(body, "freejoint", name=f"joint_{kind}")
            attrs = dict(name=f"piece_geom_{kind}", mass=".045", rgba=_text(colors[kind]), friction="1 .01 .001", solref=".006 1")
            if kind == "circle":
                ET.SubElement(body, "geom", type="cylinder", size=".020 .012", **attrs)
            elif kind == "square":
                ET.SubElement(body, "geom", type="box", size=".018 .018 .012", **attrs)
            else:
                _prism(asset, "triangle_piece", _polygon(kind), -.012, .012)
                ET.SubElement(body, "geom", type="mesh", mesh="triangle_piece", **attrs)

    def bind(self, model, data):
        super().bind(model, data)
        self.body_ids = {k: model.body(f"piece_{k}").id for k in self.kinds}

    def reference_image(self):
        image = Image.new("RGB", (440, 360), "#f3f1eb")
        draw = ImageDraw.Draw(image)
        draw.text((12, 8), "TARGET: seated pieces (+y up, +x right)", fill="black")
        for slot, kind in enumerate(self.order):
            center = np.array([260, 280 - slot * 105])
            angle = self.targets[kind][1]
            c, s = math.cos(angle), math.sin(angle)
            poly = (_polygon(kind) @ np.array([[c, s], [-s, c]])) * [1550, -1550]
            color = tuple(round(v * 255) for v in self.colors[kind][:3])
            draw.polygon([tuple(v + center) for v in poly * 1.65], fill="#454549")
            draw.polygon([tuple(v + center) for v in poly], fill=color)
            draw.text((25, int(center[1] - 7)), kind, fill="black")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def evaluate(self, simulation, answer=None):
        seated = {}
        for kind, body_id in self.body_ids.items():
            center, angle = self.targets[kind]
            rotation = self.data.xmat[body_id].reshape(3, 3)
            pos = self.data.xpos[body_id]
            c, s = math.cos(angle), math.sin(angle)
            target_rot = np.array([[c, -s], [s, c]])
            # Test the full top/bottom surface against the actual convex aperture.
            outline = _polygon(kind)
            points = np.array([[x, y, z] for z in (-.012, .012) for x, y in outline])
            world = points @ rotation.T + pos
            local = (world[:, :2] - center) @ target_rot
            aperture = _polygon(kind) * 1.035
            edges = np.roll(aperture, -1, axis=0) - aperture
            relative = local[:, None, :] - aperture[None, :, :]
            cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
            fits = bool(np.all(cross >= -1e-7))
            height = bool(np.min(world[:, 2]) >= .007 and np.max(world[:, 2]) <= .034)
            # Tight floor height excludes perched and below-board states.
            floor = abs(float(pos[2]) - .020) < .0015
            upright = rotation[2, 2] > .995
            seated[kind] = bool(fits and height and floor and upright and
                                body_is_slow(self.model, self.data, body_id) and
                                not touching_robot(self.model, self.data, [body_id]))
        count = sum(seated.values())
        return {"success": count == 3, "score": 100 * count / 3, "metrics": {"seated": seated, "count": count}}
