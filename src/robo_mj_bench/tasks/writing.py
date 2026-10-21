"""Contact-generated ink and a frozen, deliberately narrow tracing evaluator."""
import io
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageFilter

from .base import Task


class WritingTask(Task):
    task_id = "writing_yam_v1"
    approximation = "contact_writing_proxy"
    requires_stability = False
    instruction = (
        "Grasp the blue marker and write HELLO WORLD on the white paper, matching "
        "the supplied target image's position, scale, and orientation. The top of "
        "the target image corresponds to positive world Y. Ink requires the black "
        "tip to touch the paper while the marker points upward, with gentle pressure. "
        "This task scores contact tracing of a fixed template, not general handwriting."
    )
    WIDTH, HEIGHT = 240, 90
    PAPER_XY = np.array([0.375, 0.065])
    PAPER_SIZE = np.array([0.24, 0.09])
    PAPER_Z = 0.003
    INK_RADIUS = 0.0011

    def __init__(self):
        super().__init__()
        # Versioned 5x7 glyphs, not a host font, so Pillow/font updates cannot
        # change a benchmark target or scoring mask across machines.
        glyphs = {
            "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
            "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
            "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
            "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
            "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
            "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
            "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
            " ": ("00000",) * 7,
        }
        strip = np.zeros((7, 65), dtype=np.uint8)
        for i, letter in enumerate("HELLO WORLD"):
            strip[:, i*6:i*6+5] = [[int(bit)*255 for bit in row] for row in glyphs[letter]]
        strip = np.repeat(np.repeat(strip, 3, axis=0), 3, axis=1)
        target = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)
        x0, y0 = (self.WIDTH-strip.shape[1])//2, (self.HEIGHT-strip.shape[0])//2
        target[y0:y0+strip.shape[0], x0:x0+strip.shape[1]] = strip
        image = Image.fromarray(target)
        self._target = np.array(image) > 127
        self._tolerant_target = np.array(image.filter(ImageFilter.MaxFilter(5))) > 127
        self._ink = np.zeros_like(self._target)
        self._outside_samples = 0
        self._accepted_samples = 0
        self._rejected_pressure_samples = 0

    def build(self, world, root, rng):
        self.PAPER_XY = type(self).PAPER_XY + np.array([rng.uniform(-.006, .006), rng.uniform(-.006, .006)])
        ET.SubElement(world, "geom", name="writing_paper", type="box",
                      pos=f"{self.PAPER_XY[0]} {self.PAPER_XY[1]} {self.PAPER_Z / 2}",
                      size=f"{self.PAPER_SIZE[0] / 2} {self.PAPER_SIZE[1] / 2} {self.PAPER_Z / 2}",
                      rgba="0.98 0.98 0.97 1", friction="0.65 0.005 0.0001")
        for axis in (0, 1):
            for sign in (-1, 1):
                center = np.array([*self.PAPER_XY, self.PAPER_Z + .0001])
                center[axis] += sign * self.PAPER_SIZE[axis] / 2
                half = [self.PAPER_SIZE[0] / 2, self.PAPER_SIZE[1] / 2, .00005]
                half[axis] = .0005
                ET.SubElement(world, "geom", type="box", pos=" ".join(map(str, center)),
                              size=" ".join(map(str, half)), contype="0", conaffinity="0", rgba="0.02 0.02 0.02 1")
        # A freely graspable tool, initially lying horizontally below the paper.
        marker = ET.SubElement(world, "body", name="writing_marker", pos="0.32 -0.09 0.009")
        ET.SubElement(marker, "freejoint", name="writing_marker_free")
        ET.SubElement(marker, "geom", name="writing_marker_shaft", type="capsule",
                      fromto="0.009 0 0 0.105 0 0", size="0.007", mass="0.025",
                      rgba="0.06 0.22 0.85 1", friction="1 0.01 0.001")
        ET.SubElement(marker, "geom", name="writing_marker_tip", type="sphere",
                      pos="0 0 0", size="0.003", mass="0.002", rgba="0.02 0.02 0.02 1",
                      friction="0.55 0.005 0.0001", solref="0.006 1")

    def bind(self, model, data):
        super().bind(model, data)
        self._paper_id = model.geom("writing_paper").id
        self._tip_id = model.geom("writing_marker_tip").id
        self._table_id = model.geom("table").id
        self._marker_id = model.body("writing_marker").id

    def reference_image(self):
        image = Image.fromarray(np.where(self._target, 0, 255).astype(np.uint8)).convert("RGB")
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    def _record_contact(self, position, normal_force, upright):
        """Private contact accumulator, never a command or model-facing endpoint."""
        if not upright:
            return
        if not 0.03 <= normal_force <= 8.0:
            self._rejected_pressure_samples += 1
            return
        xy = np.asarray(position)[:2]
        uv = (xy - self.PAPER_XY) / self.PAPER_SIZE + 0.5
        if np.any(uv < 0) or np.any(uv > 1):
            self._outside_samples += 1
            return
        x = min(self.WIDTH - 1, int(uv[0] * self.WIDTH))
        y = min(self.HEIGHT - 1, int((1 - uv[1]) * self.HEIGHT))
        radius = self.INK_RADIUS * self.WIDTH / self.PAPER_SIZE[0]
        yy, xx = np.ogrid[max(0, y - 2):min(self.HEIGHT, y + 3), max(0, x - 2):min(self.WIDTH, x + 3)]
        self._ink[max(0, y - 2):min(self.HEIGHT, y + 3), max(0, x - 2):min(self.WIDTH, x + 3)] |= (xx-x)**2 + (yy-y)**2 <= radius**2
        self._accepted_samples += 1

    def after_step(self, simulation):
        # Local +X extends from the tip through the shaft; require a writing pose.
        upright = self.data.xmat[self._marker_id].reshape(3, 3)[2, 0] > 0.45
        force = np.zeros(6)
        for i, contact in enumerate(self.data.contact[:self.data.ncon]):
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair not in ({self._paper_id, self._tip_id}, {self._table_id, self._tip_id}):
                continue
            # Side-face collisions and paper-edge penetration cannot draw ink.
            if self._paper_id in pair and contact.pos[2] < self.PAPER_Z - .0008:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, force)
            if self._table_id in pair:
                if upright and .03 <= force[0] <= 8.0:
                    self._outside_samples += 1
                continue
            self._record_contact(contact.pos, max(0.0, float(force[0])), upright)

    def decorate_scene(self, scene, simulation):
        # Render-only ink: no collision bodies, state writes, or object teleportation.
        for y, x in np.argwhere(self._ink):
            if scene.ngeom >= scene.maxgeom:
                break
            xy = self.PAPER_XY + np.array([(x + .5) / self.WIDTH - .5,
                                          .5 - (y + .5) / self.HEIGHT]) * self.PAPER_SIZE
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                               np.array([0.00075, 0.00075, 0.00006]),
                               np.array([*xy, self.PAPER_Z + 0.0001]), np.eye(3).ravel(),
                               np.array([0.015, 0.015, 0.02, 1.0], dtype=np.float32))
            scene.ngeom += 1

    def evaluate(self, simulation, answer=None):
        ink_count = int(self._ink.sum())
        # Coverage tolerates one pixel of tool discretization; precision penalizes
        # marking the entire sheet, blank space, and excessive scribbling.
        dilated_ink = np.array(Image.fromarray(self._ink.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))) > 0
        coverage = float((dilated_ink & self._target).sum() / self._target.sum())
        precision = float((self._ink & self._tolerant_target).sum() / ink_count) if ink_count else 0.0
        bounds_ok = self._outside_samples == 0
        success = coverage >= .90 and precision >= .85 and bounds_ok
        f1 = 2 * coverage * precision / (coverage + precision) if coverage + precision else 0.0
        return {"success": success, "score": 100.0 if success else min(99.0, 100 * f1),
                "metrics": {"coverage": coverage, "precision": precision, "in_bounds": bounds_ok,
                            "ink_pixels": ink_count, "valid_contact_samples": self._accepted_samples,
                            "outside_contact_samples": self._outside_samples,
                            "rejected_pressure_samples": self._rejected_pressure_samples,
                            "recognizer": "frozen_HELLO_WORLD_mask_v1"}}
