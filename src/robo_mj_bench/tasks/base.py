"""Internal task contract. Task state and evaluation remain server-only."""

import xml.etree.ElementTree as ET

import numpy as np


class Task:
    task_id = "abstract"
    instruction = ""
    approximation = "rigid_body"
    settle_seconds = 1.0
    requires_stability = True
    accepts_answer = False

    def __init__(self):
        self.assets = {}

    def build(self, world: ET.Element, root: ET.Element, rng):
        raise NotImplementedError

    def bind(self, model, data):
        self.model, self.data = model, data

    def after_step(self, simulation):
        pass

    def evaluate(self, simulation, answer=None):
        """Return {'success': bool, 'score': 0..100, 'metrics': private dict}."""
        raise NotImplementedError

    def reference_image(self):
        """Optional PUBLIC target image; never an annotated current-state image."""
        return None

    def decorate_scene(self, scene, simulation):
        """Optional render-only geometry, e.g. contact-generated ink. No physics edits."""
        pass


def box_corners(model, data, geom_id):
    """World corners of a MuJoCo box, including its actual orientation."""
    offsets = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return data.geom_xpos[geom_id] + (offsets * model.geom_size[geom_id]) @ data.geom_xmat[geom_id].reshape(3, 3).T


def robot_body_ids(model):
    root = model.body("arm").id
    result = set()
    for body in range(1, model.nbody):
        ancestor = body
        while ancestor:
            if ancestor == root:
                result.add(body)
                break
            ancestor = int(model.body_parentid[ancestor])
    return result


def touching_robot(model, data, body_ids):
    robot = robot_body_ids(model)
    bodies = set(body_ids)
    for contact in data.contact[:data.ncon]:
        a, b = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
        if (a in bodies and b in robot) or (b in bodies and a in robot):
            return True
    return False


def body_is_slow(model, data, body_id, tolerance=0.03):
    # cvel includes rotation and translation; both must be small at evaluation.
    return bool(np.max(np.abs(data.cvel[body_id])) < tolerance)
