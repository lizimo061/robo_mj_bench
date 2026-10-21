"""Private task lifecycle over the same standard YAM robot/API."""

import hashlib
import math
import random
import xml.etree.ElementTree as ET

import mujoco
from . import __version__

from .protocol import BadRequest, SafetyStop, TIMESTEP
from .robot_assets import ROBOT_ID, provenance
from .scene import yam_scene
from .simulation import ButtonSimulation


class TaskSimulation(ButtonSimulation):
    def __init__(self, task, seed):
        self.task = task
        root = yam_scene(task.task_id)
        task.build(root.find("worldbody"), root, random.Random(seed))
        self._initialize_robot(seed, ET.tostring(root, encoding="unicode"), task.assets)
        task.bind(self.model, self.data)
        self._answer = None
        self._report = None
        self._safety_stopped = False
        self._settle_on_finish = True

    def _step(self):
        self._physics_step()
        try:
            self.task.after_step(self)
        except SafetyStop:
            self._closed = True
            self._safety_stopped = True
            raise

    def public_spec(self):
        return {"task_id": self.task.task_id, "instruction": self.task.instruction,
                "approximation": self.task.approximation,
                "reference_image_png": self.task.reference_image()}

    def set_answer(self, answer):
        if not self.task.accepts_answer:
            raise BadRequest("Task does not accept an answer")
        self._answer = answer

    def prepare_finish(self, reason):
        self._settle_on_finish = reason == "submitted"

    def finish(self):
        if self._report is not None:
            return dict(self._report)
        samples = []
        settling = 0.0
        if self._settle_on_finish and not self._closed and not self._invalid_physics:
            ticks = round(self.task.settle_seconds / TIMESTEP)
            for tick in range(ticks):
                try:
                    self._step()
                except SafetyStop:
                    break
                settling += TIMESTEP
                if self.task.requires_stability and tick >= ticks // 2:
                    samples.append(bool(self.task.evaluate(self, self._answer)["success"]))
        result = self.task.evaluate(self, self._answer)
        stable = not self.task.requires_stability or (bool(samples) and all(samples))
        safe = not self._invalid_physics and not self._closed
        success = bool(result["success"]) and stable and safe
        score = float(result["score"])
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise RuntimeError("Invalid private task score")
        score = 100.0 if success else (min(score, 99.0) if safe else 0.0)
        self._closed = True
        self._report = {
            "task": self.task.task_id, "robot": ROBOT_ID,
            "benchmark_version": __version__,
            "robot_source_commit": provenance()["commit"], "approximation": self.task.approximation,
            "success": success, "score": score, "safe": safe,
            "safety_stopped": self._safety_stopped,
            "safety_scope": "physics_validity_and_task_limits; no general damage model",
            "stable_terminal_window": stable, "settle_sim_seconds": settling,
            "metrics": result.get("metrics", {}), "answer": self._answer,
            "sim_time": float(self.data.time), "seed": self.seed,
            "final_qpos": self.data.qpos.tolist(),
            "scene_sha256": hashlib.sha256(self.xml.encode()).hexdigest(),
            "task_assets_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in self.task.assets.items()},
            "mujoco_version": mujoco.__version__,
        }
        return dict(self._report)
