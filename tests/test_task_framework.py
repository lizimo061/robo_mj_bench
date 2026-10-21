"""Shared task lifecycle, public-goal projection, answer and settling boundaries."""

import base64
import io
import json
import xml.etree.ElementTree as ET

from PIL import Image, PngImagePlugin
import pytest

from robo_mj_bench.protocol import BadRequest, SafetyStop, validate_request
from robo_mj_bench.service import EpisodeService
from robo_mj_bench.task_simulation import TaskSimulation
from robo_mj_bench.tasks.base import Task
from robo_mj_bench.tasks.registry import create_task

AUTH = "Bearer " + "x" * 32


class FixtureTask(Task):
    task_id = "fixture_task"
    instruction = "Move the object to the visible goal."

    def build(self, world, root, rng):
        self.hidden_width = rng.uniform(0.01, 0.02)
        body = ET.SubElement(world, "body", name="fixture_object", pos="0.4 0 0.02")
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom", type="box", size=f"{self.hidden_width} 0.01 0.02", mass="0.05")
        self.predicate = lambda sim: False

    def evaluate(self, simulation, answer=None):
        success = self.predicate(simulation)
        return {"success": success, "score": 100 if success else 0, "metrics": {"hidden_width": self.hidden_width}}


def test_task_finish_requires_stable_terminal_window():
    task = FixtureTask()
    sim = TaskSimulation(task, 11)
    try:
        task.predicate = lambda sim: sim.data.time > 0.9
        result = sim.finish()
        assert not result["success"]
        assert result["score"] == 99
        assert not result["stable_terminal_window"]
        assert result["sim_time"] == pytest.approx(1.0)
        assert sim.finish() == result, "Finalizing twice must not keep advancing physics"
    finally:
        sim.close()


def test_complete_stable_task_gets_full_score_without_object_telemetry():
    task = FixtureTask()
    sim = TaskSimulation(task, 11)
    try:
        task.predicate = lambda sim: True
        state = sim.robot_state()
        assert len(state["joint_positions"]) == 6 < sim.model.nq
        result = sim.finish()
        assert result["success"] and result["score"] == 100
    finally:
        sim.close()


def test_brief_failure_between_old_sampling_points_prevents_full_success():
    task = FixtureTask()
    sim = TaskSimulation(task, 11)
    try:
        task.predicate = lambda sim: not (.624 < sim.data.time < .628)
        result = sim.finish()
        assert not result["success"] and not result["stable_terminal_window"]
        assert result["score"] == 99
    finally:
        sim.close()


def test_public_goal_projection_removes_extra_fields_and_png_metadata(tmp_path):
    task = FixtureTask()
    sim = TaskSimulation(task, 11)
    service = EpisodeService(sim, "x" * 32, tmp_path)
    image = io.BytesIO()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("seed", "private-answer-937")
    Image.new("RGB", (20, 20), "blue").save(image, format="PNG", pnginfo=metadata)
    sim.public_spec = lambda: {"task_id": "fixture_task", "instruction": "Match the target.",
        "approximation": "rigid_body", "reference_image_png": image.getvalue(),
        "seed": 937, "object_poses": [1, 2, 3]}
    try:
        status, body = service.handle("/v1/task", {}, AUTH)
        assert status == 200
        assert set(body) == {"task_id", "instruction", "approximation", "reference_image_base64"}
        png = base64.b64decode(body["reference_image_base64"])
        with Image.open(io.BytesIO(png)) as clean:
            assert clean.info == {}
        assert b"private-answer-937" not in png
        assert sim.data.time == 0
    finally:
        service.close()


@pytest.mark.parametrize("dimensions", [[True, 0.1, 0.2], [float("nan"), 0.1, 0.2],
                                        [0.1, 0.2], [0, 0.1, 0.2], ["0.1", 0.1, 0.2]])
def test_answer_is_strict_finite_measurement_vector(dimensions):
    with pytest.raises(BadRequest):
        validate_request("/v1/submit", {"answer": {"dimensions_m": dimensions}})


def test_nonmeasurement_task_rejects_answer_without_closing(tmp_path):
    sim = TaskSimulation(FixtureTask(), 11)
    service = EpisodeService(sim, "x" * 32, tmp_path)
    try:
        assert service.handle("/v1/submit", {"answer": {"dimensions_m": [0.1] * 3}}, AUTH)[0] == 400
        assert service.handle("/v1/robot_state", {}, AUTH)[0] == 200
    finally:
        service.close()


def test_safety_during_submission_settling_is_scored_terminal_failure(tmp_path):
    task = FixtureTask()
    sim = TaskSimulation(task, 11)
    task.predicate = lambda sim: True
    def unsafe(sim):
        if sim.data.time > 0.1:
            raise SafetyStop()
    task.after_step = unsafe
    service = EpisodeService(sim, "x" * 32, tmp_path)
    try:
        assert service.handle("/v1/submit", {}, AUTH) == (200, {"status": "safety_stop"})
        report = json.loads((tmp_path / "result.json").read_text())
        assert report["valid"] and report["score"] == 0 and not report["success"]
        assert report["termination"] == "safety_stop"
        assert service.handle("/v1/wait", {"duration": 0.1}, AUTH)[0] == 409
    finally:
        service.close()


def test_registry_does_not_accept_module_or_file_paths():
    with pytest.raises(ValueError):
        create_task("../../simulation.py")
