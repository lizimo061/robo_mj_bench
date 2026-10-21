"""Trusted scoring fixtures and leak checks for physical metrology."""

import io
import json
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest

from robo_mj_bench.service import EpisodeService
from robo_mj_bench.tasks.registry import create_simulation


@pytest.fixture
def sim():
    simulation = create_simulation("measurement", 17)
    yield simulation
    simulation.close()


def answer(sim, offsets=None):
    values = sim.task.dimensions_m.copy()
    if offsets is not None:
        values += offsets
    return {"dimensions_m": values.tolist()}


def test_zero_partial_full_and_no_answer(sim):
    task = sim.task
    assert task.evaluate(sim)["score"] == 0
    assert task.evaluate(sim, {"dimensions_m": [.5] * 3})["score"] == 0
    assert task.evaluate(sim, answer(sim, [0, .01, .01]))["score"] == pytest.approx(100 / 3)
    assert task.evaluate(sim, answer(sim, [0, 0, .01]))["score"] == pytest.approx(200 / 3)
    result = task.evaluate(sim, answer(sim))
    assert result["success"] and result["score"] == 100
    assert len(result["metrics"]["absolute_errors_m"]) == 3


@pytest.mark.parametrize("sign", [-1, 1])
def test_inclusive_two_mm_tolerance(sim, sign):
    assert sim.task.evaluate(sim, answer(sim, [sign * .002] * 3))["success"]
    assert not sim.task.evaluate(sim, answer(sim, [sign * .002001] * 3))["success"]


@pytest.mark.parametrize("bad", [None, {}, {"dimensions_m": [1, 2]},
    {"dimensions_m": [True, .04, .05]}, {"dimensions_m": [".1", .04, .05]},
    {"dimensions_m": [float("nan"), .04, .05]}, {"dimensions_m": [float("inf"), .04, .05]},
    {"dimensions_m": [-.1, .04, .05]}, {"dimensions_m": [0, .04, .05]},
    {"dimensions_m": [2.01, .04, .05]}, {"dimensions_m": (.1, .04, .05)},
    {"dimensions_m": [.1, .04, .05], "debug": True}])
def test_invalid_answer_never_scores(sim, bad):
    result = sim.task.evaluate(sim, bad)
    assert result["score"] == 0 and not result["success"]


def test_reordered_edges_are_wrong(sim):
    values = sim.task.dimensions_m.tolist()
    values[0], values[1] = values[1], values[0]
    result = sim.task.evaluate(sim, {"dimensions_m": values})
    assert not result["success"]
    assert result["metrics"]["correct_dimensions"] == [False, False, True]


def test_goal_projection_contains_no_dimensions_seed_or_state(sim, tmp_path):
    service = EpisodeService(sim, "x" * 32, tmp_path)
    try:
        status, goal = service.handle("/v1/task", {}, "Bearer " + "x" * 32)
        assert status == 200
        assert set(goal) == {"task_id", "instruction", "approximation"}
        encoded = json.dumps(goal)
        for dimension in sim.task.dimensions_m:
            assert str(dimension) not in encoded
        assert "2 mm" in goal["instruction"] and "length,width,height" in goal["instruction"]
        assert "reference_image_base64" not in goal
        state = sim.robot_state()
        assert len(state["joint_positions"]) == 6
    finally:
        service.close()


def test_answer_submission_scores_privately_and_has_no_reward_feedback(sim, tmp_path):
    service = EpisodeService(sim, "x" * 32, tmp_path)
    try:
        status, body = service.handle("/v1/submit", {"answer": answer(sim)}, "Bearer " + "x" * 32)
        assert status == 200
        assert "score" not in body and "metrics" not in body
        report = json.loads((tmp_path / "result.json").read_text())
        assert report["success"] and report["score"] == 100
        assert report["metrics"]["answer_valid"]
    finally:
        service.close()


def test_missing_answer_submission_cannot_score(sim, tmp_path):
    service = EpisodeService(sim, "x" * 32, tmp_path)
    try:
        assert service.handle("/v1/submit", {}, "Bearer " + "x" * 32)[0] == 200
        report = json.loads((tmp_path / "result.json").read_text())
        assert not report["success"] and report["score"] == 0
    finally:
        service.close()


def test_rulers_have_visual_metric_ticks_and_constant_public_texture(sim):
    root = ET.fromstring(sim.xml)
    for name in ("table_x", "table_y", "upright"):
        body = root.find(f".//body[@name='ruler_{name}']")
        geoms = body.findall("geom")
        assert len(geoms) == 203  # collision back, scale face, 201 millimetre ticks
        assert geoms[0].attrib["size"] == ".1 .012 .001"
        assert geoms[1].attrib["material"] == "ruler_surface"
        assert len({g.attrib["pos"].split()[0] for g in geoms[2:]}) == 201
    with Image.open(io.BytesIO(sim.task.assets["ruler_scale.png"])) as scale:
        assert scale.size == (2048, 256)
        image = np.asarray(scale)
        assert np.any(image[180:230, 70:550] < 30), "SI unit text must be visible ink"
        assert np.any(image[:88, :3] < 30) and np.any(image[:88, -3:] < 30)
    other = create_simulation("measurement", 23)
    try:
        assert other.task.assets == sim.task.assets
        assert other.public_spec() == sim.public_spec()
        assert not np.array_equal(other.task.dimensions_m, sim.task.dimensions_m)
    finally:
        other.close()


def test_passive_physics_and_intrinsic_dimensions(sim):
    initial_dimensions = sim.model.geom("measured_box_geom").size.copy() * 2
    initial_qpos = sim.data.qpos.copy()
    for _ in range(600):
        sim._step()
    assert np.isfinite(sim.data.qpos).all()
    assert np.allclose(sim.task.dimensions_m, initial_dimensions)
    assert not np.array_equal(initial_qpos, sim.data.qpos), "box should settle through physics"
    assert sim.data.xpos[sim.task.box_body_id, 2] == pytest.approx(initial_dimensions[2] / 2, abs=.001)
    assert not sim.task.evaluate(sim)["success"]
    assert sim.task.evaluate(sim, answer(sim))["success"]
    assert not sim.task.requires_stability


def test_upright_ruler_has_true_table_level_origin_and_metric_orientation(sim):
    body = sim.model.body("ruler_upright").id
    rotation = sim.data.xmat[body].reshape(3, 3)
    origin = sim.data.xpos[body] + rotation @ np.array([-.1, 0, 0])
    end = sim.data.xpos[body] + rotation @ np.array([.1, 0, 0])
    assert origin[2] == pytest.approx(0, abs=1e-10)
    assert end[2] == pytest.approx(.2, abs=1e-10)
    assert rotation[:, 2] == pytest.approx([1, 0, 0], abs=1e-10)
