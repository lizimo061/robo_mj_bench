"""Private fixtures test complete compound footprint and physical release checks."""
import math

import mujoco
import numpy as np
import pytest

from robo_mj_bench.tasks.registry import create_simulation


@pytest.fixture
def sim():
    simulation = create_simulation("placement", 17)
    yield simulation
    simulation.close()


def place(sim, offset=(0, 0), height=.0119, angle=None, tilt=0):
    task = sim.task
    angle = task.target_angle if angle is None else angle
    center = task.target_center + np.asarray(offset) @ task.target_rotation.T
    address = int(sim.model.joint("placement_freejoint").qposadr[0])
    quat = np.empty(4)
    mujoco.mju_euler2Quat(quat, np.array([tilt, 0, angle]), "xyz")
    sim.data.qpos[address:address+7] = [*center, height, *quat]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)


def test_initial_zero_partial_and_full(sim):
    assert sim.task.evaluate(sim)["score"] == 0
    place(sim, offset=(.08, 0))
    partial = sim.task.evaluate(sim)
    assert 0 < partial["score"] < 99
    assert not partial["success"]
    place(sim)
    assert sim.task.evaluate(sim)["success"]
    assert sim.task.evaluate(sim)["score"] == 100


def test_tail_overhang_with_center_inside(sim):
    place(sim, offset=(-.025, 0))
    result = sim.task.evaluate(sim)
    assert not result["success"]
    assert .90 < result["metrics"]["containment_fraction"] < 1


def test_rotated_footprint_overhang(sim):
    place(sim, angle=sim.task.target_angle + math.pi / 2)
    result = sim.task.evaluate(sim)
    assert not result["success"]
    assert 0 < result["score"] < 99


@pytest.mark.parametrize("height", [.04, .20, -.05])
def test_airborne_and_under_table_rejected(sim, height):
    place(sim, height=height)
    result = sim.task.evaluate(sim)
    assert result["metrics"]["contained"]
    assert not result["success"] and result["score"] == 0


def test_upside_down_rejected(sim):
    place(sim, tilt=math.pi)
    assert sim.task.evaluate(sim)["score"] == 0


def test_moving_toy_rejected(sim):
    place(sim)
    address = int(sim.model.joint("placement_freejoint").dofadr[0])
    sim.data.qvel[address] = .15
    mujoco.mj_forward(sim.model, sim.data)
    assert not sim.task.evaluate(sim)["success"]


def test_robot_contact_rejected(sim):
    place(sim)
    assert sim.data.ncon > 0
    contact = sim.data.contact[0]
    contact.geom1 = sim.task.geom_ids[0]
    arm = sim.model.body("arm").id
    contact.geom2 = next(i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] == arm)
    result = sim.task.evaluate(sim)
    assert result["metrics"]["robot_contact"]
    assert not result["success"] and result["score"] == 0


def test_passive_settling_and_terminal_stability(sim):
    place(sim, height=.014)
    result = sim.finish()
    assert result["success"], result
    assert result["stable_terminal_window"]
    assert result["score"] == 100
    assert np.isfinite(sim.data.qpos).all()


def test_initial_scene_passive_physics_does_not_complete(sim):
    for _ in range(500):
        sim._step()
    assert sim.task.evaluate(sim)["score"] == 0
    assert np.isfinite(sim.data.qpos).all()


def test_private_layout_is_not_public_spec(sim):
    public = sim.public_spec()
    assert public["task_id"] == "placement"
    assert set(public) == {"task_id", "instruction", "approximation", "reference_image_png"}
    assert sim.task.reference_image() is None
