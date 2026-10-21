"""Trusted geometry fixtures test scoring independently of an agent policy."""
import math

import mujoco
import numpy as np
import pytest

from robo_mj_bench.task_simulation import TaskSimulation
from robo_mj_bench.tasks.tidying import TidyingTask


@pytest.fixture
def sim():
    value = TaskSimulation(TidyingTask(), 17)
    yield value
    value.close()


def place(sim, name, xyz, quaternion=(1, 0, 0, 0)):
    address = int(sim.model.joint(f"{name}_joint").qposadr[0])
    sim.data.qpos[address:address + 7] = [*xyz, *quaternion]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)


def pack(sim, count=4, lid=True):
    for name, (x, y) in zip(sim.task.block_names[:count], [(-.023, -.023), (-.023, .023), (.023, -.023), (.023, .023)]):
        place(sim, name, [*(sim.task.center + [x, y]), .023])
    if lid:
        place(sim, "tidy_lid", [*sim.task.center, .062])


def test_zero_partial_full(sim):
    assert sim.task.evaluate(sim)["score"] == 0
    pack(sim, 2)
    assert sim.task.evaluate(sim)["score"] == 40
    pack(sim)
    assert sim.task.evaluate(sim)["success"]
    assert sim.task.evaluate(sim)["score"] == 100


@pytest.mark.parametrize("z", [-.025, .055, .075])
def test_under_floor_above_rim_and_above_lid_rejected(sim, z):
    pack(sim)
    place(sim, sim.task.block_names[0], [*sim.task.center, z])
    result = sim.task.evaluate(sim)
    assert result["score"] == 60
    assert not result["success"]


def test_center_inside_but_corner_outside_rejected(sim):
    pack(sim)
    place(sim, sim.task.block_names[0], [*(sim.task.center + [.05, 0]), .023])
    assert sim.task.evaluate(sim)["score"] == 60


@pytest.mark.parametrize("angle", [.1, math.pi / 2])
def test_tilted_or_wrong_yaw_lid_rejected(sim, angle):
    pack(sim)
    quaternion = [math.cos(angle / 2), math.sin(angle / 2), 0, 0] if angle == .1 else [math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
    place(sim, "tidy_lid", [*sim.task.center, .062], quaternion)
    assert sim.task.evaluate(sim)["score"] == 80


def test_floating_and_offset_lid_rejected(sim):
    pack(sim)
    place(sim, "tidy_lid", [*sim.task.center, .08])
    assert sim.task.evaluate(sim)["score"] == 80
    place(sim, "tidy_lid", [*(sim.task.center + [.01, 0]), .062])
    assert sim.task.evaluate(sim)["score"] == 80


def test_fast_block_and_lid_rejected(sim):
    pack(sim)
    joint = sim.model.joint(f"{sim.task.block_names[0]}_joint")
    sim.data.qvel[int(joint.dofadr[0])] = .3
    mujoco.mj_forward(sim.model, sim.data)
    assert sim.task.evaluate(sim)["score"] == 60
    pack(sim)
    joint = sim.model.joint("tidy_lid_joint")
    sim.data.qvel[int(joint.dofadr[0]) + 3] = .3
    mujoco.mj_forward(sim.model, sim.data)
    assert sim.task.evaluate(sim)["score"] == 80


@pytest.mark.parametrize("body_name,expected", [("tidy_block_0", 60), ("tidy_lid", 80)])
def test_robot_support_disqualifies(sim, body_name, expected):
    pack(sim)
    assert sim.data.ncon
    contact = sim.data.contact[0]
    contact.geom1 = sim.model.geom(f"{body_name}_geom").id
    arm = sim.model.body("arm").id
    contact.geom2 = next(i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] == arm)
    assert sim.task.evaluate(sim)["score"] == expected


def test_real_floor_walls_and_lid_support_stable_full_success(sim):
    pack(sim)
    result = sim.finish()
    assert result["success"], result
    assert result["stable_terminal_window"]
    assert np.isfinite(sim.data.qpos).all()
    assert sim.data.xpos[sim.task.lid_id, 2] == pytest.approx(.062, abs=.001)


def test_passive_unpacked_scene_cannot_self_complete(sim):
    assert sim.finish()["score"] == 0


def test_physical_floor_blocks_and_walls_prevent_passage(sim):
    # Let a block drop above a wall: it must land on the rim rather than pass
    # through an imaginary scoring-only boundary.
    place(sim, sim.task.block_names[0], [*(sim.task.center + [.060, 0]), .09])
    for _ in range(600):
        sim._step()
    corners_z = sim.data.xpos[sim.task.block_ids[0], 2]
    assert corners_z > .065
    assert not sim.task.evaluate(sim)["metrics"]["packed"][0]


def test_public_spec_only_contains_plain_goal(sim):
    spec = sim.public_spec()
    assert spec["reference_image_png"] is None
    assert "0.43" not in spec["instruction"]
    assert set(spec) == {"task_id", "instruction", "approximation", "reference_image_png"}
