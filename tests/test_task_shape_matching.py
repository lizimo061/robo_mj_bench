"""Trusted state fixtures exercise the private scorer, never the agent API."""
import math

import mujoco
import numpy as np
import pytest

from robo_mj_bench.tasks.registry import create_simulation


@pytest.fixture
def sim():
    simulation = create_simulation("shape_matching", 17)
    yield simulation
    simulation.close()


def seat(sim, kinds=None, height=.020):
    for kind in kinds or sim.task.kinds:
        center, angle = sim.task.targets[kind]
        joint = sim.model.joint(f"joint_{kind}")
        address = int(joint.qposadr[0])
        sim.data.qpos[address:address + 7] = [*center, height, math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)


def test_zero_partial_full(sim):
    assert sim.task.evaluate(sim)["score"] == 0
    seat(sim, ["square"])
    assert sim.task.evaluate(sim)["score"] == pytest.approx(100 / 3)
    seat(sim)
    assert sim.task.evaluate(sim)["success"]


@pytest.mark.parametrize("height", [.045, -.02, .04])
def test_perched_above_or_under_board_rejected(sim, height):
    seat(sim, height=height)
    assert sim.task.evaluate(sim)["score"] == 0


def test_wrong_assignment_and_triangle_orientation(sim):
    seat(sim)
    tri = sim.model.joint("joint_triangle").qposadr[0]
    sim.data.qpos[tri + 3:tri + 7] = [1, 0, 0, 0]
    # Force a half turn, rather than depending on the randomized target angle.
    angle = sim.task.targets["triangle"][1] + math.pi
    sim.data.qpos[tri + 3:tri + 7] = [math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
    mujoco.mj_forward(sim.model, sim.data)
    assert not sim.task.evaluate(sim)["metrics"]["seated"]["triangle"]
    square = sim.model.joint("joint_square").qposadr[0]
    sim.data.qpos[square:square + 2] = sim.task.targets["circle"][0]
    mujoco.mj_forward(sim.model, sim.data)
    assert not sim.task.evaluate(sim)["metrics"]["seated"]["square"]


def test_velocity_rejected(sim):
    seat(sim)
    joint = sim.model.joint("joint_square")
    sim.data.qvel[int(joint.dofadr[0])] = .5
    mujoco.mj_forward(sim.model, sim.data)
    assert not sim.task.evaluate(sim)["metrics"]["seated"]["square"]


def test_real_open_sockets_stably_support_pieces(sim):
    seat(sim)
    for _ in range(600):
        sim._step()
    result = sim.task.evaluate(sim)
    assert result["success"], result
    assert np.isfinite(sim.data.qpos).all()
    for kind in sim.task.kinds:
        assert sim.data.xpos[sim.task.body_ids[kind], 2] == pytest.approx(.020, abs=.001)


def test_physical_rim_blocks_wrong_orientation(sim):
    seat(sim)
    tri = int(sim.model.joint("joint_triangle").qposadr[0])
    angle = sim.task.targets["triangle"][1] + math.pi
    sim.data.qpos[tri + 2] = .06
    sim.data.qpos[tri + 3:tri + 7] = [math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(600):
        sim._step()
    assert not sim.task.evaluate(sim)["metrics"]["seated"]["triangle"]
    assert sim.data.xpos[sim.task.body_ids["triangle"], 2] > .035


def test_reference_is_public_target_png(sim):
    assert sim.task.reference_image().startswith(b"\x89PNG\r\n\x1a\n")
    before = sim.task.reference_image()
    seat(sim)
    assert sim.task.reference_image() == before


def test_robot_contact_disqualifies_seated_piece(sim):
    seat(sim)
    # Trusted fixture injects a contact to exercise held/support rejection independently
    # of seating. The real contact iterator is used, including robot ancestry.
    assert sim.data.ncon > 0
    contact = sim.data.contact[0]
    contact.geom1 = sim.model.geom("piece_geom_square").id
    arm = sim.model.body("arm").id
    robot_geom = next(i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] == arm)
    contact.geom2 = robot_geom
    assert not sim.task.evaluate(sim)["metrics"]["seated"]["square"]


def test_randomization_changes_private_layout():
    a = create_simulation("shape_matching", 10)
    b = create_simulation("shape_matching", 11)
    try:
        assert any(not np.array_equal(a.task.targets[k][0], b.task.targets[k][0]) for k in a.task.kinds)
        assert a.task.reference_image() != b.task.reference_image()
    finally:
        a.close()
        b.close()
