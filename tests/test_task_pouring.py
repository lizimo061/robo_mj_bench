"""Trusted scorer fixtures; the agent has no access to these state mutations."""
import math

import mujoco
import numpy as np
import pytest

from robo_mj_bench.task_simulation import TaskSimulation
from robo_mj_bench.tasks.pouring import PouringTask


@pytest.fixture
def sim():
    simulation = TaskSimulation(PouringTask(), 7)
    yield simulation
    simulation.close()


def place(sim, body, position, quaternion=(1, 0, 0, 0)):
    address = sim.model.jnt_qposadr[sim.model.body_jntadr[body]]
    sim.data.qpos[address:address+7] = [*position, *quaternion]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)


def transfer(sim, count):
    task = sim.task
    origin = sim.data.xpos[task._target].copy()
    points = [(0., 0.)] + [(.015*math.cos(i*math.pi/3), .015*math.sin(i*math.pi/3)) for i in range(6)]
    for index, body in enumerate(task._beads[:count]):
        dx, dy = points[index % 7]
        place(sim, body, origin + [dx, dy, .0112 + (index//7)*.0102])


def test_empty_target_scores_zero(sim):
    result = sim.task.evaluate(sim)
    assert result["score"] == 0
    assert not result["success"]
    assert result["metrics"]["beads_in_source"] == 21


def test_partial_transfer_counts_mass(sim):
    transfer(sim, 10)
    result = sim.task.evaluate(sim)
    assert not result["success"]
    assert result["score"] == pytest.approx(1000/21)
    metrics = result["metrics"]
    assert metrics["beads_in_source"] + metrics["beads_in_target"] + metrics["beads_outside_cups"] == 21
    assert metrics["total_mass_kg"] == .021


def test_complete_transfer_and_passive_stable_physics(sim):
    transfer(sim, 21)
    assert sim.task.evaluate(sim)["success"]
    sim.wait(1.)
    result = sim.task.evaluate(sim)
    assert result["success"], result
    assert result["score"] == 100
    assert result["metrics"]["beads_in_target"] == 21
    assert sim.finish()["success"]


def test_spilled_beads_preclude_success(sim):
    transfer(sim, 19)
    for i, bead in enumerate(sim.task._beads[19:]):
        place(sim, bead, [.7, -.2+i*.03, .005])
    result = sim.task.evaluate(sim)
    assert not result["success"]
    assert result["metrics"]["beads_spilled"] == 2
    assert result["score"] < 100


def test_recovered_spill_still_counted(sim):
    bead = sim.task._beads[0]
    place(sim, bead, [.7, -.2, .005])
    sim.task.after_step(sim)
    transfer(sim, 21)
    assert sim.task.evaluate(sim)["metrics"]["beads_spilled"] == 1


def test_floating_receiver_cannot_score(sim):
    transfer(sim, 21)
    task = sim.task
    for body in [task._target, *task._beads]:
        place(sim, body, sim.data.xpos[body].copy()+[0, 0, .2])
    result = task.evaluate(sim)
    assert result["metrics"]["beads_in_target"] == 21
    assert result["score"] == 0


def test_moving_bead_cannot_score(sim):
    transfer(sim, 21)
    body = sim.task._beads[0]
    address = sim.model.jnt_dofadr[sim.model.body_jntadr[body]]
    sim.data.qvel[address] = 1
    mujoco.mj_forward(sim.model, sim.data)
    assert sim.task.evaluate(sim)["score"] == 0


def test_robot_held_receiver_cannot_score(sim):
    transfer(sim, 21)
    # Inject a trusted contact fixture, rather than guessing robot IK in a scorer test.
    robot_geom = sim.model.body("left_finger").geomadr[0]
    target_geom = sim.model.geom("pour_target_base").id
    contact = mujoco.MjContact()
    contact.geom1, contact.geom2 = int(robot_geom), target_geom
    contact.dist = -.0001
    mujoco.mj_addContact(sim.model, sim.data, contact)
    result = sim.task.evaluate(sim)
    assert not result["metrics"]["released"]
    assert result["score"] == 0


def test_invalid_state_cannot_score(sim):
    transfer(sim, 21)
    sim.data.qvel[0] = np.nan
    result = sim.task.evaluate(sim)
    assert not result["success"] and result["score"] == 0


def test_passive_source_keeps_contents(sim):
    sim.wait(1.)
    result = sim.task.evaluate(sim)
    assert result["metrics"]["beads_in_source"] == 21, result
    assert result["score"] == 0
    assert np.isfinite(sim.data.qpos).all()


def test_public_spec_has_no_private_state(sim):
    spec = sim.public_spec()
    assert spec["approximation"] == "granular_transfer_proxy"
    assert set(spec) == {"task_id", "instruction", "approximation", "reference_image_png"}


def test_overlapping_cups_do_not_double_count_or_earn_transfer_credit(sim):
    task = sim.task
    place(sim, task._target, sim.data.xpos[task._source].copy())
    result = task.evaluate(sim)
    metrics = result["metrics"]
    assert not result["success"] and result["score"] == 0
    assert not metrics["cups_separate"] and metrics["ambiguous_beads"] > 0
    assert metrics["beads_in_source"] + metrics["beads_in_target"] + metrics["beads_outside_cups"] == 21
