"""Trusted fixtures exercise scoring; these do not claim agent-policy performance."""
import math

import mujoco
import numpy as np
import pytest

from robo_mj_bench.task_simulation import TaskSimulation
from robo_mj_bench.tasks.tower import TowerTask


@pytest.fixture
def sim():
    simulation = TaskSimulation(TowerTask(), 17)
    yield simulation
    simulation.close()


def put(sim, index, position, tilt=0, yaw=0):
    address = int(sim.model.joint(f"tower_free_{index}").qposadr[0])
    quat = np.empty(4)
    mujoco.mju_euler2Quat(quat, np.array([tilt, 0, yaw]), "xyz")
    sim.data.qpos[address:address+7] = [*position, *quat]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)


def stack(sim, count=4, offset=0, gap=0):
    height = -.0001
    for i in range(count):
        half = sim.task.half_sizes[i][2]
        put(sim, i, (.40 + i*offset, 0, height + half))
        height += 2*half + gap - .0001


def test_initial_is_zero_and_partial_is_height(sim):
    assert sim.task.evaluate(sim)["score"] == 0
    stack(sim, 2)
    result = sim.task.evaluate(sim)
    assert not result["success"]
    assert result["metrics"]["stacked_blocks"] == 2
    assert result["score"] == pytest.approx(100 * .0948 / .21)


def test_full_stack(sim):
    stack(sim)
    assert sim.task.evaluate(sim)["success"]
    assert sim.task.evaluate(sim)["score"] == 100


def test_passive_full_stack_settles_and_remains_stable(sim):
    stack(sim)
    report = sim.finish()
    assert report["success"], report
    assert report["stable_terminal_window"]
    assert np.isfinite(sim.data.qpos).all()


def test_initial_passive_physics_stays_zero(sim):
    for _ in range(500):
        sim._step()
    assert sim.task.evaluate(sim)["score"] == 0


def test_airborne_and_gaps_do_not_score(sim):
    stack(sim, gap=.02)
    assert sim.task.evaluate(sim)["score"] == 0
    for i in range(4):
        put(sim, i, (.4, 0, .4 + i*.1))
    assert sim.task.evaluate(sim)["score"] == 0


def test_high_single_object_does_not_count_as_stack(sim):
    put(sim, 0, (.4, 0, .7))
    assert sim.task.evaluate(sim)["score"] == 0


@pytest.mark.parametrize("tilt", [math.pi, math.pi/2, .15])
def test_tipped_or_upside_down_top_invalidates_connected_stack(sim, tilt):
    stack(sim)
    # Reorient bottom block so the lower connected support component is invalid.
    put(sim, 0, (.4, 0, sim.task.half_sizes[0][2]), tilt=tilt)
    assert not sim.task.evaluate(sim)["success"]


def test_excessive_overhang_rejected(sim):
    stack(sim, offset=.02)
    assert sim.task.evaluate(sim)["score"] == 0


def test_velocity_invalidates_contact_component(sim):
    stack(sim)
    address = int(sim.model.joint("tower_free_3").dofadr[0])
    sim.data.qvel[address] = .2
    mujoco.mj_forward(sim.model, sim.data)
    assert sim.task.evaluate(sim)["score"] == 0


def test_robot_braced_component_rejected(sim):
    stack(sim)
    assert sim.data.ncon
    contact = sim.data.contact[0]
    contact.geom1 = sim.task.geom_ids[3]
    contact.geom2 = next(g for g in range(sim.model.ngeom)
                         if sim.model.geom_bodyid[g] in sim.task.robot_ids)
    result = sim.task.evaluate(sim)
    assert not result["success"]
    assert result["score"] == 0


def test_under_table_cannot_score(sim):
    stack(sim)
    for i in range(4):
        address = int(sim.model.joint(f"tower_free_{i}").qposadr[0])
        sim.data.qpos[address+2] -= .2
    mujoco.mj_forward(sim.model, sim.data)
    assert sim.task.evaluate(sim)["score"] == 0


def test_public_spec_omits_ground_truth(sim):
    public = sim.public_spec()
    assert set(public) == {"task_id", "instruction", "approximation", "reference_image_png"}
    assert public["task_id"] == "tower"
    assert public["reference_image_png"] is None
