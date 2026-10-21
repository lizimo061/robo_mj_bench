"""Trusted scoring fixtures plus actual MuJoCo contact-path tests."""
import io

import mujoco
import numpy as np
from PIL import Image
import pytest

from robo_mj_bench.tasks.writing import WritingTask
from robo_mj_bench.task_simulation import TaskSimulation


def test_reference_is_only_goal_and_stays_constant():
    task = WritingTask()
    before = task.reference_image()
    assert Image.open(io.BytesIO(before)).size == (240, 90)
    task._ink[:] = True
    assert task.reference_image() == before


def test_blank_and_full_sheet_cannot_pass():
    task = WritingTask()
    assert task.evaluate(None)["score"] == 0
    task._ink[:] = True
    result = task.evaluate(None)
    assert not result["success"]
    assert result["metrics"]["precision"] < .4


def test_frozen_target_and_bounds():
    task = WritingTask()
    task._ink[:] = task._target
    assert task.evaluate(None)["success"]
    task._record_contact([.1, .1, 0], .2, True)
    assert not task.evaluate(None)["success"]


def test_contact_stream_can_trace_goal():
    task = WritingTask()
    for y, x in np.argwhere(task._target):
        xy = task.PAPER_XY + np.array([(x+.5)/task.WIDTH-.5,
                                      .5-(y+.5)/task.HEIGHT])*task.PAPER_SIZE
        task._record_contact([*xy, task.PAPER_Z], .25, True)
    assert task.evaluate(None)["success"]


@pytest.mark.parametrize("force,upright", [(0, True), (.01, True), (9, True), (.3, False)])
def test_invalid_contacts_do_not_add_ink(force, upright):
    task = WritingTask()
    task._record_contact([*task.PAPER_XY, task.PAPER_Z], force, upright)
    assert not task._ink.any()


def put_marker(sim, xyz, upright=True):
    j = sim.model.joint("writing_marker_free").id
    a = sim.model.jnt_qposadr[j]
    sim.data.qpos[a:a+3] = xyz
    sim.data.qpos[a+3:a+7] = [2**-.5, 0, -2**-.5, 0] if upright else [1, 0, 0, 0]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)


def test_real_tip_paper_contact_records_ink_but_hover_does_not():
    task = WritingTask()
    sim = TaskSimulation(task, 12)
    try:
        put_marker(sim, [*task.PAPER_XY, .03])
        for _ in range(5):
            sim._step()
        assert not task._ink.any()
        put_marker(sim, [*task.PAPER_XY, task.PAPER_Z + .0028])
        for _ in range(100):
            sim._step()
        assert task._accepted_samples > 0
        assert task._ink.any()
    finally:
        sim.close()


def test_lying_marker_and_passive_initial_scene_do_not_write():
    task = WritingTask()
    sim = TaskSimulation(task, 42)
    try:
        sim.wait(1.0)
        assert np.isfinite(sim.data.qpos).all()
        assert not task._ink.any()
        assert not task.evaluate(sim)["success"]
    finally:
        sim.close()


def test_ink_decorates_scene_without_changing_physics():
    task = WritingTask()
    sim = TaskSimulation(task, 9)
    try:
        task._record_contact([*task.PAPER_XY, task.PAPER_Z], .3, True)
        scene = mujoco.MjvScene(sim.model, maxgeom=200)
        before = sim.data.qpos.copy()
        task.decorate_scene(scene, sim)
        assert scene.ngeom > 0
        assert np.array_equal(before, sim.data.qpos)
    finally:
        sim.close()


def test_real_table_contact_is_out_of_bounds_without_drawing():
    task = WritingTask()
    sim = TaskSimulation(task, 22)
    try:
        put_marker(sim, [.375, -.025, .0028])
        for _ in range(100):
            sim._step()
        assert task._outside_samples > 0
        assert not task._ink.any()
    finally:
        sim.close()


def test_scribbling_the_text_rectangle_does_not_pass():
    task = WritingTask()
    y, x = np.where(task._target)
    task._ink[y.min():y.max()+1, x.min():x.max()+1] = True
    assert not task.evaluate(None)["success"]
