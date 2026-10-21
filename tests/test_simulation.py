"""Physical smoke tests and controlled contact-stream tests of the private scorer."""

from types import SimpleNamespace
import io

import numpy as np
import pytest
from PIL import Image

from robo_mj_bench.protocol import BadRequest, SafetyStop
from robo_mj_bench.simulation import ButtonSimulation


def test_passive_button_does_not_score_and_state_is_robot_only():
    sim = ButtonSimulation(seed=19)
    try:
        state = sim.robot_state()
        assert set(state) == {"joint_positions", "joint_velocities", "gripper_opening", "sim_time"}
        assert len(state["joint_positions"]) == len(state["joint_velocities"]) == 6
        assert state["sim_time"] == 0
        sim.wait(1.0)
        assert sim.robot_state()["sim_time"] == pytest.approx(1.0)
        report = sim.finish()
        assert report["success"] is False
        assert report["score"] == 0
    finally:
        sim.close()


def test_large_instantaneous_joint_motion_is_rejected_before_physics():
    sim = ButtonSimulation(seed=19)
    try:
        targets = sim.robot_state()["joint_positions"]
        targets[0] += 0.2
        with pytest.raises(BadRequest):
            sim.move_joints(targets, 0.02)
        assert sim.robot_state()["sim_time"] == 0
    finally:
        sim.close()


def test_seed_reproduces_scene_and_physics():
    first, second = ButtonSimulation(seed=71), ButtonSimulation(seed=71)
    try:
        assert first.xml == second.xml
        for sim in (first, second):
            sim.wait(0.2)
        assert first.robot_state() == second.robot_state()
    finally:
        first.close()
        second.close()


def test_all_cameras_render_clean_rgb_without_advancing_time():
    sim = ButtonSimulation(seed=19)
    try:
        for camera_id in ("front", "left", "overhead", "wrist"):
            encoded = sim.observe(camera_id)
            assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
            with Image.open(io.BytesIO(encoded)) as image:
                assert image.size == (640, 480)
                assert image.mode == "RGB"
                assert image.info == {}
        assert sim.robot_state()["sim_time"] == 0
    finally:
        sim.close()


def test_position_command_uses_physics_and_does_not_teleport():
    sim = ButtonSimulation(seed=19)
    try:
        initial = sim.robot_state()["joint_positions"]
        targets = list(initial)
        targets[0] += 0.03
        sim.move_joints(targets, 0.02)
        actual = sim.robot_state()["joint_positions"][0]
        assert abs(actual - targets[0]) > 1e-5
        assert sim.robot_state()["sim_time"] == pytest.approx(0.02)
    finally:
        sim.close()


def scoring_sim(monkeypatch, *, travel=0.008, other_geom=2, force=1.0):
    """Isolate scoring from physics to supply exact adversarial contact streams."""
    sim = ButtonSimulation.__new__(ButtonSimulation)
    sim._cap_id = 1
    sim._finger_geoms = {2, 3}
    sim._button_qadr = 0
    sim._indicator_id = 0
    sim._pressed = False
    sim._press_seconds = sim._max_travel = sim._peak_button_force = 0.0
    sim._invalid_physics = False
    sim.model = SimpleNamespace(geom_rgba=np.zeros((1, 4)))
    contacts = [] if other_geom is None else [SimpleNamespace(geom1=1, geom2=other_geom)]
    sim.data = SimpleNamespace(qpos=np.array([travel]), qvel=np.array([0.0]), warning=[],
                               ncon=len(contacts), contact=contacts)
    monkeypatch.setattr("robo_mj_bench.simulation.mujoco.mj_step", lambda *args: None)

    def contact_force(model, data, index, output):
        output[:] = 0
        output[0] = force

    monkeypatch.setattr("robo_mj_bench.simulation.mujoco.mj_contactForce", contact_force)
    return sim


@pytest.mark.parametrize("other_geom,travel,force", [
    (None, 0.008, 1.0),  # Travel alone, including gravity, must not score.
    (99, 0.008, 1.0),  # A wrist or table contact cannot replace a finger.
    (2, 0.002, 1.0),  # Finger contact without enough travel.
    (2, 0.008, 0.1),  # Travel plus negligible contact force.
])
def test_scorer_requires_finger_force_and_button_travel(monkeypatch, other_geom, travel, force):
    sim = scoring_sim(monkeypatch, other_geom=other_geom, travel=travel, force=force)
    for _ in range(100):
        sim._step()
    assert not sim._pressed


def test_scorer_requires_continuous_press_and_latches_event(monkeypatch):
    sim = scoring_sim(monkeypatch)
    for _ in range(39):
        sim._step()
    assert not sim._pressed
    sim._step()
    assert sim._pressed
    sim.data.qpos[0] = 0
    sim._step()
    assert sim._pressed


def test_short_presses_cannot_accumulate_across_release(monkeypatch):
    sim = scoring_sim(monkeypatch)
    for _ in range(5):
        sim.data.qpos[0] = 0.008
        for _ in range(20):
            sim._step()
        sim.data.qpos[0] = 0
        sim._step()
    assert not sim._pressed


@pytest.mark.parametrize("unsafe_field,unsafe_value", [
    ("_peak_button_force", 30.001),
    ("_max_travel", 0.013001),
    ("_invalid_physics", True),
])
def test_latched_press_cannot_override_safety_failure(unsafe_field, unsafe_value):
    sim = ButtonSimulation(seed=19)
    try:
        sim._pressed = True
        setattr(sim, unsafe_field, unsafe_value)
        report = sim.finish()
        assert report["pressed"] is True
        assert report["safe"] is False
        assert report["success"] is False
        assert report["score"] == 0
    finally:
        sim.close()


@pytest.mark.parametrize("seed", [0, 9, 19])
def test_standard_yam_downward_command_respects_button_stop(seed):
    from scripts.validate_button import reference_pose

    sim = ButtonSimulation(seed)
    try:
        sim.set_gripper(0.012, 1.0)
        x, y = sim.button_xy  # Deliberately privileged mechanical validation.
        for height, duration in ((0.12, 2.0), (0.09, 1.0), (0.06, 1.0), (0.04, 1.0), (0.03, 1.0)):
            sim.move_joints(reference_pose(x, y, height), duration)
            sim.wait(0.2)
        report = sim.finish()
        assert report["pressed"] is True, "This check must actually reach the button."
        assert 0.006 <= report["max_button_travel_m"] <= 0.013
        assert report["peak_button_force_n"] <= 30
        assert report["safe"] is True
        assert report["success"] is True
    finally:
        sim.close()


def test_real_external_overload_stops_physics_and_cannot_preserve_success():
    sim = ButtonSimulation(seed=19)
    try:
        sim._pressed = True  # Model a previously successful press before damage.
        cap = sim.model.body("button_cap").id
        sim.data.xfrc_applied[cap, 2] = -100
        with pytest.raises(SafetyStop):
            sim.wait(0.2)
        stopped_time = sim.data.time
        stopped_positions = sim.data.qpos.copy()
        with pytest.raises(RuntimeError, match="closed"):
            sim.wait(0.1)
        assert sim.data.time == stopped_time
        np.testing.assert_array_equal(sim.data.qpos, stopped_positions)
        report = sim.finish()
        assert report["max_button_travel_m"] > 0.013 or report["peak_button_force_n"] > 30
        assert report["pressed"] is True
        assert report["safe"] is False
        assert report["success"] is False
        assert report["score"] == 0
    finally:
        sim.close()


@pytest.mark.parametrize("other_geom", [2, 99, None])
def test_excess_force_is_terminal_even_without_valid_finger_contact(monkeypatch, other_geom):
    sim = scoring_sim(monkeypatch, other_geom=other_geom, force=31)
    if other_geom is None:
        sim.data.qpos[0] = 0.014  # Overtravel also stops without any contact evidence.
    sim._pressed = True
    with pytest.raises(SafetyStop):
        sim._step()
    assert sim._closed is True
