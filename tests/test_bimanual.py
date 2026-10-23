"""Dual-arm fidelity, atomic API, private telemetry, and adversarial task fixtures.

Fixtures that set object poses/history test scoring, not actuator solvability.
"""

import json
from copy import deepcopy

import mujoco
import numpy as np
import pytest

from robo_mj_bench.protocol import BadRequest, SafetyStop, validate_request
from robo_mj_bench.robot_assets import MODEL_PATH
from robo_mj_bench.service import EpisodeService, Limits, _robot_projection
from robo_mj_bench.tasks.registry import BIMANUAL_TASKS, create_simulation

AUTH = "Bearer " + "x" * 32


def place(sim, name, position, quat=(1, 0, 0, 0)):
    joint = sim.model.joint(name + "_free").id
    qadr = sim.model.jnt_qposadr[joint]
    sim.data.qpos[qadr:qadr+7] = [*position, *quat]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)


def command(sim):
    state = sim.robot_state()
    return {side: {"targets": state[side]["joint_positions"], "opening": state[side]["gripper_opening"]}
            for side in ("left", "right")}


@pytest.fixture(params=BIMANUAL_TASKS)
def sim(request):
    instance = create_simulation(request.param, 17)
    yield instance
    instance.close()


def test_each_arm_preserves_upstream_dynamics_geometry_and_coupling(sim):
    source = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model = sim.model
    assert model.nu == 14 and model.neq == 2
    for side in ("left", "right"):
        for index in range(1, source.nbody):
            body = model.body(side + "_" + source.body(index).name).id
            for field in ("body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos", "body_iquat", "body_gravcomp"):
                np.testing.assert_array_equal(getattr(model, field)[body], getattr(source, field)[index])
            a = np.flatnonzero(model.geom_bodyid == body)
            b = np.flatnonzero(source.geom_bodyid == index)
            for field in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_solref", "geom_solimp"):
                np.testing.assert_array_equal(getattr(model, field)[a], getattr(source, field)[b])
        for index in range(source.nu):
            actuator = model.actuator(side + "_" + source.actuator(index).name).id
            for field in ("actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange"):
                np.testing.assert_array_equal(getattr(model, field)[actuator], getattr(source, field)[index])
        for index in range(source.njnt):
            joint = model.joint(side + "_" + source.joint(index).name).id
            for field in ("jnt_range", "jnt_axis", "jnt_type"):
                np.testing.assert_array_equal(getattr(model, field)[joint], getattr(source, field)[index])
            for field in ("dof_armature", "dof_frictionloss", "dof_damping"):
                np.testing.assert_array_equal(getattr(model, field)[model.jnt_dofadr[joint]],
                                              getattr(source, field)[source.jnt_dofadr[index]])


def test_idle_physics_never_completes_and_report_is_json_serializable(sim):
    sim.wait(.4)
    assert sim.task.evaluate(sim)["score"] == 0
    report = sim.finish()
    assert not report["success"] and report["score"] == 0
    assert report["robot"] == "dual_i2rt_yam_linear_4310"
    json.dumps(report, allow_nan=False)


def test_atomic_motion_and_slew_rejection(sim):
    commands = command(sim)
    before = sim.data.ctrl.copy()
    bad = deepcopy(commands)
    bad["left"]["targets"][0] += .01
    bad["right"]["targets"][0] += 1
    with pytest.raises(BadRequest):
        sim.move_bimanual(**bad, duration=.1)
    np.testing.assert_array_equal(sim.data.ctrl, before)
    assert sim.data.time == 0
    commands["left"]["targets"][0] += .02
    commands["right"]["targets"][0] -= .02
    sim.move_bimanual(**commands, duration=.1)
    assert sim.data.time == pytest.approx(.1), "Two arms advance once, not sequentially"
    for side in commands:
        ids = sim.arms[side][3]
        np.testing.assert_allclose(sim.data.ctrl[ids], commands[side]["targets"])


def test_gripper_slew_invalidates_entire_command(sim):
    cmds = command(sim)
    cmds["right"]["opening"] = .095
    with pytest.raises(BadRequest):
        sim.move_bimanual(**cmds, duration=.1)
    assert sim.data.time == 0


def test_gateway_projects_only_arm_telemetry_and_budgets_motion_once(sim, tmp_path):
    service = EpisodeService(sim, "x" * 32, tmp_path, Limits(max_commands=1))
    try:
        state = sim.robot_state()
        sim.robot_state = lambda: {**state, "seed": 17, "object_positions": [1, 2, 3]}
        status, public = service.handle("/v1/robot_state", {}, AUTH)
        assert status == 200 and set(public) == {"left", "right", "sim_time"}
        for side in ("left", "right"):
            assert set(public[side]) == {"joint_positions", "joint_velocities", "gripper_opening"}
        cmds = command(sim)
        payload = {**cmds, "duration": .1}
        assert service.handle("/v1/move_bimanual", payload, AUTH)[0] == 200
        assert service._commands == 1 and service._sim_seconds == pytest.approx(.1)
        assert service.handle("/v1/move_bimanual", payload, AUTH)[0] == 429
        assert service.handle("/v1/submit", {}, AUTH) == (200, {"status": "submitted"})
        assert service.handle("/v1/robot_state", {}, AUTH)[0] == 409
        report = json.loads((tmp_path / "result.json").read_text())
        assert report["valid"] and report["score"] == 0
    finally:
        service.close()


@pytest.mark.parametrize("mutation", ["extra", "missing", "nan", "bool", "length", "opening", "duration"])
def test_dual_schema_is_strict(mutation):
    payload = {side: {"targets": [0, 1, 1, 0, 0, 0], "opening": .04} for side in ("left", "right")}
    payload["duration"] = .1
    if mutation == "extra":
        payload["left"]["object_pose"] = [0, 0, 0]
    elif mutation == "missing":
        del payload["right"]["opening"]
    elif mutation == "nan":
        payload["right"]["targets"][0] = float("nan")
    elif mutation == "bool":
        payload["left"]["targets"][0] = True
    elif mutation == "length":
        payload["right"]["targets"].append(0)
    elif mutation == "opening":
        payload["left"]["opening"] = -.1
    else:
        payload["duration"] = .125
    with pytest.raises(BadRequest):
        validate_request("/v1/move_bimanual", payload)


def test_single_arm_rejects_dual_motion_without_closing(tmp_path):
    sim = create_simulation("button", 17)
    service = EpisodeService(sim, "x" * 32, tmp_path)
    payload = {side: {"targets": [0, 1, 1, 0, 0, 0], "opening": .04} for side in ("left", "right")}
    try:
        assert service.handle("/v1/move_bimanual", {**payload, "duration": .1}, AUTH)[0] == 400
        assert not service.closed and sim.data.time == 0
        with pytest.raises(BadRequest):
            sim.observe("wrist_left")
    finally:
        service.close()


def test_nonfinite_dual_telemetry_fails_closed():
    with pytest.raises(RuntimeError):
        _robot_projection({"sim_time": float("nan")}, True)


def test_handover_final_pose_without_history_fails_then_valid_history_settles():
    sim = create_simulation("bimanual_handover", 17)
    try:
        task = sim.task
        place(sim, "baton", [*task.goal, .018])
        assert task.evaluate(sim)["metrics"]["placed"]
        assert not task.evaluate(sim)["success"]
        task.stage = 3  # Trusted terminal-state fixture, not an agent rollout.
        assert sim.finish()["success"]
    finally:
        sim.close()


def test_handover_requires_ordered_dwell_and_no_support(monkeypatch):
    sim = create_simulation("bimanual_handover", 17)
    try:
        task = sim.task
        place(sim, "baton", [.35, 0, .18])
        holds = {"left": False, "right": True}
        monkeypatch.setattr(task, "gripping", lambda side, ids: holds[side])
        for _ in range(100):
            task.after_step(sim)
        assert task.stage == 0, "Right-first possession cannot satisfy left-first transfer"
        for stage, left, right, ticks in ((1, True, False, 50), (2, True, True, 50), (3, False, True, 75)):
            holds.update(left=left, right=right)
            for _ in range(ticks - 1):
                task.after_step(sim)
            assert task.stage == stage - 1
            task.after_step(sim)
            assert task.stage == stage
        place(sim, "baton", [*task.goal, .018])
        assert task.evaluate(sim)["success"]
    finally:
        sim.close()


@pytest.mark.parametrize("failure", ["support", "left_lost", "right_lost"])
def test_handover_transfer_cannot_be_interrupted(monkeypatch, failure):
    sim = create_simulation("bimanual_handover", 17)
    try:
        task = sim.task
        task.stage = 2 if failure == "right_lost" else 1
        place(sim, "baton", [.35, 0, .18])
        monkeypatch.setattr(task, "gripping", lambda side, ids: False if failure != "support" else True)
        if failure == "support":
            monkeypatch.setattr(task, "unsupported", lambda *args: False)
        task.after_step(sim)
        assert task.failed and task.evaluate(sim)["score"] == 0
    finally:
        sim.close()


def test_tray_spill_is_irreversible_and_goal_alone_is_insufficient():
    sim = create_simulation("bimanual_tray", 17)
    try:
        task = sim.task
        offset = np.array([*task.goal, .008]) - sim.data.xpos[task.tray].copy()
        cargo = sim.data.xpos[task.cargo].copy() + offset
        place(sim, "tray", [*task.goal, .008])
        for index, point in enumerate(cargo):
            place(sim, f"cargo{index}", point)
        assert not task.evaluate(sim)["success"]
        task.lifted = True
        assert task.evaluate(sim)["success"]
        saved = sim.data.xpos[task.cargo[0]].copy()
        place(sim, "cargo0", [0, 0, .012])
        task.after_step(sim)
        place(sim, "cargo0", saved)
        assert task.spilled and not task.evaluate(sim)["success"]
    finally:
        sim.close()


def test_tray_lift_requires_both_handles_and_continuous_time(monkeypatch):
    sim = create_simulation("bimanual_tray", 17)
    try:
        task = sim.task
        delta = np.array([0, 0, .13])
        cargo = sim.data.xpos[task.cargo].copy() + delta
        tray_position = sim.data.xpos[task.tray].copy() + delta
        place(sim, "tray", tray_position)
        for index, point in enumerate(cargo):
            place(sim, f"cargo{index}", point)
        holds = {"left": True, "right": False}
        monkeypatch.setattr(task, "gripping", lambda side, ids: holds[side])
        for _ in range(120):
            task.after_step(sim)
        assert not task.lifted
        holds["right"] = True
        for _ in range(99):
            task.after_step(sim)
        assert not task.lifted
        task.after_step(sim)
        assert task.lifted
    finally:
        sim.close()


def test_drawer_history_cannot_be_bypassed_and_spring_closes(monkeypatch):
    sim = create_simulation("bimanual_drawer", 17)
    try:
        task = sim.task
        address = sim.model.jnt_qposadr[task.slide]
        sim.data.qpos[address] = .14
        mujoco.mj_forward(sim.model, sim.data)
        place(sim, "insert_block", sim.data.xpos[task.drawer] + [0, 0, .024])
        assert task.block_inside()
        assert not task.evaluate(sim)["success"]
        holds = {"left": True, "right": True}
        monkeypatch.setattr(task, "gripping", lambda side, ids: holds[side])
        for _ in range(49):
            task.after_step(sim)
        assert not task.inserted
        task.after_step(sim)
        assert task.inserted and task.evaluate(sim)["success"]
        holds["left"] = False
        assert not task.evaluate(sim)["success"]
        before = sim.data.qpos[address]
        sim.wait(.2)
        assert sim.data.qpos[address] < before - .01
    finally:
        sim.close()


def test_grip_requires_two_opposing_loaded_pads(monkeypatch):
    sim = create_simulation("bimanual_handover", 17)
    try:
        task = sim.task
        sim.wait(.05)  # Real table contacts allocate contact slots.
        assert sim.data.ncon >= 2
        pads = task.pad_ids["left"]
        for index in range(2):
            contact = sim.data.contact[index]
            contact.geom1, contact.geom2 = pads[index], task.object_geom
            contact.frame[:3] = [1 if index == 0 else -1, 0, 0]
        monkeypatch.setattr(mujoco, "mj_contactForce", lambda m, d, i, force: force.__setitem__(0, .2))
        assert task.gripping("left", {task.object_geom})
        sim.data.contact[1].frame[:3] = [1, 0, 0]
        assert not task.gripping("left", {task.object_geom})
        sim.data.contact[1].frame[:3] = [-1, 0, 0]
        monkeypatch.setattr(mujoco, "mj_contactForce", lambda m, d, i, force: force.__setitem__(0, 0))
        assert not task.gripping("left", {task.object_geom})
    finally:
        sim.close()


def test_inter_arm_force_ends_episode_as_valid_failure(monkeypatch, tmp_path):
    sim = create_simulation("bimanual_handover", 17)
    service = EpisodeService(sim, "x" * 32, tmp_path)
    try:
        # Inject a solved contact, isolating the safety threshold from collision planning.
        sim.wait(.05)
        contact = sim.data.contact[0]
        contact.geom1 = sim.model.geom("left_left_pad").id
        contact.geom2 = sim.model.geom("right_left_pad").id
        monkeypatch.setattr(mujoco, "mj_step", lambda *args: None)
        monkeypatch.setattr(mujoco, "mj_contactForce", lambda m, d, i, force: force.__setitem__(0, 36))
        assert service.handle("/v1/wait", {"duration": .02}, AUTH) == (200, {"status": "safety_stop"})
        report = json.loads((tmp_path / "result.json").read_text())
        assert report["valid"] and report["score"] == 0 and not report["success"]
        assert report["peak_inter_arm_force_n"] >= 36
    finally:
        service.close()


@pytest.mark.parametrize("seed", [0, 17, 999])
@pytest.mark.parametrize("name", BIMANUAL_TASKS)
def test_public_model_reaches_representative_task_approaches(name, seed):
    from agent.robot_kinematics import RobotKinematics
    from robo_mj_bench.scene import DUAL_MOUNTS
    sim = create_simulation(name, seed)
    try:
        if name == "bimanual_handover":
            targets = [("left", [.19, 0, .115]), ("right", [.29, 0, .115])]
            # Also test initial pickup and final placement from public visual targets.
            targets += [("left", sim.data.xpos[sim.task.object_body].copy()),
                        ("right", [*sim.task.goal, .018])]
        elif name == "bimanual_tray":
            targets = [(side, sim.data.geom_xpos[gid].copy()) for side, gid in sim.task.handles.items()]
            targets += [(side, [sim.task.goal[0], sim.task.goal[1] + sign*.15, .112])
                        for side, sign in (("left", -1), ("right", 1))]
        else:
            handle = sim.data.geom_xpos[sim.task.handle].copy()
            opened_handle = handle + [0, -.14, 0]
            targets = [("left", handle), ("left", opened_handle),
                       ("right", sim.data.xpos[sim.task.block].copy()),
                       ("right", sim.data.xpos[sim.task.drawer] + [0, -.14, .05])]
        for side, target in targets:
            solver = RobotKinematics(side)
            np.testing.assert_array_equal(solver.base, DUAL_MOUNTS[side][0])
            joints = solver.downward_pose(*target)
            sim.data.qpos[sim.arms[side][0]] = joints
            mujoco.mj_forward(sim.model, sim.data)
            # Cross-check world transforms against the actual two-arm model.
            np.testing.assert_allclose(sim.data.site_xpos[sim.model.site(side + "_grasp_site").id], target, atol=2e-5)
    finally:
        sim.close()


def test_tray_terminal_fixture_survives_real_passive_settling():
    sim = create_simulation("bimanual_tray", 17)
    try:
        task = sim.task
        target = np.array([*task.goal, .008])
        cargo = sim.data.xpos[task.cargo].copy() + target - sim.data.xpos[task.tray]
        place(sim, "tray", target)
        for index, point in enumerate(cargo):
            place(sim, f"cargo{index}", point)
        task.lifted = True
        report = sim.finish()
        assert report["success"] and report["stable_terminal_window"], report
    finally:
        sim.close()


def test_tray_cannot_be_pushed_to_goal_before_cooperative_lift(monkeypatch):
    sim = create_simulation("bimanual_tray", 17)
    try:
        task = sim.task
        target = np.array([*task.goal, .13])
        cargo = sim.data.xpos[task.cargo].copy() + target - sim.data.xpos[task.tray]
        place(sim, "tray", target)
        for index, point in enumerate(cargo):
            place(sim, f"cargo{index}", point)
        monkeypatch.setattr(task, "gripping", lambda *args: True)
        for _ in range(120):
            task.after_step(sim)
        assert not task.lifted
    finally:
        sim.close()


def test_tray_requires_both_hands_until_goal(monkeypatch):
    sim = create_simulation("bimanual_tray", 17)
    try:
        task = sim.task
        cargo = sim.data.xpos[task.cargo].copy() + [0, 0, .13]
        place(sim, "tray", sim.data.xpos[task.tray].copy() + [0, 0, .13])
        for index, point in enumerate(cargo):
            place(sim, f"cargo{index}", point)
        holds = {"left": True, "right": True}
        monkeypatch.setattr(task, "gripping", lambda side, ids: holds[side])
        for _ in range(100):
            task.after_step(sim)
        assert task.lifted and not task.carry_failed
        holds["right"] = False
        task.after_step(sim)
        holds["right"] = True
        task.after_step(sim)
        assert task.carry_failed and task.evaluate(sim)["score"] == 0
    finally:
        sim.close()


def test_drawer_cannot_count_grasping_an_already_inserted_block(monkeypatch):
    sim = create_simulation("bimanual_drawer", 17)
    try:
        task = sim.task
        sim.data.qpos[sim.model.jnt_qposadr[task.slide]] = .14
        mujoco.mj_forward(sim.model, sim.data)
        place(sim, "insert_block", sim.data.xpos[task.drawer] + [0, 0, .024])
        task.after_step(sim)
        monkeypatch.setattr(task, "gripping", lambda *args: True)
        for _ in range(100):
            task.after_step(sim)
        assert not task.inserted
        place(sim, "insert_block", sim.data.xpos[task.drawer] + [0, 0, .12])
        task.after_step(sim)
        place(sim, "insert_block", sim.data.xpos[task.drawer] + [0, 0, .024])
        for _ in range(50):
            task.after_step(sim)
        assert task.inserted
    finally:
        sim.close()
