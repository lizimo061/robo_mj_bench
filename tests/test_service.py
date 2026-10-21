"""Exercise the trusted gateway without requiring a graphics driver."""

import base64
import json

import pytest

from robo_mj_bench.service import EpisodeService, Limits


TOKEN = "test-episode-secret"
AUTH = f"Bearer {TOKEN}"


class FakeSimulation:
    def __init__(self):
        self.calls = []
        self.time = 0.0

    def robot_state(self):
        self.calls.append(("state",))
        return {"joint_positions": [0.0] * 6, "joint_velocities": [0.0] * 6,
                "gripper_opening": 0.05, "sim_time": self.time}

    def observe(self, camera_id):
        self.calls.append(("observe", camera_id))
        return b"\x89PNG\r\n\x1a\nfake-image"

    def move_joints(self, targets, duration):
        self.calls.append(("move", targets, duration))
        self.time += duration

    def set_gripper(self, opening, duration):
        self.calls.append(("gripper", opening, duration))
        self.time += duration

    def wait(self, duration):
        self.calls.append(("wait", duration))
        self.time += duration

    def finish(self):
        self.calls.append(("finish",))
        return {"success": True, "reward": 1.0, "seed": 937, "button_position": [1, 2, 3]}

    def close(self):
        self.calls.append(("close",))


@pytest.fixture
def gateway(tmp_path):
    simulation = FakeSimulation()
    service = EpisodeService(simulation, token=TOKEN, audit_dir=tmp_path)
    yield service, simulation, tmp_path
    service.close()


@pytest.mark.parametrize("authorization", [None, "", TOKEN, "Bearer wrong", f"bearer {TOKEN}", f"Bearer {TOKEN} ", f"Bearer {TOKEN}\U0001f600"])
def test_authentication_precedes_simulation(gateway, authorization):
    service, simulation, _ = gateway
    status, body = service.handle("/v1/robot_state", {}, authorization)
    assert status == 401
    assert simulation.calls == []
    assert TOKEN not in json.dumps(body)


@pytest.mark.parametrize("path", ["/v1/reset", "/v1/reward", "/v1/debug", "/docs", "/openapi.json"])
def test_no_privileged_routes(gateway, path):
    service, simulation, _ = gateway
    status, _ = service.handle(path, {}, AUTH)
    assert status == 404
    assert simulation.calls == []


def test_invalid_payload_does_not_reach_simulator(gateway):
    service, simulation, _ = gateway
    status, _ = service.handle("/v1/wait", {"duration": True}, AUTH)
    assert status == 400
    assert simulation.calls == []


def test_reads_do_not_advance_physics(gateway):
    service, simulation, _ = gateway
    for _ in range(3):
        assert service.handle("/v1/robot_state", {}, AUTH)[0] == 200
        assert service.handle("/v1/observe", {"camera_id": "front"}, AUTH)[0] == 200
    assert simulation.time == 0.0


def test_observation_is_encoded_image_without_privileged_metadata(gateway):
    service, _, _ = gateway
    status, body = service.handle("/v1/observe", {"camera_id": "front"}, AUTH)
    assert status == 200
    text = json.dumps(body)
    assert base64.b64encode(b"\x89PNG\r\n\x1a\nfake-image").decode() in text
    assert not {"seed", "reward", "success", "objects", "contacts", "segmentation"}.intersection(body)


def test_submit_is_final_and_keeps_ground_truth_private(gateway):
    service, simulation, audit_dir = gateway
    status, body = service.handle("/v1/submit", {}, AUTH)
    assert status == 200
    public = json.dumps(body)
    for secret in ("reward", "success", "937", "button_position"):
        assert secret not in public
    for path, payload in [
        ("/v1/wait", {"duration": 0.1}),
        ("/v1/robot_state", {}),
        ("/v1/observe", {"camera_id": "front"}),
        ("/v1/submit", {}),
    ]:
        assert service.handle(path, payload, AUTH)[0] == 409
    assert simulation.calls.count(("finish",)) == 1
    assert list(audit_dir.iterdir()), "Private results must be persisted outside the agent."
    report = json.loads((audit_dir / "result.json").read_text())
    assert report["seed"] == 937
    assert report["valid"] is True
    assert TOKEN not in (audit_dir / "events.jsonl").read_text()
    assert (audit_dir / "result.json").stat().st_mode & 0o077 == 0
    assert (audit_dir / "events.jsonl").stat().st_mode & 0o077 == 0


def test_internal_errors_are_sanitized(gateway):
    service, simulation, _ = gateway

    def fail():
        raise RuntimeError("private-scene-seed=937 /trusted/assets/button.xml")

    simulation.robot_state = fail
    status, body = service.handle("/v1/robot_state", {}, AUTH)
    assert status == 500
    public = json.dumps(body)
    assert "937" not in public
    assert "button.xml" not in public
    assert "Traceback" not in public


@pytest.mark.parametrize("limit_name,path,payload", [
    ("max_requests", "/v1/robot_state", {}),
    ("max_observations", "/v1/observe", {"camera_id": "front"}),
    ("max_commands", "/v1/wait", {"duration": 0.02}),
    ("max_sim_seconds", "/v1/wait", {"duration": 0.02}),
    ("max_wall_seconds", "/v1/robot_state", {}),
])
def test_budget_exhaustion_still_allows_terminal_submission(tmp_path, limit_name, path, payload):
    sim = FakeSimulation()
    service = EpisodeService(sim, TOKEN, tmp_path, limits=Limits(**{limit_name: 0}))
    try:
        assert service.handle(path, payload, AUTH)[0] == 429
        assert sim.calls == []
        assert service.handle("/v1/submit", {}, AUTH) == (200, {"status": "submitted"})
        assert (tmp_path / "result.json").exists()
    finally:
        service.close()


def test_motion_budget_charged_before_partially_failed_execution(tmp_path):
    sim = FakeSimulation()

    def fail_after_motion(duration):
        sim.time += duration / 2
        from robo_mj_bench.protocol import BadRequest
        raise BadRequest("operator-private controller failure")

    sim.wait = fail_after_motion
    service = EpisodeService(sim, TOKEN, tmp_path, limits=Limits(max_commands=1))
    try:
        assert service.handle("/v1/wait", {"duration": 0.1}, AUTH)[0] == 400
        assert service.handle("/v1/wait", {"duration": 0.1}, AUTH)[0] == 429
        assert sim.time == pytest.approx(0.05)
    finally:
        service.close()


def test_projection_drops_accidental_privileged_simulator_fields(gateway):
    service, sim, _ = gateway
    original = sim.robot_state

    def contaminated_state():
        return {**original(), "seed": 937, "object_poses": [[1, 2, 3]], "success": True}

    sim.robot_state = contaminated_state
    status, body = service.handle("/v1/robot_state", {}, AUTH)
    assert status == 200
    assert set(body) == {"joint_positions", "joint_velocities", "gripper_opening", "sim_time"}


@pytest.mark.parametrize("length", [5, 7, 9])
def test_telemetry_rejects_old_or_privileged_joint_vectors(gateway, length):
    service, sim, _ = gateway
    original = sim.robot_state

    def wrong_state():
        return {**original(), "joint_positions": [0.0] * length}

    sim.robot_state = wrong_state
    status, body = service.handle("/v1/robot_state", {}, AUTH)
    assert status == 500
    assert body == {"error": "internal_error"}


@pytest.mark.parametrize("termination", ["operator_stopped", "internal_error"])
def test_unsubmitted_or_faulted_success_cannot_become_valid_score(tmp_path, termination):
    sim = FakeSimulation()
    service = EpisodeService(sim, TOKEN, tmp_path)
    try:
        if termination == "operator_stopped":
            service.close()
        else:
            def fail(camera_id):
                raise RuntimeError("rendering failed after physical success")
            sim.observe = fail
            assert service.handle("/v1/observe", {"camera_id": "front"}, AUTH)[0] == 500
        report = json.loads((tmp_path / "result.json").read_text())
        assert report["termination"] == termination
        assert report["valid"] is False
        assert report["success"] is False
        assert report["score"] is None
    finally:
        if not service._audit.closed:
            service.close()


def test_nonfinite_diagnostics_produce_complete_private_json(tmp_path):
    sim = FakeSimulation()
    sim.finish = lambda: {"success": False, "score": 0,
                          "diagnostics": [float("nan"), {"velocity": float("inf")}],
                          "negative": float("-inf")}
    service = EpisodeService(sim, TOKEN, tmp_path)
    try:
        assert service.handle("/v1/submit", {}, AUTH)[0] == 200
        report = json.loads((tmp_path / "result.json").read_text())
        assert report["valid"] is True
        assert report["diagnostics"] == [None, {"velocity": None}]
        assert report["negative"] is None
        assert not (tmp_path / "result.json.tmp").exists()
    finally:
        service.close()


def test_serialization_failure_never_leaves_truncated_result(tmp_path):
    sim = FakeSimulation()
    sim.finish = lambda: {"unsupported": object()}
    service = EpisodeService(sim, TOKEN, tmp_path)
    try:
        assert service.handle("/v1/submit", {}, AUTH)[0] == 500
        assert not (tmp_path / "result.json").exists()
        assert service.handle("/v1/robot_state", {}, AUTH)[0] == 409
    finally:
        service.close()


def test_safety_stop_overrides_latched_success_and_ends_episode(gateway):
    from robo_mj_bench.protocol import SafetyStop

    service, sim, audit_dir = gateway

    def stop(duration):
        sim.time += duration / 2
        raise SafetyStop()

    sim.wait = stop
    assert service.handle("/v1/wait", {"duration": 0.1}, AUTH) == (200, {"status": "safety_stop"})
    report = json.loads((audit_dir / "result.json").read_text())
    assert report["valid"] is True
    assert report["termination"] == "safety_stop"
    assert report["success"] is False  # Fake simulation.finish() claimed success.
    assert report["score"] == 0
    assert service.handle("/v1/wait", {"duration": 0.1}, AUTH)[0] == 409
    assert service.handle("/v1/submit", {}, AUTH)[0] == 409
    assert sim.time == pytest.approx(0.05)
    assert sim.calls.count(("finish",)) == 1
