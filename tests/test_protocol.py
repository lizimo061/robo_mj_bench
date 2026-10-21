"""Untrusted command validation must not coerce or expand the control surface."""

import math

import pytest

from robo_mj_bench.protocol import BadRequest, validate_request


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/robot_state", {}),
        ("/v1/observe", {"camera_id": "front"}),
        ("/v1/move_joints", {"targets": [0.0] * 6, "duration": 0.1}),
        ("/v1/set_gripper", {"opening": 0.025, "duration": 0.1}),
        ("/v1/wait", {"duration": 0.1}),
        ("/v1/submit", {}),
    ],
)
def test_valid_commands(path, payload):
    assert validate_request(path, payload) == payload


@pytest.mark.parametrize("path", ["/v1/reset", "/v1/reward", "/debug", "/v1/robot_state/"])
def test_non_robot_endpoints_rejected(path):
    with pytest.raises((BadRequest, KeyError)):
        validate_request(path, {})


@pytest.mark.parametrize("payload", [None, [], "{}", 1, True])
def test_payload_must_be_object(payload):
    with pytest.raises(BadRequest):
        validate_request("/v1/robot_state", payload)


@pytest.mark.parametrize("path,payload", [
    ("/v1/robot_state", {"include_objects": True}),
    ("/v1/observe", {"camera_id": "front", "segmentation": True}),
    ("/v1/move_joints", {"targets": [0] * 6, "duration": 0.1, "teleport": True}),
    ("/v1/set_gripper", {"opening": 0.02, "duration": 0.1, "force": 1000}),
    ("/v1/wait", {"duration": 0.1, "seed": 42}),
    ("/v1/submit", {"success": True}),
])
def test_unknown_fields_rejected(path, payload):
    with pytest.raises(BadRequest):
        validate_request(path, payload)


@pytest.mark.parametrize("path,payload", [
    ("/v1/observe", {}),
    ("/v1/move_joints", {"targets": [0] * 6}),
    ("/v1/move_joints", {"duration": 0.1}),
    ("/v1/set_gripper", {"duration": 0.1}),
    ("/v1/set_gripper", {"opening": 0.02}),
    ("/v1/wait", {}),
])
def test_required_fields_cannot_be_omitted(path, payload):
    with pytest.raises(BadRequest):
        validate_request(path, payload)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, False, "0.1", None])
@pytest.mark.parametrize("field", ["duration", "opening", "target"])
def test_no_nonfinite_or_coerced_numbers(field, value):
    if field == "duration":
        path, payload = "/v1/wait", {"duration": value}
    elif field == "opening":
        path, payload = "/v1/set_gripper", {"opening": value, "duration": 0.1}
    else:
        path, payload = "/v1/move_joints", {"targets": [value, 0, 0, 0, 0, 0], "duration": 0.1}
    with pytest.raises(BadRequest):
        validate_request(path, payload)


@pytest.mark.parametrize("duration", [-1, 0, 0.019, 0.021, 5.002, 1e99])
def test_duration_bounds_and_physics_tick_grid(duration):
    with pytest.raises(BadRequest):
        validate_request("/v1/wait", {"duration": duration})


@pytest.mark.parametrize("targets", [[], [0] * 5, [0] * 7, "000000", {"0": 0}, [1e9] * 6])
def test_joint_vector_shape_and_limits(targets):
    with pytest.raises(BadRequest):
        validate_request("/v1/move_joints", {"targets": targets, "duration": 0.1})


@pytest.mark.parametrize("opening", [-0.001, 0.0951, 1000])
def test_gripper_limits(opening):
    with pytest.raises(BadRequest):
        validate_request("/v1/set_gripper", {"opening": opening, "duration": 0.1})


@pytest.mark.parametrize("camera_id", ["segmentation", "depth", "../scene.xml", "front?depth=true", 0, True, None, ["front"]])
def test_camera_identifiers_are_allowlisted(camera_id):
    with pytest.raises(BadRequest):
        validate_request("/v1/observe", {"camera_id": camera_id})
