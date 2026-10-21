"""Public action contract; no simulator imports or episode-specific data."""

import math

TIMESTEP = 0.002
JOINT_LIMITS = ((-2.61799, 3.14159), (0.0, 3.66519), (0.0, 3.14159),
                (-1.69297, 1.5708), (-1.5708, 1.5708), (-2.0944, 2.0944))
HOME_JOINTS = (-0.6, 1.047, 1.047, 0.0, 0.0, 0.0)
CAMERA_IDS = ("front", "left", "overhead", "wrist")
MAX_JOINT_SPEED = 1.5
MAX_GRIPPER_SPEED = 0.05
SCHEMAS = {
    "/v1/task": set(),
    "/v1/observe": {"camera_id"},
    "/v1/robot_state": set(),
    "/v1/move_joints": {"targets", "duration"},
    "/v1/set_gripper": {"opening", "duration"},
    "/v1/wait": {"duration"},
    "/v1/submit": set(),
}


class BadRequest(ValueError):
    """Invalid public input; never include internal values in the response."""


class SafetyStop(RuntimeError):
    """Physical safety limit reached; a terminal task failure, not infrastructure."""


def number(value, low, high):
    if type(value) not in (int, float):
        raise BadRequest("invalid number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise BadRequest("invalid number") from None
    if not math.isfinite(result) or not low <= result <= high:
        raise BadRequest("number out of range")
    return result


def validate_request(path, payload):
    if path not in SCHEMAS:
        raise KeyError(path)
    answer_submission = path == "/v1/submit" and type(payload) is dict and set(payload) == {"answer"}
    if type(payload) is not dict or (set(payload) != SCHEMAS[path] and not answer_submission):
        raise BadRequest("invalid fields")
    result = dict(payload)
    if answer_submission:
        answer = result["answer"]
        if type(answer) is not dict or set(answer) != {"dimensions_m"}:
            raise BadRequest("invalid answer")
        dimensions = answer["dimensions_m"]
        if type(dimensions) is not list or len(dimensions) != 3:
            raise BadRequest("invalid dimensions")
        result["answer"] = {"dimensions_m": [number(value, 0.00001, 2.0) for value in dimensions]}
    if "duration" in result:
        duration = number(result["duration"], 0.02, 5.0)
        ticks = round(duration / TIMESTEP)
        if abs(duration - ticks * TIMESTEP) > 1e-9:
            raise BadRequest("duration must be an integer number of physics ticks")
        result["duration"] = ticks * TIMESTEP
    if "targets" in result:
        targets = result["targets"]
        if type(targets) is not list or len(targets) != len(JOINT_LIMITS):
            raise BadRequest("invalid joint targets")
        result["targets"] = [number(v, *limits) for v, limits in zip(targets, JOINT_LIMITS)]
    if "opening" in result:
        result["opening"] = number(result["opening"], 0.0, 0.095)
    if "camera_id" in result:
        if type(result["camera_id"]) is not str or result["camera_id"] not in CAMERA_IDS:
            raise BadRequest("invalid camera")
    return result
