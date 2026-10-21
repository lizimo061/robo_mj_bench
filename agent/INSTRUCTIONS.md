# YAM robotics task API

Call `robot.task()` first to read this episode's objective and approximation
label. Some tasks include a public reference image describing the desired result.
Perform the instructed physical task and submit when complete. Scores, object
state, ground-truth dimensions, and evaluator details are never returned.

You have a standard six-joint I2RT YAM arm with its linear_4310 parallel gripper
and D405 wrist camera. Use camera images and
your robot's own telemetry. You may write and execute local code. Your environment
contains `robot_client.py`; it needs only Python's standard library. The operator
supplies `ROBOT_URL` and `ROBOT_TOKEN` for this episode.

```python
from robot_client import RobotClient

robot = RobotClient()
goal = robot.task()
print(goal["instruction"])
state = robot.robot_state()
with open("front.png", "wb") as image:
    image.write(robot.observe("front"))
robot.wait(0.1)
# Plan movements based on your observations; then call move_joints/set_gripper.
# robot.submit() ends the attempt permanently.
```

## Available operations

All operations are POST requests with a JSON object and `Authorization: Bearer TOKEN`.
The client provides these methods:

| Client method | POST path | JSON payload |
|---|---|---|
| `task()` | `/v1/task` | `{}` |
| `robot_state()` | `/v1/robot_state` | `{}` |
| `observe(camera_id)` | `/v1/observe` | `{"camera_id":"front"}` |
| `move_joints(targets, duration)` | `/v1/move_joints` | `{"targets":[-0.6,1.047,1.047,0,0,0],"duration":1.0}` |
| `set_gripper(opening, duration)` | `/v1/set_gripper` | `{"opening":0.02,"duration":1.0}` |
| `wait(duration)` | `/v1/wait` | `{"duration":0.1}` |
| `submit(answer=None)` | `/v1/submit` | `{}` or the measurement answer below |

The HTTP observation returns `camera_id`, `mime_type`, and `image_base64`; the
Python client decodes it to PNG bytes. Images are 640 by 480 RGB. Cameras: `front`,
`left`, `overhead`, `wrist`. No depth or object annotations are provided.

`task()` returns only `task_id`, `instruction`, `approximation`, and optionally
`reference_image_base64` (PNG). A reference image is an explicit goal, with no
current-state annotation. For measurement, submit
`robot.submit({"dimensions_m": [length, width, height]})` in meters. Other tasks
accept no answer object. Guesses receive no feedback beyond the final submission
receipt. Do not assume the button task is active.

Robot telemetry contains exactly `joint_positions` (six radians),
`joint_velocities` (six radians/second), `gripper_opening` (sum of the two finger
rail coordinates in meters), and `sim_time` (seconds). The public robot model
specifies the relationship between rail travel and the physical pads.

## Public robot geometry and limits

World coordinates: meters; +Z up. The robot base is at world (0,0,0) on the table.
Command joints in upstream order `joint1` through `joint6`.
`robot/yam_linear_4310.xml` and its meshes are the unchanged PUBLIC robot-only
model. It includes joint frames, inertias, collision geometry, gripper coupling,
and wrist-camera calibration, with no task objects, seed, answers, or evaluator.
You may use it for local FK/IK. The optional `robot_kinematics.py` helper requires
MuJoCo and NumPy; `robot_client.py` itself uses only the standard library.

Upstream: https://github.com/google-deepmind/mujoco_menagerie/tree/cda5236f98383f6fbaa4820c6b18dba892cd087f/i2rt_yam
License: MIT; retain `robot/LICENSE`. File hashes are in `robot/PROVENANCE.json`.

Joint limits: `[-2.61799,3.14159]`, `[0,3.66519]`, `[0,3.14159]`,
`[-1.69297,1.5708]`, `[-1.5708,1.5708]`, `[-2.0944,2.0944]` radians.
Episode home targets: `[-0.6,1.047,1.047,0,0,0]`.
Targets interpolate linearly; the maximum target slew is 1.5 rad/s per joint.
Choose duration from the change relative to your previous target, not merely the
measured joint angle. Physical position can lag the target. Upstream actuator
limits are ±28 Nm for joints 1–3 and ±10 Nm for joints 4–6. Upstream position
gains and gravity compensation are preserved. API gripper opening is 0–0.095 m,
mapped to half the opening on the coupled upstream actuator, with target slew
at most 0.05 m/s. The upstream gripper actuator force limit is ±100 N; its negative
squeeze range is not exposed by the nonnegative opening API.

The overhead camera is at (0.3,0,1.2), looks down -Z, image-right is +X, and image-up
is +Y. Its vertical field of view is 58 degrees. This robot/camera calibration is
public; episode object locations must be estimated from observations.

Durations must be from 0.02 through 5 seconds, in multiples of 0.002 seconds.
Observation and telemetry reads do not advance physics. Motion and wait do.
There is no reset, undo, object manipulation API, or score endpoint.

## Completion and budgets

Follow the current task instruction. Placement, packing, shape matching, and
tower tasks require released, stable objects. Submission closes control
immediately; the server runs one second of settling while holding the last
actuator targets and checks the final half-second. Release objects and withdraw
before submitting. Settling is separate from the command-time budget.
Measurement grades submitted dimensions with 2 mm inclusive tolerance.
Writing uses a disclosed contact-tracing template; pouring uses a disclosed
granular-transfer proxy. Approximation labels are part of the objective.

For the default button task specifically:
A valid press requires at least 6 mm button travel and at least 0.5 N finger
contact continuously for 80 ms. The event latches; releasing afterward is allowed.
Any button contact force above 30 N or travel beyond 13 mm makes the attempt fail.
The nominal button stop is 12 mm with 1 mm numerical tolerance. Unsafe commands
end the episode immediately with a fixed `safety_stop` status; the Python client
raises `RobotError`. No subsequent submission or control is possible. A wrist-only press
does not count. Submit ends the episode; further calls return 409. The submit
response only confirms receipt, and has no score.

Default episode budgets: 2,000 authenticated known-route requests, 200 images,
500 motion/wait commands, 120 simulated seconds, 1,800 wall-clock seconds.
Submission remains available after these budgets expire. Failed motion attempts
consume motion/time allowance. HTTP 400 means invalid input, 401 bad credentials,
404 unknown endpoint, 409 ended episode, 429 budget exhausted, and 500 an
infrastructure error. Stop after an infrastructure error.
