# Bimanual manipulation

These three tasks require temporal cooperation, in addition to final geometry.
Select `bimanual_handover`, `bimanual_tray`, or `bimanual_drawer` on the trusted
server. All use two standard I2RT YAM arms with the linear_4310 gripper and D405
wrist camera. They are benchmark tasks, not validated real-robot procedures.

## Proposed challenges and implemented scoring

| Task | Coordination challenge | Full-credit requirements |
|---|---|---|
| `bimanual_handover` | Transfer possession without using the table as an intermediate support | Left-only airborne grasp for 0.10 s, shared grasp for 0.10 s, right-only airborne grasp for 0.15 s, then flat released placement fully inside the green pad |
| `bimanual_tray` | Maintain a shared load and cargo containment | Both arms grasp their designated handles during a level unsupported lift above 9 cm for 0.20 s, followed by stable released placement with the entire tray inside the goal and all four balls retained |
| `bimanual_drawer` | Hold against a spring while the other hand inserts | Left gripper holds the drawer at least 10 cm open while the right grips the fully inserted block for 0.10 s; at submission the block is released and stable while the left continues to hold the drawer open |

Grasp detection requires solved contact on **both opposing finger pads**, each
carrying at least 0.05 N, with opposing contact normals. A nearby end effector,
one-finger push, or open gripper does not count. Handover height must exceed
10 cm during all transfer phases. Losing the required supporting grip or
touching an external support between phases permanently invalidates the transfer.
Tray spillage is also permanent: putting a spilled ball back cannot restore
full credit. The tray must remain level during its cooperative lift (world-up
dot product above 0.98). Tray handles are at negative y for the left arm and
positive y for the right arm; placement includes handles and walls.
The lift must start within 6 cm of the tray's initial position. Both arms must
keep holding the level, unsupported tray above 6 cm until its entire footprint
is over the goal. Setting it down early or losing a handle permanently fails
the carry, so an initial lift followed by a table push cannot earn full credit.
Drawer credit requires the block to cross from outside to fully inside while
the left holds open and the right grips; grabbing an already inserted block
does not count as coordinated insertion.

Final requirements must hold throughout the last 0.5 s of a 1 s passive
submission window. Commands stop at submission; drawer holders must maintain
their commanded grasp against the spring. Handover stages award 25 points each;
tray lift and coordinated drawer insertion award 40 points each. Full credit
requires all terminal predicates. A failed handover or spilled tray scores zero.
Transient terminal success cannot earn 100 points.

## Public robot calibration and API

World z is up. The left mount is `(0, -0.32, 0)` with yaw `+0.65` radians;
the right mount is `(0, +0.32, 0)` with yaw `-0.65` radians. The standard robot
body is nested under each rigid mount without modifying its local transform.
Home joints are `[-0.6, 1.047, 1.047, 0, 0, 0]` for left and
`[+0.6, 1.047, 1.047, 0, 0, 0]` for right. The public robot-only model is the
same allowlisted model used in single-arm episodes. Optional local IK accepts
`RobotKinematics(arm="left")` or `RobotKinematics(arm="right")` and world targets.
It solves individual kinematics, not inter-arm collision avoidance or paths.

`robot_state()` returns exactly `left`, `right`, and `sim_time`. Each arm exposes
only six joint positions, six joint velocities, and gripper opening. No object
state, event-stage state, random seed, progress, or reward is exposed.

```python
from robot_client import RobotClient

robot = RobotClient()
state = robot.robot_state()
left = {"targets": state["left"]["joint_positions"],
        "opening": state["left"]["gripper_opening"]}
right = {"targets": state["right"]["joint_positions"],
         "opening": state["right"]["gripper_opening"]}
robot.move_bimanual(left, right, duration=0.2)
image = robot.observe("wrist_left")
robot.submit()
```

`POST /v1/move_bimanual` takes exactly `left`, `right`, and `duration`; each arm
takes exactly `targets` (six joint angles) and `opening` (meters). Both sides
are validated before any control or physics mutation. All 14 actuator targets
interpolate together over one duration, charging one command and one simulation
interval. Existing joint/gripper speed bounds, finite-number rules, budgets,
authentication, and terminal behavior apply. No task accepts an answer payload.
Dual episodes reject the single-arm motion endpoints. Single-arm episodes reject
dual commands. Cameras are `front`, `left`, `overhead`, `wrist_left`, and
`wrist_right`; `wrist` remains a single-arm-only alias.

## Validation and limits

Tests cover upstream dynamics/geometry fidelity for both copies, arm coupling,
atomic validation, telemetry allowlisting, contact gates, dwell times, ordered
handover, persistent spills, spring closure, and terminal fixtures. Public-model
IK checks cover representative approach targets across multiple seeds, but do
not prove collision-free paths. HTTP smoke tests exercise synchronized commands,
all five cameras, private submission, and rejection of post-submission access.

```sh
.venv/bin/python -m pytest -q tests/test_bimanual.py
.venv/bin/python scripts/smoke_suite.py --output artifacts/dual-smoke \
  --tasks bimanual_handover bimanual_tray bimanual_drawer
.venv/bin/python scripts/render_task_previews.py \
  --tasks bimanual_handover bimanual_tray bimanual_drawer
```

README GIFs are scene previews with scripted motions. Trusted scoring fixtures
may set poses or internal history directly; they are not successful policies.
Successful actuator-driven task rollouts, LLM baselines, and real-robot transfer
have **not** been validated for these three tasks.

The summed solved normal force of inter-arm contacts above 35 N terminates an
episode as a valid safety failure. This is an engineering cutoff, not a general
damage model. Tray cargo is rigid spheres; the drawer is an ideal prismatic
joint with a return spring. Grips use ordinary contact/friction, without magic
welds, attachments, scripted grasp constraints, or ground-truth action APIs.

The trusted process owns object geometry, event histories, and reports. Evaluate
agents on a separate isolated VM/host with only the public client bundle and
the authenticated gateway. Giving an agent this source tree or private run
directory reveals the evaluator; local process separation is not a sandbox.
