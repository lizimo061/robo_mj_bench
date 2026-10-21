# robo_mj_bench

A MuJoCo robotics benchmark for vision-based LLM agents, featuring a standard
YAM arm, eight tabletop tasks, and private ground-truth scoring.

An independently implemented, AA-Robotics-inspired benchmark for language-model
agents controlling a simulated robot from RGB images and robot telemetry.
Eight task families are implemented with a shared private evaluator and API.
This is not an official Artificial Analysis benchmark or a physical-score replica.

The robot is the standard **I2RT YAM with linear_4310 gripper and D405 wrist
camera**, vendored unchanged from MuJoCo Menagerie at commit
`cda5236f98383f6fbaa4820c6b18dba892cd087f`. Its six arm joints, gripper coupling,
collision geometry, inertias, gravity compensation, actuator gains, and force
limits are preserved. Task composition adds a table, button, lights, external
cameras, and a task home pose. See the
[model provenance](src/robo_mj_bench/models/i2rt_yam/PROVENANCE.json) and
[MIT license](src/robo_mj_bench/models/i2rt_yam/LICENSE).

The trusted service owns MuJoCo, procedural scene assets, camera rendering,
ground truth, scoring, and audit files. The agent receives a standard-library
HTTP client, public instructions, and the immutable robot-only model for local
kinematics. The API never returns object poses,
episode seeds, rewards, success flags, or evaluator data.

## Run locally

Python 3.11+ and an OpenGL-capable environment are required.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]' -c requirements.lock
.venv/bin/python -m pytest
.venv/bin/python -m robo_mj_bench.server --run-dir private-runs/demo --seed 7
# Select another task in a fresh episode:
.venv/bin/python -m robo_mj_bench.server --task measurement --run-dir private-runs/measure --seed 17
```

The server binds to `127.0.0.1:8765` by default. Each start requires a new run
directory and produces `agent-token`, `manifest.json`, `scene.xml`, `events.jsonl`,
observed frames, and a terminal `result.json`. The seed option is for operator
reproducibility; omitting it uses a randomly generated private seed.

In a separate **local development** terminal:

```sh
export ROBOT_URL=http://127.0.0.1:8765
export ROBOT_TOKEN="$(cat private-runs/demo/agent-token)"
PYTHONPATH=agent .venv/bin/python -c 'from robot_client import RobotClient; print(RobotClient().robot_state())'
# Optional diagnostic policy using only RGB and public robot calibration:
.venv/bin/python agent/vision_demo.py
```

Read [agent/INSTRUCTIONS.md](agent/INSTRUCTIONS.md) for the public action contract,
robot geometry, camera calibration, limits, scoring rules, and budgets.

**Local development is not a secure evaluation environment.** An agent with access
to this checkout, the server account, or `private-runs/` can read the answer.
For actual evaluations, place the agent on a different isolated host/VM, give it
only the client bundle and its own credential, and expose only the gateway through
an authenticated private tunnel or TLS proxy. Do not give it a terminal in this
operator workspace. See [docs/security.md](docs/security.md).

## Trusted validation and agent handoff

```sh
.venv/bin/python scripts/validate_button.py --output private-runs/validation --seeds 0 1 2 3 4 5 6 7 8 9
.venv/bin/python scripts/validate_button.py --output private-runs/render --seeds 7 --render
.venv/bin/python scripts/build_agent_bundle.py
.venv/bin/python scripts/e2e_button.py --output artifacts/button-videos --seeds 101 202 303 404 505
.venv/bin/python scripts/verify_button_videos.py artifacts/button-videos
.venv/bin/python scripts/smoke_suite.py --output artifacts/task-suite-smoke
```

The validation controller intentionally uses the private button position to verify
that physical pressing is possible through the same actuator API. Its results
are **solvability checks, not LLM performance**. It does not teleport robot or
object state. The agent bundle contains `robot_client.py`, `robot_kinematics.py`,
`INSTRUCTIONS.md`, and an explicitly allowlisted public `robot/` inventory with
license and provenance. It contains no task scene, benchmark implementation,
episode seeds, tests, evaluator, or results. The local IK helper requires
MuJoCo and NumPy and loads its own public robot-only model.

The end-to-end recorder runs the scripted RGB diagnostic policy in a separate
process using real HTTP, records all four cameras at 20 fps, and persists each
operator-only result and request trace. Videos play at simulation speed plus
initial/final holds. These validate integration, not LLM performance or sandbox
security. `ffmpeg` and `ffprobe` are required. The verifier decodes every frame,
checks the file hash, and replays accepted actions without rendering to require
exact final physics state and scores.

macOS uses MuJoCo's offscreen OpenGL context. On a headless Linux server configure
`MUJOCO_GL=egl` with a working EGL driver, or `MUJOCO_GL=osmesa` with the required
software-rendering libraries. Physics tests do not need rendering. Rendering
errors close the episode and produce an invalid infrastructure result.

## What is implemented

- Standard six-joint I2RT YAM, linear_4310 gripper, native wrist camera, three
  external cameras, and bounded actuator commands.
- Eight task families with randomized scenes and private geometry/contact scoring.
- Seven authenticated POST endpoints; strict schemas and finite numeric validation.
- No agent reset, generic simulator endpoint, success/reward feedback, or debug API.
- Request/image/command/time budgets; terminal submission; private atomic reports.
- Unit, transport, physics, scoring, and regression tests.

Hardware validation of this standard YAM simulation is still pending. This release
does not yet include model-provider adapters, a hosted
sandbox/VM orchestrator, a multi-model leaderboard, or validated sim-to-real
comparisons. These are follow-up milestones rather than implied capabilities.

## Task families

Choose `--task` on the trusted server; the agent reads its objective with
`RobotClient.task()`. The endpoint returns public goals and reference images
only. The same robot-only bundle works for every task.

| Selector | Objective | Simulation scope |
|---|---|---|
| `button` | Press a spring-loaded button | Rigid-body contact event |
| `placement` | Put the complete crocodile inside a rectangle | Released full-footprint containment |
| `pouring` | Transfer contents between physical cups | **Granular bead proxy**, not liquid simulation |
| `tidying` | Pack four blocks and close the lid flush | Physical floor, walls, lid and corner checks |
| `shape_matching` | Seat square, circle and triangle in matching sockets | Real cavities and public target PNG |
| `writing` | Trace HELLO WORLD with a physical marker | **Contact-writing proxy**, not general handwriting recognition |
| `tower` | Build the tallest stable freestanding stack | Physical contact/support/height scoring |
| `measurement` | Report box dimensions within 2 mm | Private random dimensions; visible physical SI rulers |

See [task documentation](docs/tasks) for scoring and limitations. Tests include
trusted terminal-state fixtures and physical contact checks. The idle HTTP smoke
script checks startup, four cameras, public goals, submission and terminal 409
responses for each task, and rejects initially-complete scenes. **These checks
do not demonstrate that an LLM can solve the seven new tasks.** Only button has
a demonstrated closed-loop RGB diagnostic policy so far.

Measurement uses `robot.submit({"dimensions_m": [length, width, height]})`;
other tasks use `robot.submit()`. Answers and scores remain private. Physical
placement tasks settle for one second after submission, with full success
required throughout the final half-second. No commands are allowed during this
period. Successful agent rollouts, held-out calibration and physical transfer
validation remain future evaluation work.

## Results and provenance

`result.json` includes `valid`, `success`, `score`, `termination`, physical
diagnostics, usage, the private seed, a scene hash, MuJoCo version, robot identity,
and upstream source revision. Historical videos
under `artifacts/button-e2e-2026-10-21` used the retired custom arm. Current YAM
results and five videos are under `artifacts/yam-button-e2e-v1`.
Explicit submission and a terminal safety stop are valid episode terminations.
Internal errors and operator interruptions
are ineligible (`valid: false`, `success: false`, `score: null`). The button task
scores 100 for a valid sustained gripper press within force limits, otherwise 0.
An unsafe physics tick ends the episode immediately as `safety_stop`, with score
0 even if the button had previously registered a press. A fixed `safety_stop`
receipt contains no score or private diagnostics; the client raises `RobotError`.
The green light is a physical observation of the latched press, not a guarantee
that all safety criteria hold. Submitted results are authoritative once persisted,
even if delivering the HTTP acknowledgment fails afterward.

The source inspiration is [AA-Robotics](https://artificialanalysis.ai/articles/aa-robotics).
Implementation references: [MuJoCo](https://mujoco.readthedocs.io/),
[privileged vs sensor observations](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html),
and [Firecracker deployment requirements](https://github.com/firecracker-microvm/firecracker/blob/main/docs/prod-host-setup.md).
