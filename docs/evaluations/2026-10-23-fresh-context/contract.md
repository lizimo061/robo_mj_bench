# Fresh-context model-control evaluation contract

Declared 2026-10-23 before any robotics episode was launched in this evaluation.

## Trial budget

Attempt one fresh episode per family first: button, placement, pouring, tidying,
shape_matching, writing, tower, measurement, bimanual_handover, bimanual_tray,
bimanual_drawer. Expand to five attempts per family (55 total planned) only if
subscription inference and robot execution remain functional. Infrastructure
failures are separate from robotics failures; never infer task difficulty from
an unattempted or infrastructure-blocked task.

## Policy and controller

Use fresh ChatGPT-signed-in CLI policy contexts with user configuration ignored,
shell/external tools disabled where supported, and explicit public inputs.
Inputs: the extracted allowlisted agent bundle, INSTRUCTIONS.md, immutable public
robot model/calibration, public robot-only IK helper, public goals/reference
images, RGB cameras, robot-only telemetry, and previous public actions/responses.
The public robot-only IK helper is permitted controller support, not a source of
object state. Log each selected action, reason, public response, and observation.
Use move_bimanual and wrist_left/wrist_right for two-arm episodes.

Do not inspect implementation code, scoring tests, procedural scene files,
private seeds/states, prior policies/pilot decisions/trajectories, vision_demo.py,
or private-state validation scripts. Launch only the trusted server entry point
with a new private run directory and omit --seed. Read only the episode transport
token from private run material. Do not expose authentication credentials.
Do not switch to paid API billing. Suppress private diagnostics during control.
After termination only, allow an operator projection of valid/success/score/
termination/usage. Keep secrets, seeds, private audits, and bulk runs out of Git.

The CLI uses the user's existing ChatGPT authentication without reading/copying
its secrets. Local same-host execution and prompt restrictions are not an
adversarial sandbox. Actual tool removal must be verified before claiming
strong isolation. If CLI inference fails, any optional in-chat fallback must be
labeled a development pilot, not an isolated or independently attested benchmark.

## Stopping

Stop affected control after an infrastructure error. Do not launch episodes
while a minimal subscription inference request cannot complete. Record
unattempted families explicitly. Report completed, valid, successful, denominators,
infrastructure-invalid attempts, and verifiable model/settings. Do not fabricate
usage or turn failures into robotics outcomes.
