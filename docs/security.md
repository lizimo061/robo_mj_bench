# Threat model and deployment boundary

Assume the evaluated agent can read, write, execute, introspect, and replace
anything inside its sandbox. Prompt instructions are not access control. The
simulator and scorer belong outside that sandbox.

## Trusted and untrusted components

Trusted: `src/robo_mj_bench`, `scripts`, private run directories, Python/MuJoCo,
the renderer, operator, and the host/VM/network enforcement around the gateway.
Untrusted: model-generated code, the agent filesystem, request bodies, credentials
held by an agent (limited to its episode), and all agent reports.

Provide only `artifacts/agent-bundle.zip` to an agent. This contains the public
I2RT YAM robot-only model and an optional local numerical IK helper, as well as
the API client and instructions. Robot geometry is legitimate calibration;
the agent's local model has no task objects or episode ground truth. Bundle
inventory is explicit and hash-verified against the pinned upstream files.
Do not install the trusted
package there or mount this repository, host `/proc`, Docker socket, private run
volume, cloud credentials, or management sockets. Private asset geometry and seeds
must remain on the trusted host even if the development source is public.

For actual evaluation use a separate agent host or hardened VM and a narrow
network allowlist permitting only its gateway and whatever model-provider access
the harness requires. Model-provider credentials should stay in an external
orchestrator if the agent does not need them. Deny cloud metadata and management
network access. Host isolation is an operator requirement, **not implemented by
this repository**. This prototype's HTTPServer is a single-episode development
service, not a public Internet edge or multi-tenant production server. Put it
behind a proxy with connection/rate limits, TLS/private transport, and authentication
enforcement. A single-threaded connection can monopolize its five-second read
timeout; the socket timeout is not a denial-of-service defense.

## API defenses

- Exact routes, exact JSON fields, exact vector sizes, finite real values only;
  booleans and strings are not coerced to numbers.
- Bounded payload, durations, actuator targets, target slew, torques, and budgets.
- One opaque bearer credential per server episode; constant-time comparison.
- No generic object serializer, simulator handle, arbitrary file/URL fetch, free
  camera, depth, segmentation, reset, reward, success, or arbitrary force API.
- Robot telemetry explicitly projects six YAM arm joints and finger opening; the
  button's joint coordinate never appears in the public joint vector.
- Rendered RGB is re-encoded as a fresh PNG without geometry labels or metadata.
- JSON rejects duplicate keys and nonfinite literals. HTTP rejects ambiguous
  content lengths, duplicate authorization, and transfer encoding.
- Error responses have fixed text and no stack traces, paths, seeds, or values.
- The client disables redirects and implicit proxies to avoid forwarding the
  episode bearer token to an unexpected destination.

Known routes and validation errors are public information. Physical observations,
including indicator color, shadows, robot tracking error, and contact-induced
motion, are legitimate sensor information. Inferring an object's location from
these observations is allowed. Private reward feedback is not returned.

`/v1/task` returns only a task identifier, natural-language instruction,
approximation label, and optionally an explicit desired-result PNG. Target
patterns are legitimate goals; this endpoint never returns current object poses,
hidden measurement answers, contact maps, evaluator state or scores. Goal PNGs
are decoded and freshly encoded to remove metadata. Measurement accepts one
strict finite `[length,width,height]` answer at final submission; no accuracy
feedback or retries are exposed. A rejected answer on another task does not
silently close or reset the episode.

## Physical validity

The API changes actuator targets, never `qpos` during an episode. A 2 ms physics
step, position servos, capped actuator forces, and target interpolation limit
obvious teleport and tunneling shortcuts. Target slew is not an actual joint
velocity guarantee; actual velocities are measured telemetry. No comprehensive
collision-damage model is implemented yet.

The scorer requires 6 mm button travel plus 0.5 N contact from a finger for 80 ms
continuously. Disconnected taps, gravity, or a wrist-only contact do not qualify.
Peak force across all cap contacts above 30 N or button travel above 13 mm
(12 mm nominal stop plus 1 mm numerical tolerance) triggers an immediate terminal
safety stop before scoring on that tick. This is a valid task failure with score
0, including if the button had already latched. The stop is enforced after each
physics tick, so one tick can overshoot the threshold; no further motion is
allowed. The API returns only a fixed `safety_stop` status, with no cause, score,
travel, or force. Do not rely on compliant joint limits as an exact rigid bound. Simulation
warnings/nonfinite dynamics stop the episode as an infrastructure failure.

Submission ends the attempt. For this event task the press latches, and no
post-submission settling is required. Other physical placement tasks implement
release/robot-support checks and one second of settling, requiring success at
every 2 ms tick throughout the final half-second. Settling uses the final actuator
targets and exposes no control or reward callback. Measurement answers and
contact-writing history do not require a placement stability window.

For the seven new tasks, `safe` in a private report means physics remained valid
and no implemented task limit triggered. It is not a general collision/damage
guarantee. Pouring uses persistent beads and conserved counts, explicitly labeled
`granular_transfer_proxy`. Writing records ink only from solved physical tip
contact and uses an explicit frozen-template tracing proxy. Neither is presented
as high-fidelity fluid dynamics or general handwriting recognition.

## Audit and remaining work

Private logs store accepted requests and outcomes, images/hashes, final geometry
state, scene hash, physics version, seed, and usage. They never log authorization
headers. A report is atomically replaced after complete serialization; nonfinite
diagnostics become JSON null. Internal errors and operator-stopped runs are ineligible;
physical safety stops are valid scored failures.
A persisted submitted result is authoritative if later acknowledgment/logging fails.
Logs are operator-controlled and are not cryptographically tamper-evident.

Tests exercise input/auth/HTTP boundaries, terminal lifecycle, budgets, telemetry
projection, deterministic physics, passive-button false positives, contact dwell,
and invalid-result regressions. The trusted reference controller checks feasibility.
These checks do not prove immunity to sandbox escape, native renderer exploits,
kernel vulnerabilities, side channels, or unknown physics exploits.

Before a public leaderboard: deploy/enforce the VM boundary, add a model harness,
automated adversarial sandbox probes and paired hidden-state tests, immutable
run manifests and signed artifact retention, held-out task generators/assets,
cross-seed physics validation, and a documented incident/rerun policy. Keep
development feedback separate from hidden evaluation. Do not reuse one agent's
scratch files/memory across zero-shot episodes.

In particular, define invalid-physics rerun/accounting before leaderboard use;
silently excluding model-induced simulator failures could reward deliberate
instability. Current fixtures and idle HTTP checks validate implementation,
not manipulation solvability, complete sandbox isolation or LLM performance.
