# Pouring: granular transfer proxy

The task uses the unchanged standard I2RT YAM robot and linear gripper. A blue,
open source cup contains 21 dynamic 1 g spheres with 5 mm radius. An orange,
open receiving cup begins empty. Both cups are free rigid bodies with physical
floors and sixteen collision walls. Each bead is a persistent dynamic body;
there is no transfer trigger, tilt-based content creation, or teleportation.
All assets are procedural and remain on the trusted simulator service.

This is explicitly `granular_transfer_proxy`, **not liquid fidelity**. It tests
physical container manipulation and granular transfer. Surface tension,
viscosity, wetting, and fluid flow are absent; these scores should not be
represented as liquid-pouring performance or combined with validated liquid
results without distinguishing the approximation.

## Public objective and scoring

Transfer at least 90% of the beads into the orange cup, with at most one spilled
bead. Return both cups upright to the table and release all task objects.
Submit only after the contents and containers have settled. The receiving cup
can be repositioned. The server checks a stable terminal window after settling.

- Full success: at least 19 of 21 complete bead volumes are inside the receiving
  cavity, no more than one bead is spilled, both cups are upright and resting at
  table level, contents are slow, and nothing is touching the robot.
- Partial score: `100 * target_count / 21`, capped at 99 if full criteria fail.
  Unreleased, airborne, moving, or invalid terminal states receive zero.
- A spill is a bead outside both cups at submission, or a bead previously observed
  at table height outside both cavities. Recovering a spilled bead does not erase
  its spill history.
- The private report contains source/target/outside counts, unique spilled count,
  and fixed total mass. Target + source + outside always sums to 21.
- Overlapping cup cavities and ambiguous double-containment cannot earn credit.

Ground truth uses actual cup transforms and bead positions on the trusted server.
Agent observations expose only the ordinary RGB views and robot telemetry. Task
instructions describe colors and the objective; they contain no episode poses,
seed, bead coordinates, or target-state metadata. No intermediate task reward or
success oracle is added.

## Validation

`tests/test_task_pouring.py` covers empty, partial, and full transfer; mass
accounting; spills and recovered-spill history; floating or robot-contacting
containers; moving contents; nonfinite state; public-spec allowlisting; and
passive physics stability in both source and destination. Test fixtures may set
trusted state directly to isolate scorer criteria. This is not an agent control
API and is not an end-to-end controller demonstration. Closed-loop visual YAM
pouring remains to be validated before claiming agent task solvability.
