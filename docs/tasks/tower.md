# Freestanding tower task

The standard YAM must stack four physical rigid cuboids on the table, release
all blocks, and withdraw. Each block has a noncolliding white top marker;
that face must remain upward (within about 2.6 degrees of vertical).
The task is a rigid-body approximation to lightweight foam blocks, not a
validated deformable-foam simulation. The robot geometry and controller are
unchanged. Initial objects occupy separated randomized tabletop positions.

Public instructions disclose the rules and the marked-face orientation, but
not episode positions or scoring state. They require at least 80% of each
upper footprint to overlap its support. Keeping the marked face upward makes
the maximum height unambiguous: the sum of the four original vertical lengths,
independent of stack order or yaw. The private evaluator knows those lengths.

## Private scoring

The evaluator builds a graph from actual MuJoCo contacts. A valid stack starts
at a block contacting the table with its bottom within 2 mm of the surface.
Every upward edge requires contact, less than 2 mm surface separation or
penetration, and exact projected-polygon overlap of at least 80% of the upper
footprint. Each supported suffix's center of mass must lie at least 2 mm
inside its supporting block. All blocks have equal mass. Velocities must be
below 0.03 in both linear and angular body velocity components.

Any contact with the robot or another external support, side contact between
blocks, excessive tilt, or moving block invalidates that entire contact
component. This also rejects indirect bracing through another block. An
isolated floating object or a single tall object earns no height credit.

Partial credit is the physical top height of the tallest valid connected
stack with at least two blocks divided by the known maximum height. Full
success requires all four blocks in one valid chain. Submission uses the
common one-second passive settling procedure, with full success required
throughout the sampled final half-second window. No scoring hook changes
positions, creates constraints, or teleports objects.

## Validation and limitations

`tests/test_task_tower.py` uses trusted state-placement fixtures to verify
partial and full stacks, gap/airborne states, overhang, under-table placement,
orientation, motion, robot bracing, and public/private separation. A complete
fixture stack is additionally stepped through real passive physics and the
terminal stability window. These tests verify task mechanics and scoring;
they are not an LLM-controlled or RGB-policy end-to-end evaluation.

The overlap and orientation rules deliberately constrain the task: creative
leaning/cantilever structures are outside this initial benchmark. Soft contact
penetration is tolerated within 2 mm. Production benchmark validation still
needs adversarial physical rollout testing and independent policy episodes.
