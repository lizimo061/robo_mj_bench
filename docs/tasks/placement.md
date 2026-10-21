# Crocodile placement

Task ID: `placement`. Uses the shared upstream YAM arm and gripper without changing robot geometry, dynamics, or global physics options.

Move the green crocodile toy entirely inside the blue painted rectangle, lay it flat on the table, release it, and withdraw the robot. The paint's inner edges define the goal. The toy is a graspable 120 g compound rigid body with a torso, head, snout, asymmetric two-part tail and four feet. Its shape is deliberately irregular so checking its center cannot solve the scoring problem. The non-colliding eye markers remain inside the head's projected footprint.

The initial toy and target positions and their yaw angles vary by seed. Targets stay in the reachable tabletop region. Initial locations are separated from the target. The outline is visual-only geometry and cannot hold an object above the table.

## Trusted scoring

The grader transforms every corner of every collision box into the oriented target frame. Full success requires all projected corners, including the tail and feet, to lie inside the rectangle. The toy must also be upright, slow, in physical contact with the table, have its bottom within 2 mm below to 3 mm above the table surface, and have no contact with any robot descendant body. Consequently a contained but held, supported by the robot, airborne, upside-down, or below-table toy earns zero.

A valid resting toy partly inside the rectangle receives `99 * component_containment_fraction`. The fraction is the sum of each component's projected intersection area divided by the sum of its projected area. This intentionally weights components, including overlapping regions; it is not the silhouette-union area. Each projection uses a convex hull of all eight transformed corners, and rectangle clipping accounts for target orientation. Any overhang prevents full success regardless of partial credit. Full success receives 100, subject to the shared one-second terminal settling and final half-second stability checks.

Only the natural-language goal is returned through the public task API. Geometry, poses, containment fractions and physical contact details stay in private evaluation. The task has no runtime state setters, teleportation, pose corrections, or reward feedback.

## Validation

`python -m pytest tests/test_task_placement.py -q` exercises zero/partial/full fixtures, centered but rotated overhang, tail-only overhang, airborne and below-table rejection, upside-down rejection, velocity, robot support/contact rejection, actual passive settling, terminal stability, and the public specification allowlist. Pose-writing fixtures are trusted tests only.
