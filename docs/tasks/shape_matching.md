# Shape matching

The standard YAM arm must place a red square, blue cylinder, and green triangular prism in matching colored tabletop sockets. Socket order, small position/angle offsets, and initial piece positions/orientations vary with the private episode seed. Each socket has a collision floor at 8 mm and a ring extending to 34 mm. Pieces are 24 mm tall. The rim consists of convex prism segments surrounding a genuine open cavity; the cavity does not use a solid collision board or object teleportation. Cylinder apertures use 48 segments. A 3.5% aperture clearance allows physical seating.

A public PNG depicts the desired occupied socket pattern, with +y upward and +x right. It contains target geometry only and does not change when the actual pieces move. The gateway exposes this explicit target; object state, seed, poses, contact identities, and partial scores remain private.

The scorer checks each complete piece outline at both top and bottom against its assigned convex aperture, requires center height within 1.5 mm of 20 mm, requires upright orientation, slow linear/angular motion, and no contact with any robot descendant. A triangular piece rotated by 180 degrees does not fit; valid rotational symmetries are accepted. Pieces above, beneath, beside, or supported by the robot do not count. Each seated piece earns one third of the score; success requires all three. The shared task lifecycle also requires a stable terminal success window during the final settling second.

MuJoCo multiple-contact collision detection is enabled for stable support of cylinder/mesh pieces. No robot geometry, actuators, or dynamics are changed. Evaluation tests use trusted state fixtures to verify scoring; these fixtures are not a demonstrated vision controller or LLM solution.

Run task tests:

```sh
.venv/bin/python -m pytest tests/test_task_shape_matching.py -q
```

Tests cover zero/partial/full scores, wrong assignment, triangle orientation, above/below-floor shortcuts, robot contact, velocity, passive physical seating, physical rim obstruction, target PNG invariance, and randomization. Full agent-through-HTTP completion remains part of benchmark integration validation.
