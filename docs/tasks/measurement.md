# Box measurement

`measurement` uses the standard YAM arm and gripper. An orange rigid box has three independently sampled private dimensions: initial x edge 75–115 mm, initial y edge 35–65 mm, and initial vertical edge 25–95 mm. Its starting position varies. These ranges describe the public task distribution; sampled answers are never part of the public task specification.

The scene provides two fixed table rulers and an upright ruler. Each ruler is physically 200 mm long with one-millimetre visible graduations, longer five- and ten-millimetre ticks, numeric twenty-millimetre labels, and an SI units legend. Its generated image is identical across all episode seeds and contains no box information. The upright ruler's zero is at table height. The box has a free joint and can be moved, grasped, rotated, or inspected using the robot and cameras. No post-step callback writes poses.

Submit through the existing API:

```json
{"answer":{"dimensions_m":[0.09,0.05,0.07]}}
```

These are illustrative values, not a reference answer. Order is **length, width, height**, in metres: initial +x edge (longer horizontal edge), initial +y edge (shorter horizontal edge), initial vertical edge. Moving the box does not change the requested ordering. Each independently correct component earns one third of the score. Success requires all three errors at most 2 mm inclusive. Empty, malformed, non-finite, string, boolean, zero, negative, oversized, or reordered answers do not produce full credit. The gateway validates submitted vectors; the task repeats validation defensively before grading. A submission without an answer earns zero.

The private report contains sampled dimensions, errors, and per-component correctness. The agent receives a generic submission acknowledgement only. No task reward, object pose, dimension, random seed, or reference image is projected into public goal metadata. Geometric quantities necessarily exist in the trusted simulator model; never mount its scene, source memory, assets, or private audit directory in the evaluated agent's environment.

Terminal stability is disabled for this task because answers describe intrinsic manufactured dimensions, independent of the object's terminal pose. Standard invalid-physics and safety-stop rejection still apply. The ordinary finish settling phase still runs.

Validation: `python -m pytest tests/test_task_measurement.py -q` covers partial/full/absent answers, ordering, inclusive tolerance, malformed input, goal projection, private submission reports, constant unit-labelled scales, physical ruler orientation and metric origin, and passive rigid-body settling. This validates the task and security contract; it does not establish vision-agent success at the 2 mm tolerance. Fixed overview cameras may require moving the wrist camera or box closer for metrology.
