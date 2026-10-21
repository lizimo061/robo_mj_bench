# Writing with the standard YAM

`writing_yam_v1` supplies an unchanged YAM and a freely graspable blue marker,
white paper, and public reference PNG. The target is uppercase **HELLO WORLD**.
The paper's position varies slightly by seed, and its perimeter is marked black.
Its image top corresponds to world positive Y. The target position, scale, and
orientation are part of the goal, available through the reference image.

This task is explicitly labeled `contact_writing_proxy`: it evaluates tracing a
frozen, embedded 5×7 bitmap alphabet, not open-ended handwriting recognition or OCR.
It is a first physics-based writing task and must not be compared as equivalent
to freeform writing on a physical robot. General handwriting requires a separate
validated recognizer and evaluation protocol.

Only solved MuJoCo contact between the marker tip and the top paper surface adds
ink. The marker shaft must point upward (vertical component over 0.45); normal
force must be 0.03–8 N. Hovering, shaft contact, text answers, and broad proximity
checks cannot draw. Contact points are rasterized without interpolating across
non-contact gaps. Table marks outside the paper invalidate bounds. Ink is
persistent render-only geometry added after camera scene updates; it does not
move bodies, change contacts, or reveal reward. The private ink raster and target
mask never enter agent responses. Only the unchanged goal PNG is public.

Success requires at least 90% target coverage (one-pixel tolerance), at least 85%
ink precision within a two-pixel target tolerance, and no out-of-bounds contact.
Blank paper scores zero; filling the paper or scribbling broadly fails precision.
Failed attempts receive bounded F1 partial credit. Task reports expose metrics
only to the trusted terminal evaluator. The marker and paper are rigid bodies;
there is no deformable paper or pressure-dependent ink model. Ink raster cells
are 1 mm; camera ink display is bounded by available renderer geometry capacity.

Tests use privileged fixtures only to validate scoring and contact generation.
These fixture manipulations are not available through the agent API, and they
are not evidence that an LLM can grasp and write successfully end-to-end.
