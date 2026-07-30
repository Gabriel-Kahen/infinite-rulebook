# Manuscript workspace

This directory is the result-safe writing workspace for the Infinite Rulebook
paper. It deliberately separates claims that already have public evidence from
registered analyses whose outcomes do not yet exist.

The current manuscript may report:

- the exact symbolic theory and implementation contracts;
- the released v1 calibration-stopped outcome;
- the publicly registered v2 design; and
- the operational finding that the local workstation is not authorized to run
  v2.

It may not report or imply v2 pilot, calibration, confirmatory, or outcome
results. Those sections remain explicitly marked `PRE-DATA` until authenticated
registered evidence is released.

## Files

- [`manuscript.md`](manuscript.md) is the paper scaffold.
- [`evidence-map.md`](evidence-map.md) maps the manuscript's principal claim
  groups to public evidence and lists claims that remain unavailable.
- [`evidence-status.json`](evidence-status.json) is the machine-readable study
  boundary enforced by the test suite.

Before editing a result statement, update the evidence map and verify the
referenced immutable artifact or release. Exploratory analyses must be labeled
post hoc and cannot be moved into a registered-results section.
