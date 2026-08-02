# Paper workspaces

## Active theory paper

The active paper is a foundations study of the bit-equivalent. It does not
depend on an empirical result.

- [`theory/manuscript.md`](theory/manuscript.md) is the new manuscript draft.
- [`theory/research-program.md`](theory/research-program.md) fixes the four-part
  theorem program and its decision gates.
- [`theory/literature-notes.md`](theory/literature-notes.md) tracks classical
  antecedents and provisional novelty boundaries.

The technical theorem and proof draft lives in
[`docs/bit-equivalent-foundations.md`](../docs/bit-equivalent-foundations.md).

## Archived experimental scaffold

The earlier Infinite Rulebook manuscript remains in the repository as a
result-safe historical scaffold. It is not the active theory paper. It
deliberately separates claims that already have public evidence from registered
analyses whose outcomes do not yet exist.

The current manuscript may report:

- the exact symbolic theory and implementation contracts;
- the released v1 calibration-stopped outcome;
- the publicly registered v2 design; and
- the operational finding that the local workstation is not authorized to run
  v2.

It may not report or imply v2 pilot, calibration, confirmatory, or outcome
results. Those sections remain explicitly marked `PRE-DATA` until authenticated
registered evidence is released.

### Files

- [`manuscript.md`](manuscript.md) is the paper scaffold.
- [`evidence-map.md`](evidence-map.md) maps the manuscript's principal claim
  groups to public evidence and lists claims that remain unavailable.
- [`evidence-status.json`](evidence-status.json) is the machine-readable study
  boundary enforced by the test suite.

Before editing a result statement, update the evidence map and verify the
referenced immutable artifact or release. Exploratory analyses must be labeled
post hoc and cannot be moved into a registered-results section.
