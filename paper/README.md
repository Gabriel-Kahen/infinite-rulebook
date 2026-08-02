# Paper workspaces

## Active theory paper

The active paper is a foundations study of the bit-equivalent. It does not
depend on an empirical result.

- [`theory/paper.typ`](theory/paper.typ) is the canonical, typeset manuscript.
- [`../output/pdf/when-does-reward-require-information.pdf`](../output/pdf/when-does-reward-require-information.pdf)
  is the compiled preprint.
- [`theory/proofs.typ`](theory/proofs.typ) contains the complete proof appendix,
  and [`theory/references.bib`](theory/references.bib) is the bibliography.
- [`theory/README.md`](theory/README.md) gives the reproducible build command.
- [`theory/manuscript.md`](theory/manuscript.md) is the earlier Markdown drafting
  surface retained for traceability.
- [`theory/research-program.md`](theory/research-program.md) fixes the four-part
  theorem program and its decision gates.
- [`theory/literature-notes.md`](theory/literature-notes.md) tracks classical
  antecedents and provisional novelty boundaries.
- [`theory/verification-report.md`](theory/verification-report.md) records the
  pre-writing numerical, assumption, proof, and literature gates.
- [`theory/theorem-ledger.md`](theory/theorem-ledger.md) and
  [`theory/assumption-stress-tests.md`](theory/assumption-stress-tests.md)
  make every theorem's assumptions and failure modes explicit.
- [`theory/systematic-literature-audit.md`](theory/systematic-literature-audit.md)
  gives the primary-source novelty audit and safe positioning language.
- [`theory/external-review-brief.md`](theory/external-review-brief.md) is a
  reviewer-ready packet; it does not imply that external review has occurred.

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
