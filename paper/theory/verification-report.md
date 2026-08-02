# Theory-paper verification report

**Status:** Canonical manuscript and rendered preprint complete; all internal
mathematical, citation, artifact, and repository gates pass. External human
mathematical review has not yet occurred.

**Date:** 2026-08-01

## Purpose

This report records the checks performed before treating the four-part theorem
program as a paper rather than a collection of plausible claims. The checks
are designed to expose errors and hidden assumptions. They do not convert
finite numerical examples into proofs and do not establish novelty by
themselves.

The four claim families are:

1. reward transformations;
2. reward-sufficient source and behavioral-action representations;
3. nondegeneracy and reward-tail collapse; and
4. independent finite and countable composition.

The proof-level statements are in
[`../../docs/bit-equivalent-foundations.md`](../../docs/bit-equivalent-foundations.md).
The assumption audit is in [`theorem-ledger.md`](theorem-ledger.md), and the
counterexamples obtained by removing assumptions are in
[`assumption-stress-tests.md`](assumption-stress-tests.md).

## Gate summary

| Gate | Evidence | Result | Residual limitation |
| --- | --- | --- | --- |
| Executable theorem checks | Eleven deterministic tests in [`../../tests/test_theory_foundations.py`](../../tests/test_theory_foundations.py) | Passed | Representative finite instances and analytic truncations, not proofs |
| Canonical paper artifact | Typst source, complete proof appendix, bibliography, 25-page PDF, and three integrity checks | Passed | Typesetting and executable checks do not replace peer review |
| Assumption removal | Constructive counterexamples and boundary cases in [`assumption-stress-tests.md`](assumption-stress-tests.md) | No core theorem refuted inside its stated contract | Standard-Borel extensions remain conditional on measurable kernels and lifts |
| Claim/proof audit | Claim-by-claim ledger in [`theorem-ledger.md`](theorem-ledger.md) | No known disqualifying error | Internal adversarial review is not independent peer review |
| Literature/novelty audit | Primary-source comparison in [`systematic-literature-audit.md`](systematic-literature-audit.md) | Classical ingredients separated from intended application-level contributions | Absence of a located antecedent is not proof of novelty |
| External review | Reviewer packet in [`external-review-brief.md`](external-review-brief.md) | Packet ready | No external human has reviewed the mathematics yet |

## Executable checks

Run from the repository root:

```bash
uv run pytest -q tests/test_theory_foundations.py
```

Observed result:

```text
...........                                                              [100%]
11 passed
```

The tests use the repository's certified finite-frontier solver where a finite
problem is available and closed-form information calculations for the
unbounded limiting witnesses.

### Reward transformations

- A three-state, three-action problem and a positive-affine reward transform
  have matching certified frontier intervals after threshold relabeling.
- Finite prefixes of the cubic classification-reversal witness attain fixed
  transformed reward while their exact information cost is
  \(\rho\log(2)/n^2\), decreasing toward zero.

These checks would detect sign errors in the affine threshold map or the
activation/information calculation. They do not numerically establish the
universal maximality theorem or the sequential open-endedness lower bound.

### Representation

- Duplicating payoff-identical source states leaves the certified frontier
  unchanged.
- Duplicating behaviorally identical actions leaves the certified frontier
  unchanged when every quotient action has a raw lift.
- Adding a useful quotient action that has no raw lift makes equality fail,
  confirming that the lift assumption is substantive.

These are finite tests of the constructions used in the proofs. They do not
settle measurable-selection questions in general spaces.

### Nondegeneracy

- In uniform binary classification at target reward \(3/4\), the certified
  frontier contains the analytic value
  \(\log 2-h(3/4)\) and lies above the Pinsker certificate
  \(2(3/4-1/2)^2=1/8\).
- A strict-margin rare-burst family holds reward fixed while its exact mutual
  information decreases as \(\rho\log(q)/M\).
- Finite truncations of the bounded noncompact boundary example attain reward
  one while the zero-information baselines increase to one and information
  decreases to zero. This checks that boundary nonattainment does not require
  tail escape.

The first check exercises a bounded problem; the second deliberately violates
uniform integrability. Together they check the intended boundary between the
noncollapse theorem and the collapse example, but not the general
uniform-integrability argument.

### Composition

- A product of two nonidentical weighted binary components matches the sum of
  its analytically allocated component frontiers.
- A finite component with a null action, strict mismatch penalty, and explicit
  positive local slope \(\kappa=\log(8/5)\) is checked for
  \(n\in\{2,3,4,5,8,16,32\}\). The values
  \(nB_1(2/n)\) are nonincreasing and reach the predicted countable value
  \(2\kappa\) once the allocation enters the linear near-zero segment.
- A hostile shared-source construction verifies that dropping independence can
  hold reward and expected support fixed while information tends to zero,
  despite a positive Donsker--Varadhan component slope.

This tests both a genuinely nonidentical finite allocation and a nonzero local
price. It does not replace the finite-coordinate lower bound, Fubini step, or
limit argument in the countable proof.

## Repository validation

The completed paper change passed the complete local gate:

```text
uv run pytest -q tests/test_theory_foundations.py tests/test_theory_paper.py  14 passed
uv run pytest -q                             833 passed in 385.31s
uv run ruff check .                          passed
uv run ruff format --check .                 162 files formatted
git diff --check                             passed
uv build                                     sdist and wheel built
typst compile paper/theory/paper.typ ...      25-page PDF built
```

The full suite was run serially to keep memory use modest. The build produced
both the source archive and the pure-Python wheel. PDF inspection found no
blank pages, clipping, overflow, missing glyphs, unresolved citations, or
unembedded fonts; all 25 rendered pages were inspected visually.

## Adversarial proof audit

An internal referee pass independently recomputed the three highest-risk
arguments:

- the one-action-lottery proof that universal frontier conjugacy forces
  positive affinity;
- the Donsker--Varadhan exponential-moment bound, alternating-policy growth,
  and exact \(dn\log2\) information identity in the cubic reversal; and
- the finite-coordinate information lower bound, absolute-convergence step,
  and product-channel upper limit in the countable local-price theorem.

It also checked the Pinsker constant, finite tensorization, source reduction,
action quotient, uniform-integrability truncation, and compact-attainment
arguments. No material mathematical finding remains under the written
contracts. Three issues found during the pass were corrected:

1. the manuscript now separates bounded boundary nonattainment from
   positive-gap tail collapse;
2. an undefined illustrative “surface rule” construction was removed in favor
   of the fully specified strict-margin example; and
3. the abstract no longer implies that a static countable frontier theorem
   constructs an open-ended learner.

This was adversarial internal review, not external peer review.

## What the checks changed

The audit fixes the scope of the paper more sharply:

- nonlinear classification reversal is proved for the specified deterministic
  bandit and finite-per-round-mean policy convention, not for every sequential
  learning model;
- action-quotient equality requires a source-independent admissible lift;
- loss of uniform integrability is necessary for positive-gap collapse, not a
  general sufficient characterization;
- bounded reward still permits nonattainment at the boundary \(\rho=R_0\) on
  a noncompact action space, so boundary nonattainment and positive-gap
  tail-escape are separate cases;
- countable composition currently requires a pointwise null action and the
  finite-expected-support integrability contract; and
- a frontier theorem is not an agent-achievability theorem.

## Readiness judgment

The canonical preprint is internally circulation-ready: each main claim has a
complete proof, an explicit assumption contract, hostile boundary cases, and
an executable sanity check where computation is informative. Two independent
final internal passes found no remaining mathematical or layout defect, and a
separate citation audit found no unresolved or materially unsupported
reference. The paper is **not externally peer-reviewed**; before journal or
conference submission, an information theorist or mathematical decision
theorist should review the packet and any resulting revisions should be
closed. Literature positioning remains conservative: several core operations
are classical, while the paper-specific contribution is their foundations
synthesis and the cubic classification-reversal consequence for the
bit-equivalent definition.
