# Manuscript evidence map

## Evidence states

| State | Meaning |
|---|---|
| `THEORETICAL` | Supported by a stated derivation and regression contract; not an empirical result. |
| `IMPLEMENTED` | Present in reviewed code and regression tests; not an empirical effect. |
| `RELEASED` | Supported by authenticated artifacts in a public release. |
| `REGISTERED` | Publicly fixed before data, but no outcome is implied. |
| `OPERATIONAL` | Resource or integrity evidence that cannot support a scientific effect. |
| `PRE-DATA` | No registered scientific outcome exists. |
| `FUTURE` | Part of the retained research scope but not yet implemented or evaluated. |

## Claims currently available

Rows group closely related prose claims; states apply only at the strength
shown and do not promote implementation or registration into scientific
evidence.

| Manuscript claim group | State | Authoritative source |
|---|---|---|
| The bit-equivalent is the infimum behavioral mutual information needed to attain a reward target. | `THEORETICAL` | [`docs/theory.md`](../docs/theory.md) |
| Latent information is only an achievable upper bound unless the induced behavioral channel is identified. | `THEORETICAL` | [`docs/research-plan.md`](../docs/research-plan.md) |
| Finite symbolic actions canonicalize sorted non-abstaining index-label entries and reject duplicate indices as ambiguous. | `IMPLEMENTED` | [`core/behavior.py`](../src/infinite_rulebook/core/behavior.py), regression tests |
| The strict negative uninformed margin prevents the intended independent frontier from collapsing. | `THEORETICAL` | [`docs/theory.md`](../docs/theory.md) |
| For one coordinate, feasibility requires \(p\ge(r+c)/(u+c)\), and the baseline initial slope is \(\log 3\). | `THEORETICAL` | [`docs/theory.md`](../docs/theory.md) |
| Independent coordinates tensorize over finite expected-support kernels, giving \(B_\infty(\rho)=\rho\log3\) for finite baseline targets. | `THEORETICAL` | [`docs/theory.md`](../docs/theory.md) |
| The symbolic environments implement independent, redundant, mixed, aleatoric, trivia, and public-reward interventions. | `IMPLEMENTED` | [`docs/symbolic-controls.md`](../docs/symbolic-controls.md), [`environments`](../src/infinite_rulebook/environments), regression tests |
| The exact symbolic implementation includes analytic, tensorized, exhaustive finite, and certified finite frontier paths. | `IMPLEMENTED` | [`docs/finite-solver.md`](../docs/finite-solver.md), regression tests |
| The suite implements fixed, expansion, reward-, novelty-, total-information-, and relevant-information-directed agents. | `IMPLEMENTED` | [`agents`](../src/infinite_rulebook/agents), regression tests |
| Recorded metrics separate reward, support, novelty, persistent information, efficiency, and frontier regret. | `IMPLEMENTED` | [`docs/information-metrics.md`](../docs/information-metrics.md), regression tests |
| Runs use domain-separated streams, hash-chained training, side-effect-free checkpoints, immutable finalized trees, and serial/parallel scientific reproduction. | `IMPLEMENTED` | [`docs/experiments.md`](../docs/experiments.md), regression tests |
| Confirmation is fail-closed behind a seal that binds registered analysis, provenance, seeds, tolerances, and environment identity. | `IMPLEMENTED` | [`docs/symbolic-confirmatory-v1.md`](../docs/symbolic-confirmatory-v1.md), regression tests |
| The bounded behavioral-estimator foundation retains feasible finite-channel upper witnesses and certified lower bounds, with descriptive synthetic calibration diagnostics. | `IMPLEMENTED` | [`docs/learned-estimator-foundation.md`](../docs/learned-estimator-foundation.md), regression tests |
| The adaptive-curriculum foundation implements deterministic oracle, estimated-frontier, value-per-information, and candidate-discovery scheduling over caller-supplied evidence. | `IMPLEMENTED` | [`docs/adaptive-curriculum.md`](../docs/adaptive-curriculum.md), regression tests |
| Reporting-host qualification fails closed across static identity, synthetic capacity, descriptor-anchored probe, and assessment records. | `IMPLEMENTED` | [`docs/symbolic-v2-host-qualification.md`](../docs/symbolic-v2-host-qualification.md), regression tests |
| V1 registered six environments, six agents, horizon 12, disjoint calibration seeds, exact canaries, distribution-free adequacy, and fail-closed selection. | `REGISTERED` | [`docs/symbolic-confirmatory-v1.md`](../docs/symbolic-confirmatory-v1.md) |
| V1 Stage 0 passed with 24 of 24 serial/parallel cell matches. | `RELEASED` | [`docs/releases/v0.1.0.md`](../docs/releases/v0.1.0.md) |
| V1 calibration completed 20,736 registered cells on each execution side, reproduced exactly, passed all 20 canaries, and recorded no deviation. | `RELEASED` | [`docs/releases/v0.1.0.md`](../docs/releases/v0.1.0.md) |
| Five of six v1 effect-adequacy gates passed; the registered relevant-over-total TRIVIA reward gate failed with interval \([0,2/3]\) and 58 of 128 favorable signs. | `RELEASED` | [`results/symbolic-calibration-v1/power.json`](../results/symbolic-calibration-v1/power.json), [`results/symbolic-calibration-v1/registered-gates.csv`](../results/symbolic-calibration-v1/registered-gates.csv) |
| V1 correctly selected no confirmatory size and created no confirmatory seal or outcome. | `RELEASED` | [`docs/releases/v0.1.0.md`](../docs/releases/v0.1.0.md) |
| V2 registers an eight-environment, six-agent, 192-by-8 calibration design with six Holm-family primary comparisons. | `REGISTERED` | [`docs/symbolic-confirmatory-v2.md`](../docs/symbolic-confirmatory-v2.md) |
| V2's compound distractor claim requires both D6 learning-path and D24 terminal effects; D12 and the legacy D6 terminal result cannot select confirmation. | `REGISTERED` | [`docs/symbolic-confirmatory-v2.md`](../docs/symbolic-confirmatory-v2.md) |
| V2 registers 27 compact canaries and a distribution-free, fail-closed sample-size selection rule. | `REGISTERED` | [`docs/symbolic-confirmatory-v2.md`](../docs/symbolic-confirmatory-v2.md) |
| The post-registration v2 ingestion probe passed execution, validation, and authenticated loading. | `OPERATIONAL` | [`docs/symbolic-v2-operational-preflight.md`](../docs/symbolic-v2-operational-preflight.md) |
| The local 32-GiB-class workstation passed the synthetic in-memory E192 analysis-capacity benchmark but failed the v2 E768 host-capacity gate. | `OPERATIONAL` | [`docs/symbolic-v2-operational-preflight.md`](../docs/symbolic-v2-operational-preflight.md) |
| No registered v2 scientific outcomes have been generated. | `PRE-DATA` | [`docs/symbolic-v2-operational-preflight.md`](../docs/symbolic-v2-operational-preflight.md) |

## Claims not yet available

The manuscript must not presently claim:

- that any v2 primary or supplemental contrast passed or failed;
- that v2 calibration selected a confirmatory environment count;
- that a v2 confirmatory seal or outcome exists;
- asymptotic open-endedness from the bounded horizon-12 symbolic study;
- statistically calibrated learned-estimator coverage, learned converse bounds,
  or large-instance identification;
- a program or neural behavioral quotient beyond the finite symbolic action
  representation;
- adaptive-curriculum, target-discovery, dynamic-state, procedural, or neural
  results; or
- robustness across unexecuted feedback, reward, representation, or structural
  sweeps.

These entries move out of `PRE-DATA` or `FUTURE` only after their own declared
gates and public evidence are complete.
