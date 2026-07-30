# Open-Ended or Just Novel?
## An Auditable Test of Reward-Relevant Information

**Manuscript status:** Methods-oriented scaffold with a released v1
calibration-stopped result and a pre-data v2 registration.

> **Evidence boundary.** No registered v2 pilot, calibration, confirmation, or
> outcome analysis has been generated. Every v2 results subsection below is
> intentionally marked `PRE-DATA`. The evidence states of the manuscript's
> principal theory, implementation, empirical, registered, and operational
> claim groups are tracked in [`evidence-map.md`](evidence-map.md).

## Abstract

Open-ended learning is often evaluated through reward growth, novelty, task
production, or total information acquisition. Those quantities can move
independently of the environment information a successful decision actually
requires. We develop Infinite Rulebook, a stationary symbolic benchmark and
auditable experimental lifecycle for studying the bit-equivalent: the infimum
mutual information between an environment and a behavioral action needed to
attain a reward threshold. The benchmark has an analytic positive control,
certified finite frontier solvers, explicit redundancy and distractor
interventions, canonical behavioral actions, and immutable scientific
artifacts. A first publicly registered construct-validation study completed
calibration but stopped as required when one of six effect-adequacy gates
failed; no confirmatory outcome was generated. We use that failure to motivate
a second, publicly registered pre-data protocol with learning-path and
high-distractor estimands. The present manuscript establishes the measurement
contract, reports the released stopped result, and fixes the boundary for the
next study. It does not yet claim v2 effects, asymptotic open-endedness,
learned-estimator validity, or neural transfer.

## 1. Introduction

An agent can encounter endless observations without accumulating useful
capability. It can also receive increasing reward because a public payoff
scale changes, memorize redundant descriptions of one latent fact, or
continually replace forgotten knowledge. Reward, novelty, task count, and
information gain therefore do not by themselves identify cumulative
reward-relevant growth.

The bit-equivalent addresses a different question. For environment
\(\Theta\), behavioral action \(A\), reward \(r_\Theta(A)\), and target
\(\rho\), define

\[
B_\rho
=
\inf_{P(A\mid\Theta):\,\mathbb E[r_\Theta(A)]\ge\rho}
I(\Theta;A).
\]

All logarithms and reported information quantities use natural units (nats);
division by \(\ln 2\) converts them to bits. The historical name
“bit-equivalent” does not change that implementation unit.

Admissible randomized kernels are supported on finite symbolic actions and
satisfy \(\mathbb E[\lVert A\rVert_0]<\infty\). This finite expected-support
condition keeps the additive expected-reward objective well defined.

This is a property of the decision problem and reward semantics, not of an
agent's parameter count or internal representation. Empirical use is difficult
because the infimum ranges over behavioral channels, irrelevant information
must be excluded, and approximate solvers must be calibrated against known
frontiers.

Infinite Rulebook turns those difficulties into interventions and failure
tests. Independent useful rules, redundant rules, persistent trivia,
aleatoric novelty, and public reward are changed separately. Exact symbolic
frontiers anchor the measurement. Bounded behavioral-estimator and
adaptive-curriculum infrastructure now provide non-study engineering
foundations. Statistical large-instance estimation, evaluated target
selection, dynamic turnover, and procedural neural transfer remain outward
extensions rather than substitutes for the anchor.

The contributions currently supported are:

1. a stationary, analytically tractable benchmark with canonical finite-support
   behavioral actions;
2. exact and certified finite reward-information frontier implementations;
3. an auditable study lifecycle with deterministic replay, immutable
   artifacts, registered canaries, and fail-closed confirmation;
4. a public v1 calibration-stopped result that preserves an unfavorable gate
   without threshold revision; and
5. a stronger, publicly registered pre-data v2 protocol and an explicit host
   qualification boundary.

## 2. Measurement target and threats to validity

### 2.1 Behavioral information

The primary quantity is \(I(\Theta;A)\). A learned latent \(Z\) can construct a
channel through \(\Theta\to Z\to A\), but \(I(\Theta;Z)\) is generally only an
achievable upper bound on behavioral information. Any approximate latent
estimator must therefore expose the induced action channel and distinguish
feasible upper bounds from converse lower bounds.

### 2.2 Canonical actions

Current symbolic deployments are finite index-label mappings. Canonicalization
sorts their non-abstaining entries, while duplicate indices are rejected as
ambiguous. Serialization order therefore does not create distinct symbolic
actions. Program or neural representations would require a separately
implemented behavioral quotient before comments, unused state, or parameter
symmetries could be excluded; unresolved equivalence cannot be silently counted
as capability.

### 2.3 Acquisition is not evaluation

Training observations update the agent through a bounded interface.
Checkpoint evaluation freezes a deployment and supplies no new observation.
Feedback noise may alter acquisition speed but not the reward-information
frontier when prior, action, and reward are fixed.

### 2.4 Infinite environments

The latent Rulebook exists from time zero and is generated lazily only as an
implementation device. Persistent information is defined through finite
closures or posterior-to-prior KL divergence, never by subtracting two infinite
entropies.

## 3. Infinite Rulebook

Each primitive rule has a \(q\)-ary label. A deployment may abstain or predict a
label at finitely many indices. Correct predictions receive \(u>0\), incorrect
predictions receive \(-c\), and abstention receives zero. Primary studies
require

\[
c>\frac{u}{q-1},
\]

so uninformed deployment has strictly negative expected reward. The baseline
uses \(q=4\) and \(u=c=1\).

The environment family includes:

- `IND`, with independent reward-relevant coordinates;
- `RED-C`, with bounded reward-sufficient latent rank;
- `MIX`, combining independent and redundant components;
- `ALEA`, with fresh reward-irrelevant observation noise;
- `TRIVIA`, with persistent reward-irrelevant latent coordinates; and
- `PUBLIC-C`, with a bounded public reward component.

The broader specification retains unrestricted redundancy, opportunity,
ephemeral-state, alternate feedback, nonlinear reward, action-representation,
adaptive-curriculum, and procedural extensions.

## 4. Exact theory and numerical anchor

For a symmetric one-coordinate deployment with conditional correctness \(p\),
define

\[
J_q(p)
=
p\log(qp)
+
(1-p)\log\left(\frac{q(1-p)}{q-1}\right)
\]

and \(g(p)=(u+c)p-c\). For \(0<r\le u\), the exact one-coordinate frontier
minimizes \(rJ_q(p)/g(p)\) subject to
\(p\ge(r+c)/(u+c)\); the constraint ensures the deployment probability
\(r/g(p)\) is at most one. At \(q=4,u=c=1\), its initial slope is \(\log 3\).
Independent coordinates tensorize, and the countably infinite finite-support
benchmark has \(B_\infty(\rho)=\rho\log3\) for every finite nonnegative
\(\rho\) over the finite expected-support kernel class.

The implementation compares analytic, tensorized, exhaustive finite, and
certified Blahut--Arimoto paths. Frontier artifacts retain feasible witnesses,
dual information, solver diagnostics, and semantic hashes. The theory and
numerical contracts are maintained in
[`docs/theory.md`](../docs/theory.md) and
[`docs/finite-solver.md`](../docs/finite-solver.md).

## 5. Agents, measurements, and artifacts

The exact symbolic suite includes fixed-target, scheduled-expansion,
reward-directed, novelty-directed, total-information-directed, and
relevant-information-directed agents. Recorded metrics distinguish reward,
support, novelty, persistent total and relevant information, useful-information
efficiency, and frontier regret.

Each run cell is a deterministic crossing of environment, feedback, reward,
agent, environment seed, and algorithm seed. Semantic random streams are
domain-separated. Training events are hash chained; checkpoints are
side-effect free; frontiers are cached separately; and finalized trees are
immutable. Serial and parallel study sides must reproduce the same scientific
content before analysis.

Confirmation is unavailable until calibration evidence passes every registered
gate and creates a self-verifying seal binding the design, analysis,
provenance, seed banks, tolerances, and execution environment.

## 6. Version 1: released calibration-stopped result

### 6.1 Registered lifecycle

V1 evaluated six environments and six agents at horizon 12. Its public
Stage-0 prerequisite tested the shared execution paths before calibration.
Calibration used disjoint registered seed banks, exact canaries, a
distribution-free adequacy split, and a fail-closed power-selection rule.

### 6.2 Integrity results

Stage 0 passed all 24 serial/parallel cell comparisons. Calibration completed
20,736 registered run cells on each execution side. Serial and parallel
scientific hashes reproduced exactly, all 20 canaries passed, both raw roots
authenticated, and the deviation log was empty.

### 6.3 Scientific disposition

Five of six registered effect-adequacy records passed. The
`relevant-over-total-trivia-hidden-reward` record did not: its simultaneous
median interval was \([0,2/3]\), which did not lie at or above the registered
minimum effect \(0.25\). Its independent favorable-sign count was \(58/128\).
These were calibration design-adequacy gates, not confirmatory effect
findings; the registered analysis marked calibration interpretation
ineligible.

The protocol therefore selected no confirmatory environment count, created no
seal, and produced no confirmatory outcomes. This is the completed registered
outcome, not missing data. The released evidence and reconstruction procedure
are recorded in [`docs/releases/v0.1.0.md`](../docs/releases/v0.1.0.md).

## 7. Version 2: registered pre-data study

### 7.1 Motivation

The failed v1 terminal estimand operated near a small reward ceiling and
produced many strict-sign ties. V2 retains every v1 condition and agent while
adding a learning-path reward estimand and a higher persistent-distractor load.
The v1 outcome motivates this design but is not reused as v2 evidence.

### 7.2 Registered design

V2 crosses eight environments, six agents, 192 calibration environment
replicas, and eight fixed algorithm replicas. The six primary comparisons form
one Holm family. The compound distractor claim requires both a D6 learning-path
effect and a D24 terminal effect; neither can rescue the other. A D12 result
and the legacy D6 terminal replication are registered outside the primary
family and cannot select confirmation.

Twenty-seven compact canaries authenticate exact frontier, path, reward
decomposition, persistent-information, and aggregate-metric identities. A
distribution-free split chooses the smallest candidate satisfying all
registered power targets, or stops without a seal.

The complete registered specification is
[`docs/symbolic-confirmatory-v2.md`](../docs/symbolic-confirmatory-v2.md).

### 7.3 Operational boundary

A post-registration ingestion probe passed execution, artifact validation, and
authenticated loading. The local 32-GiB-class workstation passed the synthetic
in-memory E192 analysis-capacity benchmark but failed E768 because the
equal-reserve and no-swap requirements were not met. These capacity benchmarks
do not exercise raw-artifact loading or full report generation. The workstation
is not authorized to execute registered v2 calibration or confirmation. This
is an operational result, not study evidence.

## 8. Version 2 results

**PRE-DATA — intentionally blank.**

No v2 effect estimate, hypothesis decision, selected sample size, seal, or
confirmatory result exists. This section may be populated only from the
authenticated registered report and must retain unfavorable and stopped
outcomes.

## 9. Implemented foundations and planned extensions

The exact symbolic frontier is the calibration anchor for approximate
behavioral and latent estimators. The current bounded behavioral estimator
retains feasible upper witnesses and certified lower bounds for small,
fully-enumerated finite problems, together with descriptive synthetic
calibration diagnostics. It does not provide statistical large-instance
coverage, a learned converse, or evidence for v2.

The current adaptive-curriculum infrastructure implements deterministic
oracle, estimated-frontier, marginal-value-per-information, and
candidate-discovery scheduling over caller-supplied evidence. It does not
generate candidates, validate the supplied estimators, or report curriculum
effects. Later studies will evaluate those policies under separately
registered gates. Reward and action semantics, ephemeral turnover, and
procedural neural composition remain independent workstreams. None is evidence
for v2.

## 10. Limitations

The released evidence is bounded, symbolic, and horizon 12. It does not
establish asymptotic open-endedness, learned-estimator validity, autonomous
target discovery, neural transfer, or robustness across all feedback and reward
semantics. Expected-performance frontiers can also reward rare-burst policies;
risk-constrained secondary frontiers remain necessary for some capability
interpretations.

V1's failed gate narrows the evidence rather than invalidating the artifact and
measurement infrastructure. V2 may also stop before confirmation. Later
procedural results, if successful, would provide transfer evidence rather than
proof that the symbolic ordering holds universally.

## 11. Reproducibility and availability

Source code, locked dependencies, configs, analysis registrations, canaries,
machine-readable reports, open tables, deterministic figures, and artifact
manifests are maintained in the public repository. The v1 release contains
manifest-ordered raw archives and reconstruction instructions. V2 publication
will use the same fail-closed boundary only after its registered lifecycle
completes.

## 12. Declarations

### Data and code availability

Code and released evidence are available from the public Infinite Rulebook
repository. V2 outcome data are unavailable because they have not been
generated.

### Author contributions

To be completed before submission.

### Competing interests

To be completed before submission.

### Acknowledgments

To be completed before submission.

## References

The submission bibliography will include the source definition of open-ended
learning, rate-distortion approaches to deciding what to learn, and standard
information-theory references listed in
[`docs/research-plan.md`](../docs/research-plan.md).
