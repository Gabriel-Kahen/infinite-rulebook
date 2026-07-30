# Adaptive curriculum infrastructure

This module is engineering infrastructure for the future adaptive-curriculum
workstream in [`research-plan.md`](research-plan.md), especially Sections
13.11--13.13 and 18. It is not part of either registered symbolic study, is not
wired into their configs or orchestration, and defines no study seed namespace.

The implementation separates the source of evidence from the scheduling rule.
A caller supplies immutable frontier and candidate bounds; a deterministic
policy returns one `hold`, `probe`, or `expand` decision; and a fail-closed
controller applies the decision to immutable structural state. This separation
prevents an oracle quantity from being silently substituted for an online
posterior estimate.

## Policies and privileges

| Policy | Required evidence | Expansion rule | Declared privilege |
|---|---|---|---|
| `OracleFrontierPolicy` | Exact frontier and exact candidate increments | Current-target frontier gap is small and the best exact marginal reward per nat is positive | Relevance mask, coordinate factorization, target hierarchy, true posterior family, exact frontier, and reward parameters |
| `EstimatedFrontierPolicy` | Posterior bounds | Residual value-of-perfect-information upper bound is small, bound widths are tolerable, and conservative marginal reward per nat is positive | Approximate frontier, target hierarchy, and reward parameters |
| `MarginalValuePerBitPolicy` | Posterior candidate bounds | Largest conservative marginal reward lower bound divided by information upper bound in nats | Approximate frontier, target hierarchy, and reward parameters |
| `CandidateGroupDiscoveryPolicy` | Discovery estimates over externally generated, unprivileged candidate groups | Probe under-observed groups first, then expand only a sufficiently precise group with positive conservative value per nat | Approximate frontier and reward parameters; no relevance mask, factorization, or declared hierarchy |

Oracle frontier-following remains an ensemble-level controller. Its estimates
must be computed from the declared matched ensemble and the chosen decision
must be shared across that ensemble. Feeding one trajectory's realized outcome
into the oracle interface would mislabel an online adaptive policy as an
oracle.

`CandidateGroupDiscoveryPolicy` implements only the scheduling/selection half
of a future target-discovery system: auditable probe-then-expand behavior over
caller-supplied candidate groups. An external, non-oracle candidate generator
must produce those groups and its provenance must be audited. This module does
not construct groups from posterior dependency graphs, causal relevance,
sparse discovery, skills, or procedural composition. The policy rejects
catalog hierarchy edges so a hidden true hierarchy cannot enter through that
interface. Full target discovery remains unimplemented.

The controller validates every policy's `CapabilityManifest` and rejects
evidence whose oracle/posterior/discovery class is not covered by that
manifest. A hierarchy-bearing catalog is rejected unless the policy declares
target-hierarchy access; candidate-group discovery rejects such a catalog
whether or not the current update happens to assess a root.

The research plan calls the third behavior “marginal value per bit.” The class
keeps that conceptual name, but all stored information, thresholds, scores, and
public fields use the repository's locked scientific unit, nats. Converting
per-nat scores and their threshold by the fixed \(\ln 2\) factor produces the
equivalent bit-denominated policy.

## State and update contract

`CurriculumCatalog` fixes stable named target increments over `TargetKey`
members. Optional parent edges express the hierarchy available to the three
hierarchy-aware policies. Catalogs reject missing parents, cycles, and overlap
patterns that could let an earlier eligible expansion consume every new member
of another target. Partial overlap remains supported when every target retains
at least one structurally protected new member; descendants may reuse ancestor
members because they cannot preempt their ancestors.

Each `CurriculumUpdate` contains:

- a monotone round index;
- at most one `CurriculumAssessment` per named target;
- lower and upper marginal-reward bounds;
- lower and upper additional-information bounds, explicitly in nats;
- an evidence count and optional discovery-probe information cost; and
- for frontier-following policies, a `FrontierEstimate` containing frontier
  gap, residual value-of-perfect-information, and uncertainty width.

The conservative value-per-nat score is

\[
\frac{\text{marginal reward lower bound}}
     {\text{additional information upper bound in nats}}.
\]

The estimated frontier policy additionally requires the current residual-value
upper bound and bound width to pass their declared thresholds. Realized
posterior KL may be used by a future estimator to create an update, but this
API does not call it population mutual information.
`MarginalValuePerBitPolicy.bound_width_tolerance` is optional: `None` applies
no width gate, while a finite nonnegative value requires candidate reward
bounds to be at most that wide.

`CurriculumState` records every decision, active target and member, cumulative
planned information, discovery-probe counts, and monotone evidence counts.
When state is restored, the controller sequentially reconstructs unique
expansions, parent-before-child order, new-member support deltas, transient
probe support, probe counts, and cumulative limits from decision history.
Candidate ordering does not affect a decision: equal scores use the stable
target name. Cumulative planned information is the exact `math.fsum` of the
decision history; a caller-supplied lower total is rejected rather than
accepted by a scale-dependent relative tolerance. Hard-limit checks compute a
full-history residual against the limit, so positive costs smaller than the
stored total's unit in the last place cannot disappear.

`CurriculumState` does not retain the original `CurriculumUpdate` records.
Consequently, restoration cannot authenticate historical evidence counts,
estimate bounds, decision scores, or a decision's submitted information cost
against the update that produced it. Restored evidence counts are checked only
for nonnegative known targets, then enforced as monotone lower bounds for
future updates. An auditable experiment must archive the complete ordered
update log alongside state and replay it through the pinned policy.

## Hard safeguards

`CurriculumController` enforces safeguards after a policy decides, so a custom
policy cannot bypass them:

- one decision per update;
- parents must already be active;
- an expansion must add new support;
- active or one-update probe support cannot exceed `maximum_support`;
- conservative information cost cannot exceed
  `maximum_planned_information_nats`;
- action costs and recorded scores must equal the submitted assessment;
- action and audit-reason combinations must be consistent;
- evidence counts cannot decrease; and
- discovery stops if its bounded probes fail to produce the required evidence.

These are curriculum-level limits. The existing `FactorizedQueryAgent`
continues to enforce the per-round query budget and observation/action binding.
`CurriculumAcquisitionPolicy` normally exposes active support. During a
one-update discovery probe it instead exposes only the selected group's
still-inactive members, then ranks those targets by expected entropy reduction
without consulting relevance metadata. Its action validator rejects an empty
probe action or one outside that group, so the controller cannot charge a
candidate-group probe while the attached agent queries ordinary active
support. The agent revalidates the same action before applying its observation,
and every action is bound to the latest curriculum round; changing the
curriculum between selection and observation fails closed.

Probe counts and costs record scheduled probe decisions. If candidate exposure
is incomplete and action validation fails, orchestration must correct the
candidate set and retry that same decision before advancing; it cannot treat
the failed selection as an observed probe.

## Synthetic use

The following illustrates posterior-bound scheduling without a study config:

```python
from infinite_rulebook.agents import (
    CurriculumAssessment,
    CurriculumCatalog,
    CurriculumController,
    CurriculumEvidence,
    CurriculumLimits,
    CurriculumTarget,
    CurriculumUpdate,
    MarginalValuePerBitPolicy,
    TargetKey,
)

catalog = CurriculumCatalog(
    (
        CurriculumTarget("initial", (TargetKey("synthetic", 1),)),
        CurriculumTarget(
            "next",
            (TargetKey("synthetic", 2),),
            parent="initial",
        ),
    )
)
controller = CurriculumController(
    catalog,
    MarginalValuePerBitPolicy(minimum_value_per_nat=0.05),
    CurriculumLimits(
        maximum_support=8,
        maximum_planned_information_nats=16.0,
    ),
)
decision = controller.update(
    CurriculumUpdate(
        round_index=0,
        assessments=(
            CurriculumAssessment(
                target="initial",
                evidence=CurriculumEvidence.POSTERIOR,
                marginal_reward_lower=0.3,
                marginal_reward_upper=0.4,
                information_nats_lower=1.0,
                information_nats_upper=1.2,
            ),
        ),
    )
)
```

Tests use only synthetic targets and environments. They cover privilege
separation, deterministic ties, uncertainty gates, hierarchy, support and
information limits, structural state restoration, discovery failure, and
acquisition-budget integration.

## Current limitations

- Frontier, value-of-perfect-information, reward, information, and uncertainty
  estimators are inputs, not implementations in this module.
- Planned information is a conservative scheduling bound, not measured
  population mutual information or a scientific result.
- Safe candidate overlap is allowed, but an assessment must price only the
  still-inactive increment; the controller counts unique support and rejects
  catalogs whose overlap can strand a target.
- A discovery probe is queryable for one curriculum update. Experiments must
  align curriculum updates and acquisition epochs explicitly. Exposing a
  multi-member group does not increase per-round query capacity:
  `FactorizedQueryAgent` still enforces its declared `query_budget`.
- The controller records the ingredients needed for a future target-expansion
  phase diagram, but it does not run the multi-axis sweep, classify regimes, or
  make noninferiority claims.
- No adaptive policy is currently a registered agent. Any later experiment
  needs its own public protocol, estimators, seeds, phase-diagram grid,
  noninferiority margin, and analysis plan before outcome generation.

In particular, this infrastructure must not be attached to the registered v2
configs, seed banks, metrics, selectors, canaries, power rule, artifacts, or
results.
