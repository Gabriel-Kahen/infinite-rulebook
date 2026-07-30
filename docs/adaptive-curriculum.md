# Adaptive curriculum infrastructure

This module is engineering infrastructure for the future adaptive-curriculum
workstream in [`research-plan.md`](research-plan.md), especially Sections
13.11--13.13 and 18. It is not part of either registered symbolic study, is not
wired into their configs or orchestration, and defines no study seed namespace.

The implementation separates the source of evidence from the scheduling rule.
A caller supplies immutable frontier and candidate bounds; a deterministic
policy returns one `hold`, `probe`, or `expand` decision; and a fail-closed
controller applies the decision to immutable structural state. This separation
makes the declared evidence class and estimand explicit and auditable; source
artifact verification remains an external requirement.

## Policies and privileges

| Policy | Required evidence | Expansion rule | Declared privilege |
|---|---|---|---|
| `OracleFrontierPolicy` | Exact frontier and exact candidate increments | Current-support frontier gap is small and the best exact marginal reward per nat is positive | Relevance mask, coordinate factorization, target hierarchy, true posterior family, exact frontier, and reward parameters |
| `EstimatedFrontierPolicy` | Posterior bounds | Residual value-of-perfect-information upper bound is small, bound widths are tolerable, and conservative marginal reward per nat is positive | Approximate frontier, target hierarchy, and reward parameters |
| `MarginalValuePerBitPolicy` | Posterior candidate bounds | Largest conservative marginal reward lower bound divided by information upper bound in nats | Approximate frontier, target hierarchy, and reward parameters |
| `CandidateGroupDiscoveryPolicy` | Discovery estimates over externally generated, unprivileged candidate groups | Probe every under-evidenced or imprecise group up to its cap, then expand only when every known group is resolved | Approximate frontier and reward parameters; no relevance mask, factorization, or declared hierarchy |

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

Those are scheduler privileges. When a controller is attached to
`FactorizedQueryAgent`, `CurriculumAcquisitionPolicy` reports the union of the
scheduler manifest and the factorized execution substrate's relevance-mask,
coordinate-factorization, true-posterior-family, and reward privileges. Thus
candidate-group generation and scheduling remain unprivileged with respect to
the hidden target structure, while the effective agent artifact honestly
records the oracle assistance used by its shared deployment machinery.

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
- one `CurriculumEvidenceBasis` that names the exact frozen
  `CurriculumEstimand`, the active-member tuple on which all estimates were
  conditioned, and the scientific-payload hash of the source evidence
  artifact;
- at most one `CurriculumAssessment` per named target;
- lower and upper marginal-reward bounds;
- lower and upper additional-information bounds, explicitly in nats;
- an evidence count and optional discovery-probe information cost; and
- for frontier-following policies, a `FrontierEstimate` containing frontier
  gap, residual value-of-perfect-information, and uncertainty width for the
  exact active support named by the evidence basis.

`CurriculumEstimand` pins the reward-spec semantic hash, reward transform,
aggregation and horizon, information random variable and conditioning,
simultaneous-confidence method, level and family, and data split. A controller
accepts only its configured estimand and its exact current active-member basis.
This makes candidate bounds comparable by declared construction and prevents a
stale frontier from unlocking a different support state. The source artifact
must contain the actual estimator inputs and outputs; a matching descriptor is
an auditable identity, not proof that those computations or coverage claims
are correct.

The estimand also declares an executable deployment-reward contract.
`CurriculumAcquisitionPolicy` accepts only
`FACTORIZED_ADDITIVE_V1`, then cross-checks the attached
`FactorizedQueryAgent`'s actual `RewardSpec` semantic hash before scoring. This
contract means the factorized additive deployment rule is authoritative;
support costs, clipping, logistic whole-action rewards, or other transforms
that change its optimum require an external adapter rather than a descriptive
string claiming compatibility.

Whenever a policy reaches candidate selection, its update must assess every
structurally eligible target. Discovery updates are stricter: they must assess
every inactive catalog group. Omitting an inconvenient candidate therefore
cannot change which target is called “best.” Candidate-group discovery keeps
probing a group while either its evidence count is below the declared minimum
or its reward bound is wider than tolerance, subject to the per-group probe
cap and global limits.

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

`CurriculumController` has read-only catalog, policy, limits, estimand, and
state references; only a validated `update` can replace its immutable state.
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
Consequently, restoration cannot authenticate historical evidence bases,
source scientific payloads, evidence counts, estimate bounds, decision scores,
or a decision's submitted information cost against the update that produced it.
Restored evidence counts are checked only for nonnegative known targets, then
enforced as monotone lower bounds for future updates. Structural restoration
does reject a candidate-discovery probe history above the configured
per-target cap. An auditable experiment must archive the complete ordered
update log alongside state and replay it through the pinned policy and
estimand.

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
- probe actions require discovery-class evidence;
- action and audit-reason combinations must be consistent;
- each evidence basis must match the pinned estimand and active support;
- selection updates cannot omit eligible candidates;
- evidence counts cannot decrease; and
- discovery emits an auditable hold if its bounded probes fail to produce the
  required evidence.

`INSUFFICIENT_DISCOVERY_EVIDENCE` is an ordinary audited `hold`, not an
irreversible controller state. It means no further probe is available under
the current catalog and policy cap. Orchestration must prespecify whether that
reason terminates a run; a later round may resume expansion only if new
external evidence satisfies the same pinned estimand and current-support
contract.

These are curriculum-level limits. The existing `FactorizedQueryAgent`
continues to enforce the per-round query budget and observation/action binding.
`CurriculumAcquisitionPolicy` normally exposes active support. During a
one-update discovery probe it instead exposes only the selected group's
still-inactive members, then ranks those targets by expected entropy reduction
without consulting relevance metadata. A zero-entropy member receives only
the smallest positive floating-point score during a scheduled probe, ensuring
that an allowed repeat probe remains executable without changing the ranking
of informative members. The context validator requires every currently
queryable member to be exposed as persistent, so partial candidate exposure
cannot steer a tie. The action validator rejects an empty probe action or one
outside that group, so the controller cannot charge a candidate-group probe
while the attached agent queries ordinary active support. Probe observations
remain excluded from deployment until an expansion activates their member.
The agent revalidates the same action before applying its observation, and
every action is bound to the latest curriculum round; changing the curriculum
between selection and observation fails closed.

Probe counts and costs record scheduled probe decisions. If candidate exposure
is incomplete and context validation fails, orchestration must correct the
candidate set and retry that same decision before advancing; it cannot treat
the failed selection as an observed probe.

## Synthetic use

The following illustrates posterior-bound scheduling without a study config:

```python
from infinite_rulebook.agents import (
    CurriculumAssessment,
    CurriculumCatalog,
    CurriculumController,
    CurriculumDeploymentRewardContract,
    CurriculumEstimand,
    CurriculumEvidence,
    CurriculumEvidenceBasis,
    CurriculumLimits,
    CurriculumTarget,
    CurriculumUpdate,
    MarginalValuePerBitPolicy,
    TargetKey,
)
from infinite_rulebook.artifacts import semantic_hash
from infinite_rulebook.core.reward import RewardSpec
from infinite_rulebook.orchestration.hashing import scientific_hash

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
estimand = CurriculumEstimand(
    name="synthetic normalized reward per acquisition epoch",
    reward_spec_semantic_hash=semantic_hash(RewardSpec()),
    deployment_reward_contract=(
        CurriculumDeploymentRewardContract.FACTORIZED_ADDITIVE_V1
    ),
    reward_transform="identity",
    reward_aggregation="fixed-horizon mean",
    reward_horizon=1,
    information_variable="next persistent query observation",
    information_conditioning="current factorized posterior and active support",
    confidence_method="simultaneous synthetic bounds",
    confidence_level=0.95,
    confidence_family="all eligible catalog targets in one update",
    data_split="synthetic development data",
)
controller = CurriculumController(
    catalog,
    MarginalValuePerBitPolicy(minimum_value_per_nat=0.05),
    CurriculumLimits(
        maximum_support=8,
        maximum_planned_information_nats=16.0,
    ),
    estimand,
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
        evidence_basis=CurriculumEvidenceBasis(
            estimand=estimand,
            active_members=controller.state.active_members,
            source_scientific_payload_hash=scientific_hash(
                {"kind": "synthetic-curriculum-evidence", "round": 0},
                domain="curriculum-evidence",
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
- Evidence descriptors bind declared estimands, support, and source artifacts;
  they do not independently establish estimator validity, simultaneous
  confidence coverage, or source-artifact truth.
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
