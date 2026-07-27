# Information ledger and core metrics

This module implements the information-accounting contract in the research
plan. All stored information is in nats.

## Finite-closure ledger

`InformationLedger` records only the unique primitive latents touched by a
history. Untouched coordinates in an infinite Rulebook remain at the prior and
are absent. For every `PosteriorBlock`, the ledger computes

\[
D_{\mathrm{KL}}(P(\Theta_{\mathcal J}\mid h)\|P(\Theta_{\mathcal J}))
\]

directly from finite joint distributions. It never obtains information by
subtracting prior and posterior entropies.

Axes in a block are canonicalized by scientific category and latent identifier,
with joint-state probabilities permuted to match. Reward-relevant primitives
and shared cores precede distractors, so their prefix contribution equals the
reward-relevant posterior projection instead of depending on caller order. The
total is computed directly from the joint distribution and audited against the
chain-rule decomposition in that order. A block can therefore represent
correlated RED or MIX posteriors even when every one-variable marginal is
unchanged.

```python
from infinite_rulebook.information import (
    InformationCategory,
    InformationLedger,
    LatentAxis,
    PosteriorBlock,
    SurfaceDependency,
)

core = PosteriorBlock(
    block_id="red-core",
    axes=(LatentAxis("z", 2, InformationCategory.SHARED_CORE),),
    prior=(0.5, 0.5),
    posterior=(1.0, 0.0),
    surface_dependencies=(
        SurfaceDependency("rule:1", ("z",)),
        SurfaceDependency("rule:1000", ("z",)),
    ),
)
breakdown = InformationLedger((core,)).breakdown
```

Both surface rules are provenance aliases for `z`; they do not create two
information contributions. Primitive latent identifiers must be globally
disjoint across blocks. Separate blocks assert both prior and posterior
factorization, so correlated variables belong in one joint block.

The additive buckets are:

- `reward_relevant_nats`: independent reward primitives;
- `shared_core_nats`: reward-relevant RED/MIX core variables;
- `persistent_distractor_nats`: persistent irrelevant variables such as
  TRIVIA;
- `dynamic_state_nats`: state-aligned dynamic information for later EPH work;
- `approximation_residual_nats`: a signed reconciliation term for approximate
  backends.

`relevant_nats` is the sum of the first two fields and is not another bucket.
ALEA has no persistent latent contribution. PUBLIC changes reward/frontier
semantics but adds no environment information. These future conditions can
construct the records above without the ledger importing an unfinished
environment implementation.

`InformationBreakdown` is a realized-history Bayesian surprise. It is not
population mutual information. `PopulationInformationEstimate` is a distinct
metric type so the two estimands cannot be passed interchangeably to
efficiency calculations.

## Bit-equivalent and frontier regret

`FrontierCurve` turns certified point bounds into a safe step-lower/chord-upper
envelope. The lower bound holds the previous certified value until the next
grid point; it does not incorrectly use a convex chord as a lower bound. The
upper chord is accepted only with an explicit witness-mixture certificate. The
record also requires an exact zero-information endpoint, a maximum-reward
endpoint, monotone bounds, nats, and the frontier semantic hash.

- `lookup_bit_equivalent` returns zero through the zero-information optimum and
  positive infinity above attainable reward.
- `integrate_bit_equivalent` integrates \(B_{\mathbb E[R_t]}\), not a
  seedwise nonlinear average. Sparse checkpoints must span `[0, horizon]` and
  declare either left-hold or linear interpolation. Elapsed-round weighting is
  mandatory.
- `frontier_regret` inverts the upper information envelope for the guaranteed
  reward bound and the lower envelope for the possible reward bound. Call it
  once with total acquired information and once with relevant information to
  populate `FrontierRegretMetrics`.
- `useful_information_efficiency` accepts only a
  `PopulationInformationEstimate`, requires a finite positive denominator,
  and emits diagnostics for an incomplete history manifest or a ratio above
  one. It does not clamp scientific violations.

`RewardMetrics`, `SupportMetrics`, and `NoveltyMetrics` are frozen validated
records. Fresh aleatoric novelty and persistent trivia novelty remain separate;
there is deliberately no scalar novelty total.

## Immutable artifacts and hashes

`RunCheckpoint` stores run-local reward samples, realized information, the
canonical deployment witness and verified semantic hash, deployment seed,
novelty, support, target size, and scientific operation counts. Wall/CPU/GPU
timings remain runtime metadata.
`CheckpointEstimate` stores pooled reward, bit-equivalent bounds, population
information, efficiency, novelty/support, both frontier regrets, uncertainty,
and named semantic hashes. They are distinct immutable schemas.

`ArtifactEnvelope` has three explicit payload boundaries:

- `semantic_payload` describes problem meaning and produces `semantic_hash`;
- `scientific_payload` contains reported values and produces
  `scientific_payload_hash`;
- `runtime_metadata` contains host, timing, path, or other execution metadata
  and affects neither scientific hash.

Canonical JSON sorts object keys, normalizes Unicode to NFC, encodes floats by
exact hexadecimal value, normalizes negative zero, tags infinities and byte
seeds, and rejects NaN. SHA-256 hashes are domain-separated, so semantic and
scientific hashes of the same bytes cannot collide by construction.

Local constructors reject malformed shapes and overlaps. Cross-record checks
return a deterministically ordered `ValidationReport` with stable diagnostic
codes such as `ABSOLUTE_CONTINUITY_VIOLATION`,
`COMPONENT_TOTAL_MISMATCH`, `INCOMPATIBLE_SEMANTIC_HASH`,
`EFFICIENCY_UNDEFINED`, and `EFFICIENCY_OUT_OF_RANGE`.
