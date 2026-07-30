# Symbolic construct-validation protocol, version 2

## Status and registration boundary

This document preregisters symbolic construct-validation v2. It retains the
complete v1 construct panel, adds two persistent-distractor loads, increases
the fixed algorithm bank, and replaces the failed v1 S2 primary with a
two-component estimand that measures both acquisition quality across the
learning path and terminal robustness under high distractor load.

No v2 pilot, calibration, freeze, or confirmatory outcome may be generated
until the implementation, this protocol, the exact configuration, analysis
plan, compact-canary specification, and supplemental-replication plan are
committed to the public repository. The first public commit containing all of
those items is the v2 registration point.

Unit tests, synthetic fixtures, and deterministic calculations used to
implement the protocol are engineering work, not study evidence. Any change
to a registered metric, selector, contrast, effect threshold, canary, seed
namespace, power rule, or candidate count after the registration point creates
a new protocol version before study data are generated.

## Motivation and v1 disclosure

V1 completed its registered calibration but stopped before confirmation. Five
of six effect-adequacy records passed. The
`relevant-over-total-trivia-hidden-reward` record failed because its
simultaneous median interval was `[0.0, 0.6666666666666666]`, below the
registered lower threshold of `0.25`; the independent favorable-sign count was
`58/128`. V1 therefore selected no confirmatory count, created no seal, and ran
no confirmatory outcomes.

The v1 terminal reward estimand was near a three-coordinate ceiling and
created many strict-sign ties even when the relevant-information policy
reached useful reward earlier. That diagnosis motivates v2 but does not
reinterpret or rescue v1.

A post-v1 exploratory learning-path calculation found a relevant-minus-total
mean of `0.5667`, a center interval of `[0.3611, 0.9167]`, and `116/128`
strict favorable held-out signs with a Clopper--Pearson lower bound of
`0.7939`. These values were inspected before v2 registration. They are
disclosed solely as design provenance and may not be reused as v2 evidence,
power observations, confirmation, or a substitute for fresh v2 calibration.

## Scientific question

Within the exact finite symbolic implementation, does the registered
measurement system distinguish:

- expanding reward-relevant support from fixed-support saturation;
- reward-relevant acquisition from reward-irrelevant persistent acquisition
  both across training and under increased distractor load;
- fresh aleatoric prediction error from persistent acquired information;
- independent useful structure from redundant useful structure; and
- hidden reward from a bounded public component?

V2 remains a construct-validation study. It does not claim asymptotic
open-endedness or neural/procedural external validity. It does not reduce the
broader research program in `docs/research-plan.md`.

## Exact implementation identities

- study contract: `bounded-symbolic-construct-validation.v2`
- adapter contract: `exact-symbolic-adapter.v2`
- calibration config hash:
  `c0f4cf5bf09e6b516379c0fec26ccd4a8780d8b6d52226093ef5a96cc0437508`
- phase-independent scientific-design hash:
  `a7d38ff66ff113f0c4a1aaae89e73a39df95294ede8c073b04437de064f88114`
- registration-component hash:
  `5f56c6fa65c0598d9cfa35d7425479b6fd21f0ab18f1f5330a101274a34a730f`
- calibration analysis-plan hash:
  `aef2100b60636a86f73f871a2b9f99346b2207762b872ec8262a512226a1f6fc`
- calibration analysis-registration hash:
  `747b53dc6fafbf354c595e20d57269270230a5f9d05a655a7df9da0e4903a1d0`
- calibration compact-canary-plan hash:
  `645a509da5f3c66563df8d88796a7a10ca87e817fd0e4549536fd68095e06a9a`
- calibration supplemental-plan hash:
  `1856efab9c7d4c0518dbb4004f3915bb9c70bb9376fbc255825bf4451c96c7f8`

The registration component transitively binds the aggregate metric definition,
compound-S2 rule, D12 role, legacy replication, expanded compact-canary
inventory, complete power-selection design, confirmatory tolerances, and
confirmatory margins into the ordinary analysis registration without changing
v1 schemas or hashes.

## Registered matrix

The exact checked-in registration artifacts are:

- `configs/symbolic-calibration-v2.json`;
- `configs/symbolic-calibration-analysis-v2.json`;
- `configs/symbolic-calibration-canaries-v2.json`; and
- `configs/symbolic-calibration-supplemental-v2.json`.

| Component | Registered value |
|---|---|
| Config name | `symbolic-construct-calibration-v2` |
| Environments | `IND`, `RED-C`, `MIX`, `ALEA`, `TRIVIA-D6`, `TRIVIA-D12`, `TRIVIA-D24`, `PUBLIC-C` |
| Agents | `fixed`, `scheduled`, `reward`, `relevant-information`, `total-information`, `novelty` |
| Conditions | 8 environments × 6 agents = 48 |
| Horizon | 12 rounds |
| Checkpoints | every integer round from 0 through 12 |
| Environment replicas | 192 |
| Fixed algorithm replicas | 8 |
| Runs per execution side | 73,728 |
| Calibration phase master | `irb-symbolic-calibration-v2` |
| Fixed algorithm master | `irb-symbolic-fixed-algorithm-bank-v2` |
| Confirmatory phase master | `irb-symbolic-confirmatory-v2` |
| Feedback | P1, q-ary symmetric error 0.1, one query per round |
| Reward | \(q=4\), \(u=1\), \(c=1\), nats |

The original v1 six-environment by six-agent panel is unchanged. D12 and D24
are additions, not replacements. `TRIVIA-D6`, `TRIVIA-D12`, and
`TRIVIA-D24` have persistent reward-irrelevant dimensions 6, 12, and 24.
Every analysis and canary selector carries exact `condition_hash` and
`agent_hash` values; selecting only `environment_kind="TRIVIA"` is ambiguous
and fails closed.

`RED-C`, `MIX`, `ALEA`, and `PUBLIC-C` retain their v1 structural settings.
The scheduled agent grows from target size 1 by one coordinate every four
rounds to a maximum of 3. The other agent semantics are unchanged.

Eight algorithm replicas form a fixed, fully crossed nuisance bank. They do
not increase the independent inferential sample size. The independent units
remain environment replicas after averaging the eight paired algorithm
differences once within each environment.

## Authenticated learning-path metric

V2 records `post_query_hidden_expected_reward` in every training event after
the query and posterior update for that round. At checkpoint \(t>0\),

\[
\texttt{post\_query\_mean\_hidden\_expected\_reward}(t)
=
\frac{1}{t}
\operatorname{fsum}_{r=1}^{t}
\texttt{post\_query\_hidden\_expected\_reward}(r).
\]

Both values are absent at checkpoint 0. At every checkpoint \(t>0\), the
checkpoint repeats the current round's authenticated
`post_query_hidden_expected_reward` and the validator requires exact equality
with training event \(t\). Training-event values, checkpoint values, the
adapter state fingerprint, run scientific hashes, exact replay, artifact
validation, and analysis loading all authenticate them. The aggregate canary
independently recomputes the mean from the registered per-round checkpoint
evidence. V1 payloads, state fingerprints, and artifact hashes remain
byte-identical.

## Stage-0 operational prerequisite

V2 retains the fail-closed requirement that calibration be receipt-bound to an
authenticated Stage-0 prerequisite. It reuses the already-public immutable v1
Stage-0 record solely because that record tests the common serial/parallel
workflow, artifact validation, deterministic replay, and six environment
paths:

- smoke config hash:
  `fae70beb1e57206d77cf192e437eb9d8baef2fb0f877a29f04181b0412edbec2`;
  and
- prerequisite evidence hash:
  `0ab32994d8c75c4ab36eb8de171f67ec802ae54c09bbb250b773918d5d892249`.

This prerequisite predates v2, is engineering evidence rather than v2
inferential data, and cannot justify a v2 effect, threshold, sample size, or
claim. It does not replace v2 adapter golden, resume, replay, aggregate-metric,
and tamper tests. The supplemental registration component binds this exact
reuse policy and evidence identity. A different, missing, or invalid
prerequisite blocks v2 calibration.

## Six-primary Holm family

All primary comparisons use checkpoint 12. The left-minus-right differences
are averaged over the eight crossed algorithm replicas within each environment
before uncertainty or testing. All six one-sided p-values form one Holm family
at \(\alpha=0.05\).

| ID | Registered comparison | Metric | Minimum effect |
|---|---|---|---:|
| S1 | `scheduled` minus `fixed` in `IND` | `hidden_expected_reward` | 0.25 reward |
| S2a | `relevant-information` minus `total-information` in `TRIVIA-D6` | `post_query_mean_hidden_expected_reward` | 0.25 reward |
| S2b | `relevant-information` minus `total-information` in `TRIVIA-D24` | `hidden_expected_reward` | 0.25 reward |
| S3 | `total-information` minus `relevant-information` in `TRIVIA-D6` | `distractor_information_nats` | 0.50 nats |
| S4 | `ALEA` minus `IND` for `novelty` | `novelty.observation_prediction_error` | 0.10 telemetry units |
| S5 | `IND` minus `RED-C` for `reward` | `relevant_information_nats` | 0.50 nats |

S4 remains telemetry-only. It is included in the registered family but cannot
be interpreted as capability growth or useful acquisition.

S5 remains conditional on the separately registered paired terminal hidden
reward equivalence gate with absolute margin `0.25`. The gate uses the same
fixed reward-directed agent and exact paired coupling as v1.

### Compound S2 rule

The construct-level S2 claim requires both S2a and S2b to pass their registered
Holm decisions and minimum-effect criteria. One component cannot rescue the
other. S2a measures reward acquisition across the training path under the
original D6 load; S2b requires a terminal separation under the higher D24
load.

## Registered evidence outside the primary family

### D12 intermediate-load comparison

The D12 relevant-minus-total terminal hidden-reward comparison is a registered
descriptive intermediate-load result outside the primary Holm family. It must
be reported regardless of direction. It cannot rescue S2a or S2b and cannot
change the selected power design.

### Legacy D6 terminal replication

V2 repeats the failed v1 D6 relevant-minus-total terminal hidden-reward
comparison using fresh v2 calibration and, if eligible, confirmatory seeds. It
is registered as `legacy-d6-terminal-hidden-reward-replication`, outside the
primary Holm family. Its exact sign result and distribution-free interval are
published in a separate supplemental plan/report. It cannot rescue either
compound-S2 component or enter confirmatory sample-size selection.

## Exact compact canaries

The v2 canary inventory contains 27 deterministic gates:

- four frontier identities: `ALEA`, D6, D12, and D24 versus `IND`;
- ten paired hidden-reward/useful-information path identities across `ALEA`,
  D6, D12, D24, and `PUBLIC-C`;
- six `PUBLIC-C` reward decompositions, one for each agent;
- six exact-zero persistent-information gates for `ALEA`, one for each agent;
  and
- one aggregate-metric derivation gate across all 48 exact registered groups.

All exact identities and ledger checks remain mandatory. A single canary
failure prohibits freezing.

To remain below hosting limits, each published v2 canary report stores
per-gate record counts, extrema, tolerances, pass/fail decisions, violation
counts, and a bounded canonical set of failure examples. Canonical detail
records are partitioned into chunks of at most 4,096 records. The report stores
each ordered chunk hash and an inventory/root hash over chunk counts, record
counts, and ordered hashes rather than embedding every detail record. The raw
serial and parallel roots remain the authoritative reconstructible evidence.
Report construction spools each chunk directly into its transactional output
directory, retains only one 4,096-record detail buffer, and derives the
aggregate gate one exact group at a time. Freeze independently recomputes and
stream-verifies the compact inventory.

## Design assurance and power

The calibration split remains fixed before inspection:

- the first 64 canonical environment replicas form the simultaneous
  median-effect adequacy split; and
- the remaining 128 form the independent favorable-sign probability split.

The v1 simultaneous confidence budget, conservative exact-sign/Holm power
logic, S5 equivalence boundaries, operating targets, and deterministic
10,000-simulation diagnostic bootstrap remain in force. The v2 directional
family has six rather than five members, so the implementation recomputes the
Bonferroni event count and the sufficient raw Holm threshold from the
registered family size.

Registered power identities:

- candidate environment counts:
  `32, 48, 64, 96, 128, 192, 256, 384, 512, 768`;
- seed: `bounded-symbolic-power-v2`;
- counter-RNG stream: `analysis.cluster-power.v2`;
- minimum individual superiority power: `0.90`;
- minimum S5 equivalence power: `0.90`;
- minimum joint power: `0.80`;
- maximum superiority global-null familywise error: `0.05`; and
- maximum false-equivalence error at either boundary: `0.05`.

The smallest candidate meeting every registered certified target is selected.
The diagnostic bootstrap never selects or rescues a candidate. If any
effect-adequacy record fails, or no candidate passes, `freeze_eligible` is
false. The largest candidate is never chosen by default.

## Freeze and confirmation

Only a complete, exactly reproduced, deviation-free calibration with all
canaries passing and a finite selected candidate may create a seal.

The seal binds:

- the exact confirmatory config and calibration-evidence hash;
- the v2 study contract, analysis registration, and registration component;
- the analysis-source, dependency-lock, and execution-environment hashes;
- all tolerances, minimum effects, and the S5 equivalence margin; and
- disjoint seed-bank identities using namespaces `calibration.v2`,
  `confirmatory.v2`, `algorithm.v2`, and `evaluation.v2`.

Confirmation uses the fresh master `irb-symbolic-confirmatory-v2`, the same
fixed eight-replica algorithm bank, and the selected environment count. A
missing, malformed, stale, or mismatched seal fails before execution. Outcomes
cannot change the config, estimands, selectors, thresholds, canaries, power
rule, exclusions, or report code.

## Execution, recovery, and publication gates

Serial and declared parallel executions use paired immutable receipts and must
contain every registered cell and checkpoint exactly once. Resume may recover
only the same receipt-bound invocation, roots, worker declaration, config, seed
banks, and finalized artifacts. Host interruption is operational only when the
authenticated resume path proves no scientific state changed.

Before interpretation or release:

1. every raw tree and frontier bundle validates;
2. serial and parallel scientific hashes agree for every cell;
3. both portable raw inventories independently authenticate;
4. all 27 compact canaries pass and reproduce exactly;
5. the formal deviation log is empty;
6. the registered analysis, supplemental replication, D12 result, and power
   report reconstruct from validated artifacts;
7. a finite candidate is selected before a seal is created;
8. the sealed confirmation, if eligible, uses disjoint registered banks; and
9. public manifests, chunked raw assets, code, lockfile, reports, open tables,
   figures, and unfavorable results receive independent review and passing CI.

Before calibration begins, the intended reporting host must pass the exact v2
calibration and maximum-candidate memory benchmarks plus the 48-condition
disjoint-seed artifact-ingestion probe in `docs/experiments.md`. A capacity
failure changes the host or execution window, never the registered matrix.

Any failed pre-freeze gate publishes a stopped calibration outcome with no
seal and no confirmatory execution. A failed hypothesis after a valid
confirmation is published as a scientific result and never triggers new seeds.

## Deliberate scope boundary

V2 strengthens the bounded exact symbolic validation without deleting any v1
condition or agent. It still does not complete the full research plan's
long-horizon scaling, learned frontier estimation, adaptive curriculum,
dynamic-state, procedural generation, or neural-agent tracks. Those remain
ambitious follow-on implementations. No conclusion about asymptotic
open-endedness follows from horizon 12.
