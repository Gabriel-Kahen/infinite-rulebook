# Symbolic construct-validation protocol, version 1

## Status and interpretation

This protocol defines a bounded symbolic construct-validation study for the
current exact finite implementation. It is deliberately called
`symbolic-confirmatory-v1`, but it is **not** the full confirmatory study
described in the research plan. Its purpose is to validate that the implemented
reward, information, novelty, redundancy, and public-reward constructs separate
in the predicted directions under a small, auditable design.

The protocol has three ordered stages:

1. rerun the integration smoke sweep;
2. run the registered calibration matrix and freeze all design-dependent
   choices; and
3. run a disjoint, sealed confirmatory matrix once.

Calibration outcomes may determine only the confirmatory environment-seed count
from the registered candidate grid. Numerical margins, practical effects, and
tolerances are preregistered; calibration verifies and seals them but may not
tune them. Confirmatory outcomes may not change the config, seed count,
contrasts, margins, tolerances, exclusions, or analysis code. A change after
unsealing data creates a new protocol version and requires a fresh confirmatory
phase seed bank.

Generated evidence fields in this document are intentionally blank until the
corresponding authenticated artifacts exist. Filling those fields records
evidence; it must not silently amend the protocol.

The foundation pilot, an implementation design probe, and earlier engineering
smoke runs predate this protocol and were used to debug the benchmark. They are
not inferential evidence. The first public Git commit containing this protocol,
the exact calibration config, and the registered analysis/canary plans is the
registration point. The receipt-bound Stage-0 rerun, calibration matrix, freeze
seal, and confirmatory matrix must be generated only after that public commit,
in that order.

## Scientific question

Within the registered finite symbolic system, do the implemented measurements
distinguish:

- target expansion from fixed-target saturation;
- reward-relevant acquisition from reward-irrelevant persistent acquisition;
- fresh aleatoric prediction error from persistent acquired information;
- independent useful structure from redundant useful structure; and
- hidden reward from a bounded public reward component?

The study addresses construct separation in this finite implementation. It
does not establish asymptotic open-endedness or external validity for learned,
neural, procedural, or adaptive systems.

## Stage 0: smoke rerun

The smoke config is `configs/pilot-smoke.json`. It crosses six environments
with four agents:

- environments: `IND`, `RED-C`, `MIX`, `ALEA`, `TRIVIA`, and `PUBLIC-C`;
- agents: `fixed`, `reward`, `novelty`, and `total-information`;
- projection size: 2;
- horizon: 4 rounds;
- checkpoints: 0, 2, and 4;
- environment replicas: 1; and
- algorithm replicas: 1.

This is an engineering smoke test, not inferential evidence. In particular:

- one environment seed and one algorithm seed provide no estimate of
  between-environment or between-algorithm variation;
- the horizon is too short to separate stable growth from transient behavior;
- only three checkpoints are observed;
- the `scheduled` and `relevant-information` agents are absent;
- the smoke `RED-C` cell has two core dimensions on a two-coordinate
  projection, so it does not provide a nondegenerate independent-versus-
  redundant comparison; and
- its results cannot justify a confirmatory seed count, an equivalence margin,
  a smallest meaningful effect, or a population claim.

Under the trusted CLI/process model, the smoke rerun is accepted only as
operational evidence that the declared serial and parallel workflows, artifact
validation, deterministic replay, and the six implemented environment paths
work end to end. Paired pre-execution receipts prevent accidental whole-root
reuse and role swaps, but they are not signatures or independent proof of
scheduler/thread usage and do not defend against a writer deliberately
regenerating declarations or copying selected subtrees.

The paired reproducibility report and both portable raw inventories are
embedded in a self-hashed `symbolic-smoke-prerequisite` record. That record
reauthenticates both roots, cross-binds their receipt pair, and records every
non-invalidating engineering anomaly. Root locations remain operational fields
outside its hashed payload so a copied public archive can be reverified at a
new location. Registered calibration execution refuses to start without this
prerequisite. The calibration report packages it, and the
calibration-evidence hash binds its config, reproducibility, inventories,
anomaly log, and evidence hash; the confirmatory seal in turn binds that
calibration-evidence hash.

### Smoke evidence

- Artifact inventory and root scientific hash:
  **[GENERATED EVIDENCE: fill after validated rerun]**
- Serial/parallel scientific-hash agreement:
  **[GENERATED EVIDENCE: fill after validated rerun]**
- Validation result:
  **[GENERATED EVIDENCE: fill after validated rerun]**
- Observed engineering anomalies:
  **[GENERATED EVIDENCE: fill after validated rerun; “none” is allowed]**
- Smoke-prerequisite evidence hash:
  **[GENERATED EVIDENCE: fill after validated rerun]**

Failure at this stage blocks calibration until the implementation is repaired.
It does not permit weakening a later scientific criterion.

## Stage 1: registered calibration

The calibration config is
`configs/symbolic-calibration-v1.json`. Its scientific matrix is:

| Component | Registered value |
|---|---|
| Config name | `symbolic-construct-calibration-v1` |
| Environments | `IND`, `RED-C`, `MIX`, `ALEA`, `TRIVIA`, `PUBLIC-C` |
| Agents | `fixed`, `scheduled`, `reward`, `relevant-information`, `total-information`, `novelty` |
| Conditions | 6 environments × 6 agents = 36 |
| Horizon | 12 rounds |
| Checkpoints | Every integer round from 0 through 12 |
| Environment replicas | 192 |
| Algorithm replicas per environment | 3 |
| Calibration phase master | `irb-symbolic-calibration-v1` |
| Fixed algorithm master | `irb-symbolic-fixed-algorithm-bank-v1` |
| Feedback | P1, q-ary symmetric error 0.1, one query per round |
| Reward | \(q=4\), \(u=1\), \(c=1\), nats |

The complete calibration contains \(36\times192\times3=20{,}736\) run cells
and 13 authenticated checkpoints per cell. All 36 condition/agent combinations
must be present; a convenient-looking subset is not a substitute for the
registered inventory.

### Nondegenerate structural settings

All environments use a three-coordinate reward projection. `RED-C` has one
core dimension and permits support on all three projected surface coordinates.
Consequently, three reward-bearing surface coordinates can share one primitive
latent; unlike the smoke setting, core dimension is strictly smaller than
projection size. `MIX` uses one redundant core dimension with redundant support
capped at one, retaining both independent and shared structure.

The scheduled agent starts with target size 1, increases by 1 every four
rounds, and is capped at size 3. The fixed agent remains at target size 1.
For the reward, relevant-information, total-information, and novelty policies,
the stored `target_size: 3` field is a nominal configuration value retained for
schema compatibility; those policies choose coordinates from their named
objective and do not consult `target_size`. This behavior is fixed across
conditions and is not tuned after calibration.

`ALEA`'s `distractor_dimensions: 32` names the size of its fresh cosmetic
alphabet. It is not a persistent latent dimension or an acquired-information
ledger bucket. Fresh cosmetic values may affect prediction-error telemetry but
must leave persistent distractor information exactly zero.

### Role of calibration

Calibration is used only to:

- verify exact canaries and complete artifact production;
- estimate the distribution of paired environment-cluster residuals;
- run the predeclared split-sample design certification and diagnostic bootstrap;
- choose the confirmatory environment-replica count from the candidate grid;
- verify the predeclared late-reward matching margin is identifiable;
- freeze all predeclared numerical tolerances and margins into the seal; and
- verify that the proposed comparisons are numerically identifiable.

Calibration is not a first confirmatory attempt. Its p-values, if computed, are
descriptive and are not reported as confirmatory discoveries. Directional
calibration outcomes must be reported even if unfavorable. They may show that a
planned comparison is infeasible, but they may not be used to swap agents,
metrics, checkpoints, or hypotheses for more favorable ones.

Favorable calibration p-values are not a continuation gate. Freezing depends on
artifact integrity, exact canaries, the fixed design, and a finite candidate
that satisfies the preregistered conservative power and error targets. An
unfavorable descriptive calibration estimate remains published and does not
authorize outcome-based abandonment or seed replacement.

## Registered estimands

All primary comparisons use the terminal checkpoint, round 12. Metric names
below are the authenticated analysis fields.

For a comparison of a left and right condition, the cell-level difference is
left minus right. Conditions are paired on environment replica and algorithm
replica. The three crossed algorithm-replica differences are averaged once
within each environment replica. The resulting environment-level values are
the independent units for uncertainty and testing.

The five primary one-sided superiority contrasts are:

| ID | Registered comparison | Agent | Metric | Alternative |
|---|---|---|---|---|
| S1 | `scheduled` minus `fixed` in `IND` | comparison names the agents | `hidden_expected_reward` | greater than 0 |
| S2 | `relevant-information` minus `total-information` in `TRIVIA` | comparison names the agents | `hidden_expected_reward` | greater than 0 |
| S3 | `total-information` minus `relevant-information` in `TRIVIA` | comparison names the agents | `distractor_information_nats` | greater than 0 |
| S4 | `ALEA` minus `IND` for the novelty agent | `novelty` | `novelty.observation_prediction_error` | greater than 0 |
| S5 | `IND` minus `RED-C` for the reward-directed agent | `reward` | `relevant_information_nats` | greater than 0, subject to the late-reward match gate |

S4 is telemetry-only. `ALEA` supplies fresh, nonpersistent cosmetic
observations. It is not a competing persistent target, and increased
prediction-error novelty in `ALEA` is not interpreted as useful acquisition,
capability growth, or a reason an agent should pursue aleatoric noise. The
companion exact-zero canary below must show that the cosmetic stream contributes
no persistent distractor information.

S5 is interpreted only if the paired terminal
`hidden_expected_reward` values for `IND` and `RED-C` satisfy a predeclared
equivalence test within an absolute margin of 0.25 reward units. This margin is
registered before calibration: it is one quarter of one unit-reward coordinate,
or one twelfth of the registered three-coordinate terminal maximum. The
matched comparison uses the reward-directed agent so the agent is not selected
after seeing which condition gives the preferred result. If the reward match
is unresolved, report the reward mismatch and the S5 effect estimate, but do
not call S5 evidence for an independent-versus-redundant information contrast.
No post hoc checkpoint or regression adjustment replaces the registered
primary result; any such analysis is labeled sensitivity-only.

The S5 sign estimands concern strict paired differences under the registered
common-random-number coupling. `IND` and `RED-C` share algorithm and query-noise
tapes but do not share latent labels. The reward-match TOST therefore tests its
registered paired sign/median functional, not equality of marginal means, and
the S5 information conclusion is conditional on that coupling. Ties count
against the asserted direction or equivalence endpoint.

### Effect sizes and uncertainty

Every contrast reports:

- the number of independent environment clusters and underlying run cells;
- the environment-cluster mean and median paired difference;
- the complete distribution of the paired cluster differences;
- a distribution-free interval for the median;
- the raw one-sided exact sign-test p-value;
- the Holm-adjusted p-value and decision; and
- the corresponding pooled checkpoint summaries for both sides.

Algorithm replicas quantify within-environment algorithm variation. They do not
increase the inferential sample size from \(E\) to \(3E\). Repeated checkpoints
are correlated trajectory measurements and are not treated as additional
replicates. The same preregistered three algorithm seeds are crossed with every
environment replica in calibration and confirmation. They are a fixed nuisance
block, not a sample from an algorithm-seed population; inference is conditional
on that bank. The phase masters, and therefore the environment-side streams,
are disjoint.

The five superiority p-values form one family. Holm step-down adjustment controls
familywise error at \(\alpha=0.05\). A contrast is supported only when its
Holm-adjusted decision rejects its registered null in the registered direction.
The late-reward equivalence check for S5 is a design-match gate, reported with
its frozen two-one-sided-test margin and interval in a separate registered
equivalence family; it is not a sixth superiority claim.

Failure to reject, an effect in the opposite direction, or an unresolved
matching gate is a scientific result and is published. It is not an artifact
integrity failure and must not trigger seed replacement.

## Exact scientific canaries

Canaries are deterministic regression properties, not statistical hypotheses.
They are evaluated on every applicable paired seed and checkpoint and must all
pass before inferential output is interpreted.

1. **TRIVIA frontier identity.** Adding persistent reward-irrelevant trivia to
   the paired `IND` base leaves the authenticated exact frontier semantic hash
   unchanged.
2. **ALEA frontier identity.** Adding fresh reward-irrelevant cosmetic
   observations to the paired `IND` base leaves the authenticated exact frontier
   semantic hash unchanged; `ALEA` is absent from the decision problem.
3. **ALEA persistent-information zero.** For every registered agent,
   `distractor_information_nats` is exactly zero for `ALEA` at every registered
   checkpoint. Numerical “nearly zero” is not substituted for this discrete
   ledger invariant.
4. **PUBLIC-C decomposition.** For every registered agent and checkpoint,
   `expected_reward = hidden_expected_reward + public_reward`, and the deployed
   maximizing public action contributes the registered bounded value 0.5.
5. **Paired reward-path identity.** For the reward-directed agent, the complete
   hidden-reward and useful-information trajectories in `ALEA`, `TRIVIA`, and
   `PUBLIC-C` equal their paired `IND` trajectories within the frozen absolute
   tolerance. This checks that the wrappers add their registered control
   feature without changing the reward-learning path.

These five properties expand to twenty machine canaries: two frontier
identities, six paired metric identities, six public decompositions, and six
exact-zero checks. Ledger reconciliation is separately enforced while
authenticating every typed checkpoint: total acquired information must reconcile
with the non-double-counted ledger buckets, including shared-core and
dynamic-state terms. Serial/parallel equality is an execution reproducibility
gate rather than an entry in the canary report. Both checks remain mandatory.

The numerical equality tolerance used for floating-point PUBLIC-C
decomposition and paired reward/useful-information path comparisons is frozen
from calibration and recorded in the seal. Hash identities and the ALEA
zero-information ledger invariant are exact.

### Canary evidence

- Canary-plan scientific hash:
  **[GENERATED EVIDENCE: fill from checked-in plan]**
- Calibration canary-report scientific hash and decision:
  **[GENERATED EVIDENCE: fill after validation]**
- Maximum observed decomposition residual:
  **[GENERATED EVIDENCE: fill after validation]**

## Power calibration and confirmatory seed count

The calibration design fixes \(E=192\) independent environment seeds and
\(A=3\) fully crossed fixed algorithm seeds. For each superiority contrast and
the S5 equivalence gate, design assurance consumes the 192 environment clusters
after averaging the three paired algorithm differences within each cluster.
Its population interpretation is conditional on the registered simulator's
environment-seed clusters being independent, stationary draws from the same
seed-generated data-generating process used for confirmation, and on the fixed
three-seed algorithm nuisance bank.

Before calibration is run, each contrast receives an externally justified
smallest scientifically meaningful difference. These values are not set equal
to the observed calibration means.

| Contrast | Smallest meaningful difference | Provenance |
|---|---:|---|
| S1 | 0.25 reward units | One quarter of one unit-reward coordinate |
| S2 | 0.25 reward units | Same registered reward scale as S1 |
| S3 | 0.50 nats | Persistent information separation of about 0.72 bits |
| S4 | 0.10 prediction-error units | Ten percent of the telemetry metric's unit scale |
| S5 | 0.50 nats | Useful-information separation of about 0.72 bits |

### Split-sample certification rule

Canonical environment IDs divide the calibration sample before inspection:
the first 64 clusters form the effect-adequacy split and the remaining 128 form
the favorable-sign-probability split. The partition is not randomized or
changed in response to outcomes.

On the first split, exact distribution-free order-statistic intervals certify
the population median. For every directional contrast, the simultaneous
interval must lie at or beyond its externally registered minimum effect in the
registered direction. For S5 reward equivalence, the interval must lie strictly
inside \((-0.25, 0.25)\). A failure makes every candidate ineligible; it is
reported rather than repaired by changing the effect threshold.

On the independent second split, strict favorable signs are counted. A
directional observation is favorable only when its paired difference is
strictly beyond the null in the registered direction. The two S5 TOST endpoints
count \(D>-0.25\) and \(D<0.25\) separately. Ties are failures. Exact one-sided
Clopper--Pearson lower limits provide conservative lower bounds on the seven
favorable-sign probabilities. The certified S5 bounds therefore target the
same stationary raw paired-difference population represented by these held-out
calibration clusters. They do not recenter the observations or claim assurance
at a hypothetical exact mean or median difference of zero.

The design confidence budget is 0.01 across 19 events by Bonferroni allocation:
12 order-statistic interval tails for the five directional medians and one
equivalence median, plus seven favorable-probability lower bounds. Thus the
reported effect-adequacy and favorable-probability bounds hold simultaneously
with confidence at least 0.99 under the registered independent/stationary
environment-seed model.

For each candidate count, exact binomial tails convert those probability lower
bounds into conservative sufficient-event power bounds:

- each directional bound uses raw threshold \(\alpha/5=0.01\), which is
  sufficient for rejection under Holm regardless of the other four p-values;
- the two S5 exact-sign TOST endpoints each use \(\alpha=0.05\), and their joint
  equivalence-power lower bound follows by a union bound;
- the registered joint-power lower bound follows by a union bound across all
  five directional decisions and both equivalence endpoints; and
- superiority-family and equivalence-boundary familywise-error bounds are the
  analytic exact-test/Holm bounds of 0.05, not fitted error rates.

Select the smallest candidate \(E_\mathrm{confirm}\) satisfying all of:

- power of at least 0.90 for every individual superiority contrast;
- power of at least 0.90 for S5 equivalence under the registered stationary raw
  paired-difference population represented by the held-out calibration split;
- joint power of at least 0.80 for all five superiority decisions and S5
  equivalence together;
- certified superiority-family global-null error no greater than 0.05; and
- certified false-equivalence error no greater than 0.05 at either margin
  boundary.

The implementation also runs 10,000 deterministic paired cluster-bootstrap
simulations per candidate, preserving shared resampled environment indices
across the registered family. It reports shifted-alternative power, global-null
error, S5 point-scenario equivalence at \(\delta^\*=0\), and both equivalence
boundaries with simultaneous Monte Carlo bounds. These quantities are
conditional working-model diagnostics only. They never select a candidate,
rescue a failed certified bound, or serve as inferential evidence. In
particular, the S5 bootstrap diagnostic is not a uniform equivalence-power claim
over the margin.

The ordered candidate grid is:
**32, 48, 64, 96, 128, 192, 256, 384, and 512 environment replicas**.

If no candidate passes, do not choose the largest candidate merely because it
was tried. Report that the design failed its power gate, extend the candidate
grid in a versioned calibration amendment, or narrow the scientific claim in a
new protocol before any confirmatory outcomes are generated. Because the rule
uses one-sided confidence bounds and union bounds, failure may be a conservative
false stop; it is not evidence that the corresponding effect is absent.

### Power evidence

- Calibration dataset scientific hash:
  **[GENERATED EVIDENCE]**
- Minimum-effect registration hash:
  **[GENERATED EVIDENCE]**
- Power-simulation seed-bank identity:
  **[GENERATED EVIDENCE]**
- Power report scientific hash:
  **[GENERATED EVIDENCE]**
- Selected \(E_\mathrm{confirm}\):
  **[GENERATED EVIDENCE]**
- Expected confirmatory run count
  \(36\times E_\mathrm{confirm}\times3\):
  **[GENERATED EVIDENCE]**

## Frozen confirmatory design

The confirmatory matrix retains all calibration scientific settings except the
master seed, phase, environment-replica count selected by certified design
assurance, and the attached freeze record. In particular, it retains:

- the same six environments and six agents;
- the same structural settings, including projection size 3 and `RED-C` core
  dimension 1;
- horizon 12 and every-round checkpoints;
- three fully crossed algorithm replicas;
- the same preregistered three-seed algorithm nuisance bank;
- the same feedback, reward, and exact-solver settings; and
- the same five superiority contrasts and S5 reward-match gate.

The immutable seal binds:

- the canonical confirmatory config hash;
- the authenticated calibration-evidence hash;
- the exact analysis-contract identifier and registered analysis-plan hash;
- the analysis-source hash, dependency-lock hash, and execution-environment
  digest;
- the authenticated smoke-prerequisite evidence hash;
- separate calibration-environment, confirmatory-environment, fixed-algorithm,
  and confirmatory-evaluation seed-bank identities;
- all numerical tolerances;
- all equivalence and practical-effect margins; and
- a self-verifying seal hash.

The calibration-environment, confirmatory-environment, fixed-algorithm, and
confirmatory-evaluation namespaces and identities must be pairwise distinct.
The sealed execution-environment digest binds the dependency lock, Python
implementation/version/build/compiler, operating-system name and release,
machine architecture, libc identity, byte order, and Python float radix,
mantissa, exponent, and rounding metadata. A platform change requires a new
seal; the current lifecycle does not claim cross-platform bit identity.
The exact confirmatory config name is
`symbolic-construct-confirmatory-v1`, and its phase master is
`irb-symbolic-confirmatory-v1`; neither value is selected after calibration.
The exact checkpoint uses a deep-copied agent state and records a
domain-separated evaluation seed; no stochastic evaluation draw is consumed in
this version. The reserved evaluation stream cannot mutate the training stream.
The config must carry `phase: "confirmatory"` and
`confirmatory_frozen: true` through its authenticated freeze record. An absent,
false, malformed, stale, or mismatched seal fails closed before any run starts.

All hashes and seed-bank identities are public protocol material. The study
depends on prior registration and immutable execution, not secrecy.

### Frozen values

- Confirmatory config path and scientific hash:
  **[GENERATED EVIDENCE]**
- Analysis-plan path, registration hash, and code version:
  **[GENERATED EVIDENCE]**
- Calibration-evidence hash:
  **[GENERATED EVIDENCE]**
- Confirmatory-training seed-bank identity:
  **[GENERATED EVIDENCE]**
- Fixed algorithm seed-bank identity:
  **[GENERATED EVIDENCE]**
- Confirmatory-evaluation seed-bank identity:
  **[GENERATED EVIDENCE]**
- Decomposition tolerance:
  **[GENERATED EVIDENCE]**
- S5 late-reward equivalence margin and provenance:
  **[GENERATED EVIDENCE]**
- Minimum practical effects:
  **[GENERATED EVIDENCE]**
- Final seal hash:
  **[GENERATED EVIDENCE]**

## Hard execution and interpretation gates

The following are pipeline gates. Every gate must pass:

1. The exact Stage-0 smoke prerequisite reauthenticates, passes, and is bound
   into calibration evidence before calibration execution.
2. The repository commit, dependency lock, scientific dirty-state digest,
   Python/numeric environment, solver contract, config, and analysis code are
   captured in run identity.
3. The expected inventory is complete: every registered
   environment/agent/environment-replica/algorithm-replica cell and every
   checkpoint from 0 through 12 is present exactly once.
4. Every raw run tree and referenced frontier bundle passes full artifact-tree
   validation, including manifests, content hashes, journal chains, witnesses,
   dual certificates, and semantic compatibility.
5. Every confirmatory record carries the same authenticated config, analysis
   registration, and freeze hash.
6. Calibration-environment, confirmatory-environment, fixed-algorithm, and
   confirmatory-evaluation seed banks are disjoint.
7. Serial and parallel scientific hashes agree on the registered reproducibility
   check.
8. Every exact scientific canary passes.
9. Solver primal/dual and bound residuals remain within the sealed numerical
   tolerances.
10. The power report selects a finite confirmatory environment count that meets
   the registered operating targets.
11. The S5 reward-match test is evaluated with its frozen margin before S5 is
    interpreted.
12. Reports and figures read only validated, finalized artifacts and reproduce
    deterministic content hashes.

A failure of gates 1–10 or 12 blocks inferential interpretation. Repairing a
scientific implementation or protocol error requires a new seal and a fresh
confirmatory run; raw finalized artifacts are never edited. Failure of gate 11
blocks only the matched interpretation of S5. Failure of a superiority
hypothesis is not a hard-gate failure and never justifies rerunning with new
seeds.

## Confirmatory analysis and reporting

The analysis loader first validates complete artifact trees, then constructs
the registered ensemble. Reward is pooled before inversion through the
authenticated certified frontier. Bit-equivalent bounds, where shown as
construct context, are computed from pooled reward; the analysis does not
average seedwise nonlinear reward-to-information ratios.

The canonical machine-readable report package collectively includes:

- dataset, config, analysis-plan, calibration-evidence, and freeze hashes;
- the authenticated Stage-0 smoke prerequisite in the calibration package;
- canonical copies of the exact config, analysis plan, and canary plan;
- complete pooled checkpoint summaries;
- all five registered contrast results;
- the S5 reward-equivalence result;
- raw and Holm-adjusted decisions;
- environment-cluster and underlying-cell counts;
- exact intervals and effect sizes;
- every canary result;
- an explicit, hashed deviation log;
- portable checksum inventories for both authenticated raw roots;
- seed-bank identities and provenance; and
- scientific hashes over every evidence object and an authenticated release
  manifest binding the exact package inventory and file bytes.

These hashes and manifests are tamper-evident content bindings, not
cryptographic signatures. The public Git commit, reviewed merge history,
release tag, and hosting platform provide the publication trust anchor.

The human-readable report and figures are deterministic projections of that
machine-readable evidence. They must show unfavorable and unresolved results
with the same prominence as favorable results.

### Confirmatory evidence

- Confirmatory dataset scientific hash:
  **[GENERATED EVIDENCE]**
- Valid run trees / expected run trees:
  **[GENERATED EVIDENCE]**
- Confirmatory canary-report hash and decision:
  **[GENERATED EVIDENCE]**
- Registered-analysis report hash:
  **[GENERATED EVIDENCE]**
- S1 result:
  **[GENERATED EVIDENCE]**
- S2 result:
  **[GENERATED EVIDENCE]**
- S3 result:
  **[GENERATED EVIDENCE]**
- S4 telemetry result:
  **[GENERATED EVIDENCE]**
- S5 reward-match decision and superiority result:
  **[GENERATED EVIDENCE]**
- Deviations:
  **[GENERATED EVIDENCE: “none” or a complete versioned account]**

## Outputs and public release

The public repository or an attached immutable release must contain:

- the smoke, calibration, and sealed confirmatory configs;
- the frozen analysis plan and exact canary plan;
- calibration, power, freeze, and confirmatory machine-readable reports;
- human-readable calibration and confirmatory summaries;
- tabular contrast, checkpoint, canary, and power evidence in open formats;
- deterministic figures with source data and build commands;
- manifests or checksums for every released raw artifact tree and frontier
  bundle;
- the code commit and dependency lock needed to reproduce the study; and
- an explicit deviation log.

Raw artifact volume may be carried by a versioned public release rather than
ordinary Git history, but its manifests and scientific hashes must be checked
in. Publication requires code review, passing continuous integration, an
independent artifact-integrity review, and an independent scientific-protocol
review. A tagged release should identify the exact merge commit and seal hash.

## Deliberate scope exclusions

This version keeps the broader research program intact but does not claim to
complete it.

- It is not the research plan’s full \(2\times3\times2\) primary factorial.
  Feedback protocol, reward alphabet/margin, and several structural factors are
  fixed rather than crossed.
- It does not run \(T=256\), held-out horizon extrapolation, finite-prefix
  scaling ladders, or the registered asymptotic classifier. No bounded,
  logarithmic, sublinear, linear, or open-ended scaling conclusion follows from
  \(T=12\).
- It makes no claim about learned or approximate frontier estimators,
  optimization variation, neural action representations, or estimator
  calibration coverage.
- It makes no neural-agent or procedural-wrapper claim.
- It does not evaluate adaptive curricula, oracle or estimated
  frontier-following schedules, target discovery, or phase diagrams.
- It does not evaluate episodic persistent-history (`EPH`) turnover or the
  separation between trajectory information and bounded current-state
  bit-equivalent.
- It does not establish robustness across multiple \(q\), reward margins,
  feedback protocols, noise levels, dependency graphs, or action
  representations.
- It does not convert the ALEA telemetry canary into a competing-target or
  capability claim.

Those components remain follow-on workstreams. The warranted conclusion from a
successful version-1 study is narrower: the exact bounded symbolic
implementation passes its integrity gates and its registered construct
separations are supported, unsupported, or partially identified as the sealed
evidence reports.
