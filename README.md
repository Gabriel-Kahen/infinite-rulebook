# Infinite Rulebook

Infinite Rulebook is a research benchmark for testing whether an agent's
performance requires a growing amount of reward-relevant information.

The project implements the experimental program described in
[*Open-Ended or Just Novel?*](docs/research-plan.md). Its primary empirical
object is the bit-equivalent:

\[
B_\rho
=
\inf_{P(A\mid\Theta):\,\mathbb E[r_\Theta(A)]\ge\rho}
I(\Theta;A).
\]

## Status

The repository implements the exact symbolic study lifecycle and its
construct-validation gates. It covers:

- a stationary, lazily generated Rulebook environment;
- canonical finite-support deployment actions;
- strict negative reward for uninformed deployment;
- the analytic one-coordinate reward-information frontier;
- finite-\(N\) tensorization and the infinite linear frontier;
- direct finite stochastic-channel problems with certified
  Blahut--Arimoto frontier bounds and information-budget inversion;
- exhaustive finite Rulebook projections;
- unrestricted, support-capped, and mixed redundancy controls;
- stationary ALEA and persistent queryable TRIVIA controls;
- exact PUBLIC-U and bounded PUBLIC-C reward transformations;
- an auditable finite-closure information ledger with correlated RED/MIX
  no-double-count guarantees;
- typed bit-equivalent, efficiency, regret, reward, support, and novelty
  metrics;
- deterministic immutable artifacts with separate semantic, scientific, and
  runtime payload boundaries;
- bounded factorized comparison agents with fixed, expanding, reward, novelty,
  total-information, and useful-information acquisition objectives;
- typed deterministic experiment configs and named seed banks;
- restart-safe immutable scientific artifacts and separately cached frontiers;
- a side-effect-free checkpoint runner with pilot, calibration, and
  fail-closed confirmatory phases;
- immutable confirmatory seals binding calibration evidence, seed-bank
  identities, registered analysis, source code, tolerances, and margins;
- environment-clustered registered analysis, exact canaries, split-sample
  certified design assurance, diagnostic power simulation, open CSV tables,
  and accessible SVG figures;
- and regression tests against analytic frontiers and control semantics.

## Development

The project uses Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
uv run infinite-rulebook pilot configs/pilot-foundation.json
uv run infinite-rulebook pilot configs/pilot-smoke.json --workers 4
uv run infinite-rulebook plan \
  configs/symbolic-calibration-v1.json \
  configs/symbolic-calibration-analysis-v1.json \
  configs/symbolic-calibration-canaries-v1.json
```

The symbolic control API and its exact identities are documented in
[docs/symbolic-controls.md](docs/symbolic-controls.md).
The non-registered deterministic curriculum API and its current limitations
are documented in
[docs/adaptive-curriculum.md](docs/adaptive-curriculum.md).
See [docs/experiments.md](docs/experiments.md) for artifact, restart, semantic
hash, validation, and execution contracts. The bounded symbolic
construct-validation protocol is
[docs/symbolic-confirmatory-v1.md](docs/symbolic-confirmatory-v1.md).

### Registered study status

Symbolic construct-validation v1 completed its public Stage-0 and registered
calibration lifecycle. Exact serial/parallel reproduction, all 20 canaries,
raw-artifact authentication, and the deviation gate passed. Five of six
effect-adequacy gates passed; the
`relevant-over-total-trivia-hidden-reward` gate did not. The fail-closed
protocol therefore selected no confirmatory sample size, created no seal, and
ran no confirmatory outcomes.

The stopped result is preserved in the
[calibration summary](results/symbolic-calibration-v1/summary.json),
[power report](results/symbolic-calibration-v1/power.json), and
[protocol disposition](docs/symbolic-confirmatory-v1.md#post-calibration-disposition-outcome-record).
The evidence-bearing publication inventory and reconstruction instructions are
in the [v0.1.0 release note](docs/releases/v0.1.0.md).

The ambitious v2 follow-on retains the original panel and adds TRIVIA D12/D24,
eight fixed algorithm replicas, authenticated post-query reward trajectories,
compact chunk-authenticated canaries, and registered supplemental evidence.
Its exact design and registration identities are in the
[v2 protocol](docs/symbolic-confirmatory-v2.md). No v2 pilot, ingestion probe,
calibration, freeze, confirmatory execution, or outcome analysis preceded this
registration. The post-registration
[operational preflight](docs/symbolic-v2-operational-preflight.md) passed the
48-run ingestion contract but failed the E768 memory gate on the local
32-GiB-class workstation. No registered v2 run is authorized there. The
unchanged workflow must move to a larger, longer-lived host and repeat the
exact capacity and ingestion gates in
[the execution guide](docs/experiments.md). The
[reporting-host runbook](docs/symbolic-v2-host-qualification.md) provides
read-only static checks, synthetic capacity evidence, and a
descriptor-anchored disjoint-seed probe gate without invoking a registered
study.

## Research scope

The full plan continues through learned frontier bounds, distractor controls,
adaptive curricula, dynamic-state extensions, and a procedural neural
benchmark. See [docs/research-plan.md](docs/research-plan.md) for definitions,
gates, and falsification criteria. The finite numerical contract is in
[docs/finite-solver.md](docs/finite-solver.md), and the information/artifact
contract is in [docs/information-metrics.md](docs/information-metrics.md).
The initial
[learned-estimator foundation](docs/learned-estimator-foundation.md) is
restricted to deterministic calibration on small synthetic finite problems;
it makes no v2 outcome or large-instance claim.

## License

MIT
