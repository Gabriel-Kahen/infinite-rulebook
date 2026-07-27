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

The repository is implementing the first exact-calibration gates. It covers:

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
- and regression tests against analytic frontiers and control semantics.

## Development

The project uses Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

The symbolic control API and its exact identities are documented in
[docs/symbolic-controls.md](docs/symbolic-controls.md).

## Research scope

The full plan continues through learned frontier bounds, distractor controls,
adaptive curricula, dynamic-state extensions, and a procedural neural
benchmark. See [docs/research-plan.md](docs/research-plan.md) for definitions,
gates, and falsification criteria. The finite numerical contract is in
[docs/finite-solver.md](docs/finite-solver.md), and the information/artifact
contract is in [docs/information-metrics.md](docs/information-metrics.md).

## License

MIT
