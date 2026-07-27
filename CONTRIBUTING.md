# Contributing

## Setup

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
```

## Scientific requirements

- Keep information quantities in nats.
- Treat deployments as canonical behaviors, not serialized descriptions.
- Do not change frontier semantics through a feedback implementation.
- Add a regression test for every analytic invariant.
- Label approximate bounds, statistical intervals, and heuristics separately.
- Keep training, fingerprint, frontier-fitting, evaluation, and final-test seeds
  disjoint.

## Pull requests

Describe the mathematical or experimental contract affected by the change and
include the validation commands used.
