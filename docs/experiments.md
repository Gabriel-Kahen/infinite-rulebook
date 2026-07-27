# Experiment artifacts

The experiment harness turns a frozen, versioned JSON config into deterministic
run cells:

```text
environment × feedback × reward × agent × environment seed × algorithm seed
```

Each cell has named seed streams for the environment, persistent distractors,
aleatoric observations, query noise, algorithm choices, deployment, evaluation,
and frontier computation. Random choices must use semantic coordinates rather
than consume a shared sequential generator. Matched conditions and agents share
the corresponding tapes for a replica.

Run the small foundation pilot with:

```bash
uv run infinite-rulebook pilot configs/pilot-foundation.json
uv run infinite-rulebook validate artifacts/pilot-foundation/<run-hash>
```

The foundation config retains the original narrow IND, RED-C, and MIX
reward-directed regression. The same runner now integrates the registered
control, ledger, metric, and comparison-agent APIs for the broader smoke sweep.

The dependency-integrated smoke sweep is:

```bash
uv run infinite-rulebook pilot configs/pilot-smoke.json --workers 4
```

It crosses IND, RED-C, MIX, ALEA, TRIVIA, and PUBLIC-C with fixed-target,
reward-directed, novelty-directed, and total-information agents. The projection,
horizon, support caps, public schedule, and replica count are intentionally
small. Per-run records contain realized Bayesian surprise only. Population-MI
efficiency and frontier regret are not emitted until complete histories are
pooled into a genuine ensemble estimate.

ALEA prediction-error telemetry includes the aligned cosmetic alphabet used by
the novelty objective. The pilot's compression-improvement field is the exact
uniform-prior posterior entropy reduction recorded by the finite ledger; fresh
cosmetic values do not enter that persistent-information quantity.

## Artifact contract

Frontiers are cached under `artifacts/_frontiers/<frontier-hash>/`, separately
from agent runs. Run directories contain:

- a fully resolved config and deterministic seed tree;
- one immutable, hash-chained file per training event;
- side-effect-free checkpoint artifacts from an independent evaluation stream;
- typed run checkpoint records and exact finite-closure ledgers;
- a reference to the validated frontier bundle;
- immutable metrics and a final manifest.

Frontier bundles retain the problem, raw curve, feasible channel witnesses,
dual certificates, and solver diagnostics. Validation checks artifact hashes,
semantic compatibility, manifest membership, witness reward/information,
recomputed dual bounds, and certificate consistency. A finalized run is valid
only while its separately cached frontier is present and matches the referenced
frontier manifest.

Appending a training event is atomic. On restart, the journal is validated and
replayed before the next missing round; an existing event key can only be reused
with identical content. Finalized runs are read-only and reject added, missing,
changed, or semantically incompatible artifacts.

Scientific hashes include only canonical scientific payloads. Runtime metadata
such as timestamps, wall time, hardware, file metadata, and output paths can be
recorded in manifests but is excluded from scientific-content hashes. Run
identity records the code commit, tracked and untracked scientific changes,
dependency lock, analysis-code digest, Python/numeric environment, and explicit
pilot solver settings, so changed scientific code cannot silently reuse a prior
run.

Every parallel cell receives a fresh adapter instance. Checkpoint evaluation
deep-copies that adapter and its training state before evaluation, and resumed
checkpoints must match the replayed state and independent seed streams.

## Pilot boundary

Configs accepted by this runner are explicitly `phase: "pilot"`. Pilot output
may inform later tolerances, matching rules, seed counts, and power analysis,
but the harness does not mark tolerances as frozen or produce a confirmatory
config. Confirmatory settings must be frozen only after the registered pilot
gates are evaluated.
