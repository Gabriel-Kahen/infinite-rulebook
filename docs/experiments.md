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
the corresponding tapes for a replica. The registered symbolic study uses
distinct phase masters for environment-side streams and one preregistered,
shared three-seed algorithm/deployment nuisance bank across calibration and
confirmation.

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
deep-copies that adapter and its training state, and resumed checkpoints must
match the replayed state. The exact symbolic checkpoint is deterministic: its
domain-separated evaluation seed is recorded and reserved, but no stochastic
evaluation draw is consumed in this version.

## Calibration and confirmatory boundary

The general `run` command accepts `pilot`, `calibration`, and `confirmatory`
configs. The legacy `pilot` command remains phase-strict. A confirmatory config
is rejected before execution unless it carries a self-verifying seal whose
config hash, calibration-evidence hash, registered-analysis hash,
analysis-source hash, dependency-lock hash, execution-environment digest,
disjoint seed-bank identities, tolerances, and margins all validate. Execution
checks all three current provenance hashes against the frozen values.

The bounded symbolic lifecycle is:

```bash
uv run infinite-rulebook reproduce \
  configs/pilot-smoke.json \
  artifacts/pilot-smoke-final-serial \
  artifacts/pilot-smoke-final-parallel \
  evidence/pilot-smoke-final-reproducibility.json \
  --workers 4
uv run infinite-rulebook inventory \
  configs/pilot-smoke.json \
  artifacts/pilot-smoke-final-serial \
  serial \
  evidence/pilot-smoke-final-serial-inventory.json
uv run infinite-rulebook inventory \
  configs/pilot-smoke.json \
  artifacts/pilot-smoke-final-parallel \
  parallel \
  evidence/pilot-smoke-final-parallel-inventory.json
uv run infinite-rulebook smoke-evidence \
  configs/pilot-smoke.json \
  evidence/pilot-smoke-final-reproducibility.json \
  evidence/pilot-smoke-final-serial-inventory.json \
  evidence/pilot-smoke-final-parallel-inventory.json \
  evidence/pilot-smoke-final-prerequisite.json
uv run infinite-rulebook plan \
  configs/symbolic-calibration-v1.json \
  configs/symbolic-calibration-analysis-v1.json \
  configs/symbolic-calibration-canaries-v1.json
uv run infinite-rulebook reproduce \
  configs/symbolic-calibration-v1.json \
  artifacts/symbolic-calibration-v1-serial \
  artifacts/symbolic-calibration-v1-parallel \
  evidence/symbolic-calibration-v1-reproducibility.json \
  --smoke-evidence evidence/pilot-smoke-final-prerequisite.json \
  --workers 4
uv run infinite-rulebook report \
  configs/symbolic-calibration-v1.json \
  configs/symbolic-calibration-analysis-v1.json \
  configs/symbolic-calibration-canaries-v1.json \
  artifacts/symbolic-calibration-v1-parallel \
  results/symbolic-calibration-v1 \
  --reproducibility-report \
  evidence/symbolic-calibration-v1-reproducibility.json \
  --smoke-evidence evidence/pilot-smoke-final-prerequisite.json
uv run infinite-rulebook freeze \
  configs/symbolic-calibration-v1.json \
  results/symbolic-calibration-v1/summary.json \
  configs/symbolic-confirmatory-v1.json \
  configs/symbolic-confirmatory-analysis-v1.json \
  configs/symbolic-confirmatory-canaries-v1.json
uv run infinite-rulebook reproduce \
  configs/symbolic-confirmatory-v1.json \
  artifacts/symbolic-confirmatory-v1-serial \
  artifacts/symbolic-confirmatory-v1-parallel \
  evidence/symbolic-confirmatory-v1-reproducibility.json \
  --workers 4
uv run infinite-rulebook report \
  configs/symbolic-confirmatory-v1.json \
  configs/symbolic-confirmatory-analysis-v1.json \
  configs/symbolic-confirmatory-canaries-v1.json \
  artifacts/symbolic-confirmatory-v1-parallel \
  results/symbolic-confirmatory-v1 \
  --reproducibility-report \
  evidence/symbolic-confirmatory-v1-reproducibility.json
```

If Stage 0 encounters a non-invalidating engineering anomaly, record each one
with a separate `--anomaly "..."` argument on `smoke-evidence`. An invalid
smoke run must be repaired and rerun; it is not converted into a passing
prerequisite by an anomaly entry. The prerequisite is required only on the two
calibration commands above. The calibration report binds it into calibration
evidence, and `freeze` binds that evidence into the confirmatory seal;
confirmatory `reproduce` and `report` therefore reject `--smoke-evidence`
rather than accepting an independent replacement.

Each fresh `reproduce` invocation is a two-root scientific equality check plus
an operational execution declaration. Both artifact roots must be absent or
empty. Before any run starts, the command writes a paired immutable receipt to
each root at
`.infinite-rulebook-reproducibility/execution-receipt.json`. The pair binds a
random invocation identity, the exact config and full current provenance, the
serial/parallel roles, and the declared worker counts. Calibration receipts also
bind the exact authenticated Stage-0 prerequisite hash before either sweep
starts. Fresh mode rejects any preexisting content and any preexisting report
output. Provenance persists the canonical execution-environment fingerprint
preimage—OS/kernel, machine, libc, Python build/compiler/version, byte order,
float metadata, and dependency-lock hash—and rederives its environment digest
during receipt authentication.

An interrupted invocation can be continued explicitly, never implicitly:

```bash
uv run infinite-rulebook reproduce \
  <config> <same-serial-root> <same-parallel-root> <same-report-output> \
  --workers <same-worker-count> \
  --resume
```

Resume acquires a paired invocation lock, requires both original receipts to
match each other and the current config, provenance, roles, and worker count,
then locks each existing tree, removes only implementation-reserved orphan
temporary files left by an interrupted atomic JSON publication, and
scientifically preflights every partial or completed tree before writing any
new scientific artifact. Other unexpected files still fail closed. A
concurrent resume is rejected. A matching completed report output may be
reused; an incompatible output is rejected before the sweeps. If either receipt
is missing/corrupt, the two-root publication was interrupted, or a partial tree
is not safely restartable, use new root names. Publishing two receipts across
two filesystems cannot be one atomic operation; caught publication failures are
rolled back, while a process or power failure inside that narrow boundary
remains fail-closed.

The ordinary `run` and legacy `pilot` paths refuse receipt-bound roots; only the
paired reproducibility workflow is allowed to continue them.

Receipts are self-authenticating integrity records for the trusted CLI/process
model. They prevent accidental whole-root reuse and serial/parallel role swaps;
they are not signatures or independent proof that an operating-system
scheduler used a particular number of threads, and they do not defend against
a writer deliberately regenerating declarations or copying selected subtrees.
Per-run identities and run/frontier scientific-content hashes remain stable
across fresh invocations and worker counts. Receipt hashes, the
reproducibility-report scientific hash, and the raw-inventory scientific hashes
are execution-evidence identities and intentionally change on a fresh
invocation because they bind its random invocation identity and receipt pair.
Downstream smoke and study evidence consequently bind the exact execution that
produced them.

`freeze` fails unless the calibration inventory is complete, both raw roots
reauthenticate, the Stage-0 prerequisite matches, the exact canaries pass, and
the preregistered distribution-free split-sample certification selects a
finite candidate environment count. The 10,000 paired-cluster bootstrap runs
are conditional diagnostics and never select or rescue a candidate. Calibration
superiority and equivalence p-values are descriptive, not continuation gates.
The calibration and confirmatory phase masters are distinct, while the three
algorithm seeds are a fixed preregistered nuisance block shared across phases.
Any post-freeze source, dependency-lock, or execution-environment change
invalidates confirmatory execution instead of silently creating a new study.

The `report` command accepts only the exact serial or parallel root declared by
the authenticated reproducibility report; a content-equivalent third copy is
not an analysis source. It canonicalizes that trusted root before later reads,
then validates every finalized run and referenced frontier before analysis. It
enforces the checked-in inventory, pools algorithm replicas within environment
clusters, inverts reward only after pooling, and emits hashed JSON/Markdown plus
open CSV tables and an accessible deterministic SVG.
It also writes an authenticated release manifest over the exact package
inventory and file bytes. Each package contains canonical copies of its config
and registered plans, an explicit deviation log, and portable serial/parallel
raw-tree inventories with per-tree byte checksums. Inventory schema v2 embeds
and binds the validated execution receipt separately from its run/frontier tree
records. Within those recorded trees, `.run.lock` is the only excluded file.
When a receipt is present, deterministic raw archives include the complete
`.infinite-rulebook-reproducibility` directory as well as the recorded run and
frontier trees, so extracted roots remain exactly re-verifiable after
relocation with `verify-inventory`.
The inference is conditional on the frozen algorithm-seed bank; population
generalization over algorithm seeds requires a later crossed-random-effects
study.

Large raw roots are packaged for the GitHub release with deterministic tar
metadata, gzip headers, and chunks below the per-asset size limit:

```bash
uv run python scripts/package_raw_release.py package \
  configs/symbolic-confirmatory-v1.json \
  results/symbolic-confirmatory-v1/raw-parallel-inventory.json \
  artifacts/symbolic-confirmatory-v1-parallel \
  raw-release/symbolic-confirmatory-v1-parallel \
  symbolic-confirmatory-v1-parallel
uv run python scripts/package_raw_release.py verify \
  raw-release/symbolic-confirmatory-v1-parallel/\
symbolic-confirmatory-v1-parallel.manifest.json
```

The checked-in asset manifest authenticates every ordered chunk and the
reconstructed compressed stream. Concatenating the ordered parts, decompressing
with gzip, and extracting the tar recreates the portable raw root.

## Analysis scale envelope

The retained in-memory analysis dataset is intentionally sized and measured
before the registered run:

```bash
uv run python scripts/benchmark_analysis_scale.py
```

On the reference Python 3.11 Linux run, the exact calibration shape—192
environment replicas, three algorithm replicas, six environments, six agents,
13 checkpoints, and 29 metrics—constructed, model-validated, sorted, and
scientifically hashed 269,568 synthetic observations in 100.075 seconds. The
registered pooling path then produced all 468 checkpoint pools in 6.264
seconds. Peak RSS was 959.31 MiB from a 29.68 MiB baseline (a 929.63 MiB
upper-bound increment). The benchmark shares repeated scientific identities
and frontier records as the real loader does while retaining per-checkpoint
metric tuples. It does not measure raw-inventory traversal, filesystem/JSON
I/O, artifact authentication, `load_run_trees`, bootstrap/statistical
reporting, or output writes, so it is an in-memory analysis benchmark rather
than an end-to-end runtime estimate.

The largest registered confirmation candidate, 512 environment replicas,
contains 718,848 observations. The measured maximum-shape run constructed and
hashed that dataset in 273.404 seconds and pooled all checkpoints in 24.782
seconds; peak RSS was 2,510.97 MiB from a 29.71 MiB baseline. Run
calibration/confirmation reporting with at least 8 GiB of available RAM and
16 GiB recommended. Re-run the benchmark with `--environment-replicas 512` on
the intended reporting host if the calibration selects that upper candidate.
These are operational capacity requirements, not changes to the registered
scientific scope.

Artifact authentication is the remaining operational risk. A report
deliberately makes three content passes while inventorying each of the two raw
roots, then validates and loads the selected root. At the largest registered
candidate this is up to 55,296 runs per root, roughly 3.3 million raw files
before repeated reads, and potentially tens of hours on local storage. This
redundancy is retained because it detects mutation across each authenticated
snapshot.

After the public registration commit and before calibration, generate the
checked-in disjoint-seed performance probe with the exact 36-condition,
horizon-12, 13-checkpoint producer shape, then measure it on the intended
filesystem:

```bash
uv run python -m scripts.run_ingestion_probe \
  configs/symbolic-calibration-v1.json \
  configs/symbolic-artifact-ingestion-probe-v1.json \
  artifacts/symbolic-artifact-ingestion-probe-v1 \
  --workers 4
uv run python scripts/benchmark_artifact_ingestion.py \
  configs/symbolic-artifact-ingestion-probe-v1.json \
  artifacts/symbolic-artifact-ingestion-probe-v1 \
  --projected-runs 55296 \
  --budget-multiplier 2
```

The projection prices two complete inventories plus one selected-root load.
The probe uses the calibration phase so validation includes exact symbolic
replay, but its master/algorithm seed namespaces are excluded from both
registered study phases. The dedicated lower-level runner accepts only the
exact derived probe and deliberately does not create study receipts or
evidence. Its outcomes are discarded and cannot tune the protocol. The
projection separates the four shared frontier trees, whose cost is fixed as
replicas grow, from warm-frontier marginal run validation, raw hashing, and
loading. It models the report's exact two-inventory/one-load pass structure and
conservatively charges any unexplained probe residual per projected run.

On the reference local NVMe host, the public-registration-commit probe produced
36 runs, 468 observations, 1,044 run files, and four unique frontier trees.
One inventory took 78.427 seconds and a selected-root load took 30.534 seconds.
One complete frontier validation set took 24.776 seconds; a cached frontier
copy took 1.350 seconds; the marginal warm-frontier costs were 0.102457 seconds
per run for validation, 0.000744 for raw hashing, and 0.115777 for loading. The
corrected maximum-candidate projection is 8.145 hours, or 16.291 hours with the
registered two-times operational factor. The calibration projection is 3.085
hours, or 6.171 hours with that factor. The earlier naive calculation that
scaled fixed frontier work with every replica is invalid and must not be used.

This remains a planning estimate, not a promise that millions of files scale
perfectly linearly. Reserve a 60-hour reporting window for the maximum
candidate, keep the raw roots on local SSD/NVMe storage, and rerun both
benchmarks on the reporting host. If the measured two-times operational budget
exceeds that window, move the unchanged sealed workflow to faster or
longer-lived infrastructure; do not reduce replicas, checkpoints, metrics, raw
verification, or the scientific scope.

The exact bounded protocol, estimands, margins, power grid, hard gates, and
scope exclusions are registered in
[symbolic-confirmatory-v1.md](symbolic-confirmatory-v1.md).
