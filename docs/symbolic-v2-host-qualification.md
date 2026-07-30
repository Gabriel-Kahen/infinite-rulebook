# Symbolic v2 reporting-host qualification

This runbook qualifies an intended reporting host without invoking a registered
study. The qualification records are operational evidence only: they cannot be
used as outcomes, power observations, or a substitute for independent review.

The local 32-GiB-class workstation is already rejected for v2. Do not run E768,
calibration, or confirmation there.

## Safety boundary

`scripts/qualify_symbolic_v2_host.py` has seven commands:

- `plan` prints the exact E192/E768 commands, the underlying disjoint-seed
  probe commands, and the qualifying `run-probe` command templates;
- `inspect` performs read-only static checks and writes optional operational
  JSON;
- `run-capacity` can launch only
  `scripts/benchmark_analysis_scale.py`, which creates a synthetic in-memory
  dataset and no study artifacts;
- `run-probe` explicitly launches one probe step through an inherited,
  no-follow directory descriptor and immediately seals its result;
- `bind-probe` can preserve imported plain probe JSON, but deliberately marks
  it nonqualifying because its write path was not descriptor-anchored;
- `assess` combines only the strict hash-sealed operational records; and
- `verify-assessment` recomputes an assessment from all five records it names.

The tool has no path that invokes `infinite-rulebook run`, `reproduce`,
`report`, or `freeze`. `run-probe` is the only path that launches the
disjoint-seed ingestion probe; it hashes artifact bytes and paths for identity
without interpreting metrics or scientific outcomes. E768 additionally
requires an explicit memory-pressure acknowledgement. Capacity commands
require the same fresh static record and recheck the approved commit, clean
tree, lockfile, checkout Python, absolute `uv` executable, tool source,
machine, and boot before launch, plus the checkout snapshot after the detached
process. Any process-monitoring exception terminates and reaps the entire
process group.

Linux `MemTotal` can be below installed capacity because of firmware and kernel
reservations. The 64-GiB check is deliberately fail-closed on visible physical
memory; a nominal 64-GiB machine may therefore fail. A 96- or 128-GiB host is
operationally safer.

Even a passing assessment sets
`registered_execution_authorized_by_this_record` to `false`. A reviewer must
approve the evidence before a separately invoked calibration.

## 1. Pin separate tool and execution checkouts

The scientific execution checkout is fixed to the reviewed public commit
`c9b6297b63b572d9e6d106de4add1dae436c00d3`. The tool rejects every other
execution SHA. Run the qualification tool from its own reviewed public
checkout so adding operational tooling does not change the approved scientific
execution tree.

Keep operational records outside both Git worktrees and outside the qualified
storage/probe tree. Record writes reject protected roots and symlinked path
components and install complete files without following the final path:

```bash
export IRB_EXECUTION_SHA="c9b6297b63b572d9e6d106de4add1dae436c00d3"
export IRB_TOOL_SHA="<full-reviewed-host-tool-merge-commit>"
export IRB_EXECUTION_REPOSITORY="/srv/infinite-rulebook-execution"
export IRB_TOOL_REPOSITORY="/srv/infinite-rulebook-host-tool"
export IRB_STORAGE_ROOT="/mnt/local-nvme/infinite-rulebook"
export IRB_OPERATIONS_ROOT="/srv/irb-host-qualification/${IRB_EXECUTION_SHA}"
export IRB_PROBE_ROOT="${IRB_STORAGE_ROOT}/qualification/probe-${IRB_EXECUTION_SHA}"

git clone https://github.com/Gabriel-Kahen/infinite-rulebook.git \
  "${IRB_EXECUTION_REPOSITORY}"
git -C "${IRB_EXECUTION_REPOSITORY}" checkout --detach "${IRB_EXECUTION_SHA}"
test "$(git -C "${IRB_EXECUTION_REPOSITORY}" rev-parse HEAD)" = \
  "${IRB_EXECUTION_SHA}"
test -z "$(git -C "${IRB_EXECUTION_REPOSITORY}" status \
  --porcelain --untracked-files=normal)"
(
  cd "${IRB_EXECUTION_REPOSITORY}"
  uv sync --frozen --all-groups
)

git clone https://github.com/Gabriel-Kahen/infinite-rulebook.git \
  "${IRB_TOOL_REPOSITORY}"
git -C "${IRB_TOOL_REPOSITORY}" checkout --detach "${IRB_TOOL_SHA}"
test "$(git -C "${IRB_TOOL_REPOSITORY}" rev-parse HEAD)" = "${IRB_TOOL_SHA}"
(
  cd "${IRB_TOOL_REPOSITORY}"
  uv sync --frozen --all-groups
)
mkdir -p "${IRB_OPERATIONS_ROOT}" "${IRB_STORAGE_ROOT}/qualification"
cd "${IRB_TOOL_REPOSITORY}"
```

The static record is valid for at most 24 hours and only on the same machine
and boot. All later records must use the same tool-source identity and static
record hash.

## 2. Declare storage and time reservations

The first local probe measured a paired-raw-root reference of
171,002,155,008 bytes and 17,104,896 run files at the maximum candidate. Those
are lower bounds, not a complete storage reservation. The intended filesystem
also needs report output, compact-canary detail chunks, filesystem metadata,
temporary publication space, and interruption-recovery headroom.

Declare positive margins beyond the fixed paired-raw references:

```bash
export IRB_ADDITIONAL_STORAGE_BYTES="<bytes-for-reports-staging-and-headroom>"
export IRB_ADDITIONAL_INODES="<inodes-for-reports-staging-and-headroom>"
export IRB_AVAILABLE_WINDOW_HOURS="<reserved-host-window-hours>"
export IRB_RECOVERY_MARGIN_HOURS="<additional-recovery-hours>"
```

The tool requires both margins to be positive and checks available resources
against each reference plus its margin. As conservative operations guidance,
not a frozen scientific threshold, prefer at least 500 GB free local NVMe and
at least 20,000,000 additional free inodes beyond the paired-raw estimate.
Larger staging or snapshot policies need larger margins. The tool cannot judge
whether the operator's allowances are scientifically prudent.

## 3. Inspect the host and print the exact plan

```bash
uv run python scripts/qualify_symbolic_v2_host.py inspect \
  --execution-commit "${IRB_EXECUTION_SHA}" \
  --repo-root "${IRB_EXECUTION_REPOSITORY}" \
  --storage-root "${IRB_STORAGE_ROOT}" \
  --additional-storage-bytes "${IRB_ADDITIONAL_STORAGE_BYTES}" \
  --additional-inodes "${IRB_ADDITIONAL_INODES}" \
  --probe-root "${IRB_PROBE_ROOT}" \
  --output "${IRB_OPERATIONS_ROOT}/host.json"

uv run python scripts/qualify_symbolic_v2_host.py plan \
  --execution-commit "${IRB_EXECUTION_SHA}" \
  --repo-root "${IRB_EXECUTION_REPOSITORY}" \
  --probe-root "${IRB_PROBE_ROOT}" \
  > "${IRB_OPERATIONS_ROOT}/plan.json"
```

`inspect` exits with status 2 if any static prerequisite fails. In particular,
the storage mount must be a locally mounted SSD/NVMe device; unknown,
networked, rotational, pseudo, or container-overlay storage does not pass.
Move the unchanged workflow to another host rather than overriding a failed
gate.

## 4. Run the two synthetic capacity gates

Run E192 first and review its record before acknowledging E768:

```bash
uv run python scripts/qualify_symbolic_v2_host.py run-capacity \
  --stage e192 \
  --host-record "${IRB_OPERATIONS_ROOT}/host.json" \
  --timeout-hours 2 \
  --output "${IRB_OPERATIONS_ROOT}/e192.json"

uv run python scripts/qualify_symbolic_v2_host.py run-capacity \
  --stage e768 \
  --acknowledge-e768-synthetic-memory-pressure \
  --host-record "${IRB_OPERATIONS_ROOT}/host.json" \
  --timeout-hours 4 \
  --output "${IRB_OPERATIONS_ROOT}/e768.json"
```

Each record binds the exact static record, tool source, machine and boot,
absolute `uv`, checkout Python, lockfile, and pre/post checkout snapshots. It
checks the exact registered shape and deterministic synthetic dataset hash.
Passing requires:

- the exact expected observations, pools, metrics, and replica count;
- no process major faults and no host swap-outs during the benchmark; and
- at least the measured peak RSS remaining again as physical `MemAvailable`.

A timeout, output-limit breach, process failure, shape mismatch, swap
dependence, reserve failure, or post-run checkout change is a failed gate.
The default hard timeout is two hours; `--timeout-hours` may set another finite,
positive limit. Monitoring intervals cannot exceed one second and never sleep
past the hard deadline.

## 5. Run the exact disjoint-seed ingestion probe

Only after both capacity records pass, fill in and invoke the two qualifying
`run-probe` templates from the tool checkout. The wrapper precreates and opens
the final artifact root without following symlinks. The child inherits that
descriptor, changes its working directory to the open inode, and passes `.`
to the approved probe module; this remains compatible with the artifact
layer's own no-follow component walker. Do not run the underlying printed
commands directly.

```bash
cd "${IRB_TOOL_REPOSITORY}"
uv run python scripts/qualify_symbolic_v2_host.py run-probe \
  --kind execution \
  --host-record "${IRB_OPERATIONS_ROOT}/host.json" \
  --timeout-hours 24 \
  --output "${IRB_OPERATIONS_ROOT}/probe-execution-record.json"

uv run python scripts/qualify_symbolic_v2_host.py run-probe \
  --kind benchmark \
  --host-record "${IRB_OPERATIONS_ROOT}/host.json" \
  --probe-execution-record \
    "${IRB_OPERATIONS_ROOT}/probe-execution-record.json" \
  --timeout-hours 24 \
  --output "${IRB_OPERATIONS_ROOT}/probe-benchmark-record.json"
```

Do not open run payloads, plot conditions, compare agents, or use probe metrics
for protocol decisions. `run-probe` records the no-symlink device/inode
identity, hashes every regular artifact file through the open directory
descriptor, and requires the execution and benchmark manifests to match. It
binds the benchmark to the execution-record hash, expected config/command,
approved checkout, tool source, machine, and boot. A plain or manually bound
probe JSON object cannot pass assessment. Each record distinguishes the
canonical planned command from the descriptor-anchored argv actually executed.
A failed execution may leave a partial root; do not reuse it—perform a fresh
static inspection with a new, absent probe root. The probe root is neither
study evidence nor a registered raw root.

## 6. Assess and review

```bash
cd "${IRB_TOOL_REPOSITORY}"
uv run python scripts/qualify_symbolic_v2_host.py assess \
  --host-record "${IRB_OPERATIONS_ROOT}/host.json" \
  --e192-record "${IRB_OPERATIONS_ROOT}/e192.json" \
  --e768-record "${IRB_OPERATIONS_ROOT}/e768.json" \
  --probe-execution-record \
    "${IRB_OPERATIONS_ROOT}/probe-execution-record.json" \
  --probe-benchmark-record \
    "${IRB_OPERATIONS_ROOT}/probe-benchmark-record.json" \
  --available-window-hours "${IRB_AVAILABLE_WINDOW_HOURS}" \
  --recovery-margin-hours "${IRB_RECOVERY_MARGIN_HOURS}" \
  --output "${IRB_OPERATIONS_ROOT}/assessment.json"

uv run python scripts/qualify_symbolic_v2_host.py verify-assessment \
  --assessment-record "${IRB_OPERATIONS_ROOT}/assessment.json" \
  --host-record "${IRB_OPERATIONS_ROOT}/host.json" \
  --e192-record "${IRB_OPERATIONS_ROOT}/e192.json" \
  --e768-record "${IRB_OPERATIONS_ROOT}/e768.json" \
  --probe-execution-record \
    "${IRB_OPERATIONS_ROOT}/probe-execution-record.json" \
  --probe-benchmark-record \
    "${IRB_OPERATIONS_ROOT}/probe-benchmark-record.json"
```

The assessment requires the static, E192, E768, probe-execution, and
probe-benchmark records to be ordered, no more than 24 hours old, and bound to
the same approved execution SHA, static-record hash, tool source, host, boot,
and artifact directory. The reserved window must cover that host's newly
measured two-times ingestion projection plus the separately declared recovery
margin. A record checksum alone is not an authenticity mechanism; reviewers
must use `verify-assessment` with the five immutable input records rather than
trusting stored check booleans. Preserve the operational JSON for review,
quarantine the probe root
outside all study, evidence, result, and raw-release paths, and do not invoke
calibration until the review is complete.

Before a selected confirmation is executed, repeat both exact capacity
benchmarks at the same pinned execution commit and review the fresh records.
The probe need not be reused as scientific evidence and must never be promoted
into a registered artifact root.
