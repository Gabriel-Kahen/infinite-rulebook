# Symbolic v2 operational preflight

This is a non-scientific operations record for public registration commit
`3909db8b9913fc0031710b9f4d4803fad00dd0bf`. It was produced only after
[PR #11](https://github.com/Gabriel-Kahen/infinite-rulebook/pull/11) merged and
its [post-merge CI](https://github.com/Gabriel-Kahen/infinite-rulebook/actions/runs/30548156462)
passed on Python 3.11, 3.12, and 3.13.

No registered v2 pilot, calibration, freeze, confirmation, or outcome analysis
was run. The disjoint-seed ingestion probe was used only for artifact timing;
its metrics were not inspected and cannot tune the protocol. The complete
machine-readable record is
[`benchmarks/symbolic-v2-operational-preflight.json`](../benchmarks/symbolic-v2-operational-preflight.json).

## Results

| Gate | Exact result | Decision |
|---|---|---|
| E192 analysis capacity | 958,464 observations; 624 pools; 3,515.719 MiB peak RSS; 370.742 s | Passed: 20,297,744,384 bytes remained physically available at the high-water sample, with zero process major faults and zero host swap-outs. |
| E768 maximum capacity | 3,833,856 observations; 624 pools; 13,747.113 MiB peak RSS; 1,538.121 s | Failed. Available physical memory fell below peak RSS and the process incurred at least 443,862 major faults after eviction. |
| V2 ingestion execution | Exact 48-run config `9120115e05a126127ce639169b15a27f9e39c09c7e277301f2e4dedd189de9aa` | Passed execute, artifact validation, and authenticated loading. |
| Maximum report-ingestion projection | 46.314 h projected; 92.628 h with the required 2× factor | Requires a longer-lived reporting host plus interruption-recovery margin. |

The capacity dataset hashes are
`f8eb1d5d130f283959c469049ea5ec1e53dbfc053e656e0c4817ae8c1534d1ee`
at E192 and
`b4de258dc8e72d78e5324c072dbfc7fab19f4534d40aaf6175c3e596873cf58a`
at E768. Authenticated probe loading produced dataset hash
`881f1c3ac959baedb22ab961ad35aaa2654ec092407da725de79fe858265a043`.

## Decision

This 32-GiB-class workstation (33,561,464,832 physical bytes, or 31.257 GiB)
is not approved for registered v2 calibration or confirmation. The unchanged
workflow must move to an intended reporting host with:

- at least 64 GiB physical RAM;
- local SSD/NVMe capacity for both complete raw roots plus reports;
- its own newly measured two-times report-ingestion projection plus
  interruption-recovery margin; and
- no swap dependence while retaining at least peak RSS again as free physical
  memory.

Both exact capacity benchmarks and the ingestion probe must be rerun on that
host before calibration. Both capacity benchmarks must be repeated again before
the selected confirmation. The 92.628-hour local projection is a reference,
not a fixed budget for a different host. Replicas, checkpoints, metrics,
authentication passes, and scientific scope must not be reduced.

## Probe disposition

The completed probe root contained 48 runs, four shared frontiers, 624 loaded
observations, 1,560 files, and 22,868,963 apparent bytes. After the benchmark,
it was moved to the desktop trash as
`symbolic-artifact-ingestion-probe-v2`; it remains recoverable but is neither
tracked nor evidence. Only structural identities and timing measurements are
recorded here.

## Exact commands

```bash
uv run python scripts/benchmark_analysis_scale.py \
  --config configs/symbolic-calibration-v2.json \
  --metrics 29 \
  --positive-checkpoint-metrics 31

uv run python scripts/benchmark_analysis_scale.py \
  --config configs/symbolic-calibration-v2.json \
  --environment-replicas 768 \
  --metrics 29 \
  --positive-checkpoint-metrics 31

uv run python -m scripts.run_ingestion_probe \
  configs/symbolic-calibration-v2.json \
  configs/symbolic-artifact-ingestion-probe-v2.json \
  artifacts/symbolic-artifact-ingestion-probe-v2 \
  --workers 4

uv run python scripts/benchmark_artifact_ingestion.py \
  configs/symbolic-artifact-ingestion-probe-v2.json \
  artifacts/symbolic-artifact-ingestion-probe-v2 \
  --projected-runs 294912 \
  --budget-multiplier 2
```
