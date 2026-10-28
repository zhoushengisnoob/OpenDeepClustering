"""Regenerate reported aggregates from a committed benchmark record.

The record under docs/benchmarks/records/ holds every per-seed metric, so each
table in the corresponding report can be rebuilt without server access. Run
with --check to confirm the stored aggregates match a fresh recomputation from
the per-seed values.

    python tools/report_benchmark_record.py \
        docs/benchmarks/records/mnist-20seed-2026-09-18.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

METRICS = ("acc", "f1", "nmi", "ari")
TOLERANCE = 1e-9

T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
}


def recompute(values: list[float]) -> dict:
    mean = st.mean(values)
    sample_sd = st.stdev(values) if len(values) > 1 else 0.0
    half_width = (
        T_CRITICAL_95.get(len(values) - 1, 1.96) * sample_sd / math.sqrt(len(values))
        if len(values) > 1 else 0.0
    )
    return {
        "mean": mean,
        "std_population": st.pstdev(values),
        "std_sample": sample_sd,
        "median": st.median(values),
        "min": min(values),
        "max": max(values),
        "ci95_half_width": half_width,
        "ci95": [mean - half_width, mean + half_width],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument(
        "--check", action="store_true",
        help="Verify stored aggregates match recomputation and exit non-zero on drift.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.record.read_text(encoding="utf-8"))
    problems: list[str] = []

    print(f"Record: {args.record}")
    print(f"Source: {manifest['source_directory']}")
    print(f"Files audited: {manifest['files_audited']}")
    stored_problems = manifest.get("integrity_problems", [])
    print(f"Integrity problems recorded at audit time: {len(stored_problems)}")
    for problem in stored_problems:
        print("  -", problem)
    print()

    for estimator, block in sorted(manifest["estimators"].items()):
        per_seed = block["per_seed"]
        seeds = [entry["seed"] for entry in per_seed]
        print(f"## {estimator}")
        print(f"  runs: {len(per_seed)}  seeds: {min(seeds)}-{max(seeds)}")
        print(f"  commit: {block['git_sha']}")
        print(f"  torch: {block['environment']['torch']}  "
              f"python: {block['environment']['python']}")
        print(f"  dataset: {block['dataset']['samples']} x "
              f"{block['dataset']['features']}  "
              f"features sha256: {block['dataset']['features_sha256'][:16]}...")
        print(f"  stop reasons: {block['stop_reasons']}")
        print(f"  distinct label digests: {block['distinct_label_digests']}"
              f"/{len(per_seed)}")

        if len(set(seeds)) != len(seeds):
            problems.append(f"{estimator}: duplicate seeds")
        if block["n_runs"] != len(per_seed):
            problems.append(
                f"{estimator}: n_runs {block['n_runs']} != {len(per_seed)} records"
            )
        if block["distinct_label_digests"] != len(per_seed):
            problems.append(
                f"{estimator}: only {block['distinct_label_digests']} distinct "
                f"label digests across {len(per_seed)} runs"
            )

        for metric in METRICS:
            values = [entry["metrics"][metric] for entry in per_seed]
            fresh = recompute(values)
            stored = block["aggregates"][metric]
            for key, value in fresh.items():
                reference = stored[key]
                if isinstance(value, list):
                    drift = max(abs(a - b) for a, b in zip(value, reference))
                else:
                    drift = abs(value - reference)
                if drift > TOLERANCE:
                    problems.append(
                        f"{estimator}.{metric}.{key}: stored {reference} != "
                        f"recomputed {value}"
                    )
            print("  %-4s %.4f +- %.4f (sample sd)  median %.4f  "
                  "95%% CI [%.4f, %.4f]" % (
                      metric, fresh["mean"], fresh["std_sample"], fresh["median"],
                      fresh["ci95"][0], fresh["ci95"][1]))
        print()

    if problems:
        print(f"FAILED: {len(problems)} discrepancies")
        for problem in problems:
            print("  -", problem)
        return 1

    print("All stored aggregates match recomputation from per-seed records.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
