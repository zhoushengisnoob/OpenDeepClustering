"""Audit the 60-run MNIST reference batch and emit a compact committed record.

Verifies seed completeness, configuration consistency and provenance agreement
across every per-seed JSON, then writes one machine-readable manifest that is
small enough to commit and from which every reported aggregate can be
regenerated without server access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

METRICS = ("acc", "f1", "nmi", "ari")
EXPECTED_SEEDS = set(range(20))


def t_critical_95(dof: int) -> float:
    """Two-sided 95% Student-t critical value for small samples."""
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086,
    }
    return table.get(dof, 1.96)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.results.glob("*.json"))
    groups: dict[str, list] = defaultdict(list)
    problems: list[str] = []

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        estimator = payload["config"]["estimator"]
        runs = payload.get("runs", [])
        if len(runs) != 1:
            problems.append(f"{path.name}: expected 1 run, found {len(runs)}")
            continue
        run = runs[0]
        groups[estimator].append(
            {
                "file": path.name,
                "seed": run["seed"],
                "metrics": {metric: run["metrics"][metric] for metric in METRICS},
                "duration_seconds": run["duration_seconds"],
                "n_iter": run.get("n_iter"),
                "converged": run.get("converged"),
                "stop_reason": run.get("stop_reason"),
                "labels_sha256": run["labels_sha256"],
                "git_sha": payload.get("git_sha"),
                "features_sha256": payload["dataset"]["features_sha256"],
                "labels_digest": payload["dataset"]["labels_sha256"],
                "samples": payload["dataset"]["samples"],
                "features": payload["dataset"]["features"],
                "environment": payload["environment"],
                "parameters": payload["config"].get("parameters", {}),
                "dataset_config": payload["config"]["dataset"],
                "schema_version": payload.get("schema_version"),
            }
        )

    manifest: dict = {
        "record_version": 1,
        "source_directory": str(args.results),
        "estimators": {},
    }

    for estimator in sorted(groups):
        entries = sorted(groups[estimator], key=lambda item: item["seed"])
        seeds = [entry["seed"] for entry in entries]

        if len(set(seeds)) != len(seeds):
            problems.append(f"{estimator}: duplicate seeds {seeds}")
        missing = EXPECTED_SEEDS - set(seeds)
        if missing:
            problems.append(f"{estimator}: missing seeds {sorted(missing)}")

        # Every run of one estimator must share provenance and configuration.
        for field in ("git_sha", "features_sha256", "labels_digest", "samples",
                      "features", "schema_version"):
            values = {json.dumps(entry[field], sort_keys=True) for entry in entries}
            if len(values) != 1:
                problems.append(f"{estimator}: inconsistent {field}: {sorted(values)}")
        for field in ("parameters", "dataset_config", "environment"):
            values = {json.dumps(entry[field], sort_keys=True) for entry in entries}
            # random_state legitimately varies with the seed.
            if field == "parameters":
                values = {
                    json.dumps(
                        {k: v for k, v in json.loads(value).items()
                         if k != "random_state"},
                        sort_keys=True,
                    )
                    for value in values
                }
            if len(values) != 1:
                problems.append(f"{estimator}: inconsistent {field}")

        # Distinct label digests confirm the seeds produced distinct solutions.
        label_digests = {entry["labels_sha256"] for entry in entries}

        aggregates = {}
        for metric in METRICS:
            values = [entry["metrics"][metric] for entry in entries]
            mean = st.mean(values)
            sample_sd = st.stdev(values) if len(values) > 1 else 0.0
            half_width = (
                t_critical_95(len(values) - 1) * sample_sd / math.sqrt(len(values))
                if len(values) > 1 else 0.0
            )
            aggregates[metric] = {
                "mean": mean,
                "std_population": st.pstdev(values),
                "std_sample": sample_sd,
                "median": st.median(values),
                "min": min(values),
                "max": max(values),
                "ci95_half_width": half_width,
                "ci95": [mean - half_width, mean + half_width],
            }

        durations = [entry["duration_seconds"] for entry in entries]
        reference = entries[0]
        manifest["estimators"][estimator] = {
            "n_runs": len(entries),
            "seeds": seeds,
            "git_sha": reference["git_sha"],
            "schema_version": reference["schema_version"],
            "dataset": {
                "samples": reference["samples"],
                "features": reference["features"],
                "features_sha256": reference["features_sha256"],
                "labels_sha256": reference["labels_digest"],
                "config": reference["dataset_config"],
            },
            "environment": reference["environment"],
            "parameters": reference["parameters"],
            "duration_seconds": {
                "mean": st.mean(durations),
                "median": st.median(durations),
                "total": sum(durations),
            },
            "stop_reasons": {
                reason: sum(1 for e in entries if e["stop_reason"] == reason)
                for reason in sorted({e["stop_reason"] for e in entries})
            },
            "converged": sorted({e["converged"] for e in entries}),
            "distinct_label_digests": len(label_digests),
            "aggregates": aggregates,
            "per_seed": [
                {
                    "seed": entry["seed"],
                    "metrics": entry["metrics"],
                    "duration_seconds": entry["duration_seconds"],
                    "n_iter": entry["n_iter"],
                    "stop_reason": entry["stop_reason"],
                    "labels_sha256": entry["labels_sha256"],
                }
                for entry in entries
            ],
        }

    manifest["files_audited"] = len(files)
    manifest["integrity_problems"] = problems

    serialized = json.dumps(manifest, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")

    print("files audited:", len(files))
    print("integrity problems:", len(problems))
    for problem in problems:
        print("  -", problem)
    print()
    header = "%-20s n  %-24s %-24s %-10s"
    print(header % ("estimator", "acc mean [95% CI]", "nmi mean [95% CI]", "median min"))
    for estimator, block in sorted(manifest["estimators"].items()):
        acc = block["aggregates"]["acc"]
        nmi = block["aggregates"]["nmi"]
        print("%-20s %-2d %.4f [%.4f,%.4f]  %.4f [%.4f,%.4f]  %.2f" % (
            estimator, block["n_runs"],
            acc["mean"], acc["ci95"][0], acc["ci95"][1],
            nmi["mean"], nmi["ci95"][0], nmi["ci95"][1],
            block["duration_seconds"]["median"] / 60,
        ))
        print("     distinct label digests: %d/%d  stop_reasons: %s" % (
            block["distinct_label_digests"], block["n_runs"], block["stop_reasons"]))
    print()
    print("manifest sha256:", hashlib.sha256(serialized.encode()).hexdigest())
    print("manifest bytes:", len(serialized))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
