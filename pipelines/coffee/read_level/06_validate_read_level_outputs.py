# -*- coding: utf-8 -*-
"""
Validate paired-read synthetic outputs before interpretation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from read_level_common import iter_paired_fastq, sha256_file, write_json


def add_check(
    checks: List[Dict],
    name: str,
    observed,
    expected,
    ok: bool,
    critical: bool = True,
    note: str = "",
) -> None:
    checks.append(
        {
            "name": name,
            "observed": observed,
            "expected": expected,
            "ok": bool(ok),
            "critical": bool(critical),
            "note": note,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_fastq_manifest", required=True)
    parser.add_argument("--truth_manifest", required=True)
    parser.add_argument("--build_audit", required=True)
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument("--summary_dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--expected_scenarios", default="primary,conservative")
    parser.add_argument("--expected_replicates", type=int, default=5)
    parser.add_argument("--expected_modes", default="r1,paired")
    parser.add_argument("--read_pairs_per_sample", type=int, default=6000)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checks: List[Dict] = []

    generated = pd.read_csv(args.generated_fastq_manifest, sep="\t")
    truth = pd.read_csv(args.truth_manifest, sep="\t")
    build_audit = json.loads(Path(args.build_audit).read_text(encoding="utf-8"))
    run_metrics = pd.read_csv(Path(args.analysis_dir) / "run_metrics.tsv", sep="\t")
    master = json.loads(
        (Path(args.summary_dir) / "read_level_master_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    expected_scenarios = sorted(
        [x.strip() for x in args.expected_scenarios.split(",") if x.strip()]
    )
    observed_scenarios = sorted(generated["scenario"].astype(str).unique().tolist())
    add_check(
        checks,
        "Scenario set",
        observed_scenarios,
        expected_scenarios,
        observed_scenarios == expected_scenarios,
    )

    observed_reps = sorted(generated["replicate"].astype(int).unique().tolist())
    expected_reps = list(range(1, args.expected_replicates + 1))
    add_check(
        checks,
        "Replicate set",
        observed_reps,
        expected_reps,
        observed_reps == expected_reps,
    )

    expected_modes = sorted(
        [x.strip() for x in args.expected_modes.split(",") if x.strip()]
    )
    observed_modes = sorted(run_metrics["analysis_mode"].astype(str).unique().tolist())
    add_check(
        checks,
        "Analysis mode set",
        observed_modes,
        expected_modes,
        observed_modes == expected_modes,
    )

    add_check(
        checks,
        "Build audit status",
        build_audit.get("status"),
        "PASS",
        build_audit.get("status") == "PASS",
    )
    add_check(
        checks,
        "Source read-pair reuse",
        build_audit.get("read_reuse_within_run"),
        False,
        build_audit.get("read_reuse_within_run") is False,
    )
    add_check(
        checks,
        "Duplicate source assignments",
        build_audit.get("duplicate_source_pair_assignments"),
        0,
        int(build_audit.get("duplicate_source_pair_assignments", -1)) == 0,
    )

    observed_read_pair_counts = sorted(
        generated["read_pairs"].astype(int).unique().tolist()
    )
    add_check(
        checks,
        "Generated read-pair depth",
        observed_read_pair_counts,
        [int(args.read_pairs_per_sample)],
        observed_read_pair_counts == [int(args.read_pairs_per_sample)],
    )

    merged = generated.merge(
        truth[["scenario", "replicate", "sample_id", "read_pairs", "class_label"]],
        on=["scenario", "replicate", "sample_id", "read_pairs", "class_label"],
        how="outer",
        indicator=True,
    )
    add_check(
        checks,
        "Generated/truth manifest match",
        merged["_merge"].value_counts().to_dict(),
        {"both": len(generated)},
        bool((merged["_merge"] == "both").all()),
    )

    fastq_failures = []
    observed_hash_pairs = []
    for row in generated.itertuples(index=False):
        r1 = Path(str(row.r1_path))
        r2 = Path(str(row.r2_path))
        if not r1.exists() or not r2.exists():
            fastq_failures.append(f"{row.sample_id}: missing FASTQ")
            continue
        actual_pairs = 0
        try:
            for _idx, _a, _b in iter_paired_fastq(r1, r2):
                actual_pairs += 1
        except Exception as exc:
            fastq_failures.append(f"{row.sample_id}: {exc}")
            continue
        if actual_pairs != int(row.read_pairs):
            fastq_failures.append(
                f"{row.sample_id}: expected {int(row.read_pairs)}, observed {actual_pairs}"
            )
        observed_r1 = sha256_file(r1)
        observed_r2 = sha256_file(r2)
        if observed_r1 != str(row.r1_sha256) or observed_r2 != str(row.r2_sha256):
            fastq_failures.append(f"{row.sample_id}: checksum mismatch")
        observed_hash_pairs.append((observed_r1, observed_r2))

    add_check(
        checks,
        "Generated FASTQ integrity",
        fastq_failures[:20],
        [],
        len(fastq_failures) == 0,
        note=f"Total failures: {len(fastq_failures)}",
    )
    add_check(
        checks,
        "Generated library checksum uniqueness",
        len(set(observed_hash_pairs)),
        len(observed_hash_pairs),
        len(set(observed_hash_pairs)) == len(observed_hash_pairs),
    )

    expected_run_count = (
        len(expected_scenarios) * args.expected_replicates * len(expected_modes)
    )
    add_check(
        checks,
        "Analysis run count",
        int(len(run_metrics)),
        expected_run_count,
        int(len(run_metrics)) == expected_run_count,
    )
    add_check(
        checks,
        "Graph connectivity",
        {
            "connected_runs": int(run_metrics["graph_connected"].astype(bool).sum()),
            "total_runs": int(len(run_metrics)),
            "component_counts": sorted(
                run_metrics["graph_components"].astype(int).unique().tolist()
            ),
        },
        "reported",
        True,
        critical=False,
        note=(
            "Connectivity is reported rather than enforced because the neighborhood "
            "parameter is held fixed for comparability."
        ),
    )

    metric_columns = [
        "roc_auc",
        "average_precision",
        "best_f1",
        "best_threshold",
        "tms_entropy_spearman_rho",
    ]
    nonfinite = {}
    for column in metric_columns:
        values = pd.to_numeric(run_metrics[column], errors="coerce").to_numpy(dtype=float)
        bad = int((~np.isfinite(values)).sum())
        if bad:
            nonfinite[column] = bad
    add_check(
        checks,
        "Finite run metrics",
        nonfinite,
        {},
        not nonfinite,
    )

    add_check(
        checks,
        "Master metrics status",
        master.get("status"),
        "COMPLETE",
        master.get("status") == "COMPLETE",
    )

    critical_failures = [
        item for item in checks if item["critical"] and not item["ok"]
    ]
    status = "PASS" if not critical_failures else "FAIL"
    payload = {
        "status": status,
        "critical_failure_count": len(critical_failures),
        "checks": checks,
    }
    write_json(outdir / "read_level_validation_audit.json", payload)

    lines = [f"STATUS: {status}", ""]
    for item in checks:
        marker = "PASS" if item["ok"] else "FAIL"
        lines.append(
            f"[{marker}] {item['name']}: observed={item['observed']}; "
            f"expected={item['expected']} {item['note']}".rstrip()
        )
    (outdir / "read_level_validation_audit.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    if status != "PASS":
        raise SystemExit(1)
    (outdir / "READ_LEVEL_VALIDATION_PASS.txt").write_text(
        "PASS\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
