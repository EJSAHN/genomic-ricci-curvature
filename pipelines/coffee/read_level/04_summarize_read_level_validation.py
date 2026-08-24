# -*- coding: utf-8 -*-
"""
Aggregate paired-read validation results into machine-readable master metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from read_level_common import ensure_dir, write_json


METRIC_COLUMNS = [
    "roc_auc",
    "average_precision",
    "best_f1",
    "best_threshold",
    "tms_entropy_spearman_rho",
    "tms_entropy_spearman_p",
    "tms_minor_fraction_spearman_rho",
    "tms_minor_fraction_spearman_p",
]


def summarize_numeric(
    frame: pd.DataFrame,
    group_columns: List[str],
    metric_columns: List[str],
) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys))
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": float(values.mean()) if len(values) else float("nan"),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(values.median()) if len(values) else float("nan"),
                    "min": float(values.min()) if len(values) else float("nan"),
                    "max": float(values.max()) if len(values) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument("--truth_manifest", required=True)
    parser.add_argument("--build_audit", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--baseline_audit_json", default="")
    parser.add_argument("--primary_scenario", default="primary")
    parser.add_argument("--primary_mode", default="r1")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    outdir = ensure_dir(args.outdir)
    run_metrics = pd.read_csv(analysis_dir / "run_metrics.tsv", sep="\t")
    comparator_metrics = pd.read_csv(analysis_dir / "comparator_metrics.tsv", sep="\t")
    ratio_metrics = pd.read_csv(analysis_dir / "ratio_metrics.tsv", sep="\t")
    node_scores = pd.read_csv(analysis_dir / "node_scores_all.tsv", sep="\t")
    truth = pd.read_csv(args.truth_manifest, sep="\t")
    build_audit = json.loads(Path(args.build_audit).read_text(encoding="utf-8"))

    run_summary = summarize_numeric(
        run_metrics,
        ["scenario", "analysis_mode"],
        METRIC_COLUMNS,
    )
    comparator_summary = summarize_numeric(
        comparator_metrics,
        ["scenario", "analysis_mode", "score_name"],
        ["roc_auc", "average_precision", "best_f1", "best_threshold"],
    )
    ratio_summary = summarize_numeric(
        ratio_metrics,
        ["scenario", "analysis_mode", "pattern_id"],
        ["roc_auc", "average_precision", "best_f1", "mean_tms", "median_tms"],
    )

    primary_runs = run_metrics[
        (run_metrics["scenario"].astype(str) == args.primary_scenario)
        & (run_metrics["analysis_mode"].astype(str) == args.primary_mode)
    ].copy()
    if primary_runs.empty:
        raise SystemExit(
            f"[ERR] Primary result not found: scenario={args.primary_scenario}, "
            f"mode={args.primary_mode}"
        )

    primary = {}
    for metric in METRIC_COLUMNS:
        values = pd.to_numeric(primary_runs[metric], errors="coerce").dropna()
        primary[metric] = {
            "n": int(len(values)),
            "mean": float(values.mean()) if len(values) else None,
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(values.median()) if len(values) else None,
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }

    baseline = None
    if args.baseline_audit_json:
        baseline_path = Path(args.baseline_audit_json)
        if baseline_path.exists():
            baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline = baseline_payload.get("observed", {})
            comparison_rows = []
            synthetic = baseline.get("synthetic", {})
            for metric, baseline_key in [
                ("roc_auc", "roc_auc"),
                ("average_precision", "average_precision"),
                ("best_f1", "best_f1"),
                ("best_threshold", "best_threshold"),
                ("tms_entropy_spearman_rho", "score_entropy_spearman_rho"),
            ]:
                comparison_rows.append(
                    {
                        "metric": metric,
                        "sketch_space_submission_baseline": synthetic.get(baseline_key),
                        "paired_read_primary_mean": primary.get(metric, {}).get("mean"),
                        "paired_read_primary_sd": primary.get(metric, {}).get("sd"),
                    }
                )
            pd.DataFrame(comparison_rows).to_csv(
                outdir / "baseline_vs_paired_read.tsv",
                sep="\t",
                index=False,
            )

    payload = {
        "status": "COMPLETE",
        "primary_scenario": args.primary_scenario,
        "primary_mode": args.primary_mode,
        "primary_metrics": primary,
        "n_run_combinations": int(len(run_metrics)),
        "scenarios": sorted(run_metrics["scenario"].astype(str).unique().tolist()),
        "analysis_modes": sorted(run_metrics["analysis_mode"].astype(str).unique().tolist()),
        "replicates_per_group": {
            f"{scenario}|{mode}": int(len(group))
            for (scenario, mode), group in run_metrics.groupby(
                ["scenario", "analysis_mode"], sort=True
            )
        },
        "build_audit": build_audit,
        "submission_baseline": baseline,
    }
    write_json(outdir / "read_level_master_metrics.json", payload)

    run_summary.to_csv(outdir / "run_summary.tsv", sep="\t", index=False)
    comparator_summary.to_csv(
        outdir / "comparator_summary.tsv", sep="\t", index=False
    )
    ratio_summary.to_csv(outdir / "ratio_summary.tsv", sep="\t", index=False)

    workbook_path = outdir / "read_level_master_metrics.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        run_summary.to_excel(writer, sheet_name="run_summary", index=False)
        run_metrics.to_excel(writer, sheet_name="run_metrics", index=False)
        comparator_summary.to_excel(writer, sheet_name="comparator_summary", index=False)
        comparator_metrics.to_excel(writer, sheet_name="comparator_runs", index=False)
        ratio_summary.to_excel(writer, sheet_name="ratio_summary", index=False)
        ratio_metrics.to_excel(writer, sheet_name="ratio_runs", index=False)
        truth.to_excel(writer, sheet_name="truth_manifest", index=False)
        node_scores.to_excel(writer, sheet_name="node_scores", index=False)

    text_lines = [
        "Paired-read synthetic validation",
        "================================",
        "",
        f"Primary scenario: {args.primary_scenario}",
        f"Primary analysis mode: {args.primary_mode}",
        f"Replicates: {len(primary_runs)}",
        "",
    ]
    for metric in [
        "roc_auc",
        "average_precision",
        "best_f1",
        "tms_entropy_spearman_rho",
    ]:
        values = primary[metric]
        text_lines.append(
            f"{metric}: mean={values['mean']:.6f}, sd={values['sd']:.6f}, "
            f"range={values['min']:.6f}–{values['max']:.6f}"
        )
    (outdir / "READ_LEVEL_RESULTS_SUMMARY.txt").write_text(
        "\n".join(text_lines) + "\n",
        encoding="utf-8",
    )

    print(f"[DONE] Master JSON: {outdir / 'read_level_master_metrics.json'}")
    print(f"[DONE] Master workbook: {workbook_path}")
    print(f"[DONE] Summary: {outdir / 'READ_LEVEL_RESULTS_SUMMARY.txt'}")


if __name__ == "__main__":
    main()
