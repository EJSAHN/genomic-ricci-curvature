# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crossfit_common import SCORE_VARIANTS, ensure_dir, evaluate_binary, spearman_safe, write_json


def top_jaccard(frame: pd.DataFrame, a: str, b: str, n: int = 5) -> float:
    sa = set(frame.nlargest(n, a)["sample_id"].astype(str))
    sb = set(frame.nlargest(n, b)["sample_id"].astype(str))
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossfit_scores", required=True)
    ap.add_argument("--crossfit_pass", required=True)
    ap.add_argument("--reference_free_batch_nodes", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    if not Path(args.crossfit_pass).is_file():
        raise SystemExit("[ERROR] Cross-fit scoring PASS marker is absent")
    outdir = ensure_dir(args.outdir)
    scores = pd.read_csv(args.crossfit_scores, sep="\t")
    rf = pd.read_csv(args.reference_free_batch_nodes, sep="\t")
    rf = rf[rf["analysis_mode"].astype(str) == "paired"].copy()
    if len(scores) != 560 or len(rf) != 560:
        raise SystemExit(f"Expected 560 score rows and 560 paired reference-free rows; got {len(scores)} and {len(rf)}")

    keep_rf = [
        "sample_id","replicate","class_label","category","pattern_id","is_mixture",
        "tms","betweenness","negative_orc_incidence","mean_incident_distance",
        "real_bridge_score","pca_distance","lof_score",
    ]
    merged = scores.merge(
        rf[keep_rf],
        on=["sample_id","replicate","class_label"],
        how="inner",
        validate="one_to_one",
        suffixes=("","_rf"),
    )
    if len(merged) != 560:
        raise SystemExit(f"Reference/reference-free merge yielded {len(merged)} rows")
    merged.to_csv(outdir / "generated_reference_vs_reference_free_scores.tsv", sep="\t", index=False)

    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for replicate, group in merged.groupby("replicate", sort=True):
        y = (group["class_label"].astype(str) == "synthetic_mixture").astype(int).to_numpy()
        tms_metrics = evaluate_binary(y, group["tms"].to_numpy(float))
        metric_rows.append({"replicate":int(replicate),"score_name":"reference_free_tms",**tms_metrics})
        for variant in SCORE_VARIANTS:
            metrics = evaluate_binary(y, group[variant].to_numpy(float))
            rho, p = spearman_safe(group[variant], group["tms"])
            metric_rows.append({"replicate":int(replicate),"score_name":variant,**metrics})
            comparison_rows.append({
                "replicate": int(replicate),
                "score_name": variant,
                "reference_roc_auc": metrics["roc_auc"],
                "reference_average_precision": metrics["average_precision"],
                "tms_roc_auc": tms_metrics["roc_auc"],
                "tms_average_precision": tms_metrics["average_precision"],
                "reference_minus_tms_auc": metrics["roc_auc"] - tms_metrics["roc_auc"],
                "score_tms_spearman_rho": rho,
                "score_tms_spearman_p": p,
                "top5_jaccard": top_jaccard(group, variant, "tms", n=5),
            })

        controls = group[group["class_label"].astype(str) == "single_source_control"]
        for category in sorted(c for c in group["category"].dropna().astype(str).unique() if c != "control"):
            subset = pd.concat([controls, group[group["category"].astype(str) == category]], ignore_index=True)
            yy = (subset["class_label"].astype(str) == "synthetic_mixture").astype(int).to_numpy()
            for variant in ["reference_qc_crossfit","reference_qc_crossfit_no_pca","reference_mapping_crossfit","reference_marker_crossfit","tms"]:
                category_rows.append({
                    "replicate": int(replicate),
                    "category": category,
                    "score_name": variant,
                    **evaluate_binary(yy, subset[variant].to_numpy(float)),
                })

    metrics_df = pd.DataFrame(metric_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    category_df = pd.DataFrame(category_rows)
    metrics_df.to_csv(outdir / "batch_reference_and_tms_metrics.tsv", sep="\t", index=False)
    comparison_df.to_csv(outdir / "batch_reference_vs_tms_comparison.tsv", sep="\t", index=False)
    category_df.to_csv(outdir / "batch_reference_category_metrics.tsv", sep="\t", index=False)

    summary_rows = []
    for score_name, group in metrics_df.groupby("score_name", sort=True):
        summary_rows.append({
            "score_name": score_name,
            "replicate_count": int(group["replicate"].nunique()),
            "roc_auc_mean": float(group["roc_auc"].mean()),
            "roc_auc_sd": float(group["roc_auc"].std(ddof=0)),
            "roc_auc_min": float(group["roc_auc"].min()),
            "roc_auc_max": float(group["roc_auc"].max()),
            "average_precision_mean": float(group["average_precision"].mean()),
            "average_precision_sd": float(group["average_precision"].std(ddof=0)),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "batch_reference_and_tms_summary.tsv", sep="\t", index=False)
    write_json({"status":"PASS","merged_rows":len(merged),"metric_rows":len(metrics_df),"comparison_rows":len(comparison_df)}, outdir / "batch_comparison_audit.json")
    (outdir / "BATCH_COMPARISON_PASS.txt").write_text("PASS\n", encoding="utf-8")
    print("[DONE] Cross-fitted reference scores compared with locked paired reference-free batch scores.")


if __name__ == "__main__":
    main()
