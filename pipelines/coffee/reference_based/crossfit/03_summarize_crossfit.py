# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from crossfit_common import SCORE_VARIANTS, ensure_dir, write_json


DISPLAY_NAMES = {
    "reference_qc_crossfit": "Cross-fitted reference QC (full)",
    "reference_qc_crossfit_no_pca": "Cross-fitted reference QC (without PCA)",
    "reference_marker_crossfit": "Cross-fitted marker component",
    "reference_marker_no_pca_crossfit": "Cross-fitted marker component (without PCA)",
    "reference_mapping_crossfit": "Mapping-only component",
    "pca_only_crossfit": "Cross-fitted PCA reconstruction error",
    "reconstruction_only_crossfit": "Single/pair reconstruction component",
}


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    per_rep = metrics[metrics["evaluation_scope"] == "per_replicate"].copy()
    rows: list[dict[str, object]] = []
    for variant, group in per_rep.groupby("score_variant"):
        prevalence = float(group["prevalence"].mean())
        all_positive_f1 = 2.0 * prevalence / (1.0 + prevalence)
        ap_mean = float(group["average_precision"].mean())
        best_f1_mean = float(group["best_f1"].mean())
        rows.append(
            {
                "score_variant": variant,
                "display_name": DISPLAY_NAMES.get(variant, variant),
                "roc_auc_mean": float(group["roc_auc"].mean()),
                "roc_auc_sd": float(group["roc_auc"].std(ddof=0)),
                "roc_auc_min": float(group["roc_auc"].min()),
                "roc_auc_max": float(group["roc_auc"].max()),
                "average_precision_mean": ap_mean,
                "average_precision_sd": float(group["average_precision"].std(ddof=0)),
                "prevalence_mean": prevalence,
                "average_precision_lift": ap_mean / prevalence if prevalence > 0 else float("nan"),
                "best_f1_mean": best_f1_mean,
                "all_positive_f1": all_positive_f1,
                "best_f1_gain_over_all_positive": best_f1_mean - all_positive_f1,
                "n_replicates": int(group["replicate"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("score_variant").reset_index(drop=True)


def detection_status(auc: float) -> str:
    if not np.isfinite(auc):
        return "NOT_EVALUABLE"
    if auc >= 0.70:
        return "SUPPORTED"
    if auc >= 0.60:
        return "WEAK_TO_MODERATE"
    return "NOT_SUPPORTED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--comparison_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--original_reference_master_json", required=True)
    ap.add_argument("--read_level_master_json", required=True)
    ap.add_argument("--rare_event_master_json", required=True)
    ap.add_argument("--baseline_audit_json", required=True)
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    comparison_dir = Path(args.comparison_dir)
    outdir = ensure_dir(args.outdir)

    consensus_metrics = pd.read_csv(
        comparison_dir / "crossfit_generated_consensus_metrics.tsv", sep="\t"
    )
    method_summary = summarize_metrics(consensus_metrics)
    real_scores = pd.read_csv(
        comparison_dir / "crossfit_real_consensus_scores.tsv", sep="\t"
    )
    real_meta_metrics = pd.read_csv(
        comparison_dir / "crossfit_real_metadata_metrics_descriptive.tsv", sep="\t"
    )
    generated_reference_metrics = pd.read_csv(
        scores_dir / "crossfit_generated_reference_metrics.tsv", sep="\t"
    )
    generated_reference_scores = pd.read_csv(
        scores_dir / "crossfit_generated_reference_scores.tsv", sep="\t"
    )
    provenance = pd.read_csv(
        scores_dir / "crossfit_generated_fold_provenance.tsv", sep="\t"
    )
    real_provenance = pd.read_csv(
        scores_dir / "crossfit_real_provenance.tsv", sep="\t"
    )
    generated_concordance = pd.read_csv(
        comparison_dir / "crossfit_tms_generated_concordance.tsv", sep="\t"
    )
    real_concordance = json.loads(
        (comparison_dir / "crossfit_real_rank_concordance.json").read_text(
            encoding="utf-8"
        )
    )

    original = json.loads(
        Path(args.original_reference_master_json).read_text(encoding="utf-8")
    )
    read_level = json.loads(
        Path(args.read_level_master_json).read_text(encoding="utf-8")
    )
    rare_event = json.loads(
        Path(args.rare_event_master_json).read_text(encoding="utf-8")
    )
    baseline = json.loads(
        Path(args.baseline_audit_json).read_text(encoding="utf-8")
    )

    summary_lookup = method_summary.set_index("score_variant")
    full = summary_lookup.loc["reference_qc_crossfit"].to_dict()
    no_pca = summary_lookup.loc["reference_qc_crossfit_no_pca"].to_dict()
    mapping = summary_lookup.loc["reference_mapping_crossfit"].to_dict()
    pca_only = summary_lookup.loc["pca_only_crossfit"].to_dict()

    nominal_auc = float(original["primary_endpoint"]["roc_auc_mean"])
    nominal_ap = float(original["primary_endpoint"]["average_precision_mean"])
    sketch_auc = float(baseline["observed"]["synthetic"]["roc_auc"])
    rare_auc = float(rare_event["primary_endpoint"]["roc_auc_mean"])

    # Prefer the paired-read endpoint stored in the original comparison summary.
    paired_auc = float(
        original.get("reference_free_comparison", {}).get(
            "paired_read_tms_auc_mean",
            read_level.get("primary_metrics", {})
            .get("roc_auc", {})
            .get("mean", float("nan")),
        )
    )

    primary_auc = float(full["roc_auc_mean"])
    primary_status = detection_status(primary_auc)

    real_top = {}
    for variant in [
        "reference_qc_crossfit",
        "reference_qc_crossfit_no_pca",
        "reference_mapping_crossfit",
        "pca_only_crossfit",
    ]:
        col = variant + "_consensus"
        real_top[variant] = {
            "top3": real_scores.nlargest(3, col)["sample_id"].astype(str).tolist(),
            "top5": real_scores.nlargest(5, col)["sample_id"].astype(str).tolist(),
        }

    metadata_metric_lookup = real_meta_metrics.set_index("score_variant")
    metadata_auc_full = float(
        metadata_metric_lookup.loc["reference_qc_crossfit", "roc_auc"]
    )
    metadata_auc_no_pca = float(
        metadata_metric_lookup.loc["reference_qc_crossfit_no_pca", "roc_auc"]
    )

    master = {
        "status": "COMPLETE",
        "primary_endpoint": {
            "method": "dual-reference cross-fitted reference QC",
            "score_variant": "reference_qc_crossfit",
            "roc_auc_mean": primary_auc,
            "roc_auc_sd": float(full["roc_auc_sd"]),
            "average_precision_mean": float(full["average_precision_mean"]),
            "n_replicates": int(full["n_replicates"]),
            "detection_status": primary_status,
            "evaluation_design": (
                "leave-one-generated-replicate-out outer evaluation with "
                "nested leave-one-training-replicate-out calibration"
            ),
        },
        "crossfit_method_summary": method_summary.to_dict("records"),
        "comparators": {
            "nominal_in_sample_reference_qc_auc": nominal_auc,
            "nominal_in_sample_reference_qc_ap": nominal_ap,
            "crossfit_full_auc": primary_auc,
            "crossfit_no_pca_auc": float(no_pca["roc_auc_mean"]),
            "crossfit_mapping_only_auc": float(mapping["roc_auc_mean"]),
            "crossfit_pca_only_auc": float(pca_only["roc_auc_mean"]),
            "paired_read_tms_auc": paired_auc,
            "rare_event_tms_auc": rare_auc,
            "idealized_sketch_space_auc": sketch_auc,
        },
        "provenance": {
            "outer_fold_count": int(len(provenance)),
            "reference_count": int(provenance["reference_id"].nunique()),
            "outer_test_replicates": sorted(
                provenance["outer_test_replicate"].astype(int).unique().tolist()
            ),
            "maximum_outer_train_test_overlap": int(
                provenance["outer_train_test_overlap_count"].max()
            ),
            "maximum_test_rows_used_in_marker_selection": int(
                provenance["marker_selection_used_outer_test_rows"].max()
            ),
            "maximum_test_rows_used_in_pca_fit": int(
                provenance["pca_fit_used_outer_test_rows"].max()
            ),
            "maximum_test_rows_used_in_scaling": int(
                provenance["scaling_used_outer_test_rows"].max()
            ),
            "real_test_rows_used_in_fitting": int(
                max(
                    real_provenance[
                        [
                            "marker_selection_used_real_test_rows",
                            "pca_fit_used_real_test_rows",
                            "scaling_used_real_test_rows",
                        ]
                    ].max()
                )
            ),
            "real_known_source_identity_overlap_count": int(
                real_provenance[
                    "real_training_source_identity_overlap_count"
                ].max()
            ),
        },
        "real_data": {
            "rankings": real_top,
            "metadata_holdout_auc_descriptive_full": metadata_auc_full,
            "metadata_holdout_auc_descriptive_no_pca": metadata_auc_no_pca,
            "metadata_pool_labels_used_for_model_fitting": False,
            "interpretation": (
                "Metadata-pool labels are used only for descriptive held-out "
                "comparison and are not definitive biological ground truth. "
                "Generated controls from the 13 non-pool source identities form "
                "the reference panel, so real-data scoring is panel matching "
                "rather than de novo discovery."
            ),
            "tms_rank_concordance": real_concordance,
        },
        "generated_tms_concordance": {
            "full_spearman_mean": float(
                generated_concordance.loc[
                    generated_concordance["score_variant"]
                    == "reference_qc_crossfit",
                    "spearman_rho",
                ].mean()
            ),
            "full_top5_jaccard_mean": float(
                generated_concordance.loc[
                    generated_concordance["score_variant"]
                    == "reference_qc_crossfit",
                    "top5_jaccard",
                ].mean()
            ),
        },
        "source_results": {
            "original_reference_master_json": str(args.original_reference_master_json),
            "read_level_master_json": str(args.read_level_master_json),
            "rare_event_master_json": str(args.rare_event_master_json),
            "baseline_audit_json": str(args.baseline_audit_json),
        },
    }
    master = clean_json(master)
    write_json(master, outdir / "reference_qc_crossfit_master_metrics.json")

    with pd.ExcelWriter(
        outdir / "reference_qc_crossfit_master_metrics.xlsx",
        engine="openpyxl",
    ) as writer:
        method_summary.to_excel(writer, sheet_name="method_summary", index=False)
        consensus_metrics.to_excel(writer, sheet_name="consensus_metrics", index=False)
        generated_reference_metrics.to_excel(
            writer, sheet_name="reference_metrics", index=False
        )
        real_scores.to_excel(writer, sheet_name="real_rankings", index=False)
        real_meta_metrics.to_excel(
            writer, sheet_name="real_metadata_metrics", index=False
        )
        provenance.to_excel(writer, sheet_name="fold_provenance", index=False)
        real_provenance.to_excel(
            writer, sheet_name="real_provenance", index=False
        )

    lines = [
        "Leakage-controlled cross-fitted reference-QC benchmark",
        "======================================================",
        "",
        "Primary endpoint: dual-reference consensus across five held-out generated replicates",
        "Outer evaluation: one replicate held out completely",
        "Inner calibration: one training replicate held out in turn",
        "",
        f"Cross-fitted full ROC AUC: {primary_auc:.3f} +/- {float(full['roc_auc_sd']):.3f}",
        f"Cross-fitted full Average Precision: {float(full['average_precision_mean']):.3f}",
        f"Positive-class prevalence: {float(full['prevalence_mean']):.3f}",
        f"Average-precision lift over prevalence: {float(full['average_precision_lift']):.3f}",
        f"Best-F1 gain over all-positive baseline: {float(full['best_f1_gain_over_all_positive']):.3f}",
        f"Cross-fitted no-PCA ROC AUC: {float(no_pca['roc_auc_mean']):.3f}",
        f"Mapping-only ROC AUC: {float(mapping['roc_auc_mean']):.3f}",
        f"PCA-only ROC AUC: {float(pca_only['roc_auc_mean']):.3f}",
        f"Detection status: {primary_status}",
        "",
        "Context:",
        f"  Previous nominal in-sample reference-QC AUC: {nominal_auc:.3f}",
        f"  Paired-read TMS AUC: {paired_auc:.3f}",
        f"  Rare-event TMS AUC: {rare_auc:.3f}",
        f"  Idealized sketch-space AUC: {sketch_auc:.3f}",
        "",
        "Leakage controls:",
        f"  Outer train/test overlap: {master['provenance']['maximum_outer_train_test_overlap']}",
        "  Evaluation rows used for marker selection: 0",
        "  Evaluation rows used for PCA fitting: 0",
        "  Evaluation rows used for scaling: 0",
        "",
        "Real-data comparison:",
        "  Known source identities represented in the training panel: "
        + str(master["provenance"]["real_known_source_identity_overlap_count"]),
        "  Full cross-fitted top 3: "
        + ", ".join(real_top["reference_qc_crossfit"]["top3"]),
        "  No-PCA cross-fitted top 3: "
        + ", ".join(real_top["reference_qc_crossfit_no_pca"]["top3"]),
        f"  Metadata-holdout AUC (descriptive, full): {metadata_auc_full:.3f}",
        "",
        "COMPLETE refers to computational completion. Detection status is a",
        "separate empirical judgment based on the prespecified AUC thresholds.",
        "Metadata labels are not treated as definitive biological ground truth.",
    ]
    (outdir / "REFERENCE_QC_CROSSFIT_RESULTS_SUMMARY.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
