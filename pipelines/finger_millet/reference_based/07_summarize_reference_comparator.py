# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from crossfit_common import SCORE_VARIANTS, ensure_dir, write_json


PRIMARY_SCORE = "reference_qc_crossfit"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def status_from_auc(auc: float, weak: float, supported: float) -> str:
    if not np.isfinite(auc):
        return "NOT_EVALUABLE"
    if auc >= supported:
        return "SUPPORTED"
    if auc >= weak:
        return "WEAK_TO_MODERATE"
    return "NOT_SUPPORTED"


def row_to_dict(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_manifest", required=True)
    ap.add_argument("--marker_summary", required=True)
    ap.add_argument("--feature_summary", required=True)
    ap.add_argument("--crossfit_scores_dir", required=True)
    ap.add_argument("--comparison_dir", required=True)
    ap.add_argument("--rare_dir", required=True)
    ap.add_argument("--external_master", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    refs = pd.read_csv(args.reference_manifest, sep="\t")
    marker = load_json(args.marker_summary)
    feature = load_json(args.feature_summary)
    config = load_json(args.config)
    external = load_json(args.external_master)

    scores_dir = Path(args.crossfit_scores_dir)
    comparison_dir = Path(args.comparison_dir)
    rare_dir = Path(args.rare_dir)

    crossfit_metrics = pd.read_csv(scores_dir / "generated_crossfit_metrics.tsv", sep="\t")
    provenance = pd.read_csv(scores_dir / "crossfit_fold_provenance.tsv", sep="\t")
    batch_summary = pd.read_csv(comparison_dir / "batch_reference_and_tms_summary.tsv", sep="\t")
    rare_summary = pd.read_csv(rare_dir / "reference_rare_event_summary.tsv", sep="\t")
    rare_replicates = pd.read_csv(rare_dir / "reference_rare_event_replicate_summary.tsv", sep="\t")

    primary_rare = rare_summary[
        (rare_summary["injection_count"].astype(int) == 1)
        & (rare_summary["score_name"].astype(str) == PRIMARY_SCORE)
    ]
    if len(primary_rare) != 1:
        raise SystemExit(
            f"[ERROR] Expected exactly one primary rare-event summary row; found {len(primary_rare)}"
        )
    primary = row_to_dict(primary_rare.iloc[0])

    threshold_cfg = config.get("performance_thresholds", {})
    supported_threshold = float(threshold_cfg.get("supported", 0.70))
    weak_threshold = float(threshold_cfg.get("weak_to_moderate", 0.60))
    primary_auc = finite_float(primary.get("roc_auc_mean"))
    empirical_status = status_from_auc(primary_auc, weak_threshold, supported_threshold)

    batch_rows: dict[str, dict[str, Any]] = {}
    for _, row in batch_summary.iterrows():
        batch_rows[str(row["score_name"])] = row_to_dict(row)

    variant_rows: list[dict[str, Any]] = []
    for score_name in SCORE_VARIANTS:
        subset = rare_summary[rare_summary["score_name"].astype(str) == score_name]
        for _, row in subset.sort_values("injection_count").iterrows():
            variant_rows.append(row_to_dict(row))
    pd.DataFrame(variant_rows).to_csv(
        outdir / "reference_comparator_rare_event_summary.tsv", sep="\t", index=False
    )

    crossfit_summary_rows: list[dict[str, Any]] = []
    for score_name, group in crossfit_metrics.groupby("score_variant", sort=True):
        crossfit_summary_rows.append(
            {
                "score_name": str(score_name),
                "replicate_count": int(group["replicate"].nunique()),
                "roc_auc_mean": float(group["roc_auc"].mean()),
                "roc_auc_sd": float(group["roc_auc"].std(ddof=0)),
                "roc_auc_min": float(group["roc_auc"].min()),
                "roc_auc_max": float(group["roc_auc"].max()),
                "average_precision_mean": float(group["average_precision"].mean()),
                "average_precision_sd": float(group["average_precision"].std(ddof=0)),
                "prevalence_mean": float(group["prevalence"].mean()),
            }
        )
    crossfit_summary_df = pd.DataFrame(crossfit_summary_rows)
    crossfit_summary_df.to_csv(
        outdir / "reference_comparator_batch_crossfit_summary.tsv", sep="\t", index=False
    )

    leakage = {
        "fold_count": int(len(provenance)),
        "outer_train_test_overlap_max": int(
            pd.to_numeric(provenance["outer_train_test_overlap_count"], errors="coerce").max()
        ),
        "outer_test_rows_used_for_marker_selection": int(
            pd.to_numeric(provenance["marker_selection_used_outer_test_rows"], errors="coerce").sum()
        ),
        "outer_test_rows_used_for_pca_fitting": int(
            pd.to_numeric(provenance["pca_fit_used_outer_test_rows"], errors="coerce").sum()
        ),
        "outer_test_rows_used_for_scaling": int(
            pd.to_numeric(provenance["scaling_used_outer_test_rows"], errors="coerce").sum()
        ),
        "outer_test_labels_used_for_model_fitting": bool(
            provenance["outer_test_labels_used_for_model_fitting"].astype(str).str.lower().isin(["true", "1", "yes"]).any()
        ),
        "mixture_labels_used_for_parameter_tuning": bool(
            provenance["mixture_labels_used_for_parameter_tuning"].astype(str).str.lower().isin(["true", "1", "yes"]).any()
        ),
        "training_control_role_used": bool(
            provenance["training_control_role_used_for_model_fitting"].astype(str).str.lower().isin(["true", "1", "yes"]).all()
        ),
        "evaluation_unit": "held-out generated read replicate; source identities recur across replicates but physical read pairs are disjoint",
    }

    tms_primary = external.get("primary_endpoint", {})
    tms_batch = external.get("batch_primary_mode", {})
    reference_batch_primary = batch_rows.get(PRIMARY_SCORE, {})

    interpretation = {
        "primary_reference_result": (
            "Reference-based discrimination is interpreted using the same prespecified AUC thresholds "
            "as the locked reference-free benchmark."
        ),
        "scope": (
            "The comparison evaluates technical mixed-library screening under locked true read-level "
            "mixtures. It does not identify the biological cause of heterogeneity in archived samples."
        ),
        "crossfit_boundary": (
            "Outer folds hold out one generated-read replicate completely. Marker discovery uses separate "
            "physical source reads and no mixture labels; source identities are represented across replicates."
        ),
    }

    master: Dict[str, Any] = {
        "status": "COMPLETE",
        "dataset": "finger_millet_PRJNA791522",
        "method": "leakage_controlled_reference_qc_crossfit",
        "reference_count": int(len(refs)),
        "references": refs.to_dict(orient="records"),
        "marker_panel": marker,
        "feature_extraction": feature,
        "primary_endpoint": {
            **primary,
            "endpoint_definition": "one true read-level mixture injected among 28 controls",
            "score_name": PRIMARY_SCORE,
            "analysis_mode": "paired",
        },
        "reference_detection_status": empirical_status,
        "performance_thresholds": {
            "SUPPORTED": f"mean primary AUC >= {supported_threshold:.2f}",
            "WEAK_TO_MODERATE": f"{weak_threshold:.2f} <= mean primary AUC < {supported_threshold:.2f}",
            "NOT_SUPPORTED": f"mean primary AUC < {weak_threshold:.2f}",
        },
        "reference_batch_primary": reference_batch_primary,
        "reference_batch_crossfit_variants": crossfit_summary_df.to_dict(orient="records"),
        "reference_rare_event_variants": variant_rows,
        "reference_free_context": {
            "primary_rare_event_tms": tms_primary,
            "paired_batch_tms": tms_batch,
            "detection_status": external.get("external_reference_free_detection_status"),
            "master_lock_sha256": external.get("master_lock_sha256"),
        },
        "leakage_controls": leakage,
        "interpretation": interpretation,
        "row_counts": {
            "crossfit_metric_rows": int(len(crossfit_metrics)),
            "crossfit_fold_rows": int(len(provenance)),
            "rare_event_summary_rows": int(len(rare_summary)),
            "rare_event_replicate_rows": int(len(rare_replicates)),
        },
    }
    write_json(master, outdir / "external_reference_comparator_master_metrics.json")

    tms_auc = finite_float(tms_primary.get("roc_auc_mean"))
    tms_ap = finite_float(tms_primary.get("average_precision_mean"))
    ref_ap = finite_float(primary.get("average_precision_mean"))
    ref_lift = finite_float(primary.get("ap_lift_over_prevalence_mean"))
    batch_ref_auc = finite_float(reference_batch_primary.get("roc_auc_mean"))
    batch_tms_auc = finite_float(tms_batch.get("roc_auc_mean"))

    lines = [
        "Finger millet leakage-controlled reference-based comparator",
        "============================================================",
        "",
        "Primary endpoint: one true read-level mixture among 28 controls",
        "Outer evaluation: one generated-read replicate held out completely",
        "Marker discovery: independent source read pairs with zero overlap with generated libraries",
        "",
        f"Reference assembly: {refs.iloc[0]['accession']} ({refs.iloc[0]['label']})",
        f"Independent marker panel: {int(marker.get('marker_count', 0)):,} markers",
        f"Cross-fitted primary rare-event ROC AUC: {primary_auc:.3f} +/- {finite_float(primary.get('roc_auc_sd')):.3f}",
        f"Cross-fitted primary Average Precision: {ref_ap:.3f}",
        f"Positive-class prevalence: {finite_float(primary.get('chance_top1')):.3f}",
        f"AP lift over prevalence: {ref_lift:.3f}",
        f"Top-1 capture: {finite_float(primary.get('top1_capture_rate')):.3f} (chance {finite_float(primary.get('chance_top1')):.3f})",
        f"Any mixture in Top-3: {finite_float(primary.get('top3_capture_rate')):.3f} (chance {finite_float(primary.get('chance_any_top3')):.3f})",
        f"Any mixture in Top-5: {finite_float(primary.get('top5_capture_rate')):.3f} (chance {finite_float(primary.get('chance_any_top5')):.3f})",
        f"Mean mixture rank percentile: {finite_float(primary.get('mean_mixture_rank_percentile')):.3f}",
        f"Reference-based detection status: {empirical_status}",
        "",
        "Locked reference-free context:",
        f"  Rare-event TMS ROC AUC: {tms_auc:.3f}",
        f"  Rare-event TMS Average Precision: {tms_ap:.3f}",
        f"  Paired batch TMS ROC AUC: {batch_tms_auc:.3f}",
        "",
        "Reference-based batch context:",
        f"  Full cross-fitted score ROC AUC: {batch_ref_auc:.3f}",
        "",
        "Leakage controls:",
        f"  Outer train/test overlap: {leakage['outer_train_test_overlap_max']}",
        f"  Evaluation rows used for marker selection: {leakage['outer_test_rows_used_for_marker_selection']}",
        f"  Evaluation rows used for PCA fitting: {leakage['outer_test_rows_used_for_pca_fitting']}",
        f"  Evaluation rows used for scaling: {leakage['outer_test_rows_used_for_scaling']}",
        f"  Mixture labels used for parameter tuning: {leakage['mixture_labels_used_for_parameter_tuning']}",
        "",
        "COMPLETE denotes computational completion. The empirical detection status is",
        "a separate judgment based on the prespecified primary AUC thresholds.",
        "Source identities recur across read replicates; held-out evaluation is therefore",
        "a disjoint-read replicate test rather than a source-identity holdout.",
    ]
    summary_text = "\n".join(lines) + "\n"
    (outdir / "REFERENCE_COMPARATOR_RESULTS_SUMMARY.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
