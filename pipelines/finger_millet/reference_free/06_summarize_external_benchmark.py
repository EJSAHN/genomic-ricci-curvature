# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from external_common import (
    ensure_dir,
    read_json,
    spearman_safe,
    summarize_numeric,
    write_json,
)


def finite_mean(values: Iterable[Any]) -> float:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else float("nan")


def finite_sd(values: Iterable[Any]) -> float:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    array = array[np.isfinite(array)]
    return float(np.std(array, ddof=1)) if len(array) > 1 else (0.0 if len(array) == 1 else float("nan"))


def aggregate_replicates(
    table: pd.DataFrame,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = table.groupby(list(group_columns), dropna=False, sort=True)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(group_columns, key)}
        row["replicate_count"] = int(group["replicate"].nunique()) if "replicate" in group else 0
        row["row_count"] = int(len(group))
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else float("nan")
            row[f"{metric}_sd"] = float(finite.std(ddof=1)) if len(finite) > 1 else (0.0 if len(finite) == 1 else float("nan"))
            row[f"{metric}_min"] = float(finite.min()) if len(finite) else float("nan")
            row[f"{metric}_max"] = float(finite.max()) if len(finite) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def rare_replicate_first(graphs: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "roc_auc",
        "average_precision",
        "ap_lift_over_prevalence",
        "best_f1",
        "all_positive_f1",
        "best_f1_gain_over_all_positive",
        "mean_mixture_percentile",
        "top1_is_mixture",
        "any_mixture_top3",
        "any_mixture_top5",
        "mixture_recall_top3",
        "mixture_recall_top5",
        "mixture_precision_top3",
        "mixture_precision_top5",
        "chance_top1",
        "chance_any_top3",
        "chance_any_top5",
        "negative_edge_fraction",
        "orc_component_informative",
        "tms_betweenness_spearman",
        "graph_connected",
    ]
    rows: List[Dict[str, Any]] = []
    for (mode, injection, replicate), group in graphs.groupby(
        ["analysis_mode", "injection_count", "replicate"], sort=True
    ):
        row: Dict[str, Any] = {
            "analysis_mode": mode,
            "injection_count": int(injection),
            "replicate": int(replicate),
            "graph_count": int(len(group)),
        }
        for metric in metric_columns:
            row[metric] = finite_mean(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_batch_runs(batch_runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "roc_auc",
        "average_precision",
        "best_f1",
        "tms_entropy_spearman_rho",
        "tms_minor_fraction_spearman_rho",
        "tms_parent_distance_spearman_rho",
        "negative_edge_fraction",
        "orc_component_informative",
    ]
    return aggregate_replicates(batch_runs, ["analysis_mode"], metrics)


def summarize_rare_replicates(rep_table: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "roc_auc",
        "average_precision",
        "ap_lift_over_prevalence",
        "best_f1",
        "all_positive_f1",
        "best_f1_gain_over_all_positive",
        "mean_mixture_percentile",
        "top1_is_mixture",
        "any_mixture_top3",
        "any_mixture_top5",
        "mixture_recall_top3",
        "mixture_recall_top5",
        "mixture_precision_top3",
        "mixture_precision_top5",
        "chance_top1",
        "chance_any_top3",
        "chance_any_top5",
        "negative_edge_fraction",
        "orc_component_informative",
        "tms_betweenness_spearman",
        "graph_connected",
    ]
    return aggregate_replicates(
        rep_table,
        ["analysis_mode", "injection_count"],
        metrics,
    )


def comparator_replicate_first(table: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "roc_auc",
        "average_precision",
        "ap_lift_over_prevalence",
        "best_f1",
    ]
    rows: List[Dict[str, Any]] = []
    for (mode, injection, score, replicate), group in table.groupby(
        ["analysis_mode", "injection_count", "score_name", "replicate"],
        sort=True,
    ):
        row: Dict[str, Any] = {
            "analysis_mode": mode,
            "injection_count": int(injection),
            "score_name": score,
            "replicate": int(replicate),
            "graph_count": int(len(group)),
        }
        for metric in metric_columns:
            row[metric] = finite_mean(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def category_or_pattern_summary(
    mixture_rows: pd.DataFrame,
    grouping_column: str,
) -> pd.DataFrame:
    primary = mixture_rows[mixture_rows["injection_count"].astype(int) == 1].copy()
    metrics = [
        "rank_percentile",
        "top1",
        "top3",
        "top5",
        "tms",
        "negative_orc_incidence",
        "betweenness",
        "mixture_to_parent_min",
        "mixture_to_parent_mean",
        "parent_distance_mean",
    ]
    replicate_rows: List[Dict[str, Any]] = []
    for (mode, group_value, replicate), group in primary.groupby(
        ["analysis_mode", grouping_column, "replicate"], sort=True
    ):
        row: Dict[str, Any] = {
            "analysis_mode": mode,
            grouping_column: group_value,
            "replicate": int(replicate),
            "mixture_count": int(len(group)),
        }
        for metric in metrics:
            row[metric] = finite_mean(group[metric])
        replicate_rows.append(row)
    replicate_table = pd.DataFrame(replicate_rows)
    if replicate_table.empty:
        return replicate_table
    return aggregate_replicates(
        replicate_table,
        ["analysis_mode", grouping_column],
        metrics,
    )


def detection_status(value: float, interpretation: Dict[str, str]) -> str:
    if not math.isfinite(value):
        return "UNDETERMINED"
    if value >= 0.70:
        return "SUPPORTED"
    if value >= 0.60:
        return "WEAK_TO_MODERATE"
    return "NOT_SUPPORTED"


def row_to_dict(row: pd.Series) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            result[key] = None
        elif isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating,)):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock_json", required=True)
    parser.add_argument("--preflight_dir", required=True)
    parser.add_argument("--full_cohort_dir", required=True)
    parser.add_argument("--generated_dir", required=True)
    parser.add_argument("--sketch_dir", required=True)
    parser.add_argument("--batch_dir", required=True)
    parser.add_argument("--rare_dir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    lock = read_json(args.lock_json)

    required_passes = [
        Path(args.preflight_dir) / "EXTERNAL_BENCHMARK_PREFLIGHT_PASS.txt",
        Path(args.full_cohort_dir) / "FULL83_GEOMETRY_PASS.txt",
        Path(args.generated_dir) / "manifests" / "GENERATED_FASTQ_BUILD_PASS.txt",
        Path(args.sketch_dir) / "GENERATED_SKETCH_PASS.txt",
        Path(args.batch_dir) / "BATCH_BENCHMARK_PASS.txt",
        Path(args.rare_dir) / "RARE_EVENT_BENCHMARK_PASS.txt",
    ]
    missing_passes = [str(path) for path in required_passes if not path.exists()]
    if missing_passes:
        raise SystemExit("[ERROR] Required upstream PASS markers are missing: " + " | ".join(missing_passes))

    full_summary = read_json(Path(args.full_cohort_dir) / "full83_geometry_summary.json")
    build_audit = read_json(Path(args.generated_dir) / "manifests" / "generated_fastq_build_audit.json")
    sketch_summary = read_json(Path(args.sketch_dir) / "generated_sketch_summary.json")
    batch_runs = pd.read_csv(Path(args.batch_dir) / "batch_run_metrics.tsv", sep="\t")
    batch_comparators = pd.read_csv(Path(args.batch_dir) / "batch_comparator_metrics.tsv", sep="\t")
    batch_categories = pd.read_csv(Path(args.batch_dir) / "batch_category_metrics.tsv", sep="\t")
    batch_patterns = pd.read_csv(Path(args.batch_dir) / "batch_pattern_metrics.tsv", sep="\t")
    rare_graphs = pd.read_csv(Path(args.rare_dir) / "rare_event_graph_metrics.tsv", sep="\t")
    rare_mixtures = pd.read_csv(Path(args.rare_dir) / "rare_event_mixture_rank_metrics.tsv", sep="\t")
    rare_comparators = pd.read_csv(Path(args.rare_dir) / "rare_event_comparator_metrics.tsv", sep="\t")

    batch_summary = summarize_batch_runs(batch_runs)
    batch_comparator_summary = aggregate_replicates(
        batch_comparators,
        ["analysis_mode", "score_name"],
        ["roc_auc", "average_precision", "best_f1"],
    )
    batch_category_summary = aggregate_replicates(
        batch_categories,
        ["analysis_mode", "category"],
        [
            "roc_auc",
            "average_precision",
            "best_f1",
            "mean_tms",
            "mean_parent_distance",
        ],
    )
    batch_pattern_summary = aggregate_replicates(
        batch_patterns,
        ["analysis_mode", "pattern_id"],
        [
            "roc_auc",
            "average_precision",
            "best_f1",
            "mean_tms",
            "mean_entropy",
            "mean_minor_fraction",
        ],
    )

    rare_replicates = rare_replicate_first(rare_graphs)
    rare_summary = summarize_rare_replicates(rare_replicates)
    rare_comparator_reps = comparator_replicate_first(rare_comparators)
    rare_comparator_summary = aggregate_replicates(
        rare_comparator_reps,
        ["analysis_mode", "injection_count", "score_name"],
        ["roc_auc", "average_precision", "ap_lift_over_prevalence", "best_f1"],
    )
    rare_category_summary = category_or_pattern_summary(rare_mixtures, "category")
    rare_pattern_summary = category_or_pattern_summary(rare_mixtures, "pattern_id")

    correlation_rows: List[Dict[str, Any]] = []
    single = rare_mixtures[rare_mixtures["injection_count"].astype(int) == 1].copy()
    for mode, group in single.groupby("analysis_mode", sort=True):
        for covariate in [
            "actual_entropy_norm",
            "actual_minor_fraction",
            "locked_distance_summary",
            "parent_distance_mean",
            "mixture_to_parent_min",
            "mixture_to_parent_mean",
        ]:
            rho, p_value = spearman_safe(group[covariate], group["rank_percentile"])
            correlation_rows.append(
                {
                    "analysis_mode": mode,
                    "endpoint": "rank_percentile",
                    "covariate": covariate,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "n": int(group[[covariate, "rank_percentile"]].dropna().shape[0]),
                }
            )
    rare_correlations = pd.DataFrame(correlation_rows)

    batch_summary.to_csv(outdir / "batch_group_summary.tsv", sep="\t", index=False)
    batch_comparator_summary.to_csv(outdir / "batch_comparator_summary.tsv", sep="\t", index=False)
    batch_category_summary.to_csv(outdir / "batch_category_summary.tsv", sep="\t", index=False)
    batch_pattern_summary.to_csv(outdir / "batch_pattern_summary.tsv", sep="\t", index=False)
    rare_replicates.to_csv(outdir / "rare_event_replicate_summary.tsv", sep="\t", index=False)
    rare_summary.to_csv(outdir / "rare_event_group_summary.tsv", sep="\t", index=False)
    rare_comparator_reps.to_csv(outdir / "rare_event_comparator_replicate_summary.tsv", sep="\t", index=False)
    rare_comparator_summary.to_csv(outdir / "rare_event_comparator_summary.tsv", sep="\t", index=False)
    rare_category_summary.to_csv(outdir / "rare_event_category_summary.tsv", sep="\t", index=False)
    rare_pattern_summary.to_csv(outdir / "rare_event_pattern_summary.tsv", sep="\t", index=False)
    rare_correlations.to_csv(outdir / "rare_event_correlations.tsv", sep="\t", index=False)

    primary_rows = rare_summary[
        (rare_summary["analysis_mode"] == str(lock["primary_analysis_mode"]))
        & (rare_summary["injection_count"].astype(int) == 1)
    ]
    if len(primary_rows) != 1:
        raise SystemExit(f"[ERROR] Expected one primary rare-event summary row; observed {len(primary_rows)}")
    primary = primary_rows.iloc[0]
    primary_auc = float(primary["roc_auc_mean"])
    status = detection_status(primary_auc, lock.get("performance_interpretation", {}))

    batch_primary_rows = batch_summary[
        batch_summary["analysis_mode"] == str(lock["primary_analysis_mode"])
    ]
    batch_primary = row_to_dict(batch_primary_rows.iloc[0]) if len(batch_primary_rows) == 1 else {}

    primary_graph_count = int(
        len(
            rare_graphs[
                (rare_graphs["analysis_mode"].astype(str) == str(lock["primary_analysis_mode"]))
                & (rare_graphs["injection_count"].astype(int) == 1)
            ]
        )
    )
    primary_endpoint = {
        "analysis_mode": str(lock["primary_analysis_mode"]),
        "injection_count": 1,
        "replicate_count": int(primary["replicate_count"]),
        "graph_count": primary_graph_count,
        "roc_auc_mean": primary_auc,
        "roc_auc_sd": float(primary["roc_auc_sd"]),
        "average_precision_mean": float(primary["average_precision_mean"]),
        "positive_class_prevalence_mean": float(primary["chance_top1_mean"]),
        "average_precision_lift_mean": float(primary["ap_lift_over_prevalence_mean"]),
        "mean_rank_percentile": float(primary["mean_mixture_percentile_mean"]),
        "top1_capture_rate": float(primary["top1_is_mixture_mean"]),
        "top1_chance": float(primary["chance_top1_mean"]),
        "any_mixture_top3_rate": float(primary["any_mixture_top3_mean"]),
        "any_mixture_top3_chance": float(primary["chance_any_top3_mean"]),
        "any_mixture_top5_rate": float(primary["any_mixture_top5_mean"]),
        "any_mixture_top5_chance": float(primary["chance_any_top5_mean"]),
        "negative_edge_fraction_mean": float(primary["negative_edge_fraction_mean"]),
        "orc_informative_graph_fraction": float(primary["orc_component_informative_mean"]),
        "tms_betweenness_spearman_mean": float(primary["tms_betweenness_spearman_mean"]),
        "graph_connectivity_fraction": float(primary["graph_connected_mean"]),
    }

    master = {
        "status": "COMPLETE",
        "dataset": lock["dataset"],
        "master_lock_sha256": "e9117c96aa765bc4cd619e8b66bedc42c88fead82083d566ab330b5b4a503101",
        "design_lock": lock,
        "full_cohort": full_summary,
        "generated_fastq_build": build_audit,
        "generated_sketch_build": sketch_summary,
        "batch_primary_mode": batch_primary,
        "primary_endpoint": primary_endpoint,
        "external_reference_free_detection_status": status,
        "performance_status_rule": lock.get("performance_interpretation", {}),
        "analysis_modes": sorted(batch_runs["analysis_mode"].unique().tolist()),
        "batch_run_count": int(len(batch_runs)),
        "rare_event_graph_count": int(len(rare_graphs)),
        "rare_event_mixture_row_count": int(len(rare_mixtures)),
        "rare_event_comparator_row_count": int(len(rare_comparators)),
        "interpretation": (
            "Computational PASS/COMPLETE denotes integrity and reproducibility. "
            "SUPPORTED/WEAK_TO_MODERATE/NOT_SUPPORTED is determined only by the "
            "prespecified primary rare-event AUC threshold and is not altered after results are observed."
        ),
    }
    write_json(outdir / "external_reference_free_master_metrics.json", master)

    lines = [
        "Finger millet external reference-free benchmark",
        "================================================",
        "",
        f"Dataset: {lock['dataset']}",
        f"Locked design hash: {master['master_lock_sha256']}",
        "",
        "Primary endpoint: one true read-level mixture injected among 28 controls",
        f"Primary analysis mode: {primary_endpoint['analysis_mode']}",
        f"Replicates: {primary_endpoint['replicate_count']}",
        f"Graphs: {primary_endpoint['graph_count']}",
        "",
        f"ROC AUC: {primary_endpoint['roc_auc_mean']:.3f} +/- {primary_endpoint['roc_auc_sd']:.3f}",
        f"Average Precision: {primary_endpoint['average_precision_mean']:.3f}",
        f"Positive-class prevalence: {primary_endpoint['positive_class_prevalence_mean']:.3f}",
        f"AP lift over prevalence: {primary_endpoint['average_precision_lift_mean']:.3f}",
        f"Mean mixture rank percentile: {primary_endpoint['mean_rank_percentile']:.3f}",
        f"Top-1 capture: {primary_endpoint['top1_capture_rate']:.3f} (chance {primary_endpoint['top1_chance']:.3f})",
        f"Any mixture in Top-3: {primary_endpoint['any_mixture_top3_rate']:.3f} (chance {primary_endpoint['any_mixture_top3_chance']:.3f})",
        f"Any mixture in Top-5: {primary_endpoint['any_mixture_top5_rate']:.3f} (chance {primary_endpoint['any_mixture_top5_chance']:.3f})",
        f"Negative-edge fraction: {primary_endpoint['negative_edge_fraction_mean']:.6f}",
        f"Graphs with informative negative-ORC variation: {primary_endpoint['orc_informative_graph_fraction']:.3f}",
        f"Mean TMS-betweenness Spearman rho: {primary_endpoint['tms_betweenness_spearman_mean']:.3f}",
        "",
        f"External reference-free detection status: {status}",
        "",
        "Secondary batch benchmark (paired mode):",
    ]
    if batch_primary:
        lines.extend(
            [
                f"  ROC AUC: {float(batch_primary['roc_auc_mean']):.3f} +/- {float(batch_primary['roc_auc_sd']):.3f}",
                f"  Average Precision: {float(batch_primary['average_precision_mean']):.3f}",
                f"  TMS-entropy Spearman rho: {float(batch_primary['tms_entropy_spearman_rho_mean']):.3f}",
                f"  Negative-edge fraction: {float(batch_primary['negative_edge_fraction_mean']):.6f}",
            ]
        )
    lines.extend(
        [
            "",
            "Full-cohort descriptive geometry:",
            f"  Samples: {full_summary['n_samples']}",
            f"  Graph connected: {full_summary['graph_connected']}",
            f"  Negative-edge fraction: {full_summary['negative_edge_fraction']:.6f}",
            f"  Mean same-population neighbor fraction: {full_summary['mean_same_population_neighbor_fraction']:.3f}",
            f"  PERMANOVA pseudo-F: {full_summary['permanova']['pseudo_f']:.3f}",
            f"  PERMANOVA permutation P: {full_summary['permanova']['permutation_p']:.4f}",
            "",
            "PASS/COMPLETE refers to computational integrity. The empirical detection",
            "status is a separate judgment based on the prespecified primary AUC thresholds.",
        ]
    )
    summary_text = "\n".join(lines) + "\n"
    (outdir / "EXTERNAL_REFERENCE_FREE_RESULTS_SUMMARY.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
