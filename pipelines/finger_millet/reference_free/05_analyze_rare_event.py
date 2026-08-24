# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from external_common import (
    compute_node_scores,
    ensure_dir,
    evaluate_score,
    exact_any_topk_chance,
    read_json,
    sha256_file,
    sha256_text,
    spearman_safe,
    write_json,
)


SCORE_COLUMNS = {
    "tms": "tms",
    "betweenness": "betweenness",
    "negative_orc": "negative_orc_incidence",
    "mean_incident_distance": "mean_incident_distance",
    "real_bridge_score": "real_bridge_score",
    "pca_distance": "pca_distance",
    "local_outlier_factor": "lof_score",
    "raw_betweenness_plus_negative_orc": "raw_sum",
}


def semicolon_values(value: Any) -> List[str]:
    return [part for part in str(value).split(";") if part]


def part_marker_valid(marker_path: Path, input_signature: str) -> bool:
    if not marker_path.exists():
        return False
    try:
        marker = read_json(marker_path)
    except Exception:
        return False
    if marker.get("status") != "COMPLETE":
        return False
    if marker.get("input_signature") != input_signature:
        return False
    for field in (
        "graph_metrics_path",
        "mixture_metrics_path",
        "comparator_metrics_path",
        "edge_diagnostics_path",
    ):
        if not Path(marker.get(field, "")).exists():
            return False
    return True


def process_group(
    replicate: int,
    mode: str,
    schedule_records: List[Dict[str, Any]],
    truth_path: str,
    definitions_path: str,
    distance_path: str,
    part_dir_path: str,
    locked_knn: int,
    alpha: float,
) -> Dict[str, Any]:
    part_dir = ensure_dir(part_dir_path)
    marker_path = part_dir / "PART_COMPLETE.json"
    input_signature = sha256_text(
        "|".join(
            [
                str(replicate),
                mode,
                sha256_file(truth_path),
                sha256_file(definitions_path),
                sha256_file(distance_path),
                sha256_text(json.dumps(schedule_records, sort_keys=True)),
                str(locked_knn),
                str(alpha),
            ]
        )
    )
    if part_marker_valid(marker_path, input_signature):
        marker = read_json(marker_path)
        marker["cache_hit"] = True
        return marker

    for path in part_dir.glob("*.tsv"):
        path.unlink()
    if marker_path.exists():
        marker_path.unlink()

    distance_full = pd.read_csv(distance_path, index_col=0)
    truth = pd.read_csv(truth_path, sep="\t")
    definitions = pd.read_csv(definitions_path, sep="\t")
    definition_columns = [
        "mixture_definition_id",
        "base_set_id",
        "anchor_population",
        "distance_summary",
        "target_entropy_norm",
        "target_minor_fraction",
    ]
    truth = truth.merge(
        definitions[definition_columns],
        on="mixture_definition_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_definition"),
    )
    for column in definition_columns[1:]:
        definition_column = f"{column}_definition"
        if definition_column in truth.columns:
            if column in truth.columns:
                truth[column] = truth[column].where(
                    truth[column].notna(), truth[definition_column]
                )
            else:
                truth[column] = truth[definition_column]
            truth = truth.drop(columns=[definition_column])
    truth_rep = truth[truth["replicate"].astype(int) == int(replicate)].copy()
    truth_lookup = truth_rep.set_index("sample_id", drop=False)
    controls = truth_rep[truth_rep["class_label"] == "single_source_control"]
    run_to_control: Dict[str, str] = {}
    for row in controls.itertuples(index=False):
        for run in semicolon_values(getattr(row, "actual_parent_runs", getattr(row, "parent_runs", ""))):
            run_to_control[run] = str(row.sample_id)

    graph_rows: List[Dict[str, Any]] = []
    mixture_rows: List[Dict[str, Any]] = []
    comparator_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []

    total_graphs = len(schedule_records)
    for graph_index, schedule_row in enumerate(schedule_records, start=1):
        if graph_index == 1 or graph_index % 25 == 0 or graph_index == total_graphs:
            print(
                f"[RARE R{replicate:02d} {mode} {graph_index}/{total_graphs}] "
                f"{schedule_row['graph_id']}",
                flush=True,
            )
        control_ids = semicolon_values(schedule_row["control_sample_ids"])
        mixture_ids = semicolon_values(schedule_row["mixture_sample_ids"])
        names = control_ids + mixture_ids
        missing = [name for name in names if name not in distance_full.index]
        if missing:
            raise ValueError(
                f"Rare-event graph {schedule_row['graph_id']} is missing samples: {missing}"
            )
        distance = distance_full.loc[names, names]
        node_table, edge_table, graph = compute_node_scores(
            distance, k=int(locked_knn), alpha=float(alpha)
        )
        node_table["raw_sum"] = (
            node_table["betweenness"]
            + node_table["negative_orc_incidence"]
        )
        node_table["is_mixture"] = node_table["sample_id"].isin(mixture_ids).astype(int)
        labels = node_table["is_mixture"].to_numpy(dtype=int)
        n_nodes = len(node_table)
        n_mixtures = int(labels.sum())
        prevalence = n_mixtures / n_nodes

        ordered = node_table.sort_values(
            ["tms", "sample_id"], ascending=[False, True]
        ).reset_index(drop=True)
        exact_rank = {sample_id: index + 1 for index, sample_id in enumerate(ordered["sample_id"])}
        average_ranks_array = rankdata(
            -node_table["tms"].to_numpy(dtype=float), method="average"
        )
        average_rank = {
            sample_id: float(rank)
            for sample_id, rank in zip(node_table["sample_id"], average_ranks_array)
        }
        exact_mixture_ranks = [exact_rank[sample_id] for sample_id in mixture_ids]
        average_mixture_ranks = [average_rank[sample_id] for sample_id in mixture_ids]
        top1_ids = set(ordered.head(1)["sample_id"])
        top3_ids = set(ordered.head(min(3, n_nodes))["sample_id"])
        top5_ids = set(ordered.head(min(5, n_nodes))["sample_id"])
        top1_hits = len(top1_ids.intersection(mixture_ids))
        top3_hits = len(top3_ids.intersection(mixture_ids))
        top5_hits = len(top5_ids.intersection(mixture_ids))

        tms_metrics = evaluate_score(labels, node_table["tms"].to_numpy(float))
        all_positive_f1 = 2 * n_mixtures / (2 * n_mixtures + (n_nodes - n_mixtures))
        negative_edge_count = int((edge_table["orc"] < 0).sum()) if len(edge_table) else 0
        negative_edge_fraction = (
            negative_edge_count / len(edge_table) if len(edge_table) else 0.0
        )
        negative_node_variance = float(
            node_table["negative_orc_incidence"].var(ddof=0)
        )
        tms_betweenness_rho, _ = spearman_safe(
            node_table["tms"], node_table["betweenness"]
        )

        graph_rows.append(
            {
                "graph_id": schedule_row["graph_id"],
                "replicate": int(replicate),
                "analysis_mode": mode,
                "injection_count": int(schedule_row["injection_count"]),
                "n_controls": len(control_ids),
                "n_mixtures": n_mixtures,
                "n_nodes": n_nodes,
                "prevalence": prevalence,
                "graph_connected": bool(nx.is_connected(graph)),
                "graph_components": int(nx.number_connected_components(graph)),
                "graph_edges": int(graph.number_of_edges()),
                "roc_auc": tms_metrics["roc_auc"],
                "average_precision": tms_metrics["average_precision"],
                "ap_lift_over_prevalence": (
                    tms_metrics["average_precision"] / prevalence
                    if prevalence > 0
                    else float("nan")
                ),
                "best_f1": tms_metrics["best_f1"],
                "all_positive_f1": all_positive_f1,
                "best_f1_gain_over_all_positive": (
                    tms_metrics["best_f1"] - all_positive_f1
                ),
                "mean_mixture_rank": float(np.mean(exact_mixture_ranks)),
                "median_mixture_rank": float(np.median(exact_mixture_ranks)),
                "mean_mixture_average_rank": float(np.mean(average_mixture_ranks)),
                "mean_mixture_percentile": float(
                    np.mean(
                        [
                            1.0 - (rank - 1.0) / (n_nodes - 1.0)
                            for rank in average_mixture_ranks
                        ]
                    )
                )
                if n_nodes > 1
                else 1.0,
                "top1_is_mixture": int(top1_hits > 0),
                "any_mixture_top3": int(top3_hits > 0),
                "any_mixture_top5": int(top5_hits > 0),
                "mixture_recall_top3": top3_hits / n_mixtures,
                "mixture_recall_top5": top5_hits / n_mixtures,
                "mixture_precision_top3": top3_hits / min(3, n_nodes),
                "mixture_precision_top5": top5_hits / min(5, n_nodes),
                "chance_top1": prevalence,
                "chance_any_top3": exact_any_topk_chance(n_nodes, n_mixtures, 3),
                "chance_any_top5": exact_any_topk_chance(n_nodes, n_mixtures, 5),
                "chance_recall_top3": min(3, n_nodes) / n_nodes,
                "chance_recall_top5": min(5, n_nodes) / n_nodes,
                "negative_edge_count": negative_edge_count,
                "negative_edge_fraction": negative_edge_fraction,
                "negative_orc_node_variance": negative_node_variance,
                "negative_orc_nonzero_nodes": int(
                    (node_table["negative_orc_incidence"] > 0).sum()
                ),
                "orc_component_informative": int(negative_node_variance > 1e-12),
                "tms_betweenness_spearman": tms_betweenness_rho,
            }
        )

        for score_name, score_column in SCORE_COLUMNS.items():
            metrics = evaluate_score(labels, node_table[score_column].to_numpy(float))
            comparator_rows.append(
                {
                    "graph_id": schedule_row["graph_id"],
                    "replicate": int(replicate),
                    "analysis_mode": mode,
                    "injection_count": int(schedule_row["injection_count"]),
                    "score_name": score_name,
                    "prevalence": prevalence,
                    **metrics,
                    "ap_lift_over_prevalence": (
                        metrics["average_precision"] / prevalence
                        if prevalence > 0
                        else float("nan")
                    ),
                }
            )

        for mixture_id in mixture_ids:
            truth_row = truth_lookup.loc[mixture_id]
            parent_runs = semicolon_values(truth_row["actual_parent_runs"])
            parent_controls = [
                run_to_control[parent_run]
                for parent_run in parent_runs
                if parent_run in run_to_control
            ]
            parent_distances: List[float] = []
            for first_index in range(len(parent_controls)):
                for second_index in range(first_index + 1, len(parent_controls)):
                    parent_distances.append(
                        float(
                            distance_full.loc[
                                parent_controls[first_index],
                                parent_controls[second_index],
                            ]
                        )
                    )
            mixture_to_parent = [
                float(distance_full.loc[mixture_id, parent_control])
                for parent_control in parent_controls
            ]
            node_row = node_table[node_table["sample_id"] == mixture_id].iloc[0]
            rank_exact = exact_rank[mixture_id]
            rank_average = average_rank[mixture_id]
            mixture_rows.append(
                {
                    "graph_id": schedule_row["graph_id"],
                    "replicate": int(replicate),
                    "analysis_mode": mode,
                    "injection_count": int(schedule_row["injection_count"]),
                    "sample_id": mixture_id,
                    "mixture_definition_id": truth_row["mixture_definition_id"],
                    "base_set_id": truth_row["base_set_id"],
                    "category": truth_row["category"],
                    "anchor_population": truth_row["anchor_population"],
                    "pattern_id": truth_row["pattern_id"],
                    "n_parents": int(truth_row["n_parents"]),
                    "parents": truth_row["parents"],
                    "actual_parent_runs": truth_row["actual_parent_runs"],
                    "actual_entropy_norm": float(truth_row["actual_entropy_norm"]),
                    "actual_minor_fraction": float(truth_row["actual_minor_fraction"]),
                    "locked_distance_summary": float(truth_row["distance_summary"]),
                    "rank_exact": rank_exact,
                    "rank_average": rank_average,
                    "rank_percentile": (
                        1.0 - (rank_average - 1.0) / (n_nodes - 1.0)
                        if n_nodes > 1
                        else 1.0
                    ),
                    "top1": int(rank_exact <= 1),
                    "top3": int(rank_exact <= 3),
                    "top5": int(rank_exact <= 5),
                    "tms": float(node_row["tms"]),
                    "real_bridge_score": float(node_row["real_bridge_score"]),
                    "betweenness": float(node_row["betweenness"]),
                    "negative_orc_incidence": float(
                        node_row["negative_orc_incidence"]
                    ),
                    "mean_incident_distance": float(
                        node_row["mean_incident_distance"]
                    ),
                    "mean_incident_orc": float(node_row["mean_incident_orc"]),
                    "pca_distance": float(node_row["pca_distance"]),
                    "lof_score": float(node_row["lof_score"]),
                    "parent_controls_present": sum(
                        parent_control in names for parent_control in parent_controls
                    ),
                    "parent_controls_total": len(parent_controls),
                    "parent_distance_min": (
                        float(np.min(parent_distances))
                        if parent_distances
                        else float("nan")
                    ),
                    "parent_distance_mean": (
                        float(np.mean(parent_distances))
                        if parent_distances
                        else float("nan")
                    ),
                    "parent_distance_max": (
                        float(np.max(parent_distances))
                        if parent_distances
                        else float("nan")
                    ),
                    "mixture_to_parent_min": (
                        float(np.min(mixture_to_parent))
                        if mixture_to_parent
                        else float("nan")
                    ),
                    "mixture_to_parent_mean": (
                        float(np.mean(mixture_to_parent))
                        if mixture_to_parent
                        else float("nan")
                    ),
                    "mixture_to_parent_max": (
                        float(np.max(mixture_to_parent))
                        if mixture_to_parent
                        else float("nan")
                    ),
                }
            )

        edge_rows.append(
            {
                "graph_id": schedule_row["graph_id"],
                "replicate": int(replicate),
                "analysis_mode": mode,
                "injection_count": int(schedule_row["injection_count"]),
                "edge_count": len(edge_table),
                "negative_edge_count": negative_edge_count,
                "negative_edge_fraction": negative_edge_fraction,
                "orc_min": float(edge_table["orc"].min())
                if len(edge_table)
                else float("nan"),
                "orc_median": float(edge_table["orc"].median())
                if len(edge_table)
                else float("nan"),
                "orc_max": float(edge_table["orc"].max())
                if len(edge_table)
                else float("nan"),
            }
        )

    graph_path = part_dir / "graph_metrics.tsv"
    mixture_path = part_dir / "mixture_rank_metrics.tsv"
    comparator_path = part_dir / "comparator_metrics.tsv"
    edge_path = part_dir / "edge_diagnostics.tsv"
    pd.DataFrame(graph_rows).to_csv(graph_path, sep="\t", index=False)
    pd.DataFrame(mixture_rows).to_csv(mixture_path, sep="\t", index=False)
    pd.DataFrame(comparator_rows).to_csv(comparator_path, sep="\t", index=False)
    pd.DataFrame(edge_rows).to_csv(edge_path, sep="\t", index=False)

    marker = {
        "status": "COMPLETE",
        "input_signature": input_signature,
        "replicate": int(replicate),
        "analysis_mode": mode,
        "graph_rows": len(graph_rows),
        "mixture_rows": len(mixture_rows),
        "comparator_rows": len(comparator_rows),
        "edge_rows": len(edge_rows),
        "graph_metrics_path": str(graph_path),
        "mixture_metrics_path": str(mixture_path),
        "comparator_metrics_path": str(comparator_path),
        "edge_diagnostics_path": str(edge_path),
        "cache_hit": False,
    }
    write_json(marker_path, marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--truth_manifest", required=True)
    parser.add_argument("--mixture_definitions", required=True)
    parser.add_argument("--batch_dir", required=True)
    parser.add_argument("--lock_json", required=True)
    parser.add_argument("--batch_pass", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--modes", default="paired,r1")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if not Path(args.batch_pass).exists():
        raise SystemExit("[ERROR] Batch benchmark PASS marker is absent.")

    outdir = ensure_dir(args.outdir)
    parts_root = ensure_dir(outdir / "parts")
    schedule = pd.read_csv(args.schedule, sep="\t")
    if len(schedule) != 735:
        raise SystemExit(f"[ERROR] Rare-event schedule has {len(schedule)} rows, expected 735.")
    lock = read_json(args.lock_json)
    locked_knn = int(lock["locked_knn_synthetic"])
    alpha = float(lock["orc_alpha"])
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]

    tasks: List[Tuple[int, str, List[Dict[str, Any]], str, str, str, str, int, float]] = []
    for replicate in sorted(schedule["replicate"].astype(int).unique()):
        records = schedule[schedule["replicate"].astype(int) == int(replicate)].to_dict(
            orient="records"
        )
        if len(records) != 147:
            raise SystemExit(
                f"[ERROR] Replicate {replicate} rare-event schedule contains {len(records)} rows, expected 147."
            )
        for mode in modes:
            distance_path = (
                Path(args.batch_dir)
                / "runs"
                / f"rep_{int(replicate):02d}"
                / mode
                / "js_distance.csv"
            )
            if not distance_path.exists():
                raise SystemExit(f"[ERROR] Missing batch distance matrix: {distance_path}")
            part_dir = parts_root / f"rep_{int(replicate):02d}" / mode
            tasks.append(
                (
                    int(replicate),
                    mode,
                    records,
                    args.truth_manifest,
                    args.mixture_definitions,
                    str(distance_path),
                    str(part_dir),
                    locked_knn,
                    alpha,
                )
            )

    markers: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(process_group, *task): (task[0], task[1]) for task in tasks}
        completed = 0
        for future in as_completed(futures):
            marker = future.result()
            markers.append(marker)
            completed += 1
            replicate, mode = futures[future]
            print(
                f"[RARE GROUP COMPLETE {completed}/{len(tasks)}] "
                f"replicate={replicate:02d}, mode={mode}, "
                f"{'cache' if marker.get('cache_hit') else 'computed'}",
                flush=True,
            )

    graph_parts = [pd.read_csv(marker["graph_metrics_path"], sep="\t") for marker in markers]
    mixture_parts = [pd.read_csv(marker["mixture_metrics_path"], sep="\t") for marker in markers]
    comparator_parts = [
        pd.read_csv(marker["comparator_metrics_path"], sep="\t") for marker in markers
    ]
    edge_parts = [pd.read_csv(marker["edge_diagnostics_path"], sep="\t") for marker in markers]

    graph_table = pd.concat(graph_parts, ignore_index=True).sort_values(
        ["analysis_mode", "replicate", "injection_count", "graph_id"]
    )
    mixture_table = pd.concat(mixture_parts, ignore_index=True).sort_values(
        ["analysis_mode", "replicate", "injection_count", "graph_id", "sample_id"]
    )
    comparator_table = pd.concat(comparator_parts, ignore_index=True).sort_values(
        ["analysis_mode", "replicate", "injection_count", "score_name", "graph_id"]
    )
    edge_table = pd.concat(edge_parts, ignore_index=True).sort_values(
        ["analysis_mode", "replicate", "injection_count", "graph_id"]
    )

    graph_table.to_csv(outdir / "rare_event_graph_metrics.tsv", sep="\t", index=False)
    mixture_table.to_csv(
        outdir / "rare_event_mixture_rank_metrics.tsv", sep="\t", index=False
    )
    comparator_table.to_csv(
        outdir / "rare_event_comparator_metrics.tsv", sep="\t", index=False
    )
    edge_table.to_csv(
        outdir / "rare_event_edge_diagnostics.tsv", sep="\t", index=False
    )

    expected_graph_rows = len(schedule) * len(modes)
    expected_mixture_rows = 2520 if len(modes) == 2 else 1260
    failures = []
    if len(graph_table) != expected_graph_rows:
        failures.append("graph_row_count")
    if len(mixture_table) != expected_mixture_rows:
        failures.append("mixture_row_count")
    if len(comparator_table) != expected_graph_rows * len(SCORE_COLUMNS):
        failures.append("comparator_row_count")
    status = "PASS" if not failures else "FAIL"
    summary = {
        "status": status,
        "analysis_modes": modes,
        "locked_knn": locked_knn,
        "orc_alpha": alpha,
        "schedule_rows": len(schedule),
        "graph_metric_rows": len(graph_table),
        "mixture_metric_rows": len(mixture_table),
        "comparator_metric_rows": len(comparator_table),
        "part_groups": len(markers),
        "part_cache_hits": sum(bool(marker.get("cache_hit")) for marker in markers),
        "failures": failures,
    }
    write_json(outdir / "rare_event_analysis_summary.json", summary)
    print(
        f"[DONE] Rare-event analysis: graphs={len(graph_table)}, "
        f"mixture rows={len(mixture_table)}, status={status}"
    )
    marker = outdir / "RARE_EVENT_BENCHMARK_PASS.txt"
    if status == "PASS":
        marker.write_text("PASS\n", encoding="utf-8")
    else:
        if marker.exists():
            marker.unlink()
        raise SystemExit(5)


if __name__ == "__main__":
    main()
