# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd

from external_common import (
    compute_node_scores,
    ensure_dir,
    evaluate_score,
    pairwise_js_distance_matrix,
    read_json,
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


def load_signature(cache_path: str, mode: str) -> np.ndarray:
    data = np.load(cache_path, allow_pickle=False)
    if mode == "paired":
        return np.asarray(data["paired_signature"], dtype=np.float64)
    if mode == "r1":
        return np.asarray(data["r1_signature"], dtype=np.float64)
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketch_manifest", required=True)
    parser.add_argument("--truth_manifest", required=True)
    parser.add_argument("--mixture_definitions", required=True)
    parser.add_argument("--lock_json", required=True)
    parser.add_argument("--sketch_pass", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--modes", default="paired,r1")
    args = parser.parse_args()

    if not Path(args.sketch_pass).exists():
        raise SystemExit("[ERROR] Generated sketch PASS marker is absent.")

    outdir = ensure_dir(args.outdir)
    lock = read_json(args.lock_json)
    sketches = pd.read_csv(args.sketch_manifest, sep="\t")
    truth = pd.read_csv(args.truth_manifest, sep="\t")
    definitions = pd.read_csv(args.mixture_definitions, sep="\t")
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    if sorted(modes) != ["paired", "r1"]:
        raise SystemExit(f"[ERROR] Expected paired and r1 modes; observed {modes}")

    merged = sketches.merge(
        truth,
        on=[
            "replicate",
            "sample_id",
            "class_label",
            "category",
            "pattern_id",
            "read_pairs",
        ],
        how="left",
        suffixes=("", "_truth"),
        validate="one_to_one",
    )
    if merged["actual_entropy_norm"].isna().any():
        raise SystemExit("[ERROR] Generated sketch and truth manifests do not match.")
    definition_lookup = definitions[
        [
            "mixture_definition_id",
            "base_set_id",
            "anchor_population",
            "distance_summary",
            "target_entropy_norm",
            "target_minor_fraction",
        ]
    ]
    merged = merged.merge(
        definition_lookup,
        on="mixture_definition_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_definition"),
    )

    locked_knn = int(lock["locked_knn_synthetic"])
    alpha = float(lock["orc_alpha"])
    run_rows: List[Dict[str, Any]] = []
    comparator_rows: List[Dict[str, Any]] = []
    category_rows: List[Dict[str, Any]] = []
    pattern_rows: List[Dict[str, Any]] = []
    node_tables: List[pd.DataFrame] = []

    for replicate in sorted(merged["replicate"].astype(int).unique()):
        replicate_table = (
            merged[merged["replicate"].astype(int) == int(replicate)]
            .sort_values("sample_id")
            .reset_index(drop=True)
        )
        if len(replicate_table) != 112:
            raise SystemExit(
                f"[ERROR] Replicate {replicate} contains {len(replicate_table)} libraries, expected 112."
            )
        names = replicate_table["sample_id"].astype(str).tolist()

        for mode in modes:
            print(
                f"[BATCH] replicate={int(replicate):02d}, mode={mode}, n={len(names)}",
                flush=True,
            )
            signatures = [
                load_signature(path, mode)
                for path in replicate_table["sketch_cache"].astype(str)
            ]
            distance = pairwise_js_distance_matrix(signatures, names)
            node_table, edge_table, graph = compute_node_scores(
                distance,
                k=locked_knn,
                alpha=alpha,
            )
            node_table["raw_sum"] = (
                node_table["betweenness"]
                + node_table["negative_orc_incidence"]
            )
            truth_columns = [
                "sample_id",
                "class_label",
                "mixture_definition_id",
                "base_set_id",
                "category",
                "anchor_population",
                "pattern_id",
                "n_parents",
                "parents",
                "parent_runs",
                "weights",
                "actual_weights",
                "actual_entropy_norm",
                "actual_minor_fraction",
                "distance_summary",
                "read_pairs",
            ]
            node_table = node_table.merge(
                replicate_table[truth_columns],
                on="sample_id",
                how="left",
                validate="one_to_one",
            )
            node_table["replicate"] = int(replicate)
            node_table["analysis_mode"] = mode
            node_table["is_mixture"] = (
                node_table["class_label"].astype(str) == "synthetic_mixture"
            ).astype(int)

            run_dir = ensure_dir(
                outdir / "runs" / f"rep_{int(replicate):02d}" / mode
            )
            distance.to_csv(run_dir / "js_distance.csv")
            node_table.to_csv(run_dir / "node_scores.tsv", sep="\t", index=False)
            edge_table.to_csv(run_dir / "edge_scores.tsv", sep="\t", index=False)

            labels = node_table["is_mixture"].to_numpy(dtype=int)
            tms_metrics = evaluate_score(labels, node_table["tms"].to_numpy(float))
            mixture_nodes = node_table[node_table["is_mixture"] == 1]
            entropy_rho, entropy_p = spearman_safe(
                mixture_nodes["actual_entropy_norm"], mixture_nodes["tms"]
            )
            minor_rho, minor_p = spearman_safe(
                mixture_nodes["actual_minor_fraction"], mixture_nodes["tms"]
            )
            distance_rho, distance_p = spearman_safe(
                mixture_nodes["distance_summary"], mixture_nodes["tms"]
            )
            negative_edge_count = int((edge_table["orc"] < 0).sum())
            negative_edge_fraction = (
                negative_edge_count / len(edge_table) if len(edge_table) else 0.0
            )
            negative_node_variance = float(
                node_table["negative_orc_incidence"].var(ddof=0)
            )
            run_payload = {
                "replicate": int(replicate),
                "analysis_mode": mode,
                "n_samples": int(len(node_table)),
                "n_controls": int((labels == 0).sum()),
                "n_mixtures": int((labels == 1).sum()),
                "graph_connected": bool(nx.is_connected(graph)),
                "graph_components": int(nx.number_connected_components(graph)),
                "graph_edges": int(graph.number_of_edges()),
                "locked_knn": locked_knn,
                "orc_alpha": alpha,
                **tms_metrics,
                "tms_entropy_spearman_rho": entropy_rho,
                "tms_entropy_spearman_p": entropy_p,
                "tms_minor_fraction_spearman_rho": minor_rho,
                "tms_minor_fraction_spearman_p": minor_p,
                "tms_parent_distance_spearman_rho": distance_rho,
                "tms_parent_distance_spearman_p": distance_p,
                "negative_edge_count": negative_edge_count,
                "negative_edge_fraction": negative_edge_fraction,
                "negative_orc_node_variance": negative_node_variance,
                "orc_component_informative": int(negative_node_variance > 1e-12),
            }
            write_json(run_dir / "run_metrics.json", run_payload)
            run_rows.append(run_payload)

            for score_name, column in SCORE_COLUMNS.items():
                metrics = evaluate_score(labels, node_table[column].to_numpy(float))
                comparator_rows.append(
                    {
                        "replicate": int(replicate),
                        "analysis_mode": mode,
                        "score_name": score_name,
                        **metrics,
                    }
                )

            controls = node_table[node_table["is_mixture"] == 0]
            for category, category_mixtures in mixture_nodes.groupby(
                "category", sort=True
            ):
                comparison = pd.concat(
                    [controls, category_mixtures], ignore_index=True
                )
                metrics = evaluate_score(
                    comparison["is_mixture"].to_numpy(int),
                    comparison["tms"].to_numpy(float),
                )
                category_rows.append(
                    {
                        "replicate": int(replicate),
                        "analysis_mode": mode,
                        "category": category,
                        "n_controls": len(controls),
                        "n_mixtures": len(category_mixtures),
                        "mean_tms": float(category_mixtures["tms"].mean()),
                        "median_tms": float(category_mixtures["tms"].median()),
                        "mean_parent_distance": float(
                            category_mixtures["distance_summary"].mean()
                        ),
                        **metrics,
                    }
                )

            for pattern_id, pattern_mixtures in mixture_nodes.groupby(
                "pattern_id", sort=True
            ):
                comparison = pd.concat(
                    [controls, pattern_mixtures], ignore_index=True
                )
                metrics = evaluate_score(
                    comparison["is_mixture"].to_numpy(int),
                    comparison["tms"].to_numpy(float),
                )
                pattern_rows.append(
                    {
                        "replicate": int(replicate),
                        "analysis_mode": mode,
                        "pattern_id": pattern_id,
                        "n_controls": len(controls),
                        "n_mixtures": len(pattern_mixtures),
                        "mean_tms": float(pattern_mixtures["tms"].mean()),
                        "median_tms": float(pattern_mixtures["tms"].median()),
                        "mean_entropy": float(
                            pattern_mixtures["actual_entropy_norm"].mean()
                        ),
                        "mean_minor_fraction": float(
                            pattern_mixtures["actual_minor_fraction"].mean()
                        ),
                        **metrics,
                    }
                )

            node_tables.append(node_table)

    run_metrics = pd.DataFrame(run_rows).sort_values(
        ["analysis_mode", "replicate"]
    )
    comparator_metrics = pd.DataFrame(comparator_rows).sort_values(
        ["analysis_mode", "score_name", "replicate"]
    )
    category_metrics = pd.DataFrame(category_rows).sort_values(
        ["analysis_mode", "category", "replicate"]
    )
    pattern_metrics = pd.DataFrame(pattern_rows).sort_values(
        ["analysis_mode", "pattern_id", "replicate"]
    )
    all_nodes = pd.concat(node_tables, ignore_index=True).sort_values(
        ["analysis_mode", "replicate", "sample_id"]
    )

    run_metrics.to_csv(outdir / "batch_run_metrics.tsv", sep="\t", index=False)
    comparator_metrics.to_csv(
        outdir / "batch_comparator_metrics.tsv", sep="\t", index=False
    )
    category_metrics.to_csv(
        outdir / "batch_category_metrics.tsv", sep="\t", index=False
    )
    pattern_metrics.to_csv(
        outdir / "batch_pattern_metrics.tsv", sep="\t", index=False
    )
    all_nodes.to_csv(outdir / "batch_node_scores_all.tsv", sep="\t", index=False)

    parameters = {
        "status": "PASS",
        "analysis_modes": modes,
        "replicates": sorted(run_metrics["replicate"].unique().tolist()),
        "locked_knn": locked_knn,
        "orc_alpha": alpha,
        "kmer": int(lock["kmer"]),
        "sketch_dimension": int(lock["sketch_dimension"]),
        "run_count": len(run_metrics),
        "node_score_rows": len(all_nodes),
        "parameter_tuning_on_mixture_labels": False,
    }
    write_json(outdir / "batch_analysis_parameters.json", parameters)
    (outdir / "BATCH_BENCHMARK_PASS.txt").write_text("PASS\n", encoding="utf-8")
    print(
        f"[DONE] Batch benchmark: runs={len(run_metrics)}, "
        f"node rows={len(all_nodes)}, locked kNN={locked_knn}"
    )


if __name__ == "__main__":
    main()
