# -*- coding: utf-8 -*-
"""
Analyze paired-read synthetic libraries with the same hashed k-mer, JS-distance,
kNN, Ollivier-Ricci, and TMS definitions used by the idealized calibration baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

from read_level_common import (
    compute_node_scores,
    ensure_dir,
    evaluate_score,
    kmer_sketch_probability,
    pairwise_js_distance_matrix,
    read_fastq_sequences,
    spearman_safe,
    write_json,
)


SCORE_COLUMNS = {
    "tms": "tms",
    "betweenness": "betweenness",
    "negative_orc": "negative_orc_incidence",
    "mean_incident_distance": "mean_incident_distance",
    "pca_distance": "pca_distance",
    "local_outlier_factor": "lof_score",
    "raw_betweenness_plus_negative_orc": "raw_sum",
}


def sketch_library(r1_path: str, r2_path: str, mode: str, kmer: int, sketch: int) -> np.ndarray:
    sig1 = kmer_sketch_probability(
        read_fastq_sequences(r1_path),
        k=kmer,
        sketch_size=sketch,
    )
    if mode == "r1":
        return sig1
    if mode != "paired":
        raise ValueError(f"Unsupported analysis mode: {mode}")
    sig2 = kmer_sketch_probability(
        read_fastq_sequences(r2_path),
        k=kmer,
        sketch_size=sketch,
    )
    combined = (sig1 + sig2) / 2.0
    return combined / combined.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_fastq_manifest", required=True)
    parser.add_argument("--truth_manifest", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--modes", default="r1,paired")
    parser.add_argument("--kmer", type=int, default=17)
    parser.add_argument("--sketch", type=int, default=16384)
    parser.add_argument("--knn", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    generated = pd.read_csv(args.generated_fastq_manifest, sep="\t")
    truth = pd.read_csv(args.truth_manifest, sep="\t")
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    invalid = [x for x in modes if x not in {"r1", "paired"}]
    if invalid:
        raise SystemExit(f"[ERR] Invalid modes: {invalid}")

    merged = generated.merge(
        truth[
            [
                "scenario",
                "replicate",
                "sample_id",
                "class_label",
                "pattern_id",
                "n_parents",
                "parents",
                "target_entropy_norm",
                "target_minor_fraction",
                "actual_entropy_norm",
                "actual_minor_fraction",
                "read_pairs",
            ]
        ],
        on=["scenario", "replicate", "sample_id", "class_label", "read_pairs"],
        how="left",
        validate="one_to_one",
    )
    if merged["pattern_id"].isna().any():
        raise SystemExit("[ERR] Generated FASTQ manifest and truth manifest do not match")

    run_metric_rows: List[Dict] = []
    comparator_rows: List[Dict] = []
    ratio_rows: List[Dict] = []
    node_rows: List[pd.DataFrame] = []

    groups = merged.groupby(["scenario", "replicate"], sort=True)
    for (scenario, replicate), group in groups:
        group = group.sort_values("sample_id").reset_index(drop=True)
        names = group["sample_id"].astype(str).tolist()

        for mode in modes:
            print(
                f"[RUN] scenario={scenario}, replicate={int(replicate):02d}, "
                f"mode={mode}, n={len(group)}"
            )
            signatures = []
            for row in group.itertuples(index=False):
                signatures.append(
                    sketch_library(
                        r1_path=str(row.r1_path),
                        r2_path=str(row.r2_path),
                        mode=mode,
                        kmer=args.kmer,
                        sketch=args.sketch,
                    )
                )

            distance = pairwise_js_distance_matrix(signatures, names)
            node_df, edge_df, graph = compute_node_scores(
                distance=distance,
                k=args.knn,
                alpha=args.alpha,
            )
            node_df["raw_sum"] = (
                node_df["betweenness"] + node_df["negative_orc_incidence"]
            )
            node_df = node_df.merge(
                group[
                    [
                        "sample_id",
                        "class_label",
                        "pattern_id",
                        "n_parents",
                        "parents",
                        "target_entropy_norm",
                        "target_minor_fraction",
                        "actual_entropy_norm",
                        "actual_minor_fraction",
                        "read_pairs",
                    ]
                ],
                on="sample_id",
                how="left",
                validate="one_to_one",
            )
            node_df["scenario"] = scenario
            node_df["replicate"] = int(replicate)
            node_df["analysis_mode"] = mode
            node_df["is_mixture"] = (
                node_df["class_label"].astype(str) == "synthetic_mixture"
            ).astype(int)

            run_dir = ensure_dir(
                outdir / "runs" / str(scenario) / f"rep_{int(replicate):02d}" / mode
            )
            distance.to_csv(run_dir / "js_distance.csv")
            node_df.to_csv(run_dir / "node_scores.tsv", sep="\t", index=False)
            edge_df.to_csv(run_dir / "edge_scores.tsv", sep="\t", index=False)

            y = node_df["is_mixture"].to_numpy(dtype=int)
            tms_metrics = evaluate_score(y, node_df["tms"].to_numpy(dtype=float))
            mixtures = node_df[node_df["is_mixture"] == 1]
            entropy_rho, entropy_p = spearman_safe(
                mixtures["actual_entropy_norm"],
                mixtures["tms"],
            )
            minor_rho, minor_p = spearman_safe(
                mixtures["actual_minor_fraction"],
                mixtures["tms"],
            )

            run_payload = {
                "scenario": str(scenario),
                "replicate": int(replicate),
                "analysis_mode": mode,
                "n_samples": int(len(node_df)),
                "n_mixtures": int(y.sum()),
                "n_controls": int((1 - y).sum()),
                "graph_connected": bool(nx.is_connected(graph)),
                "graph_components": int(nx.number_connected_components(graph)),
                "graph_edges": int(graph.number_of_edges()),
                "kmer": args.kmer,
                "sketch": args.sketch,
                "knn": args.knn,
                "alpha": args.alpha,
                "roc_auc": tms_metrics["roc_auc"],
                "average_precision": tms_metrics["average_precision"],
                "best_f1": tms_metrics["best_f1"],
                "best_threshold": tms_metrics["best_threshold"],
                "tms_entropy_spearman_rho": entropy_rho,
                "tms_entropy_spearman_p": entropy_p,
                "tms_minor_fraction_spearman_rho": minor_rho,
                "tms_minor_fraction_spearman_p": minor_p,
            }
            write_json(run_dir / "run_metrics.json", run_payload)
            run_metric_rows.append(run_payload)

            for score_name, column in SCORE_COLUMNS.items():
                metrics = evaluate_score(y, node_df[column].to_numpy(dtype=float))
                comparator_rows.append(
                    {
                        "scenario": scenario,
                        "replicate": int(replicate),
                        "analysis_mode": mode,
                        "score_name": score_name,
                        **metrics,
                    }
                )

            for pattern_id, pattern_group in mixtures.groupby("pattern_id", sort=True):
                combined = pd.concat(
                    [
                        pattern_group,
                        node_df[node_df["is_mixture"] == 0],
                    ],
                    ignore_index=True,
                )
                metrics = evaluate_score(
                    combined["is_mixture"].to_numpy(dtype=int),
                    combined["tms"].to_numpy(dtype=float),
                )
                ratio_rows.append(
                    {
                        "scenario": scenario,
                        "replicate": int(replicate),
                        "analysis_mode": mode,
                        "pattern_id": pattern_id,
                        "n_mixtures": int(len(pattern_group)),
                        "mean_tms": float(pattern_group["tms"].mean()),
                        "median_tms": float(pattern_group["tms"].median()),
                        **metrics,
                    }
                )

            node_rows.append(node_df)

    run_metrics = pd.DataFrame(run_metric_rows).sort_values(
        ["scenario", "analysis_mode", "replicate"]
    )
    comparators = pd.DataFrame(comparator_rows).sort_values(
        ["scenario", "analysis_mode", "score_name", "replicate"]
    )
    ratios = pd.DataFrame(ratio_rows).sort_values(
        ["scenario", "analysis_mode", "pattern_id", "replicate"]
    )
    nodes = pd.concat(node_rows, ignore_index=True).sort_values(
        ["scenario", "analysis_mode", "replicate", "sample_id"]
    )

    run_metrics.to_csv(outdir / "run_metrics.tsv", sep="\t", index=False)
    comparators.to_csv(outdir / "comparator_metrics.tsv", sep="\t", index=False)
    ratios.to_csv(outdir / "ratio_metrics.tsv", sep="\t", index=False)
    nodes.to_csv(outdir / "node_scores_all.tsv", sep="\t", index=False)

    parameters = {
        "modes": modes,
        "kmer": args.kmer,
        "sketch": args.sketch,
        "knn": args.knn,
        "alpha": args.alpha,
        "n_runs": int(len(run_metrics)),
    }
    write_json(outdir / "analysis_parameters.json", parameters)

    print(f"[DONE] Run metrics: {outdir / 'run_metrics.tsv'}")
    print(f"[DONE] Comparator metrics: {outdir / 'comparator_metrics.tsv'}")
    print(f"[DONE] Node scores: {outdir / 'node_scores_all.tsv'}")


if __name__ == "__main__":
    main()
