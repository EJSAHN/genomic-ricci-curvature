# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd

from external_common import (
    classical_mds,
    compute_node_scores,
    diffusion_coordinates,
    distance_population_summary,
    ensure_dir,
    permanova_pseudo_f,
    read_json,
    read_tsv,
    write_json,
    write_tsv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_distance", required=True)
    parser.add_argument("--full_manifest", required=True)
    parser.add_argument("--lock_json", required=True)
    parser.add_argument("--preflight_pass", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--permutations", type=int, default=999)
    args = parser.parse_args()

    if not Path(args.preflight_pass).exists():
        raise SystemExit("[ERROR] External benchmark preflight PASS marker is absent.")

    outdir = ensure_dir(args.outdir)
    lock = read_json(args.lock_json)
    manifest_rows = read_tsv(args.full_manifest)
    manifest = pd.DataFrame(manifest_rows)
    distance = pd.read_csv(args.full_distance, index_col=0)

    names = manifest["sample_accession"].astype(str).tolist()
    if len(names) != 83:
        raise SystemExit(f"[ERROR] Full cohort manifest contains {len(names)} rows, expected 83.")
    if set(names) != set(distance.index.astype(str)) or set(distance.index) != set(distance.columns):
        raise SystemExit("[ERROR] Full cohort distance matrix and manifest sample sets differ.")
    distance = distance.loc[names, names].astype(float)
    if not np.allclose(distance.to_numpy(), distance.to_numpy().T, atol=1e-12):
        raise SystemExit("[ERROR] Full cohort distance matrix is not symmetric.")
    if not np.allclose(np.diag(distance.to_numpy()), 0.0, atol=1e-12):
        raise SystemExit("[ERROR] Full cohort distance matrix diagonal is not zero.")

    knn = int(lock["locked_knn_full_cohort"])
    alpha = float(lock["orc_alpha"])
    node_table, edge_table, graph = compute_node_scores(distance, k=knn, alpha=alpha)

    metadata_columns = [
        "sample_accession",
        "sample_alias",
        "run_accession",
        "population",
        "panel_order",
        "pair_count",
    ]
    metadata = manifest[metadata_columns].rename(columns={"sample_accession": "sample_id"})
    node_table = node_table.merge(metadata, on="sample_id", how="left", validate="one_to_one")

    population_lookup = dict(zip(node_table["sample_id"], node_table["population"]))
    purity_rows: List[Dict[str, Any]] = []
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        same_count = sum(
            population_lookup.get(neighbor) == population_lookup.get(node)
            for neighbor in neighbors
        )
        purity_rows.append(
            {
                "sample_id": node,
                "population": population_lookup[node],
                "degree": len(neighbors),
                "same_population_neighbors": same_count,
                "same_population_neighbor_fraction": (
                    same_count / len(neighbors) if neighbors else float("nan")
                ),
            }
        )
    purity_table = pd.DataFrame(purity_rows)
    node_table = node_table.merge(
        purity_table[
            ["sample_id", "degree", "same_population_neighbor_fraction"]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    mds = classical_mds(distance, dimensions=5).merge(
        metadata, on="sample_id", how="left", validate="one_to_one"
    )
    diffusion = diffusion_coordinates(graph, dimensions=3).merge(
        metadata, on="sample_id", how="left", validate="one_to_one"
    )

    pair_table, distance_summary = distance_population_summary(
        distance, population_lookup
    )
    population_labels = [population_lookup[name] for name in names]
    permanova = permanova_pseudo_f(
        distance,
        labels=population_labels,
        permutations=int(args.permutations),
        seed=int(lock["design_seed"]),
    )

    edge_table["u_population"] = edge_table["u"].map(population_lookup)
    edge_table["v_population"] = edge_table["v"].map(population_lookup)
    edge_table["same_population"] = (
        edge_table["u_population"] == edge_table["v_population"]
    ).astype(int)

    node_table = node_table.sort_values(
        ["real_bridge_score", "sample_id"], ascending=[False, True]
    ).reset_index(drop=True)
    node_table["real_bridge_rank"] = np.arange(1, len(node_table) + 1)
    tms_rank = (
        node_table[["sample_id", "tms"]]
        .sort_values(["tms", "sample_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    tms_rank["tms_rank"] = np.arange(1, len(tms_rank) + 1)
    node_table = node_table.merge(
        tms_rank[["sample_id", "tms_rank"]], on="sample_id", how="left"
    )

    negative_edge_count = int((edge_table["orc"] < 0).sum())
    edge_count = int(len(edge_table))
    population_counts = Counter(node_table["population"].astype(str))
    purity_by_population = (
        purity_table.groupby("population", as_index=False)
        .agg(
            n=("sample_id", "size"),
            mean_same_population_neighbor_fraction=(
                "same_population_neighbor_fraction",
                "mean",
            ),
            median_same_population_neighbor_fraction=(
                "same_population_neighbor_fraction",
                "median",
            ),
        )
        .sort_values("population")
    )

    node_table.to_csv(outdir / "full83_node_scores.tsv", sep="\t", index=False)
    edge_table.to_csv(outdir / "full83_edge_scores.tsv", sep="\t", index=False)
    pair_table.to_csv(outdir / "full83_pairwise_population_distances.tsv", sep="\t", index=False)
    purity_table.to_csv(outdir / "full83_neighbor_population_purity.tsv", sep="\t", index=False)
    purity_by_population.to_csv(
        outdir / "full83_neighbor_population_purity_by_population.tsv",
        sep="\t",
        index=False,
    )
    mds.to_csv(outdir / "full83_mds_coordinates.tsv", sep="\t", index=False)
    diffusion.to_csv(outdir / "full83_diffusion_coordinates.tsv", sep="\t", index=False)
    distance.to_csv(outdir / "full83_js_distance_locked.csv")

    top_real = node_table.head(10)[
        [
            "real_bridge_rank",
            "sample_id",
            "run_accession",
            "sample_alias",
            "population",
            "real_bridge_score",
            "tms",
            "betweenness",
            "negative_orc_incidence",
            "mean_incident_distance",
            "mean_incident_orc",
        ]
    ].to_dict(orient="records")
    top_tms = (
        node_table.sort_values(["tms", "sample_id"], ascending=[False, True])
        .head(10)[
            [
                "tms_rank",
                "sample_id",
                "run_accession",
                "sample_alias",
                "population",
                "tms",
                "real_bridge_score",
                "betweenness",
                "negative_orc_incidence",
            ]
        ]
        .to_dict(orient="records")
    )

    summary = {
        "status": "PASS",
        "dataset": lock["dataset"],
        "n_samples": len(node_table),
        "population_counts": dict(sorted(population_counts.items())),
        "locked_knn": knn,
        "orc_alpha": alpha,
        "graph_connected": bool(nx.is_connected(graph)),
        "graph_components": int(nx.number_connected_components(graph)),
        "graph_edges": edge_count,
        "negative_edge_count": negative_edge_count,
        "negative_edge_fraction": (
            negative_edge_count / edge_count if edge_count else 0.0
        ),
        "orc_min": float(edge_table["orc"].min()) if edge_count else float("nan"),
        "orc_median": float(edge_table["orc"].median()) if edge_count else float("nan"),
        "orc_max": float(edge_table["orc"].max()) if edge_count else float("nan"),
        "mean_same_population_neighbor_fraction": float(
            purity_table["same_population_neighbor_fraction"].mean()
        ),
        "distance_population_summary": distance_summary,
        "permanova": permanova,
        "top10_real_bridge_score": top_real,
        "top10_synthetic_tms": top_tms,
        "interpretation": (
            "Full-cohort scores are descriptive structural rankings. The 83 archived "
            "libraries do not provide validated mixture labels for performance evaluation."
        ),
    }
    write_json(outdir / "full83_geometry_summary.json", summary)

    text = [
        "Finger millet full-cohort reference-free geometry",
        "=================================================",
        "",
        "Status: PASS",
        f"Samples: {len(node_table)}",
        f"Population counts: {dict(sorted(population_counts.items()))}",
        f"Locked kNN: {knn}",
        f"ORC alpha: {alpha}",
        f"Graph connected: {nx.is_connected(graph)}",
        f"Graph edges: {edge_count}",
        f"Negative-edge fraction: {summary['negative_edge_fraction']:.6f}",
        f"Mean same-population neighbor fraction: {summary['mean_same_population_neighbor_fraction']:.3f}",
        f"Within-population JS mean: {distance_summary['within_mean']:.6f}",
        f"Between-population JS mean: {distance_summary['between_mean']:.6f}",
        f"PERMANOVA pseudo-F: {permanova['pseudo_f']:.3f}",
        f"PERMANOVA permutation P: {permanova['permutation_p']:.4f}",
        "",
        "These full-cohort rankings are descriptive. Performance is evaluated only",
        "with the locked true read-level mixtures in the batch and rare-event analyses.",
    ]
    (outdir / "FULL83_GEOMETRY_SUMMARY.txt").write_text(
        "\n".join(text) + "\n", encoding="utf-8"
    )
    (outdir / "FULL83_GEOMETRY_PASS.txt").write_text("PASS\n", encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
