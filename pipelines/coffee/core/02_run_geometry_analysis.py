# -*- coding: utf-8 -*-
"""
Geometry analysis from a precomputed Jensen–Shannon distance matrix.

What this script does
---------------------
1. Reads kmer_js_distance.csv from the preprocessing step.
2. Builds a symmetric k-nearest-neighbor graph.
3. Computes Ollivier–Ricci and Forman curvature.
4. Computes diffusion-map coordinates and topology summaries.
5. Writes an Excel workbook for downstream interpretation.

Primary workbook
----------------
<outdir>/Supplementary_Data_S2_Geometry.xlsx
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_distance_matrix(results_dir: Path) -> pd.DataFrame:
    dist_path = results_dir / "kmer_js_distance.csv"
    if not dist_path.exists():
        raise FileNotFoundError(f"Missing distance matrix: {dist_path}")
    dist = pd.read_csv(dist_path, index_col=0)
    if dist.shape[0] != dist.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if (dist.index != dist.columns).any():
        dist = dist.loc[dist.index, dist.index]
    dist = dist.apply(pd.to_numeric, errors="coerce")
    np.fill_diagonal(dist.values, 0.0)
    return dist


def build_symmetric_knn_graph(dist_df: pd.DataFrame, k: int = 4) -> Tuple[nx.Graph, int]:
    labels = list(dist_df.index)
    n = len(labels)
    dist = dist_df.values.copy()
    np.fill_diagonal(dist, np.inf)

    k_use = min(max(1, k), n - 1)
    while True:
        G = nx.Graph()
        G.add_nodes_from(labels)

        for i, u in enumerate(labels):
            nn_idx = np.argsort(dist[i])[:k_use]
            for j in nn_idx:
                v = labels[j]
                d = float(dist_df.loc[u, v])
                if not math.isfinite(d) or u == v:
                    continue
                if not G.has_edge(u, v):
                    G.add_edge(u, v, length=d, distance=d)

        if nx.is_connected(G) or k_use >= n - 1:
            return G, k_use
        k_use += 1


def transport_w1(cost: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    m, n = cost.shape
    c = cost.ravel()

    A_eq = []
    b_eq = []

    for i in range(m):
        row = np.zeros(m * n, dtype=float)
        row[i * n : (i + 1) * n] = 1.0
        A_eq.append(row)
        b_eq.append(float(a[i]))

    for j in range(n):
        col = np.zeros(m * n, dtype=float)
        col[j::n] = 1.0
        A_eq.append(col)
        b_eq.append(float(b[j]))

    A_eq = np.vstack(A_eq)
    b_eq = np.array(b_eq, dtype=float)
    bounds = [(0.0, None)] * (m * n)

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")
    return float(res.fun)


def ollivier_ricci_curvature(
    G: nx.Graph,
    alpha: float = 0.0,
    weight: str = "length",
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    sp = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
    edge_kappa: Dict[Tuple[str, str], float] = {}

    for u, v in G.edges():
        Nu = list(G.neighbors(u))
        Nv = list(G.neighbors(v))
        supp_u = [u] + Nu
        supp_v = [v] + Nv

        du = len(Nu)
        dv = len(Nv)

        mu = np.zeros(len(supp_u), dtype=float)
        mv = np.zeros(len(supp_v), dtype=float)
        mu[0] = alpha
        mv[0] = alpha
        if du > 0:
            mu[1:] = (1.0 - alpha) / du
        if dv > 0:
            mv[1:] = (1.0 - alpha) / dv

        cost = np.zeros((len(supp_u), len(supp_v)), dtype=float)
        for i, a in enumerate(supp_u):
            for j, b in enumerate(supp_v):
                cost[i, j] = sp[a][b]

        W1 = transport_w1(cost, mu, mv)
        d_uv = sp[u][v]
        kappa = 0.0 if d_uv == 0 else 1.0 - (W1 / d_uv)
        edge_kappa[(u, v)] = float(kappa)

    node_kappa: Dict[str, float] = {}
    for u in G.nodes():
        vals = []
        for v in G.neighbors(u):
            if (u, v) in edge_kappa:
                vals.append(edge_kappa[(u, v)])
            elif (v, u) in edge_kappa:
                vals.append(edge_kappa[(v, u)])
        node_kappa[u] = float(np.mean(vals)) if vals else float("nan")

    return edge_kappa, node_kappa


def forman_edge_curvature_unweighted(G: nx.Graph) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    edge_F: Dict[Tuple[str, str], float] = {}
    for u, v in G.edges():
        tri = len(set(G.neighbors(u)).intersection(G.neighbors(v)))
        F = 4 - G.degree(u) - G.degree(v) + 3 * tri
        edge_F[(u, v)] = float(F)

    node_F: Dict[str, float] = {}
    for u in G.nodes():
        vals = []
        for v in G.neighbors(u):
            if (u, v) in edge_F:
                vals.append(edge_F[(u, v)])
            elif (v, u) in edge_F:
                vals.append(edge_F[(v, u)])
        node_F[u] = float(np.mean(vals)) if vals else float("nan")
    return edge_F, node_F


def diffusion_map_embedding(
    G: nx.Graph,
    nodes: List[str],
    t: int = 1,
    n_components: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    n = len(nodes)
    index = {u: i for i, u in enumerate(nodes)}

    edge_d = [float(data.get("distance", data.get("length", 1.0))) for _, _, data in G.edges(data=True)]
    sigma = float(np.median(edge_d)) if edge_d else 1.0
    if sigma <= 0:
        sigma = float(np.mean(edge_d)) if edge_d else 1.0

    W = np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        i = index[u]
        j = index[v]
        d = float(data.get("distance", data.get("length", 1.0)))
        a = math.exp(-(d * d) / (2.0 * sigma * sigma))
        W[i, j] = a
        W[j, i] = a

    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    P = W / row_sums[:, None]

    evals, evecs = np.linalg.eig(P.T)
    evals = np.real(evals)
    evecs = np.real(evecs)
    idx = np.argsort(-evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    coords = {}
    for comp in range(1, n_components + 1):
        coords[f"DM{comp}"] = (evals[comp] ** t) * evecs[:, comp]

    coords_df = pd.DataFrame(coords, index=nodes)
    eig_df = pd.DataFrame(
        {"eigenvalue": evals[: n_components + 1]},
        index=[f"eig{i}" for i in range(n_components + 1)],
    )
    return coords_df, eig_df, sigma


def von_neumann_entropy_from_graph(G: nx.Graph, nodes: List[str]) -> float:
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=float)
    deg = np.sum(A, axis=1)
    L = np.diag(deg) - A
    tr = float(np.trace(L))
    if tr <= 0:
        return 0.0
    rho = L / tr
    eig = np.linalg.eigvalsh(rho)
    eig = eig[eig > 1e-12]
    return float(-np.sum(eig * np.log(eig)))


def topology_curve(dist_df: pd.DataFrame, n_thresholds: int = 21) -> pd.DataFrame:
    nodes = list(dist_df.index)
    n = len(nodes)
    off = dist_df.values[~np.eye(n, dtype=bool)].ravel()
    qs = np.quantile(off, np.linspace(0.05, 0.95, n_thresholds))

    rows = []
    for t in qs:
        Gt = nx.Graph()
        Gt.add_nodes_from(nodes)
        for i, u in enumerate(nodes):
            for j in range(i + 1, n):
                v = nodes[j]
                d = float(dist_df.iat[i, j])
                if d <= t:
                    Gt.add_edge(u, v)

        V = n
        E = Gt.number_of_edges()
        components = nx.number_connected_components(Gt)

        clique_counts: Dict[int, int] = {}
        for clique in nx.enumerate_all_cliques(Gt):
            s = len(clique)
            clique_counts[s] = clique_counts.get(s, 0) + 1

        euler = 0
        for s, cnt in clique_counts.items():
            euler += ((-1) ** (s - 1)) * cnt

        beta1_graph = E - V + components
        vn = von_neumann_entropy_from_graph(Gt, nodes)
        max_clique = max(clique_counts.keys()) if clique_counts else 0

        rows.append(
            {
                "threshold": float(t),
                "V": int(V),
                "E": int(E),
                "components": int(components),
                "beta1_graph": int(beta1_graph),
                "euler_clique_complex": int(euler),
                "vn_entropy": float(vn),
                "max_clique_size": int(max_clique),
            }
        )
    return pd.DataFrame(rows)


def zscore(s: pd.Series) -> pd.Series:
    mu = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd <= 0:
        sd = 1.0
    return (s - mu) / sd


def compute_node_table(
    G: nx.Graph,
    node_orc: Dict[str, float],
    node_forman: Dict[str, float],
) -> pd.DataFrame:
    deg = dict(G.degree())
    bet = nx.betweenness_centrality(G, weight="length", normalized=True)
    clo = nx.closeness_centrality(G, distance="length")

    mean_edge_len = {}
    for u in G.nodes():
        lens = [float(G[u][v]["length"]) for v in G.neighbors(u)]
        mean_edge_len[u] = float(np.mean(lens)) if lens else float("nan")

    df = pd.DataFrame(
        {
            "degree": pd.Series(deg),
            "betweenness": pd.Series(bet),
            "closeness": pd.Series(clo),
            "mean_edge_length": pd.Series(mean_edge_len),
            "orc_scalar": pd.Series(node_orc),
            "forman_scalar": pd.Series(node_forman),
        }
    )
    df["betweenness_z"] = zscore(df["betweenness"])
    df["mean_edge_length_z"] = zscore(df["mean_edge_length"])
    df["orc_scalar_z"] = zscore(df["orc_scalar"])
    df["mixture_score"] = df["betweenness_z"] + df["mean_edge_length_z"] - df["orc_scalar_z"]
    df = df.sort_values("mixture_score", ascending=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="Folder containing kmer_js_distance.csv")
    ap.add_argument("--outdir", required=True, help="Output folder for workbook")
    ap.add_argument("--knn", type=int, default=4, help="Initial k for symmetric kNN graph")
    ap.add_argument("--alpha", type=float, default=0.0, help="Idleness for Ollivier–Ricci curvature")
    ap.add_argument("--n_thresholds", type=int, default=21, help="Number of topology thresholds")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    dist = read_distance_matrix(results_dir)
    nodes = list(dist.index)

    G, k_used = build_symmetric_knn_graph(dist, k=args.knn)
    edge_orc, node_orc = ollivier_ricci_curvature(G, alpha=args.alpha, weight="length")
    edge_forman, node_forman = forman_edge_curvature_unweighted(G)
    node_table = compute_node_table(G, node_orc, node_forman)
    coords, eig_df, sigma = diffusion_map_embedding(G, nodes, t=1, n_components=3)
    topo = topology_curve(dist, n_thresholds=args.n_thresholds)

    edge_rows = []
    for (u, v), kappa in edge_orc.items():
        d = float(G[u][v]["length"])
        tri = len(set(G.neighbors(u)).intersection(G.neighbors(v)))
        F = edge_forman.get((u, v), edge_forman.get((v, u), float("nan")))
        edge_rows.append(
            {
                "u": u,
                "v": v,
                "distance": d,
                "orc_kappa": float(kappa),
                "forman_F": float(F),
                "triangles": tri,
            }
        )
    edge_table = pd.DataFrame(edge_rows).sort_values("orc_kappa").reset_index(drop=True)
    candidates = node_table.reset_index().rename(columns={"index": "sample"}).copy()
    candidates["rank"] = np.arange(1, len(candidates) + 1)

    params_df = pd.DataFrame(
        [
            {
                "requested_knn": args.knn,
                "used_knn": k_used,
                "orc_alpha": args.alpha,
                "n_thresholds": args.n_thresholds,
                "n_samples": len(nodes),
                "n_edges": G.number_of_edges(),
                "connected": nx.is_connected(G),
                "diffusion_sigma": sigma,
            }
        ]
    )

    out_xlsx = outdir / "Supplementary_Data_S2_Geometry.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        params_df.to_excel(xw, sheet_name="run_params", index=False)
        dist.to_excel(xw, sheet_name="kmer_js_distance")
        topo.to_excel(xw, sheet_name="topology_curve", index=False)
        node_table.reset_index().rename(columns={"index": "sample"}).to_excel(xw, sheet_name="node_metrics", index=False)
        edge_table.to_excel(xw, sheet_name="edge_metrics", index=False)
        coords.reset_index().rename(columns={"index": "sample"}).to_excel(xw, sheet_name="diffusion_coords", index=False)
        eig_df.to_excel(xw, sheet_name="diffusion_eigs")
        candidates.to_excel(xw, sheet_name="ranked_candidates", index=False)
        candidates.to_excel(xw, sheet_name="mixture_ranking", index=False)

    print("\n[DONE] Geometry analysis complete.")
    print(f"  -> Workbook: {out_xlsx}")
    print(f"  -> Graph used k = {k_used} with {G.number_of_edges()} edges")


if __name__ == "__main__":
    main()
