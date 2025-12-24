# -*- coding: utf-8 -*-
"""
Gauss–Euler–Hawking Pipeline for Alignment‑Free GBS
(Distance -> Topology / Curvature / Heat Flow)

WHAT THIS DOES:
  1) Reads an existing k-mer Jensen–Shannon distance matrix (kmer_js_distance.csv)
  2) Builds a symmetric kNN graph (geometry on genomes)
  3) Computes:
       - Ollivier–Ricci curvature (optimal transport; "Gauss" flavor)
       - Forman–Ricci curvature (fast local curvature)
       - Diffusion Map embedding (heat flow; "Hawking" flavor)
  4) Computes a multiscale topological signature over distance thresholds:
       - Euler characteristic curve of the Vietoris–Rips clique complex
       - Von Neumann entropy of the graph Laplacian
  5) Produces paper-ready figures and an Excel supplement (S2)

USAGE EXAMPLE:
  python 02_run_geometry_analysis.py \
    --results_dir "./results" \
    --outdir "./results/gauss_euler" \
    --knn 4 \
    --alpha 0.0 \
    --n_thresholds 21

INPUT REQUIRED:
  - {results_dir}/kmer_js_distance.csv  (square matrix, header+index are sample IDs)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networkx as nx
from scipy.optimize import linprog


# -----------------------------
# I/O helpers
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_fig(out_base: Path, dpi: int = 300) -> None:
    """
    Saves the *current* matplotlib figure to out_base.[png,pdf]
    """
    out_png = out_base.with_suffix(".png")
    out_pdf = out_base.with_suffix(".pdf")
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"[FIG] {out_png}")
    print(f"[FIG] {out_pdf}")
    plt.close()

def read_distance_matrix(results_dir: Path) -> pd.DataFrame:
    dist_path = results_dir / "kmer_js_distance.csv"
    if not dist_path.exists():
        raise FileNotFoundError(f"Missing distance matrix: {dist_path}")
    dist = pd.read_csv(dist_path, index_col=0)
    # basic sanity checks
    if dist.shape[0] != dist.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if (dist.index != dist.columns).any():
        # try to align if same labels but different order
        dist = dist.loc[dist.index, dist.index]
    # ensure numeric
    dist = dist.apply(pd.to_numeric, errors="coerce")
    # force diagonal zeros
    np.fill_diagonal(dist.values, 0.0)
    return dist

def try_read_qc(results_dir: Path) -> pd.DataFrame | None:
    p = results_dir / "qc_metrics_sampled.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


# -----------------------------
# Graph construction
# -----------------------------
def build_symmetric_knn_graph(dist_df: pd.DataFrame, k: int = 4) -> Tuple[nx.Graph, int]:
    """
    Symmetric kNN graph from a distance matrix.
    Edge attribute:
      - length: distance
      - distance: distance
    Increases k until connected (up to n-1).
    """
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
                if not math.isfinite(d):
                    continue
                if u == v:
                    continue
                if not G.has_edge(u, v):
                    G.add_edge(u, v, length=d, distance=d)

        if nx.is_connected(G) or k_use >= n - 1:
            return G, k_use
        k_use += 1


# -----------------------------
# Optimal transport for Ollivier–Ricci curvature
# -----------------------------
def transport_w1(cost: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Solve min <cost, P> s.t. P 1 = a, P^T 1 = b, P>=0 (Earth mover distance).
    Uses scipy.optimize.linprog (HiGHS).
    """
    m, n = cost.shape
    c = cost.ravel()

    A_eq = []
    b_eq = []

    # Row sums
    for i in range(m):
        row = np.zeros(m * n, dtype=float)
        row[i * n:(i + 1) * n] = 1.0
        A_eq.append(row)
        b_eq.append(float(a[i]))

    # Column sums
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

def ollivier_ricci_curvature(G: nx.Graph, alpha: float = 0.0, weight: str = "length"
                             ) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Edge Ollivier–Ricci curvature κ(u,v)=1 - W1(μ_u,μ_v)/d(u,v)
    where μ_u puts alpha mass on u and (1-alpha) uniformly on neighbors(u).
    """
    # all-pairs shortest paths on the graph metric
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


# -----------------------------
# Forman curvature (fast local curvature)
# -----------------------------
def forman_edge_curvature_unweighted(G: nx.Graph) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Unweighted Forman–Ricci curvature (triangle-aware):
      F(e) = 4 - deg(u) - deg(v) + 3 * (#triangles containing e)
    This is a popular fast surrogate of curvature on graphs.
    """
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


# -----------------------------
# Diffusion map (heat flow embedding)
# -----------------------------
def diffusion_map_embedding(G: nx.Graph, nodes: List[str], t: int = 1, n_components: int = 3
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Build a Gaussian affinity from edge distances and compute diffusion map coords.
    """
    n = len(nodes)
    index = {u: i for i, u in enumerate(nodes)}

    # edge distance list for sigma
    edge_d = []
    for _, _, data in G.edges(data=True):
        edge_d.append(float(data.get("distance", data.get("length", 1.0))))
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

    # Markov normalization
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    P = W / row_sums[:, None]

    # Eigen decomposition (small n => dense eig ok)
    evals, evecs = np.linalg.eig(P.T)  # use transpose to get stable real ordering
    evals = np.real(evals)
    evecs = np.real(evecs)

    idx = np.argsort(-evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    coords = {}
    for comp in range(1, n_components + 1):
        coords[f"DM{comp}"] = (evals[comp] ** t) * evecs[:, comp]

    coords_df = pd.DataFrame(coords, index=nodes)
    eig_df = pd.DataFrame({"eigenvalue": evals[:n_components + 1]},
                          index=[f"eig{i}" for i in range(n_components + 1)])
    return coords_df, eig_df, sigma


# -----------------------------
# Multiscale topology + von Neumann entropy
# -----------------------------
def von_neumann_entropy_from_graph(G: nx.Graph, nodes: List[str]) -> float:
    """
    Von Neumann entropy of the graph Laplacian density matrix:
      ρ = L / tr(L),  S = -tr(ρ log ρ)
    """
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
    """
    Euler characteristic curve over distance thresholds using clique enumeration.
    Also returns components and graph cycle rank.
    """
    nodes = list(dist_df.index)
    n = len(nodes)
    off = dist_df.values[~np.eye(n, dtype=bool)].ravel()
    qs = np.quantile(off, np.linspace(0.05, 0.95, n_thresholds))

    rows = []
    for t in qs:
        Gt = nx.Graph()
        Gt.add_nodes_from(nodes)
        # edges at threshold
        for i, u in enumerate(nodes):
            for j in range(i + 1, n):
                v = nodes[j]
                d = float(dist_df.iat[i, j])
                if d <= t:
                    Gt.add_edge(u, v)

        V = n
        E = Gt.number_of_edges()
        components = nx.number_connected_components(Gt)

        # clique counts by size => Euler characteristic
        clique_counts: Dict[int, int] = {}
        for clq in nx.enumerate_all_cliques(Gt):
            s = len(clq)
            clique_counts[s] = clique_counts.get(s, 0) + 1

        euler = 0
        for s, cnt in clique_counts.items():
            euler += ((-1) ** (s - 1)) * cnt

        # graph cycle rank (β1 of the graph)
        beta1_graph = E - V + components

        vn = von_neumann_entropy_from_graph(Gt, nodes)
        max_clique = max(clique_counts.keys()) if clique_counts else 0

        rows.append({
            "threshold": float(t),
            "V": int(V),
            "E": int(E),
            "components": int(components),
            "beta1_graph": int(beta1_graph),
            "euler_clique_complex": int(euler),
            "vn_entropy": float(vn),
            "max_clique_size": int(max_clique)
        })

    return pd.DataFrame(rows)


# -----------------------------
# Node metrics + "mixture/outlier" score (unsupervised)
# -----------------------------
def zscore(s: pd.Series) -> pd.Series:
    mu = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd <= 0:
        sd = 1.0
    return (s - mu) / sd

def compute_node_table(G: nx.Graph,
                       node_orc: Dict[str, float],
                       node_forman: Dict[str, float]) -> pd.DataFrame:
    """
    Centrality + curvature, with an interpretable combined score.
    """
    deg = dict(G.degree())
    bet = nx.betweenness_centrality(G, weight="length", normalized=True)
    clo = nx.closeness_centrality(G, distance="length")

    mean_edge_len = {}
    for u in G.nodes():
        lens = [float(G[u][v]["length"]) for v in G.neighbors(u)]
        mean_edge_len[u] = float(np.mean(lens)) if lens else float("nan")

    df = pd.DataFrame({
        "degree": pd.Series(deg),
        "betweenness": pd.Series(bet),
        "closeness": pd.Series(clo),
        "mean_edge_length": pd.Series(mean_edge_len),
        "orc_scalar": pd.Series(node_orc),
        "forman_scalar": pd.Series(node_forman),
    })

    # Combined "mixture/outlier score":
    #   - high betweenness: sits between clusters (bridge)
    #   - high mean edge length: less "tight" neighborhood
    #   - low ORC: tends to be a saddle/bridge (negative curvature edges nearby)
    df["betweenness_z"] = zscore(df["betweenness"])
    df["mean_edge_length_z"] = zscore(df["mean_edge_length"])
    df["orc_scalar_z"] = zscore(df["orc_scalar"])

    df["mixture_score"] = df["betweenness_z"] + df["mean_edge_length_z"] - df["orc_scalar_z"]
    df = df.sort_values("mixture_score", ascending=False)
    return df


# -----------------------------
# Plotting
# -----------------------------
def plot_euler_curve(topo: pd.DataFrame, out_base: Path) -> None:
    plt.figure(figsize=(9, 6))
    plt.plot(topo["threshold"], topo["euler_clique_complex"], marker="o")
    plt.xlabel("Distance threshold (Jensen–Shannon, k-mer)")
    plt.ylabel("Euler characteristic χ (clique complex)")
    plt.title("Euler Characteristic Curve (Topology fingerprint)")
    plt.grid(True, linestyle=":", alpha=0.5)
    save_fig(out_base)

def plot_entropy_curve(topo: pd.DataFrame, out_base: Path) -> None:
    plt.figure(figsize=(9, 6))
    plt.plot(topo["threshold"], topo["vn_entropy"], marker="o")
    plt.xlabel("Distance threshold (Jensen–Shannon, k-mer)")
    plt.ylabel("Von Neumann entropy S(L)")
    plt.title("Graph Von Neumann Entropy Across Scales (Heat/Quantum view)")
    plt.grid(True, linestyle=":", alpha=0.5)
    save_fig(out_base)

def plot_diffusion(coords: pd.DataFrame, node_table: pd.DataFrame, out_base: Path,
                   top_k: int = 6) -> None:
    # highlight top-k mixture_score
    top_nodes = list(node_table.index[:top_k])

    plt.figure(figsize=(9, 7))
    x = coords["DM1"].values
    y = coords["DM2"].values

    # base scatter
    plt.scatter(x, y, s=60)

    # labels
    for i, name in enumerate(coords.index):
        plt.text(x[i], y[i], name, fontsize=8)

    # overlay top nodes as stars
    for name in top_nodes:
        plt.scatter(coords.loc[name, "DM1"], coords.loc[name, "DM2"], marker="*", s=250,
                    edgecolors="black", linewidths=0.8)

    plt.xlabel("Diffusion map 1")
    plt.ylabel("Diffusion map 2")
    plt.title("Diffusion Geometry (heat flow) + top mixture/outlier candidates (*)")
    plt.grid(True, linestyle=":", alpha=0.5)
    save_fig(out_base)

def plot_curvature_network(G: nx.Graph, edge_orc: Dict[Tuple[str, str], float],
                           node_table: pd.DataFrame, out_base: Path,
                           top_k: int = 6) -> None:
    """
    Network plot with:
      - node size ~ mixture_score (scaled)
      - edge color ~ Ollivier–Ricci curvature
      - top_k nodes outlined
    """
    nodes = list(G.nodes())
    # layout
    pos = nx.spring_layout(G, seed=42, weight="length")

    # edge colors from ORC
    edges = list(G.edges())
    edge_vals = []
    for u, v in edges:
        if (u, v) in edge_orc:
            edge_vals.append(edge_orc[(u, v)])
        else:
            edge_vals.append(edge_orc.get((v, u), 0.0))

    # node sizes
    ms = node_table.loc[nodes, "mixture_score"]
    # scale sizes into a readable range
    s = (ms - ms.min()) / (ms.max() - ms.min() + 1e-12)
    node_sizes = 200 + 1200 * s

    plt.figure(figsize=(10, 8))
    # draw edges (colored)
    nx.draw_networkx_edges(G, pos, edge_color=edge_vals, width=2.0, alpha=0.9)

    # draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes)

    # labels
    nx.draw_networkx_labels(G, pos, font_size=8)

    # outline top nodes
    top_nodes = list(node_table.index[:top_k])
    nx.draw_networkx_nodes(G, pos, nodelist=top_nodes, node_size=node_sizes[[nodes.index(n) for n in top_nodes]],
                           node_color="none", edgecolors="black", linewidths=2.5)

    plt.title("Ollivier–Ricci curvature network (edge color) with mixture/outlier score (node size)")
    plt.axis("off")
    save_fig(out_base)

def plot_mixture_ranking(node_table: pd.DataFrame, out_base: Path, top_n: int = 12) -> None:
    df = node_table.head(top_n).copy()
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, df["mixture_score"].values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mixture / Outlier score (z-composite)")
    plt.title("Top mixture/outlier candidates (unsupervised)")
    plt.grid(True, axis="y", linestyle=":", alpha=0.5)
    save_fig(out_base)

def plot_curvature_hist(edge_orc: Dict[Tuple[str, str], float], out_base: Path, title: str) -> None:
    vals = np.array(list(edge_orc.values()), dtype=float)
    plt.figure(figsize=(9, 6))
    plt.hist(vals, bins=20)
    plt.xlabel("Edge curvature value")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.5)
    save_fig(out_base)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="Folder containing kmer_js_distance.csv")
    ap.add_argument("--outdir", required=True, help="Output folder for figures + Excel")
    ap.add_argument("--knn", type=int, default=4, help="k for symmetric kNN graph (auto-increases until connected)")
    ap.add_argument("--alpha", type=float, default=0.0, help="Idleness for Ollivier–Ricci (0.0 gives strongest contrast)")
    ap.add_argument("--n_thresholds", type=int, default=21, help="How many thresholds in topology curve (5%%..95%% quantiles)")
    ap.add_argument("--topk", type=int, default=6, help="How many top candidates to mark with '*'")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)

    ensure_dir(outdir)
    fig_main = outdir / "figures" / "main"
    fig_sup  = outdir / "figures" / "supplementary"
    ensure_dir(fig_main)
    ensure_dir(fig_sup)

    # Load distance matrix
    dist = read_distance_matrix(results_dir)
    nodes = list(dist.index)
    n = len(nodes)
    print(f"[OK] Loaded distance matrix: {n} samples")

    # Build graph
    G, k_used = build_symmetric_knn_graph(dist, k=args.knn)
    print(f"[OK] Built symmetric kNN graph: k={k_used}, edges={G.number_of_edges()}, connected={nx.is_connected(G)}")

    # Curvature
    edge_orc, node_orc = ollivier_ricci_curvature(G, alpha=args.alpha, weight="length")
    edge_forman, node_forman = forman_edge_curvature_unweighted(G)
    print("[OK] Curvature computed (Ollivier–Ricci + Forman)")

    # Node table (unsupervised mixture/outlier)
    node_table = compute_node_table(G, node_orc, node_forman)

    # Diffusion embedding
    coords, eig_df, sigma = diffusion_map_embedding(G, nodes, t=1, n_components=3)
    print(f"[OK] Diffusion map computed (sigma={sigma:.6g})")

    # Topology curve
    topo = topology_curve(dist, n_thresholds=args.n_thresholds)
    print("[OK] Topology curve computed")

    # ----------------- FIGURES (>=5 main) -----------------
    plot_euler_curve(topo, fig_main / "EulerCharacteristicCurve")
    plot_entropy_curve(topo, fig_main / "VonNeumannEntropyCurve")
    plot_diffusion(coords, node_table, fig_main / "DiffusionGeometry", top_k=args.topk)
    plot_curvature_network(G, edge_orc, node_table, fig_main / "RicciCurvatureNetwork", top_k=args.topk)
    plot_mixture_ranking(node_table, fig_main / "MixtureOutlierRanking", top_n=min(12, n))

    # ----------------- SUPPLEMENTARY FIGURES -----------------
    plot_curvature_hist(edge_orc, fig_sup / "ORC_Histogram",
                        title="Edge Ollivier–Ricci curvature distribution")
    plot_curvature_hist(edge_forman, fig_sup / "Forman_Histogram",
                        title="Edge Forman–Ricci curvature distribution")

    # ----------------- EXCEL S2 -----------------
    # Edge table
    edge_rows = []
    for (u, v), kappa in edge_orc.items():
        d = float(G[u][v]["length"])
        # triangle count
        tri = len(set(G.neighbors(u)).intersection(G.neighbors(v)))
        F = edge_forman.get((u, v), edge_forman.get((v, u), float("nan")))
        edge_rows.append({"u": u, "v": v, "distance": d, "orc_kappa": kappa, "forman_F": F, "triangles": tri})
    edge_table = pd.DataFrame(edge_rows).sort_values("orc_kappa")

    # candidate table
    candidates = node_table.reset_index().rename(columns={"index": "sample"}).copy()
    candidates["rank"] = np.arange(1, len(candidates) + 1)

    out_xlsx = outdir / "Supplementary_Data_S2_GaussEuler.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        dist.to_excel(xw, sheet_name="kmer_js_distance")
        topo.to_excel(xw, sheet_name="topology_curve", index=False)
        node_table.reset_index().rename(columns={"index": "sample"}).to_excel(xw, sheet_name="node_metrics", index=False)
        edge_table.to_excel(xw, sheet_name="edge_metrics", index=False)
        coords.reset_index().rename(columns={"index": "sample"}).to_excel(xw, sheet_name="diffusion_coords", index=False)
        eig_df.to_excel(xw, sheet_name="diffusion_eigs")
        candidates.to_excel(xw, sheet_name="mixture_ranking", index=False)

    print(f"[DONE] Excel written: {out_xlsx}")
    print(f"[DONE] Figures: {outdir / 'figures' / 'main'} and {outdir / 'figures' / 'supplementary'}")
    print()
    print("INTERPRETATION TIP:")
    print("  - High mixture_score nodes are *bridges* in the genomic geometry.")
    print("  - Edges with negative ORC curvature often indicate 'bottlenecks'/admixture bridges.")
    print("  - The Euler curve + entropy curve show at which distance scale structure 'turns on'.")


if __name__ == "__main__":
    main()
