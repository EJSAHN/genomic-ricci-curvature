# -*- coding: utf-8 -*-
"""
utils_visualization.py

A standalone utility to visualize high-dimensional k-mer matrices.
It performs the following steps:
1. Loads a k-mer feature matrix (CSV/TSV or .npz).
2. Normalizes rows to probability distributions.
3. Computes pairwise distances (Jensen-Shannon or Nei-like).
4. Performs Principal Coordinate Analysis (PCoA).
5. (Optional) Builds a Neighbor-Joining (NJ) tree.
6. Generates 2D scatter plots, optionally colored by metadata or mixture scores.

Dependencies: numpy, pandas, matplotlib

Usage Example:
  python utils_visualization.py \
    --matrix "./results/kmer_matrix.npz" \
    --out_prefix "./results/figures/big_picture" \
    --metric jsd \
    --make_nj
"""

import os
import math
import argparse
import numpy as np

# Set matplotlib backend to Agg (non-interactive) to prevent display errors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_table_auto(path: str):
    """Reads a table, auto-detecting comma or tab delimiter."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.readline()
    sep = "," if head.count(",") >= head.count("\t") else "\t"
    import pandas as pd
    return pd.read_csv(path, sep=sep)

def load_matrix(path: str, id_col: str = "sample_id"):
    """Loads feature matrix from .npz or .csv/.tsv."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        # Accept flexible keys
        if "X" in data:
            X = data["X"]
        elif "matrix" in data:
            X = data["matrix"]
        else:
            raise ValueError(f"[ERROR] NPZ must contain key 'X' (or 'matrix'). Keys={list(data.keys())}")

        if "ids" in data:
            ids = data["ids"]
        elif "samples" in data:
            ids = data["samples"]
        else:
            raise ValueError(f"[ERROR] NPZ must contain key 'ids' (or 'samples'). Keys={list(data.keys())}")

        ids = [str(x) for x in ids.tolist()]
        X = np.asarray(X, dtype=float)
        return ids, X

    # CSV/TSV
    df = read_table_auto(path)
    cols = list(df.columns)
    if id_col in cols:
        ids = df[id_col].astype(str).tolist()
        feat_df = df.drop(columns=[id_col])
    else:
        # Fallback: assume first column is ID
        ids = df.iloc[:, 0].astype(str).tolist()
        feat_df = df.iloc[:, 1:]

    X = feat_df.to_numpy(dtype=float)
    return ids, X

def normalize_rows_to_prob(X: np.ndarray, eps: float = 1e-12):
    if np.any(X < 0):
        raise ValueError("[ERROR] Matrix has negative values. Expected counts or non-negative features.")
    row_sums = X.sum(axis=1, keepdims=True)
    # Check for zero rows
    if np.any(row_sums <= 0):
        bad = np.where(row_sums.squeeze() <= 0)[0][:10]
        print(f"[WARN] Some rows sum to 0. Indices: {bad}. Adding epsilon.")
        row_sums[row_sums <= 0] = 1.0  # Avoid division by zero
    
    P = X / row_sums
    # Clip to avoid log(0)
    P = np.clip(P, eps, 1.0)
    # Renormalize
    P = P / P.sum(axis=1, keepdims=True)
    return P

def cosine_similarity_matrix(P: np.ndarray):
    # P: (n, d)
    norms = np.linalg.norm(P, axis=1)
    norms = np.where(norms == 0, 1e-12, norms)
    S = (P @ P.T) / np.outer(norms, norms)
    return np.clip(S, -1.0, 1.0)

def nei_like_distance(P: np.ndarray, eps: float = 1e-12):
    # Nei-like genetic identity ~ cosine similarity; D = -ln(I)
    S = cosine_similarity_matrix(P)
    S = np.clip(S, eps, 1.0)
    D = -np.log(S)
    np.fill_diagonal(D, 0.0)
    return D

def js_divergence(p: np.ndarray, q: np.ndarray):
    # Jensen-Shannon divergence (natural log base)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def jsd_distance(P: np.ndarray, sqrt: bool = True):
    n = P.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        pi = P[i]
        for j in range(i + 1, n):
            d = js_divergence(pi, P[j])
            if sqrt:
                d = math.sqrt(max(d, 0.0))
            D[i, j] = d
            D[j, i] = d
    return D

def pcoa(D: np.ndarray, n_components: int = 2):
    # Classical MDS / PCoA
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (J @ D2 @ J)

    # Eigen decomposition
    evals, evecs = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Keep positive eigenvalues
    pos = evals > 1e-12
    evals = evals[pos]
    evecs = evecs[:, pos]

    k = min(n_components, len(evals))
    coords = evecs[:, :k] * np.sqrt(evals[:k])
    return coords, evals

def neighbor_joining_newick(D: np.ndarray, labels):
    # Basic Neighbor-Joining implementation; returns Newick string
    labels = [str(x) for x in labels]
    n = len(labels)

    dist = D.copy().astype(float)
    taxa = labels[:]
    subtree = {t: t for t in taxa}

    while n > 2:
        r = dist.sum(axis=1)
        Q = np.full((n, n), np.inf, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] = (n - 2) * dist[i, j] - r[i] - r[j]

        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i > j:
            i, j = j, i

        ti, tj = taxa[i], taxa[j]
        dij = dist[i, j]
        delta = (r[i] - r[j]) / (n - 2) if n > 2 else 0
        limb_i = 0.5 * (dij + delta)
        limb_j = 0.5 * (dij - delta)

        limb_i = max(limb_i, 0.0)
        limb_j = max(limb_j, 0.0)

        new_label = f"NJ_{len(subtree)}"
        new_subtree = f"({subtree[ti]}:{limb_i:.6f},{subtree[tj]}:{limb_j:.6f})"

        new_row = []
        for k in range(n):
            if k in (i, j):
                continue
            dik = dist[i, k]
            djk = dist[j, k]
            duk = 0.5 * (dik + djk - dij)
            new_row.append(duk)

        # Update distance matrix
        keep = [k for k in range(n) if k not in (i, j)]
        dist_keep = dist[np.ix_(keep, keep)]
        n2 = dist_keep.shape[0]

        dist_new = np.zeros((n2 + 1, n2 + 1), dtype=float)
        dist_new[:n2, :n2] = dist_keep
        dist_new[n2, :n2] = new_row
        dist_new[:n2, n2] = new_row

        taxa_keep = [taxa[k] for k in keep]
        for t in (ti, tj):
            subtree.pop(t, None)
        subtree[new_label] = new_subtree
        taxa = taxa_keep + [new_label]
        dist = dist_new
        n = len(taxa)

    a, b = taxa[0], taxa[1]
    dab = dist[0, 1]
    dab = max(dab, 0.0)
    newick = f"({subtree[a]}:{dab/2:.6f},{subtree[b]}:{dab/2:.6f});"
    return newick

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, help="k-mer feature matrix: .csv/.tsv or .npz (keys: X, ids)")
    ap.add_argument("--id_col", default="sample_id", help="ID column name if using CSV/TSV")
    ap.add_argument("--orc_csv", default=None, help="CSV/TSV with mixture score (needs sample_id + mix_col)")
    ap.add_argument("--mix_col", default="mixture_score", help="mixture score column name in orc_csv")
    ap.add_argument("--meta_csv", default=None, help="optional metadata CSV/TSV (sample_id + group_col)")
    ap.add_argument("--group_col", default=None, help="optional grouping column (e.g., species)")
    ap.add_argument("--metric", choices=["jsd", "nei"], default="jsd", help="distance metric")
    ap.add_argument("--pcoa_dims", type=int, default=2)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--make_nj", action="store_true", help="also output NJ tree (Newick)")
    args = ap.parse_args()

    if not os.path.exists(args.matrix):
        raise SystemExit(f"[ERROR] matrix not found: {args.matrix}")

    ids, X = load_matrix(args.matrix, id_col=args.id_col)
    print(f"[INFO] loaded matrix: n_samples={len(ids)} n_features={X.shape[1]}")

    P = normalize_rows_to_prob(X)

    if args.metric == "jsd":
        D = jsd_distance(P, sqrt=True)
        dist_name = "JSD_sqrt"
    else:
        D = nei_like_distance(P)
        dist_name = "Nei_like(-ln cosine)"

    # Save distance matrix
    out_dist = args.out_prefix + f".distance_{args.metric}.tsv"
    with open(out_dist, "w", encoding="utf-8") as f:
        f.write("sample_id\t" + "\t".join(ids) + "\n")
        for i, sid in enumerate(ids):
            row = "\t".join(f"{D[i,j]:.8f}" for j in range(len(ids)))
            f.write(sid + "\t" + row + "\n")
    print(f"[OK] distance matrix saved: {out_dist} ({dist_name})")

    # PCoA
    coords, evals = pcoa(D, n_components=args.pcoa_dims)
    out_pcoa = args.out_prefix + f".pcoa_{args.metric}.csv"
    
    import pandas as pd
    df_pcoa = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
    df_pcoa.insert(0, "sample_id", ids)
    df_pcoa.to_csv(out_pcoa, index=False)
    print(f"[OK] PCoA coords saved: {out_pcoa}")

    # Optional: NJ tree
    if args.make_nj:
        newick = neighbor_joining_newick(D, ids)
        out_nwk = args.out_prefix + f".nj_{args.metric}.nwk"
        with open(out_nwk, "w", encoding="utf-8") as f:
            f.write(newick + "\n")
        print(f"[OK] NJ tree saved: {out_nwk}")

    # Merge metadata for plotting
    df = df_pcoa.copy()
    df["sample_id"] = df["sample_id"].astype(str)

    if args.orc_csv and os.path.exists(args.orc_csv):
        orc = read_table_auto(args.orc_csv)
        orc["sample_id"] = orc["sample_id"].astype(str)
        if args.mix_col in orc.columns:
            df = df.merge(orc[["sample_id", args.mix_col]], on="sample_id", how="left")

    if args.meta_csv and os.path.exists(args.meta_csv) and args.group_col:
        meta = read_table_auto(args.meta_csv)
        meta["sample_id"] = meta["sample_id"].astype(str)
        if args.group_col in meta.columns:
            df = df.merge(meta[["sample_id", args.group_col]], on="sample_id", how="left")

    # Save merged table
    out_merged = args.out_prefix + f".pcoa_{args.metric}.merged.csv"
    df.to_csv(out_merged, index=False)
    print(f"[OK] merged table saved: {out_merged}")

    # Plot
    xcol = "PC1"
    ycol = "PC2" if "PC2" in df.columns else None
    if ycol is None:
        print("[WARN] Only 1D PCoA requested; skipping 2D plot.")
        return

    plt.figure(figsize=(9, 7))
    if args.orc_csv and args.mix_col in df.columns:
        sc = plt.scatter(df[xcol], df[ycol], c=df[args.mix_col], s=35, cmap="viridis")
        plt.colorbar(sc, label=args.mix_col)
    else:
        plt.scatter(df[xcol], df[ycol], s=35)

    # Label groups if available
    if args.group_col and args.group_col in df.columns:
        g = df.dropna(subset=[args.group_col]).groupby(args.group_col)
        for name, sub in g:
            cx, cy = sub[xcol].mean(), sub[ycol].mean()
            plt.text(cx, cy, str(name), fontsize=9, fontweight='bold', ha='center')

    plt.title(f"k-mer {dist_name} PCoA")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.tight_layout()

    out_png = args.out_prefix + f".pcoa_{args.metric}.png"
    plt.savefig(out_png, dpi=300)
    print(f"[OK] plot saved: {out_png}")

if __name__ == "__main__":
    main()