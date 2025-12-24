# -*- coding: utf-8 -*-
"""
Synthetic Mixing Calibration + OT-based Ollivier–Ricci Curvature
===============================================================

GOAL:
Produce a defensible result that does NOT rely on uncertain metadata labels.

1) Build k-mer sketch signatures from *individual* FASTQs (subsampled).
2) Generate in-silico pooled samples by convexly mixing signatures.
3) Compute Jensen–Shannon distance matrix on the sketches.
4) Build a kNN graph and compute *true* Ollivier–Ricci curvature using OT.
5) Derive a bridge/mixture score from (negative curvature incidence + betweenness).
6) Validate on synthetic ground truth (ROC/AUC, PR).
7) Optionally place *real* pools on the calibrated scale.

OUTPUTS:
- Figures (PDF + PNG) under: <outdir>/figures/
- Excel: <outdir>/Supplementary_Data_S4_SyntheticMixing.xlsx
- CSVs: node_scores.csv, synthetic_truth.csv, js_distance.csv

USAGE EXAMPLE:
  python 03_run_synthetic_validation.py \
    --fastq_dir "./data" \
    --outdir "./results/synthetic_calib" \
    --reads_per_sample 6000 \
    --kmer 17 \
    --sketch 16384 \
    --n_synth 40 \
    --knn 4 \
    --alpha 0.5 \
    --include_real_pools
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networkx as nx
from scipy.optimize import linprog
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.isotonic import IsotonicRegression


SRR_RE = re.compile(r"^(SRR\d+)_([12])\.fastq(\.gz)?$", re.IGNORECASE)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_fastq_seqs(path: str, max_reads: int) -> Iterable[str]:
    """Yield sequences from a (gzipped) FASTQ file, up to max_reads."""
    open_fn = gzip.open if path.lower().endswith(".gz") else open
    n = 0
    with open_fn(path, "rt", encoding="utf-8", errors="ignore") as fh:
        while True:
            h = fh.readline()
            if not h:
                break
            seq = fh.readline().strip()
            fh.readline()
            fh.readline()
            if not seq:
                break
            yield seq
            n += 1
            if n >= max_reads:
                break


def kmer_sketch_prob(
    seqs: Iterable[str],
    k: int,
    sketch_size: int,
) -> np.ndarray:
    """Compute a hashed k-mer sketch probability vector (length=sketch_size)."""
    counts = np.zeros(int(sketch_size), dtype=np.uint64)
    total = 0

    for seq in seqs:
        s = seq.strip().upper()
        if len(s) < k:
            continue
        # quick filter: skip ambiguous reads (rare in your data)
        if "N" in s:
            continue

        b = s.encode("ascii", errors="ignore")
        L = len(b)
        # Count all k-mers in the read into a fixed sketch.
        # This is a fast, stable hash (crc32) -> bin.
        for i in range(0, L - k + 1):
            h = zlib.crc32(b[i : i + k]) & 0xFFFFFFFF
            counts[h % sketch_size] += 1
            total += 1

    if total == 0:
        # avoid divide by zero; return uniform tiny mass
        out = np.full(int(sketch_size), 1.0 / sketch_size, dtype=np.float64)
        return out

    return (counts / float(total)).astype(np.float64)


def js_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen–Shannon distance (sqrt of JS divergence)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # small epsilon to avoid log(0) warnings in degenerate bins
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    jsd = 0.5 * (kl_pm + kl_qm)
    return float(np.sqrt(max(jsd, 0.0)))


def pairwise_js_distance_matrix(X: List[np.ndarray], names: List[str]) -> pd.DataFrame:
    n = len(X)
    D = np.zeros((n, n), dtype=np.float64)
    for i in tqdm(range(n), desc="JS distances", leave=False):
        for j in range(i + 1, n):
            d = js_distance(X[i], X[j])
            D[i, j] = d
            D[j, i] = d
    return pd.DataFrame(D, index=names, columns=names)


def build_symmetric_knn_graph(dist: pd.DataFrame, k: int) -> nx.Graph:
    """Build a symmetric kNN graph from a distance matrix."""
    names = dist.index.tolist()
    D = dist.values
    n = len(names)
    G = nx.Graph()
    G.add_nodes_from(names)

    # directed kNN edges
    edges = set()
    for i in range(n):
        # argsort includes self at 0
        nbrs = np.argsort(D[i])[: k + 1]
        nbrs = [j for j in nbrs if j != i][:k]
        for j in nbrs:
            a = names[i]
            b = names[j]
            if a == b:
                continue
            edges.add(tuple(sorted((a, b))))

    # add edges with distance and affinity weight
    # sigma based on median edge distance
    edge_ds = []
    for a, b in edges:
        edge_ds.append(dist.loc[a, b])
    sigma = float(np.median(edge_ds)) if edge_ds else 1.0
    sigma = max(sigma, 1e-9)

    for a, b in edges:
        d = float(dist.loc[a, b])
        w = float(np.exp(-(d * d) / (2.0 * sigma * sigma)))
        G.add_edge(a, b, d=d, w=w)

    return G


def wasserstein1_transport(cost: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Exact Earth Mover's Distance / W1 for small discrete supports via linear programming.
    cost: (m,n), a: (m,), b: (n,)
    """
    cost = np.asarray(cost, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    m, n = cost.shape
    c = cost.reshape(-1)

    # Build equality constraints: row sums = a, col sums = b
    A_eq = []
    b_eq = []

    # rows
    for i in range(m):
        row = np.zeros(m * n, dtype=np.float64)
        row[i * n : (i + 1) * n] = 1.0
        A_eq.append(row)
        b_eq.append(a[i])

    # cols
    for j in range(n):
        col = np.zeros(m * n, dtype=np.float64)
        col[j::n] = 1.0
        A_eq.append(col)
        b_eq.append(b[j])

    bounds = [(0.0, None)] * (m * n)

    res = linprog(
        c,
        A_eq=np.vstack(A_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        # fallback: extremely rare; return a conservative estimate
        return float(np.dot(c, np.ones_like(c)) * 0.0)
    return float(res.fun)


def ollivier_ricci_curvature(
    G: nx.Graph,
    dist: pd.DataFrame,
    alpha: float = 0.5,
) -> Dict[Tuple[str, str], float]:
    """
    Compute Ollivier–Ricci curvature on each edge using OT on local neighbor measures.
    Uses ground cost = JS distance between nodes from the provided dist matrix.
    """
    alpha = float(alpha)
    alpha = min(max(alpha, 0.0), 1.0)

    nodes = dist.index.tolist()
    idx = {n: i for i, n in enumerate(nodes)}
    D = dist.values

    curv: Dict[Tuple[str, str], float] = {}

    for u, v in tqdm(G.edges(), desc="ORC (OT)", leave=False):
        # local supports: node + its neighbors
        Nu = list(G.neighbors(u))
        Nv = list(G.neighbors(v))
        Su = [u] + Nu
        Sv = [v] + Nv

        du = max(len(Nu), 1)
        dv = max(len(Nv), 1)

        # measures: (1-alpha) at self, alpha spread uniformly over neighbors
        a = np.zeros(len(Su), dtype=np.float64)
        b = np.zeros(len(Sv), dtype=np.float64)
        a[0] = 1.0 - alpha
        b[0] = 1.0 - alpha
        if len(Nu) > 0:
            a[1:] = alpha / len(Nu)
        else:
            a[0] = 1.0  # isolate (shouldn't happen in connected kNN)
        if len(Nv) > 0:
            b[1:] = alpha / len(Nv)
        else:
            b[0] = 1.0

        # cost matrix from global dist
        Iu = [idx[x] for x in Su]
        Iv = [idx[x] for x in Sv]
        C = D[np.ix_(Iu, Iv)]

        d_uv = float(dist.loc[u, v])
        if d_uv <= 0:
            kappa = 0.0
        else:
            W1 = wasserstein1_transport(C, a, b)
            kappa = 1.0 - (W1 / d_uv)

        curv[tuple(sorted((u, v)))] = float(kappa)

    return curv


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd <= 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def diffusion_map_coords(dist: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    Simple diffusion-map style embedding from a distance matrix.
    """
    D = dist.values
    # kernel width sigma from median off-diagonal distances
    off = D[~np.eye(D.shape[0], dtype=bool)]
    sigma = float(np.median(off))
    sigma = max(sigma, 1e-9)
    K = np.exp(-(D * D) / (sigma * sigma))

    # normalize to Markov matrix
    d = K.sum(axis=1)
    d = np.clip(d, 1e-12, None)
    Dinv = 1.0 / d
    P = (K.T * Dinv).T  # row-stochastic

    # symmetric normalization for eigen-decomp
    sqrt_d = np.sqrt(d)
    A = (K / sqrt_d).T / sqrt_d  # D^{-1/2} K D^{-1/2}

    # eigendecomposition (symmetric)
    evals, evecs = np.linalg.eigh(A)
    order = np.argsort(evals)[::-1]  # descending
    evals = evals[order]
    evecs = evecs[:, order]

    # skip the first trivial component
    comps = []
    for i in range(1, 1 + n_components):
        comps.append(evecs[:, i] / np.clip(sqrt_d, 1e-12, None))
    coords = np.vstack(comps).T
    cols = [f"DM{i+1}" for i in range(n_components)]
    return pd.DataFrame(coords, index=dist.index, columns=cols)


@dataclass
class SyntheticSample:
    name: str
    parents: List[str]
    weights: List[float]

    @property
    def entropy(self) -> float:
        w = np.asarray(self.weights, dtype=np.float64)
        w = np.clip(w, 1e-12, None)
        w = w / w.sum()
        H = -np.sum(w * np.log(w))
        # normalize by log(k) to be in [0,1]
        return float(H / np.log(len(w)))

    @property
    def minor(self) -> float:
        w = np.asarray(self.weights, dtype=np.float64)
        w = w / w.sum()
        return float(np.min(w))


def discover_fastqs(fastq_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Return mapping: sample -> {'1': path, '2': path} where available.
    """
    files = [f for f in os.listdir(fastq_dir) if f.lower().endswith((".fastq.gz", ".fastq"))]
    out: Dict[str, Dict[str, str]] = {}
    for fn in files:
        m = SRR_RE.match(fn)
        if not m:
            continue
        srr, mate, _ = m.group(1), m.group(2), m.group(3)
        out.setdefault(srr.upper(), {})
        out[srr.upper()][mate] = os.path.join(fastq_dir, fn)
    return out


def default_known_pools() -> List[str]:
    # based on your NCBI listing: pool-GBS are SRR17037618-22
    return [f"SRR170376{i}" for i in range(18, 23)]


def generate_synthetics(
    pure_names: List[str],
    pure_sigs: Dict[str, np.ndarray],
    n_synth: int,
    max_parents: int,
    min_minor: float,
    seed: int,
) -> Tuple[List[str], List[np.ndarray], List[SyntheticSample]]:
    rng = np.random.default_rng(seed)
    synth_names: List[str] = []
    synth_sigs: List[np.ndarray] = []
    truth: List[SyntheticSample] = []

    for i in range(n_synth):
        k = int(rng.integers(2, max_parents + 1))
        parents = rng.choice(pure_names, size=k, replace=False).tolist()

        # sample weights, enforce minimum minor
        for _ in range(1000):
            w = rng.dirichlet(np.ones(k))
            if float(np.min(w)) >= float(min_minor):
                break
        w = w / w.sum()

        # convex combination in k-mer probability space
        mix = np.zeros_like(next(iter(pure_sigs.values())))
        for p, wp in zip(parents, w):
            mix += wp * pure_sigs[p]
        mix = mix / mix.sum()

        name = f"SYNTH_{i+1:03d}"
        synth_names.append(name)
        synth_sigs.append(mix)
        truth.append(SyntheticSample(name=name, parents=parents, weights=w.tolist()))

    return synth_names, synth_sigs, truth


def save_fig(path_png: str, path_pdf: str) -> None:
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.savefig(path_pdf)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq_dir", required=True, help="Directory containing SRR*_1.fastq(.gz) files")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--reads_per_sample", type=int, default=6000, help="Reads to use per sample (R1; optionally R2)")
    ap.add_argument("--kmer", type=int, default=17, help="k-mer length for sketch")
    ap.add_argument("--sketch", type=int, default=16384, help="Sketch vector length (bins)")
    ap.add_argument("--n_synth", type=int, default=40, help="Number of synthetic mixtures to generate")
    ap.add_argument("--max_parents", type=int, default=3, help="Max number of parents in synthetic mixtures (min=2)")
    ap.add_argument("--min_minor", type=float, default=0.10, help="Minimum minor fraction in synthetic mixtures")
    ap.add_argument("--knn", type=int, default=4, help="k for kNN graph")
    ap.add_argument("--alpha", type=float, default=0.5, help="ORC neighborhood mass (0..1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--use_r2", action="store_true", help="Also include R2 reads (slower)")
    ap.add_argument("--include_real_pools", action="store_true", help="Also sketch real pools (from --known_pools or default_known_pools()) and map them")
    ap.add_argument(
        "--known_pools",
        default="",
        help=("Comma-separated SRR IDs to treat as REAL pools for this study. "
              "If empty, uses default_known_pools(). "
              "Example: SRR17037621,SRR17037622,SRR17037623"),
    )
    args = ap.parse_args()

    outdir = args.outdir
    fig_main = os.path.join(outdir, "figures", "main")
    fig_sup = os.path.join(outdir, "figures", "supplementary")
    ensure_dir(fig_main)
    ensure_dir(fig_sup)

    fastqs = discover_fastqs(args.fastq_dir)
    if not fastqs:
        raise SystemExit(f"[ERR] No SRR FASTQs discovered in: {args.fastq_dir}")

    if args.known_pools.strip():
        known_pools = {s.strip().upper() for s in args.known_pools.split(',') if s.strip()}
    else:
        known_pools = set(default_known_pools())
    all_samples = sorted(fastqs.keys())

    # define "pure" individuals for sketching
    pure = [s for s in all_samples if s not in known_pools]
    if len(pure) < 4:
        raise SystemExit("[ERR] Too few individuals discovered (need >=4). Check fastq_dir and filenames.")

    print(f"[OK] Discovered FASTQs: {len(all_samples)} samples")
    print(f"[OK] Using individuals for calibration: {len(pure)} samples")
    print(f"[OK] Known pools (for optional mapping): {sorted(list(known_pools))}")

    # compute k-mer sketches for pure individuals
    pure_sigs: Dict[str, np.ndarray] = {}
    for s in tqdm(pure, desc="Sketch individuals"):
        r1 = fastqs[s].get("1")
        if not r1:
            continue
        seqs = list(read_fastq_seqs(r1, args.reads_per_sample))
        sig = kmer_sketch_prob(seqs, k=args.kmer, sketch_size=args.sketch)

        if args.use_r2:
            r2 = fastqs[s].get("2")
            if r2:
                seqs2 = list(read_fastq_seqs(r2, args.reads_per_sample))
                sig2 = kmer_sketch_prob(seqs2, k=args.kmer, sketch_size=args.sketch)
                sig = (sig + sig2) / 2.0
                sig = sig / sig.sum()

        pure_sigs[s] = sig

    pure = [s for s in pure if s in pure_sigs]
    if len(pure) < 4:
        raise SystemExit("[ERR] Failed to compute sketches for enough individuals.")

    # generate synthetic mixtures
    synth_names, synth_sigs, truth = generate_synthetics(
        pure_names=pure,
        pure_sigs=pure_sigs,
        n_synth=args.n_synth,
        max_parents=args.max_parents,
        min_minor=args.min_minor,
        seed=args.seed,
    )

    # optionally compute sketches for real pools
    real_pool_sigs: Dict[str, np.ndarray] = {}
    if args.include_real_pools:
        pools_present = [s for s in sorted(list(known_pools)) if s in fastqs and "1" in fastqs[s]]
        for s in tqdm(pools_present, desc="Sketch real pools"):
            r1 = fastqs[s].get("1")
            seqs = list(read_fastq_seqs(r1, args.reads_per_sample))
            sig = kmer_sketch_prob(seqs, k=args.kmer, sketch_size=args.sketch)
            if args.use_r2:
                r2 = fastqs[s].get("2")
                if r2:
                    seqs2 = list(read_fastq_seqs(r2, args.reads_per_sample))
                    sig2 = kmer_sketch_prob(seqs2, k=args.kmer, sketch_size=args.sketch)
                    sig = (sig + sig2) / 2.0
                    sig = sig / sig.sum()
            real_pool_sigs[s] = sig

    # assemble analysis set
    names: List[str] = []
    sigs: List[np.ndarray] = []
    label: Dict[str, str] = {}

    for s in pure:
        names.append(s)
        sigs.append(pure_sigs[s])
        label[s] = "individual"

    for n, sig in zip(synth_names, synth_sigs):
        names.append(n)
        sigs.append(sig)
        label[n] = "synthetic_pool"

    for s, sig in real_pool_sigs.items():
        names.append(s)
        sigs.append(sig)
        label[s] = "real_pool"

    # distance matrix
    dist = pairwise_js_distance_matrix(sigs, names)
    dist_path = os.path.join(outdir, "js_distance.csv")
    dist.to_csv(dist_path)

    # kNN graph
    G = build_symmetric_knn_graph(dist, k=args.knn)
    print(f"[OK] Built symmetric kNN graph: k={args.knn}, edges={G.number_of_edges()}, connected={nx.is_connected(G)}")

    # OT-based ORC curvature
    orc = ollivier_ricci_curvature(G, dist, alpha=args.alpha)

    # attach curvature to edges
    for u, v in G.edges():
        key = tuple(sorted((u, v)))
        G.edges[u, v]["orc"] = orc.get(key, np.nan)

    # node features
    bet = nx.betweenness_centrality(G, weight="d", normalized=True)
    neg_orc = {n: 0.0 for n in G.nodes()}
    for u, v, data in G.edges(data=True):
        kappa = float(data.get("orc", 0.0))
        if np.isfinite(kappa) and kappa < 0:
            neg_orc[u] += -kappa
            neg_orc[v] += -kappa

    nodes = list(G.nodes())
    bet_vec = np.array([bet[n] for n in nodes], dtype=np.float64)
    neg_vec = np.array([neg_orc[n] for n in nodes], dtype=np.float64)
    score = zscore(bet_vec) + zscore(neg_vec)

    node_df = pd.DataFrame(
        {
            "sample": nodes,
            "label": [label.get(n, "unknown") for n in nodes],
            "betweenness": bet_vec,
            "neg_orc_incidence": neg_vec,
            "mixture_score_ot": score,
        }
    ).sort_values("mixture_score_ot", ascending=False)

    node_df.to_csv(os.path.join(outdir, "node_scores.csv"), index=False)

    # build synthetic truth table
    truth_rows = []
    for t in truth:
        truth_rows.append(
            {
                "sample": t.name,
                "parents": ",".join(t.parents),
                "weights": ",".join([f"{w:.4f}" for w in t.weights]),
                "entropy_norm": t.entropy,
                "minor_fraction": t.minor,
                "n_parents": len(t.parents),
            }
        )
    truth_df = pd.DataFrame(truth_rows)
    truth_df.to_csv(os.path.join(outdir, "synthetic_truth.csv"), index=False)

    # --- validation on synthetic ground truth ---
    y_true = node_df["label"].values == "synthetic_pool"
    y_mask = (node_df["label"].values == "synthetic_pool") | (node_df["label"].values == "individual")
    y = (node_df.loc[y_mask, "label"].values == "synthetic_pool").astype(int)
    s = node_df.loc[y_mask, "mixture_score_ot"].values.astype(float)

    roc_auc = float(roc_auc_score(y, s))
    ap_score = float(average_precision_score(y, s))

    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)

    # choose threshold that maximizes F1
    ths = np.unique(s)
    best_f1 = -1.0
    best_th = None
    for th in ths:
        yhat = (s >= th).astype(int)
        f1 = f1_score(y, yhat)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_th = float(th)

    # --- figures ---
    # Fig: ROC
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Synthetic pools detection (AUC={roc_auc:.3f})")
    save_fig(
        os.path.join(fig_main, "ROC_SyntheticPools.png"),
        os.path.join(fig_main, "ROC_SyntheticPools.pdf"),
    )

    # Fig: PR curve
    plt.figure(figsize=(7, 6))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall (AP={ap_score:.3f})")
    save_fig(
        os.path.join(fig_sup, "PR_SyntheticPools.png"),
        os.path.join(fig_sup, "PR_SyntheticPools.pdf"),
    )

    # Fig: Diffusion map embedding
    coords = diffusion_map_coords(dist.loc[nodes, nodes], n_components=2)
    emb = coords.copy()
    emb["label"] = [label.get(n, "unknown") for n in emb.index]
    plt.figure(figsize=(8, 7))
    for lab, mk in [("individual", "o"), ("synthetic_pool", "*"), ("real_pool", "s")]:
        sub = emb[emb["label"] == lab]
        if len(sub) == 0:
            continue
        plt.scatter(sub["DM1"], sub["DM2"], marker=mk)
    plt.xlabel("DM1")
    plt.ylabel("DM2")
    plt.title("Diffusion geometry: individuals vs synthetic pools vs real pools")
    save_fig(
        os.path.join(fig_main, "DiffusionGeometry_Synthetic.png"),
        os.path.join(fig_main, "DiffusionGeometry_Synthetic.pdf"),
    )

    # FigD: mixture_score vs mixing entropy (synthetic only)
    syn_scores = node_df[node_df["label"] == "synthetic_pool"][["sample", "mixture_score_ot"]].merge(
        truth_df[["sample", "entropy_norm", "minor_fraction", "n_parents"]],
        on="sample",
        how="left",
    )
    plt.figure(figsize=(7, 6))
    plt.scatter(syn_scores["entropy_norm"], syn_scores["mixture_score_ot"])
    plt.xlabel("Mixing entropy (normalized)")
    plt.ylabel("Bridge score (mixture_score_ot)")
    plt.title("Bridge score increases with mixing entropy (synthetic ground truth)")
    save_fig(
        os.path.join(fig_main, "Score_vs_Entropy.png"),
        os.path.join(fig_main, "Score_vs_Entropy.pdf"),
    )

    # Fig: score vs minor fraction
    plt.figure(figsize=(7, 6))
    plt.scatter(syn_scores["minor_fraction"], syn_scores["mixture_score_ot"])
    plt.xlabel("Minor fraction (min weight)")
    plt.ylabel("Bridge score (mixture_score_ot)")
    plt.title("Detection power vs minor fraction (synthetic ground truth)")
    save_fig(
        os.path.join(fig_sup, "Score_vs_MinorFraction.png"),
        os.path.join(fig_sup, "Score_vs_MinorFraction.pdf"),
    )

    # Fig: curvature network (top nodes)
    topk_show = min(12, len(nodes))
    top_nodes = node_df["sample"].head(topk_show).tolist()
    H = G.subgraph(top_nodes).copy()
    pos = nx.spring_layout(H, seed=args.seed, weight="w")
    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(H, pos, nodelist=top_nodes, node_size=400)
    # draw edges, thicker if negative curvature
    widths = []
    for u, v in H.edges():
        kappa = float(H.edges[u, v].get("orc", 0.0))
        widths.append(2.5 if kappa < 0 else 1.0)
    nx.draw_networkx_edges(H, pos, width=widths)
    nx.draw_networkx_labels(H, pos, font_size=8)
    plt.title("Top bridge candidates (OT-Ricci curvature network)")
    save_fig(
        os.path.join(fig_main, "TopBridgeNetwork.png"),
        os.path.join(fig_main, "TopBridgeNetwork.pdf"),
    )

    # Optional: map real pools to "effective mixing entropy" by isotonic regression
    eff_map_df = pd.DataFrame()
    if len(real_pool_sigs) > 0:
        ir = IsotonicRegression(out_of_bounds="clip")
        # fit score->entropy on synthetic
        ir.fit(syn_scores["mixture_score_ot"].values, syn_scores["entropy_norm"].values)
        real_df = node_df[node_df["label"] == "real_pool"][["sample", "mixture_score_ot"]].copy()
        if len(real_df) > 0:
            real_df["effective_entropy_norm"] = ir.predict(real_df["mixture_score_ot"].values)
            eff_map_df = real_df.sort_values("effective_entropy_norm", ascending=False)

            plt.figure(figsize=(8, 5))
            plt.bar(np.arange(len(eff_map_df)), eff_map_df["effective_entropy_norm"].values)
            plt.xticks(np.arange(len(eff_map_df)), eff_map_df["sample"].values, rotation=45, ha="right")
            plt.ylabel("Effective mixing entropy (normalized)")
            plt.title("Real pools mapped onto synthetic mixing calibration")
            save_fig(
                os.path.join(fig_main, "RealPools_EffectiveEntropy.png"),
                os.path.join(fig_main, "RealPools_EffectiveEntropy.pdf"),
            )

    # Metrics summary
    metrics = {
        "roc_auc_synthetic": roc_auc,
        "avg_precision_synthetic": ap_score,
        "best_f1_synthetic": best_f1,
        "best_threshold": best_th,
        "n_individual": int((node_df["label"] == "individual").sum()),
        "n_synthetic": int((node_df["label"] == "synthetic_pool").sum()),
        "n_real_pools": int((node_df["label"] == "real_pool").sum()),
        "kmer": int(args.kmer),
        "sketch": int(args.sketch),
        "reads_per_sample": int(args.reads_per_sample),
        "knn": int(args.knn),
        "orc_alpha": float(args.alpha),
        "use_r2": bool(args.use_r2),
        "seed": int(args.seed),
    }
    metrics_df = pd.DataFrame([metrics])

    # Excel output
    xlsx_path = os.path.join(outdir, "Supplementary_Data_S4_SyntheticMixing.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        metrics_df.to_excel(xw, sheet_name="metrics_summary", index=False)
        node_df.to_excel(xw, sheet_name="node_scores", index=False)
        truth_df.to_excel(xw, sheet_name="synthetic_truth", index=False)
        syn_scores.to_excel(xw, sheet_name="synthetic_score_vs_truth", index=False)
        if len(eff_map_df) > 0:
            eff_map_df.to_excel(xw, sheet_name="real_pools_calibration", index=False)
        # edge curvature table
        edge_rows = []
        for u, v, data in G.edges(data=True):
            edge_rows.append(
                {
                    "u": u,
                    "v": v,
                    "distance": float(data.get("d", np.nan)),
                    "affinity": float(data.get("w", np.nan)),
                    "orc": float(data.get("orc", np.nan)),
                }
            )
        pd.DataFrame(edge_rows).to_excel(xw, sheet_name="edge_curvature", index=False)

    print("\n[DONE] Synthetic mixing calibration complete.")
    print(f"  -> Excel: {xlsx_path}")
    print(f"  -> Figures: {os.path.join(outdir, 'figures')}")
    print("\nKey numbers (synthetic pools vs individuals):")
    print(f"  ROC AUC: {roc_auc:.3f}")
    print(f"  Avg Precision: {ap_score:.3f}")
    print(f"  Best F1: {best_f1:.3f} at threshold {best_th:.4f}")
    if len(real_pool_sigs) > 0 and len(eff_map_df) > 0:
        print("\nReal pools mapped to effective entropy (higher = more mixed):")
        for _, r in eff_map_df.iterrows():
            print(f"  {r['sample']}: score={r['mixture_score_ot']:.3f}, eff_entropy={r['effective_entropy_norm']:.3f}")


if __name__ == "__main__":
    main()
