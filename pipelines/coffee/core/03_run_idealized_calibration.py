# -*- coding: utf-8 -*-
"""
Synthetic validation for curvature-based mixed-signal screening.

What this script does
---------------------
1. Builds hashed k-mer sketches for reference libraries.
2. Generates synthetic mixtures as convex combinations of reference sketches.
3. Computes a kNN graph and Ollivier–Ricci curvature on the combined cohort.
4. Derives a curvature-based mixture score.
5. Evaluates score performance against synthetic ground truth.
6. Optionally maps user-specified real candidate IDs onto the synthetic scale.

Primary workbook
----------------
<outdir>/Supplementary_Data_S3_SyntheticValidation.xlsx
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm


SRR_RE = re.compile(r"^(.+?)_([12])\.f(ast)?q(\.gz)?$", re.IGNORECASE)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_fastq_seqs(path: str, max_reads: int) -> Iterable[str]:
    open_fn = gzip.open if path.lower().endswith(".gz") else open
    n = 0
    with open_fn(path, "rt", encoding="utf-8", errors="ignore") as fh:
        while True:
            header = fh.readline()
            if not header:
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


def kmer_sketch_prob(seqs: Iterable[str], k: int, sketch_size: int) -> np.ndarray:
    counts = np.zeros(int(sketch_size), dtype=np.uint64)
    total = 0
    for seq in seqs:
        s = seq.strip().upper()
        if len(s) < k or "N" in s:
            continue
        b = s.encode("ascii", errors="ignore")
        L = len(b)
        for i in range(0, L - k + 1):
            h = zlib.crc32(b[i : i + k]) & 0xFFFFFFFF
            counts[h % sketch_size] += 1
            total += 1
    if total == 0:
        return np.full(int(sketch_size), 1.0 / sketch_size, dtype=np.float64)
    return (counts / float(total)).astype(np.float64)


def js_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
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
    names = dist.index.tolist()
    D = dist.values
    n = len(names)
    G = nx.Graph()
    G.add_nodes_from(names)

    edges = set()
    for i in range(n):
        nbrs = np.argsort(D[i])[: k + 1]
        nbrs = [j for j in nbrs if j != i][:k]
        for j in nbrs:
            a = names[i]
            b = names[j]
            if a != b:
                edges.add(tuple(sorted((a, b))))

    edge_ds = [dist.loc[a, b] for a, b in edges]
    sigma = float(np.median(edge_ds)) if edge_ds else 1.0
    sigma = max(sigma, 1e-9)

    for a, b in edges:
        d = float(dist.loc[a, b])
        w = float(np.exp(-(d * d) / (2.0 * sigma * sigma)))
        G.add_edge(a, b, d=d, w=w)

    return G


def wasserstein1_transport(cost: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    cost = np.asarray(cost, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    m, n = cost.shape
    c = cost.reshape(-1)

    A_eq = []
    b_eq = []

    for i in range(m):
        row = np.zeros(m * n, dtype=np.float64)
        row[i * n : (i + 1) * n] = 1.0
        A_eq.append(row)
        b_eq.append(a[i])

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
        return 0.0
    return float(res.fun)


def ollivier_ricci_curvature(G: nx.Graph, dist: pd.DataFrame, alpha: float = 0.5) -> Dict[Tuple[str, str], float]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    nodes = dist.index.tolist()
    idx = {n: i for i, n in enumerate(nodes)}
    D = dist.values
    curv: Dict[Tuple[str, str], float] = {}

    for u, v in tqdm(G.edges(), desc="ORC (OT)", leave=False):
        Nu = list(G.neighbors(u))
        Nv = list(G.neighbors(v))
        Su = [u] + Nu
        Sv = [v] + Nv

        a = np.zeros(len(Su), dtype=np.float64)
        b = np.zeros(len(Sv), dtype=np.float64)
        a[0] = 1.0 - alpha
        b[0] = 1.0 - alpha
        if len(Nu) > 0:
            a[1:] = alpha / len(Nu)
        else:
            a[0] = 1.0
        if len(Nv) > 0:
            b[1:] = alpha / len(Nv)
        else:
            b[0] = 1.0

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
    D = dist.values
    off = D[~np.eye(D.shape[0], dtype=bool)]
    sigma = float(np.median(off))
    sigma = max(sigma, 1e-9)
    K = np.exp(-(D * D) / (sigma * sigma))
    d = K.sum(axis=1)
    d = np.clip(d, 1e-12, None)
    sqrt_d = np.sqrt(d)
    A = (K / sqrt_d).T / sqrt_d
    evals, evecs = np.linalg.eigh(A)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

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
        return float(H / np.log(len(w)))

    @property
    def minor(self) -> float:
        w = np.asarray(self.weights, dtype=np.float64)
        w = w / w.sum()
        return float(np.min(w))


def discover_fastqs(fastq_dir: str) -> Dict[str, Dict[str, str]]:
    files = [f for f in os.listdir(fastq_dir) if f.lower().endswith((".fastq.gz", ".fastq", ".fq.gz", ".fq"))]
    out: Dict[str, Dict[str, str]] = {}
    for fn in files:
        m = SRR_RE.match(fn)
        if not m:
            continue
        sample_id, mate = m.group(1), m.group(2)
        out.setdefault(sample_id, {})
        out[sample_id][mate] = os.path.join(fastq_dir, fn)
    return out


def generate_synthetics(
    reference_names: List[str],
    reference_sigs: Dict[str, np.ndarray],
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
        parents = rng.choice(reference_names, size=k, replace=False).tolist()

        for _ in range(1000):
            w = rng.dirichlet(np.ones(k))
            if float(np.min(w)) >= float(min_minor):
                break
        w = w / w.sum()

        mix = np.zeros_like(next(iter(reference_sigs.values())))
        for p, wp in zip(parents, w):
            mix += wp * reference_sigs[p]
        mix = mix / mix.sum()

        name = f"SYNTH_{i+1:03d}"
        synth_names.append(name)
        synth_sigs.append(mix)
        truth.append(SyntheticSample(name=name, parents=parents, weights=w.tolist()))
    return synth_names, synth_sigs, truth


def parse_real_pool_ids(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq_dir", required=True, help="Directory containing paired FASTQ files")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--reads_per_sample", type=int, default=6000, help="Reads to use per sample")
    ap.add_argument("--kmer", type=int, default=17, help="k-mer length for sketch")
    ap.add_argument("--sketch", type=int, default=16384, help="Sketch vector length")
    ap.add_argument("--n_synth", type=int, default=40, help="Number of synthetic mixtures")
    ap.add_argument("--max_parents", type=int, default=3, help="Maximum number of parents in synthetic mixtures")
    ap.add_argument("--min_minor", type=float, default=0.10, help="Minimum minor fraction in synthetic mixtures")
    ap.add_argument("--knn", type=int, default=4, help="k for kNN graph")
    ap.add_argument("--alpha", type=float, default=0.5, help="Neighborhood mass for ORC")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--use_r2", action="store_true", help="Also include R2 reads")
    ap.add_argument("--include_real_pools", action="store_true", help="Sketch and calibrate user-specified real candidate IDs")
    ap.add_argument("--real_pool_ids", default="", help="Comma-separated IDs for optional real candidate calibration")
    args = ap.parse_args()

    outdir = args.outdir
    ensure_dir(outdir)

    fastqs = discover_fastqs(args.fastq_dir)
    if not fastqs:
        raise SystemExit(f"[ERR] No paired FASTQ files discovered in: {args.fastq_dir}")

    real_pool_ids = parse_real_pool_ids(args.real_pool_ids)
    real_pool_set = set(real_pool_ids)

    if args.include_real_pools and not real_pool_ids:
        raise SystemExit("[ERR] --include_real_pools was set but --real_pool_ids is empty.")

    all_samples = sorted(fastqs.keys())
    reference_samples = [s for s in all_samples if s not in real_pool_set]
    if len(reference_samples) < 4:
        raise SystemExit("[ERR] Too few reference libraries discovered (need at least 4).")

    print(f"[OK] Discovered FASTQs: {len(all_samples)} samples")
    print(f"[OK] Using reference libraries for synthetic calibration: {len(reference_samples)} samples")

    reference_sigs: Dict[str, np.ndarray] = {}
    for sample in tqdm(reference_samples, desc="Sketch reference libraries"):
        r1 = fastqs[sample].get("1")
        if not r1:
            continue
        seqs = list(read_fastq_seqs(r1, args.reads_per_sample))
        sig = kmer_sketch_prob(seqs, k=args.kmer, sketch_size=args.sketch)

        if args.use_r2:
            r2 = fastqs[sample].get("2")
            if r2:
                seqs2 = list(read_fastq_seqs(r2, args.reads_per_sample))
                sig2 = kmer_sketch_prob(seqs2, k=args.kmer, sketch_size=args.sketch)
                sig = (sig + sig2) / 2.0
                sig = sig / sig.sum()

        reference_sigs[sample] = sig

    reference_samples = [s for s in reference_samples if s in reference_sigs]
    if len(reference_samples) < 4:
        raise SystemExit("[ERR] Failed to compute sketches for enough reference libraries.")

    synth_names, synth_sigs, truth = generate_synthetics(
        reference_names=reference_samples,
        reference_sigs=reference_sigs,
        n_synth=args.n_synth,
        max_parents=args.max_parents,
        min_minor=args.min_minor,
        seed=args.seed,
    )

    real_pool_sigs: Dict[str, np.ndarray] = {}
    if args.include_real_pools:
        present_real = [s for s in real_pool_ids if s in fastqs and "1" in fastqs[s]]
        for sample in tqdm(present_real, desc="Sketch real candidate libraries"):
            r1 = fastqs[sample]["1"]
            seqs = list(read_fastq_seqs(r1, args.reads_per_sample))
            sig = kmer_sketch_prob(seqs, k=args.kmer, sketch_size=args.sketch)
            if args.use_r2 and "2" in fastqs[sample]:
                seqs2 = list(read_fastq_seqs(fastqs[sample]["2"], args.reads_per_sample))
                sig2 = kmer_sketch_prob(seqs2, k=args.kmer, sketch_size=args.sketch)
                sig = (sig + sig2) / 2.0
                sig = sig / sig.sum()
            real_pool_sigs[sample] = sig

    names: List[str] = []
    sigs: List[np.ndarray] = []
    label: Dict[str, str] = {}

    for sample in reference_samples:
        names.append(sample)
        sigs.append(reference_sigs[sample])
        label[sample] = "reference"

    for name, sig in zip(synth_names, synth_sigs):
        names.append(name)
        sigs.append(sig)
        label[name] = "synthetic_mixture"

    for sample, sig in real_pool_sigs.items():
        names.append(sample)
        sigs.append(sig)
        label[sample] = "real_candidate"

    dist = pairwise_js_distance_matrix(sigs, names)
    dist_path = os.path.join(outdir, "js_distance.csv")
    dist.to_csv(dist_path)

    G = build_symmetric_knn_graph(dist, k=args.knn)
    orc = ollivier_ricci_curvature(G, dist, alpha=args.alpha)

    for u, v in G.edges():
        key = tuple(sorted((u, v)))
        G.edges[u, v]["orc"] = orc.get(key, np.nan)

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
            "negative_orc_incidence": neg_vec,
            "mixture_score": score,
        }
    ).sort_values("mixture_score", ascending=False)
    node_df.to_csv(os.path.join(outdir, "node_scores.csv"), index=False)

    truth_rows = []
    for item in truth:
        truth_rows.append(
            {
                "sample": item.name,
                "parents": ",".join(item.parents),
                "weights": ",".join([f"{w:.4f}" for w in item.weights]),
                "entropy_norm": item.entropy,
                "minor_fraction": item.minor,
                "n_parents": len(item.parents),
            }
        )
    truth_df = pd.DataFrame(truth_rows)
    truth_df.to_csv(os.path.join(outdir, "synthetic_truth.csv"), index=False)

    y_mask = (node_df["label"].values == "synthetic_mixture") | (node_df["label"].values == "reference")
    y = (node_df.loc[y_mask, "label"].values == "synthetic_mixture").astype(int)
    s = node_df.loc[y_mask, "mixture_score"].values.astype(float)

    roc_auc = float(roc_auc_score(y, s))
    ap_score = float(average_precision_score(y, s))
    fpr, tpr, roc_th = roc_curve(y, s)
    prec, rec, pr_th = precision_recall_curve(y, s)

    ths = np.unique(s)
    best_f1 = -1.0
    best_th: Optional[float] = None
    for th in ths:
        yhat = (s >= th).astype(int)
        f1 = f1_score(y, yhat)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_th = float(th)

    coords = diffusion_map_coords(dist.loc[nodes, nodes], n_components=2)
    emb = coords.copy()
    emb["label"] = [label.get(n, "unknown") for n in emb.index]

    syn_scores = (
        node_df[node_df["label"] == "synthetic_mixture"][["sample", "mixture_score"]]
        .merge(truth_df[["sample", "entropy_norm", "minor_fraction", "n_parents"]], on="sample", how="left")
        .sort_values("mixture_score", ascending=False)
        .reset_index(drop=True)
    )

    eff_map_df = pd.DataFrame()
    if len(real_pool_sigs) > 0:
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(syn_scores["mixture_score"].values, syn_scores["entropy_norm"].values)
        real_df = node_df[node_df["label"] == "real_candidate"][["sample", "mixture_score"]].copy()
        if len(real_df) > 0:
            real_df["effective_entropy_norm"] = ir.predict(real_df["mixture_score"].values)
            eff_map_df = real_df.sort_values("effective_entropy_norm", ascending=False).reset_index(drop=True)

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
    edge_df = pd.DataFrame(edge_rows).sort_values("orc").reset_index(drop=True)

    metrics_df = pd.DataFrame(
        [
            {
                "roc_auc_synthetic": roc_auc,
                "avg_precision_synthetic": ap_score,
                "best_f1_synthetic": best_f1,
                "best_threshold": best_th,
                "n_reference": int((node_df["label"] == "reference").sum()),
                "n_synthetic": int((node_df["label"] == "synthetic_mixture").sum()),
                "n_real_candidates": int((node_df["label"] == "real_candidate").sum()),
                "kmer": int(args.kmer),
                "sketch": int(args.sketch),
                "reads_per_sample": int(args.reads_per_sample),
                "knn": int(args.knn),
                "orc_alpha": float(args.alpha),
                "use_r2": bool(args.use_r2),
                "seed": int(args.seed),
            }
        ]
    )

    curve_roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    curve_pr_df = pd.DataFrame({"recall": rec, "precision": prec})

    xlsx_path = os.path.join(outdir, "Supplementary_Data_S3_SyntheticValidation.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        metrics_df.to_excel(xw, sheet_name="metrics_summary", index=False)
        node_df.to_excel(xw, sheet_name="node_scores", index=False)
        truth_df.to_excel(xw, sheet_name="synthetic_truth", index=False)
        syn_scores.to_excel(xw, sheet_name="synthetic_score_vs_truth", index=False)
        edge_df.to_excel(xw, sheet_name="edge_curvature", index=False)
        dist.to_excel(xw, sheet_name="js_distance")
        emb.reset_index().rename(columns={"index": "sample"}).to_excel(xw, sheet_name="diffusion_coords", index=False)
        curve_roc_df.to_excel(xw, sheet_name="roc_curve", index=False)
        curve_pr_df.to_excel(xw, sheet_name="pr_curve", index=False)
        if len(eff_map_df) > 0:
            eff_map_df.to_excel(xw, sheet_name="real_candidate_calibration", index=False)

    print("\n[DONE] Synthetic validation complete.")
    print(f"  -> Workbook: {xlsx_path}")
    print(f"  -> ROC AUC: {roc_auc:.3f}")
    print(f"  -> Average precision: {ap_score:.3f}")
    print(f"  -> Best F1: {best_f1:.3f} at threshold {best_th:.4f}")


if __name__ == "__main__":
    main()
