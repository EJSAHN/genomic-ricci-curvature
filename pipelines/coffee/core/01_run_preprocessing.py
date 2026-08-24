# -*- coding: utf-8 -*-
"""
Preprocessing workflow for paired-end FASTQ libraries.

What this script does
---------------------
1. Discovers paired FASTQ files in a folder.
2. Computes sampled QC summaries.
3. Builds hashed k-mer composition vectors.
4. Computes a Jensen–Shannon distance matrix.
5. Generates Excel-friendly summary tables for downstream analysis.

Primary workbook
----------------
<outdir>/Supplementary_Data_S1_Preprocessing.xlsx
"""
from __future__ import annotations

import argparse
import gzip
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline=None)
    return open(path, "rt", encoding="utf-8", errors="replace", newline=None)


def iter_fastq(path: Path, max_reads: Optional[int] = None):
    """Yield (sequence, quality) tuples from FASTQ or FASTQ.GZ."""
    n = 0
    with open_maybe_gzip(path) as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq = fh.readline().strip()
            fh.readline()
            qual = fh.readline().strip()
            if not qual:
                break
            yield seq, qual
            n += 1
            if max_reads is not None and n >= max_reads:
                break


def phred_scores(qual: str) -> np.ndarray:
    return (np.frombuffer(qual.encode("ascii", "ignore"), dtype=np.uint8) - 33).astype(np.int16)


@dataclass
class QcResult:
    reads_used: int
    bases_used: int
    mean_read_len: float
    sd_read_len: float
    min_read_len: int
    max_read_len: int
    gc_frac: float
    n_frac: float
    mean_phred: float
    q30_frac: float
    per_pos_mean_q: np.ndarray


def qc_fastq(path: Path, max_reads: Optional[int] = 200000) -> QcResult:
    lens: List[int] = []
    gc = 0
    ncount = 0
    qsum_total = 0
    qcount_total = 0
    q30 = 0
    per_pos_sum = np.zeros(0, dtype=np.float64)
    per_pos_cnt = np.zeros(0, dtype=np.int64)

    reads = 0
    for seq, qual in iter_fastq(path, max_reads=max_reads):
        reads += 1
        s = seq.upper()
        L = len(s)
        lens.append(L)
        gc += s.count("G") + s.count("C")
        ncount += s.count("N")

        q = phred_scores(qual)
        if q.size != L:
            m = min(q.size, L)
            q = q[:m]
            L = m

        qsum_total += int(q.sum())
        qcount_total += int(q.size)
        q30 += int((q >= 30).sum())

        if per_pos_sum.size < L:
            per_pos_sum = np.pad(per_pos_sum, (0, L - per_pos_sum.size))
            per_pos_cnt = np.pad(per_pos_cnt, (0, L - per_pos_cnt.size))
        per_pos_sum[:L] += q
        per_pos_cnt[:L] += 1

    if reads == 0:
        raise RuntimeError(f"No reads parsed from {path}")

    lens_arr = np.array(lens, dtype=np.int32)
    bases_used = int(lens_arr.sum())
    mean_len = float(lens_arr.mean())
    sd_len = float(lens_arr.std(ddof=1)) if lens_arr.size > 1 else 0.0
    min_len = int(lens_arr.min())
    max_len = int(lens_arr.max())

    gc_frac = gc / bases_used if bases_used else float("nan")
    n_frac = ncount / bases_used if bases_used else float("nan")
    mean_phred = qsum_total / qcount_total if qcount_total else float("nan")
    q30_frac = q30 / qcount_total if qcount_total else float("nan")

    per_pos_mean = np.divide(
        per_pos_sum,
        per_pos_cnt,
        out=np.zeros_like(per_pos_sum),
        where=per_pos_cnt > 0,
    )

    return QcResult(
        reads_used=reads,
        bases_used=bases_used,
        mean_read_len=mean_len,
        sd_read_len=sd_len,
        min_read_len=min_len,
        max_read_len=max_len,
        gc_frac=gc_frac,
        n_frac=n_frac,
        mean_phred=mean_phred,
        q30_frac=q30_frac,
        per_pos_mean_q=per_pos_mean,
    )


def find_pairs(input_dir: Path) -> pd.DataFrame:
    r1_files = sorted(list(input_dir.glob("*_1.fastq.gz")) + list(input_dir.glob("*_1.fastq")))
    if not r1_files:
        raise FileNotFoundError(f"No *_1.fastq(.gz) files found in {input_dir}")
    rows = []
    for r1 in r1_files:
        sample = r1.name.replace("_1.fastq.gz", "").replace("_1.fastq", "")
        r2 = input_dir / f"{sample}_2.fastq.gz"
        if not r2.exists():
            r2 = input_dir / f"{sample}_2.fastq"
        if not r2.exists():
            raise FileNotFoundError(f"Missing mate for {r1.name}: expected {r2.name}")
        rows.append(
            {
                "sample": sample,
                "r1": str(r1),
                "r2": str(r2),
                "r1_bytes": int(r1.stat().st_size),
                "r2_bytes": int(r2.stat().st_size),
                "total_bytes": int(r1.stat().st_size + r2.stat().st_size),
            }
        )
    return pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)


def hash_kmer_to_bin(kmer: str, dim: int) -> int:
    return (zlib.crc32(kmer.encode("ascii")) & 0xFFFFFFFF) % dim


def kmer_feature_from_fastq(path: Path, k: int, dim: int, step: int, max_reads: Optional[int]) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float64)
    total = 0
    for seq, _ in iter_fastq(path, max_reads=max_reads):
        s = seq.upper()
        if len(s) < k:
            continue
        for i in range(0, len(s) - k + 1, step):
            kmer = s[i : i + k]
            if "N" in kmer:
                continue
            vec[hash_kmer_to_bin(kmer, dim)] += 1.0
            total += 1
    if total == 0:
        return vec
    vec /= vec.sum()
    return vec


def js_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(np.float64, copy=False)
    q = q.astype(np.float64, copy=False)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + eps) / (b[mask] + eps))))

    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(np.sqrt(max(js, 0.0)))


def pairwise_js_distance_matrix(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = js_distance(X[i], X[j])
            D[i, j] = D[j, i] = d
    return D


def shannon_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(np.float64, copy=False)
    p = p / max(p.sum(), eps)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask] + eps)))


def assign_depth_groups(manifest: pd.DataFrame, random_state: int = 0) -> Dict[str, str]:
    x = manifest[["total_bytes"]].values.astype(np.float64)
    xlog = np.log1p(x)
    km = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    lab = km.fit_predict(xlog)
    means = {c: float(x[lab == c].mean()) for c in [0, 1]}
    high_cluster = max(means, key=means.get)
    return {sample: ("HiDepth" if c == high_cluster else "LoDepth") for sample, c in zip(manifest["sample"], lab)}


def build_similarity_network_summary(D: np.ndarray, labels: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dmax = float(D.max()) if float(D.max()) > 0 else 1.0
    S = 1.0 - (D / dmax)

    vals = S[~np.eye(len(labels), dtype=bool)]
    thr = float(np.quantile(vals, 0.90))

    G = nx.Graph()
    G.add_nodes_from(labels)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if S[i, j] >= thr:
                G.add_edge(labels[i], labels[j], weight=float(S[i, j]))

    comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    node2c: Dict[str, int] = {}
    for idx, cset in enumerate(comms):
        for node in cset:
            node2c[node] = idx + 1

    net_metrics = pd.DataFrame(
        [
            {
                "threshold_similarity": thr,
                "n_nodes": G.number_of_nodes(),
                "n_edges": G.number_of_edges(),
                "n_communities": len(comms),
                "is_connected": nx.is_connected(G) if G.number_of_nodes() > 0 else False,
            }
        ]
    )
    node_table = pd.DataFrame(
        [{"sample": sample, "community_id": node2c.get(sample, np.nan)} for sample in labels]
    )
    return net_metrics, node_table


def cluster_reference_libraries(D: np.ndarray, labels: List[str], depth_groups: Dict[str, str], n_groups: int = 5):
    ref_idx = [i for i, s in enumerate(labels) if depth_groups[s] == "HiDepth"]
    if len(ref_idx) < 3:
        return {}, ref_idx

    n_groups = max(2, min(n_groups, len(ref_idx)))
    D_ref = D[np.ix_(ref_idx, ref_idx)]
    model = AgglomerativeClustering(n_clusters=n_groups, metric="precomputed", linkage="average")
    ref_labels = model.fit_predict(D_ref)
    return {labels[i]: f"G{ref_labels[k] + 1}" for k, i in enumerate(ref_idx)}, ref_idx


def deconvolve_low_depth_libraries(
    X: np.ndarray,
    labels: List[str],
    depth_groups: Dict[str, str],
    ref_groups: Dict[str, str],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    refs = [s for s in labels if depth_groups[s] == "HiDepth"]
    lows = [s for s in labels if depth_groups[s] == "LoDepth"]
    if not refs or not lows:
        return None, None

    idx_ref = [labels.index(s) for s in refs]
    idx_low = [labels.index(s) for s in lows]

    A = X[idx_ref].T
    weight_rows = []
    for sample, idx in zip(lows, idx_low):
        b = X[idx]
        w, rnorm = nnls(A, b)
        if w.sum() > 0:
            w = w / w.sum()
        row = {"sample": sample, "residual_norm": float(rnorm)}
        for ref_s, weight in zip(refs, w):
            row[ref_s] = float(weight)
        weight_rows.append(row)

    W = pd.DataFrame(weight_rows).set_index("sample")
    group_names = sorted(set(ref_groups.values())) if ref_groups else []
    agg = pd.DataFrame(index=W.index, columns=group_names, dtype=float)
    for group_name in group_names:
        members = [s for s in refs if ref_groups.get(s) == group_name]
        agg[group_name] = W[members].sum(axis=1) if members else 0.0

    return W, agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq_dir", dest="input", required=False, help="Folder with *_1.fastq(.gz) and *_2.fastq(.gz) files")
    ap.add_argument("--outdir", dest="out", required=False, help="Output folder")
    ap.add_argument("--input", required=False, help="Alias of --fastq_dir")
    ap.add_argument("--out", required=False, help="Alias of --outdir")
    ap.add_argument("--qc_reads", type=int, default=200000, help="Reads sampled per FASTQ for QC (0 = all reads)")
    ap.add_argument("--kmer_reads", type=int, default=50000, help="Reads sampled per FASTQ for k-mer composition (0 = all reads)")
    ap.add_argument("--k", type=int, default=21, help="k-mer length")
    ap.add_argument("--dim", type=int, default=16384, help="Hashed feature dimension")
    ap.add_argument("--step", type=int, default=4, help="Stride for k-mer sampling within reads")
    ap.add_argument("--n_groups", type=int, default=5, help="Maximum number of auto-inferred reference groups")
    args = ap.parse_args()

    if args.input is None or args.out is None:
        ap.error("Provide input and output folders using --fastq_dir/--outdir or --input/--out.")

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    qc_reads = None if args.qc_reads == 0 else args.qc_reads
    kmer_reads = None if args.kmer_reads == 0 else args.kmer_reads

    manifest = find_pairs(in_dir)
    manifest.to_csv(out_dir / "sample_manifest.csv", index=False)

    depth_groups = assign_depth_groups(manifest, random_state=0)

    qc_rows = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="QC"):
        sample = row["sample"]
        r1 = Path(row["r1"])
        r2 = Path(row["r2"])
        qc1 = qc_fastq(r1, max_reads=qc_reads)
        qc2 = qc_fastq(r2, max_reads=qc_reads)
        qc_rows.append(
            {
                "sample": sample,
                "depth_group": depth_groups[sample],
                "R1_reads_used": qc1.reads_used,
                "R2_reads_used": qc2.reads_used,
                "total_reads_used": qc1.reads_used + qc2.reads_used,
                "R1_bases_used": qc1.bases_used,
                "R2_bases_used": qc2.bases_used,
                "total_bases_used": qc1.bases_used + qc2.bases_used,
                "R1_mean_len": qc1.mean_read_len,
                "R2_mean_len": qc2.mean_read_len,
                "mean_read_len": 0.5 * (qc1.mean_read_len + qc2.mean_read_len),
                "R1_min_len": qc1.min_read_len,
                "R2_min_len": qc2.min_read_len,
                "min_read_len": min(qc1.min_read_len, qc2.min_read_len),
                "R1_max_len": qc1.max_read_len,
                "R2_max_len": qc2.max_read_len,
                "max_read_len": max(qc1.max_read_len, qc2.max_read_len),
                "R1_gc_frac": qc1.gc_frac,
                "R2_gc_frac": qc2.gc_frac,
                "gc_frac": 0.5 * (qc1.gc_frac + qc2.gc_frac),
                "R1_n_frac": qc1.n_frac,
                "R2_n_frac": qc2.n_frac,
                "n_frac": 0.5 * (qc1.n_frac + qc2.n_frac),
                "R1_mean_phred": qc1.mean_phred,
                "R2_mean_phred": qc2.mean_phred,
                "mean_phred": 0.5 * (qc1.mean_phred + qc2.mean_phred),
                "R1_q30_frac": qc1.q30_frac,
                "R2_q30_frac": qc2.q30_frac,
                "q30_frac": 0.5 * (qc1.q30_frac + qc2.q30_frac),
            }
        )
    qc_df = pd.DataFrame(qc_rows).sort_values("sample").reset_index(drop=True)
    qc_df.to_csv(out_dir / "qc_metrics_sampled.csv", index=False)

    feats = []
    labels = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="k-mers"):
        sample = row["sample"]
        r1 = Path(row["r1"])
        r2 = Path(row["r2"])
        v1 = kmer_feature_from_fastq(r1, k=args.k, dim=args.dim, step=args.step, max_reads=kmer_reads)
        v2 = kmer_feature_from_fastq(r2, k=args.k, dim=args.dim, step=args.step, max_reads=kmer_reads)
        v = v1 + v2
        if v.sum() > 0:
            v = v / v.sum()
        feats.append(v)
        labels.append(sample)
    X = np.vstack(feats)

    entropy = {sample: shannon_entropy(X[i]) for i, sample in enumerate(labels)}

    D = pairwise_js_distance_matrix(X)
    dist_df = pd.DataFrame(D, index=labels, columns=labels)
    dist_df.to_csv(out_dir / "kmer_js_distance.csv")

    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    coords = PCA(n_components=2, random_state=0).fit_transform(Xs)
    embed_df = pd.DataFrame(
        {
            "sample": labels,
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "depth_group": [depth_groups[s] for s in labels],
        }
    )

    net_metrics, net_nodes = build_similarity_network_summary(D, labels)

    ref_groups, _ = cluster_reference_libraries(D, labels, depth_groups, n_groups=args.n_groups)
    W, agg = deconvolve_low_depth_libraries(X, labels, depth_groups, ref_groups)

    entropy_df = pd.DataFrame(
        {
            "sample": labels,
            "depth_group": [depth_groups[s] for s in labels],
            "kmer_entropy": [entropy[s] for s in labels],
        }
    ).sort_values("sample").reset_index(drop=True)

    auto_groups_df = pd.DataFrame(
        [{"sample": s, "auto_group": ref_groups.get(s, "")} for s in labels]
    ).sort_values("sample").reset_index(drop=True)

    params_df = pd.DataFrame(
        [
            {
                "k": args.k,
                "dim": args.dim,
                "step": args.step,
                "qc_reads": args.qc_reads,
                "kmer_reads": args.kmer_reads,
                "n_groups": args.n_groups,
            }
        ]
    )

    xlsx_path = out_dir / "Supplementary_Data_S1_Preprocessing.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xl:
        manifest.to_excel(xl, sheet_name="manifest", index=False)
        qc_df.to_excel(xl, sheet_name="qc_sampled", index=False)
        dist_df.to_excel(xl, sheet_name="kmer_js_distance")
        embed_df.to_excel(xl, sheet_name="embedding_pca", index=False)
        entropy_df.to_excel(xl, sheet_name="entropy", index=False)
        params_df.to_excel(xl, sheet_name="params", index=False)
        auto_groups_df.to_excel(xl, sheet_name="auto_groups", index=False)
        net_metrics.to_excel(xl, sheet_name="network_summary", index=False)
        net_nodes.to_excel(xl, sheet_name="network_nodes", index=False)
        if W is not None:
            W.to_excel(xl, sheet_name="weights_by_reference")
        if agg is not None:
            agg.to_excel(xl, sheet_name="weights_by_group")

    print("\n[DONE] Preprocessing complete.")
    print(f"  -> Output directory: {out_dir}")
    print(f"  -> Workbook: {xlsx_path}")
    print("  -> Intermediates: sample_manifest.csv, qc_metrics_sampled.csv, kmer_js_distance.csv")


if __name__ == "__main__":
    main()
