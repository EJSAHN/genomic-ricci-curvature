# -*- coding: utf-8 -*-
"""
GBS Q1-style "physics/geometry/graph/portfolio" starter pipeline (alignment-free) for paired FASTQ(.gz).

Inputs (expected in one folder):
  SRRxxxxxxx_1.fastq.gz
  SRRxxxxxxx_2.fastq.gz

Outputs:
  results/Supplementary_Data_S1.xlsx (multi-sheet)
  results/figures/main/*.png + *.pdf (300 dpi png)
  results/figures/supplementary/*.png + *.pdf
"""
import argparse, gzip, zlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import networkx as nx


def open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline=None)
    return open(path, "rt", encoding="utf-8", errors="replace", newline=None)


def iter_fastq(path: Path, max_reads: Optional[int] = None):
    """Yield (seq, qual) tuples from FASTQ(.gz)."""
    n = 0
    with open_maybe_gzip(path) as fh:
        while True:
            h = fh.readline()
            if not h:
                break
            seq = fh.readline().strip()
            _plus = fh.readline()
            qual = fh.readline().strip()
            if not qual:
                break
            yield seq, qual
            n += 1
            if max_reads is not None and n >= max_reads:
                break


def phred_scores(qual: str) -> np.ndarray:
    # Phred+33
    return (np.frombuffer(qual.encode("ascii", "ignore"), dtype=np.uint8) - 33).astype(np.int16)


@dataclass
class QcResult:
    reads_used: int
    bases_used: int
    mean_read_len: float
    sd_read_len: float
    gc_frac: float
    n_frac: float
    mean_phred: float
    q30_frac: float
    per_pos_mean_q: np.ndarray


def qc_fastq(path: Path, max_reads: Optional[int] = 200000) -> QcResult:
    lens = []
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

    gc_frac = gc / bases_used if bases_used else float("nan")
    n_frac = ncount / bases_used if bases_used else float("nan")
    mean_phred = qsum_total / qcount_total if qcount_total else float("nan")
    q30_frac = q30 / qcount_total if qcount_total else float("nan")

    per_pos_mean = np.divide(per_pos_sum, per_pos_cnt, out=np.zeros_like(per_pos_sum), where=per_pos_cnt > 0)

    return QcResult(
        reads_used=reads,
        bases_used=bases_used,
        mean_read_len=mean_len,
        sd_read_len=sd_len,
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
        r2 = input_dir / (sample + "_2.fastq.gz")
        if not r2.exists():
            r2 = input_dir / (sample + "_2.fastq")
        if not r2.exists():
            raise FileNotFoundError(f"Missing mate for {r1.name}: expected {r2.name}")

        rows.append(
            dict(
                sample=sample,
                r1=str(r1),
                r2=str(r2),
                r1_bytes=r1.stat().st_size,
                r2_bytes=r2.stat().st_size,
                total_bytes=r1.stat().st_size + r2.stat().st_size,
            )
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
    # Jensen-Shannon distance = sqrt(JS divergence)
    p = p.astype(np.float64, copy=False)
    q = q.astype(np.float64, copy=False)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)

    def kl(a, b):
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


def save_fig(fig: plt.Figure, out_png: Path, out_pdf: Path, dpi: int = 300):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_depth_bar(manifest: pd.DataFrame, outdir: Path):
    df = manifest.sort_values("total_bytes", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df["sample"], df["total_bytes"] / (1024**2))
    ax.set_ylabel("Total file size (MB) [proxy for sequencing depth]")
    ax.set_title("Sequencing depth proxy per sample (R1+R2 gz size)")
    ax.tick_params(axis="x", labelrotation=90)
    save_fig(fig, outdir / "figures" / "main" / "Fig_depth_proxy.png", outdir / "figures" / "main" / "Fig_depth_proxy.pdf")


def plot_quality_profiles(per_pos_dict: Dict[Tuple[str, str], np.ndarray], groups: Dict[str, str], outdir: Path):
    def group_mean(read: str, group: str) -> np.ndarray:
        arrs = [per_pos_dict[(sample, read)] for sample, g in groups.items() if g == group and (sample, read) in per_pos_dict]
        if not arrs:
            return np.array([])
        L = max(len(a) for a in arrs)
        mat = np.full((len(arrs), L), np.nan, dtype=np.float64)
        for i, a in enumerate(arrs):
            mat[i, : len(a)] = a
        return np.nanmean(mat, axis=0)

    unique_groups = sorted(set(groups.values()))
    fig, ax = plt.subplots(figsize=(10, 5))
    for g in unique_groups:
        for read in ["R1", "R2"]:
            m = group_mean(read, g)
            if m.size == 0:
                continue
            ax.plot(np.arange(1, m.size + 1), m, label=f"{g}-{read}")
    ax.axhline(30, linestyle="--", linewidth=1)
    ax.set_xlabel("Position (bp)")
    ax.set_ylabel("Mean Phred quality")
    ax.set_title("Mean per-base quality profile (group-averaged)")
    ax.legend()
    save_fig(fig, outdir / "figures" / "main" / "Fig_quality_profiles.png", outdir / "figures" / "main" / "Fig_quality_profiles.pdf")


def plot_heatmap_dendrogram(D: np.ndarray, labels: List[str], outdir: Path):
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    dendro = dendrogram(Z, orientation="left", labels=labels, ax=ax1)
    ax1.set_xticks([])

    order = dendro["leaves"]
    D_ord = D[np.ix_(order, order)]
    labels_ord = [labels[i] for i in order]

    ax2 = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    im = ax2.imshow(D_ord, aspect="auto")
    ax2.set_xticks(range(len(labels_ord)))
    ax2.set_xticklabels(labels_ord, rotation=90, fontsize=7)
    ax2.set_yticks(range(len(labels_ord)))
    ax2.set_yticklabels(labels_ord, fontsize=7)
    ax2.set_title("Jensen–Shannon distance heatmap (hashed k-mers)")

    cax = fig.add_axes([0.92, 0.1, 0.02, 0.6])
    fig.colorbar(im, cax=cax, label="JS distance")

    save_fig(fig, outdir / "figures" / "main" / "kmer_JS_heatmap.png", outdir / "figures" / "main" / "kmer_JS_heatmap.pdf")
    return order, Z


def plot_embedding(X: np.ndarray, labels: List[str], groups: Dict[str, str], outdir: Path) -> pd.DataFrame:
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(7, 6))
    for sample, (x, y) in zip(labels, coords):
        ax.scatter(x, y)
        ax.text(x, y, sample, fontsize=7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Geometry of samples in hashed k-mer space (PCA)")
    save_fig(fig, outdir / "figures" / "main" / "embedding_PCA.png", outdir / "figures" / "main" / "embedding_PCA.pdf")

    return pd.DataFrame(
        {
            "sample": labels,
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "depth_group": [groups[s] for s in labels],
        }
    )


def assign_depth_groups(manifest: pd.DataFrame, random_state: int = 0) -> Dict[str, str]:
    x = manifest[["total_bytes"]].values.astype(np.float64)
    xlog = np.log1p(x)
    km = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    lab = km.fit_predict(xlog)

    means = {c: float(x[lab == c].mean()) for c in [0, 1]}
    hi = max(means, key=means.get)
    return {sample: ("HiDepth" if c == hi else "LoDepth") for sample, c in zip(manifest["sample"], lab)}


def plot_entropy_gc(qc_df: pd.DataFrame, entropy: Dict[str, float], outdir: Path):
    df = qc_df.copy()
    df["entropy"] = df["sample"].map(entropy)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["gc_frac"] * 100, df["entropy"])
    for _, r in df.iterrows():
        ax.text(r["gc_frac"] * 100, r["entropy"], r["sample"], fontsize=7)
    ax.set_xlabel("GC (%) [sampled]")
    ax.set_ylabel("Shannon entropy (hashed k-mers)")
    ax.set_title("GC vs k-mer entropy (alignment-free complexity)")
    save_fig(fig, outdir / "figures" / "supplementary" / "GC_entropy.png", outdir / "figures" / "supplementary" / "GC_entropy.pdf")


def build_similarity_network(D: np.ndarray, labels: List[str], outdir: Path):
    Dmax = float(D.max()) if float(D.max()) > 0 else 1.0
    S = 1.0 - (D / Dmax)

    vals = S[~np.eye(len(labels), dtype=bool)]
    thr = float(np.quantile(vals, 0.90))

    G = nx.Graph()
    for a in labels:
        G.add_node(a)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if S[i, j] >= thr:
                G.add_edge(labels[i], labels[j], weight=float(S[i, j]))

    comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    node2c = {}
    for idx, cset in enumerate(comms):
        for n in cset:
            node2c[n] = idx

    pos = nx.spring_layout(G, seed=0)
    fig, ax = plt.subplots(figsize=(8, 7))
    for n in G.nodes():
        x, y = pos[n]
        ax.scatter(x, y)
        ax.text(x, y, n, fontsize=7)
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], linewidth=1)

    ax.set_title(f"Similarity network (thr={thr:.3f}, communities={len(comms)})")
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(fig, outdir / "figures" / "supplementary" / "similarity_network.png", outdir / "figures" / "supplementary" / "similarity_network.pdf")

    net_metrics = pd.DataFrame([{"edges": G.number_of_edges(), "threshold": thr, "communities": len(set(node2c.values()))}])
    return net_metrics, node2c


def cluster_individuals(D: np.ndarray, labels: List[str], depth_groups: Dict[str, str], n_groups: int = 5):
    indiv_idx = [i for i, s in enumerate(labels) if depth_groups[s] == "HiDepth"]
    if len(indiv_idx) < 3:
        return {}, indiv_idx

    n_groups = max(2, min(n_groups, len(indiv_idx)))
    D_ind = D[np.ix_(indiv_idx, indiv_idx)]

    model = AgglomerativeClustering(n_clusters=n_groups, metric="precomputed", linkage="average")
    ind_labels = model.fit_predict(D_ind)

    return {labels[i]: f"G{ind_labels[k] + 1}" for k, i in enumerate(indiv_idx)}, indiv_idx


def deconvolve_pools(X: np.ndarray, labels: List[str], depth_groups: Dict[str, str], indiv_groups: Dict[str, str], outdir: Path):
    indiv = [s for s in labels if depth_groups[s] == "HiDepth"]
    pools = [s for s in labels if depth_groups[s] == "LoDepth"]
    if not pools or not indiv:
        return None, None

    idx_ind = [labels.index(s) for s in indiv]
    idx_pool = [labels.index(s) for s in pools]

    A = X[idx_ind].T
    weights_rows = []
    for s, idx in zip(pools, idx_pool):
        b = X[idx]
        w, rnorm = nnls(A, b)
        if w.sum() > 0:
            w = w / w.sum()
        row = {"pool_sample": s, "residual_norm": float(rnorm)}
        for ind_s, ww in zip(indiv, w):
            row[ind_s] = float(ww)
        weights_rows.append(row)

    W = pd.DataFrame(weights_rows).set_index("pool_sample")

    group_names = sorted(set(indiv_groups.values())) if indiv_groups else []
    agg = pd.DataFrame(index=W.index, columns=group_names, dtype=float)
    for g in group_names:
        inds = [s for s in indiv if indiv_groups.get(s) == g]
        agg[g] = W[inds].sum(axis=1) if inds else 0.0

    if group_names:
        fig, ax = plt.subplots(figsize=(10, 5))
        bottom = np.zeros(len(agg), dtype=float)
        x = np.arange(len(agg))
        for g in group_names:
            vals = agg[g].values.astype(float)
            ax.bar(x, vals, bottom=bottom, label=g)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(list(agg.index), rotation=90, fontsize=8)
        ax.set_ylabel("Estimated proportion")
        ax.set_title("Pool → cultivar-group deconvolution (NNLS on hashed k-mers)")
        ax.legend(ncols=min(5, len(group_names)))
        save_fig(fig, outdir / "figures" / "main" / "pool_deconvolution.png", outdir / "figures" / "main" / "pool_deconvolution.pdf")

    return W, agg


def main():
    ap = argparse.ArgumentParser()
    # Accept both the original flags (--input/--out) and README-friendly aliases (--fastq_dir/--outdir)
    ap.add_argument("--input", required=False, help="Folder with *_1.fastq(.gz) and *_2.fastq(.gz) files (alias: --fastq_dir)")
    ap.add_argument("--out", required=False, help="Output folder (alias: --outdir)")
    ap.add_argument("--fastq_dir", dest="input", required=False, help="Alias of --input")
    ap.add_argument("--outdir", dest="out", required=False, help="Alias of --out")
    ap.add_argument("--qc_reads", type=int, default=200000, help="Reads sampled per FASTQ for QC (0 = use ALL reads)")
    ap.add_argument("--kmer_reads", type=int, default=50000, help="Reads sampled per FASTQ for k-mers (0 = use ALL reads)")
    ap.add_argument("--k", type=int, default=21, help="k-mer length")
    ap.add_argument("--dim", type=int, default=16384, help="Hashed feature dimension")
    ap.add_argument("--step", type=int, default=4, help="Stride for k-mer sampling within reads")
    ap.add_argument("--n_groups", type=int, default=5, help="How many cultivar groups to infer among HiDepth samples")
    args = ap.parse_args()

    if args.input is None or args.out is None:
        ap.error("You must provide input/output folders using either --input/--out or --fastq_dir/--outdir.")

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    qc_reads = None if args.qc_reads == 0 else args.qc_reads
    kmer_reads = None if args.kmer_reads == 0 else args.kmer_reads

    manifest = find_pairs(in_dir)
    manifest.to_csv(out_dir / "file_manifest.csv", index=False)

    depth_groups = assign_depth_groups(manifest, random_state=0)

    # QC
    qc_rows = []
    per_pos = {}
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="QC"):
        s = row["sample"]
        r1 = Path(row["r1"])
        r2 = Path(row["r2"])
        qc1 = qc_fastq(r1, max_reads=qc_reads)
        qc2 = qc_fastq(r2, max_reads=qc_reads)
        per_pos[(s, "R1")] = qc1.per_pos_mean_q
        per_pos[(s, "R2")] = qc2.per_pos_mean_q
        qc_rows.append(
            dict(
                sample=s,
                depth_group=depth_groups[s],
                R1_reads_used=qc1.reads_used,
                R2_reads_used=qc2.reads_used,
                R1_mean_len=qc1.mean_read_len,
                R2_mean_len=qc2.mean_read_len,
                R1_gc_frac=qc1.gc_frac,
                R2_gc_frac=qc2.gc_frac,
                R1_n_frac=qc1.n_frac,
                R2_n_frac=qc2.n_frac,
                R1_mean_phred=qc1.mean_phred,
                R2_mean_phred=qc2.mean_phred,
                R1_q30_frac=qc1.q30_frac,
                R2_q30_frac=qc2.q30_frac,
            )
        )
    qc_df = pd.DataFrame(qc_rows).sort_values("sample").reset_index(drop=True)
    qc_df["gc_frac"] = 0.5 * (qc_df["R1_gc_frac"] + qc_df["R2_gc_frac"])
    qc_df["mean_phred"] = 0.5 * (qc_df["R1_mean_phred"] + qc_df["R2_mean_phred"])
    qc_df.to_csv(out_dir / "qc_metrics_sampled.csv", index=False)

    # main figs
    plot_depth_bar(manifest, out_dir)
    plot_quality_profiles(per_pos, depth_groups, out_dir)

    # k-mer features
    feats = []
    labels = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="k-mers"):
        s = row["sample"]
        r1 = Path(row["r1"])
        r2 = Path(row["r2"])
        v1 = kmer_feature_from_fastq(r1, k=args.k, dim=args.dim, step=args.step, max_reads=kmer_reads)
        v2 = kmer_feature_from_fastq(r2, k=args.k, dim=args.dim, step=args.step, max_reads=kmer_reads)
        v = v1 + v2
        if v.sum() > 0:
            v = v / v.sum()
        feats.append(v)
        labels.append(s)
    X = np.vstack(feats)

    # entropy + Fig
    ent = {s: shannon_entropy(X[i]) for i, s in enumerate(labels)}
    plot_entropy_gc(qc_df, ent, out_dir)

    # distances + Fig
    D = pairwise_js_distance_matrix(X)
    dist_df = pd.DataFrame(D, index=labels, columns=labels)
    dist_df.to_csv(out_dir / "kmer_js_distance.csv")
    _order, _Z = plot_heatmap_dendrogram(D, labels, out_dir)

    # Fig embedding
    embed_df = plot_embedding(X, labels, depth_groups, out_dir)

    # network + Fig
    net_metrics, node2c = build_similarity_network(D, labels, out_dir)

    # cultivar grouping + Fig
    indiv_groups, _ = cluster_individuals(D, labels, depth_groups, n_groups=args.n_groups)
    W, agg = deconvolve_pools(X, labels, depth_groups, indiv_groups, out_dir)

    # Excel (Supplementary Data S1)
    xlsx_path = out_dir / "Supplementary_Data_S1.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xl:
        manifest.to_excel(xl, sheet_name="manifest", index=False)
        qc_df.to_excel(xl, sheet_name="qc_sampled", index=False)
        dist_df.to_excel(xl, sheet_name="kmer_js_distance")
        embed_df.to_excel(xl, sheet_name="embedding_pca", index=False)
        pd.DataFrame(
            {"sample": labels, "depth_group": [depth_groups[s] for s in labels], "kmer_entropy": [ent[s] for s in labels]}
        ).to_excel(xl, sheet_name="entropy", index=False)
        pd.DataFrame(
            [{"k": args.k, "dim": args.dim, "step": args.step, "qc_reads": args.qc_reads, "kmer_reads": args.kmer_reads}]
        ).to_excel(xl, sheet_name="params", index=False)
        pd.DataFrame([{"sample": s, "auto_group": indiv_groups.get(s, "")} for s in labels]).to_excel(
            xl, sheet_name="auto_cultivar_groups", index=False
        )
        net_metrics.to_excel(xl, sheet_name="network_metrics", index=False)
        if W is not None:
            W.to_excel(xl, sheet_name="pool_weights_individual")
        if agg is not None:
            agg.to_excel(xl, sheet_name="pool_weights_group")

    print("\n[DONE] All outputs written:")
    print(f"  -> {out_dir}")
    print(f"  -> Excel: {xlsx_path}")
    print("  -> Figures: results/figures/main + results/figures/supplementary\n")


if __name__ == "__main__":
    main()
