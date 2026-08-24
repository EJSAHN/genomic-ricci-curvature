# -*- coding: utf-8 -*-
"""
Shared utilities for paired-read synthetic validation.

The functions in this module are independent of dataset-specific labels.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA


FASTQ_SUFFIXES = (".fastq.gz", ".fq.gz", ".fastq", ".fq")


@dataclass(frozen=True)
class FastqRecord:
    header: str
    sequence: str
    plus: str
    quality: str


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join([str(base_seed), *[str(x) for x in parts]])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def sha256_file(path: str | os.PathLike[str], block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def open_text(path: str | os.PathLike[str], mode: str = "rt"):
    p = str(path)
    if p.lower().endswith(".gz"):
        return gzip.open(p, mode, encoding="utf-8", errors="strict", newline="")
    return open(p, mode, encoding="utf-8", errors="strict", newline="")


def iter_fastq(path: str | os.PathLike[str]) -> Iterator[FastqRecord]:
    with open_text(path, "rt") as fh:
        line_no = 0
        while True:
            h = fh.readline()
            if not h:
                return
            s = fh.readline()
            p = fh.readline()
            q = fh.readline()
            line_no += 4
            if not (s and p and q):
                raise ValueError(f"Truncated FASTQ record near line {line_no} in {path}")
            h = h.rstrip("\r\n")
            s = s.rstrip("\r\n")
            p = p.rstrip("\r\n")
            q = q.rstrip("\r\n")
            if not h.startswith("@"):
                raise ValueError(f"Invalid FASTQ header near line {line_no-3} in {path}: {h[:80]}")
            if not p.startswith("+"):
                raise ValueError(f"Invalid FASTQ plus line near line {line_no-1} in {path}: {p[:80]}")
            if len(s) != len(q):
                raise ValueError(
                    f"Sequence/quality length mismatch near line {line_no} in {path}: "
                    f"{len(s)} != {len(q)}"
                )
            yield FastqRecord(h, s, p, q)


def normalize_read_id(header: str) -> str:
    token = header[1:] if header.startswith("@") else header
    token = token.split()[0]
    token = re.sub(r"([/._-])?[12]$", "", token)
    return token


def iter_paired_fastq(
    r1_path: str | os.PathLike[str],
    r2_path: str | os.PathLike[str],
) -> Iterator[Tuple[int, FastqRecord, FastqRecord]]:
    it1 = iter_fastq(r1_path)
    it2 = iter_fastq(r2_path)
    idx = 0
    while True:
        try:
            a = next(it1)
            end1 = False
        except StopIteration:
            end1 = True
            a = None
        try:
            b = next(it2)
            end2 = False
        except StopIteration:
            end2 = True
            b = None
        if end1 and end2:
            return
        if end1 != end2:
            raise ValueError(f"R1/R2 record counts differ: {r1_path} vs {r2_path}")
        assert a is not None and b is not None
        if normalize_read_id(a.header) != normalize_read_id(b.header):
            raise ValueError(
                f"Paired read identifiers differ at pair {idx}: "
                f"{a.header[:100]} vs {b.header[:100]}"
            )
        yield idx, a, b
        idx += 1


def write_fastq_record(fh, record: FastqRecord, new_header: str) -> None:
    fh.write(new_header + "\n")
    fh.write(record.sequence + "\n")
    fh.write("+\n")
    fh.write(record.quality + "\n")


def deterministic_gzip(
    source_path: str | os.PathLike[str],
    dest_path: str | os.PathLike[str],
    compresslevel: int = 6,
) -> None:
    source = Path(source_path)
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as src, open(dest, "wb") as raw_out:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_out,
            compresslevel=compresslevel,
            mtime=0,
        ) as gz_out:
            shutil.copyfileobj(src, gz_out, length=1024 * 1024)


def largest_remainder_counts(weights: Sequence[float], total: int) -> List[int]:
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError(f"Invalid weights: {weights}")
    w = w / w.sum()
    raw = w * int(total)
    counts = np.floor(raw).astype(int)
    remainder = int(total) - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for idx in order[:remainder]:
        counts[idx] += 1
    if int(counts.sum()) != int(total):
        raise AssertionError("Largest-remainder allocation did not preserve total")
    return counts.tolist()


def normalized_entropy(weights: Sequence[float]) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    if len(w) <= 1:
        return 0.0
    w = w / w.sum()
    return float(-np.sum(w * np.log(w)) / np.log(len(w)))


def discover_fastq_pairs(
    fastq_root: str | os.PathLike[str],
    sample_ids: Sequence[str] | None = None,
) -> Dict[str, Dict[str, str]]:
    root = Path(fastq_root)
    wanted = None if sample_ids is None else {str(x) for x in sample_ids}
    result: Dict[str, Dict[str, str]] = {}
    patterns = [
        re.compile(r"^(.+?)_([12])\.(?:fastq|fq)(\.gz)?$", re.IGNORECASE),
        re.compile(r"^(.+?)[._-]R([12])\.(?:fastq|fq)(\.gz)?$", re.IGNORECASE),
    ]
    for path in sorted(root.iterdir()):
        if not path.is_file() or not path.name.lower().endswith(FASTQ_SUFFIXES):
            continue
        match = None
        for pat in patterns:
            match = pat.match(path.name)
            if match:
                break
        if not match:
            continue
        sample, mate = match.group(1), match.group(2)
        if wanted is not None and sample not in wanted:
            continue
        result.setdefault(sample, {})[mate] = str(path.resolve())
    return result


def read_fastq_sequences(path: str | os.PathLike[str], max_reads: int = 0) -> Iterator[str]:
    n = 0
    for record in iter_fastq(path):
        yield record.sequence
        n += 1
        if max_reads > 0 and n >= max_reads:
            return


def kmer_sketch_probability(
    sequences: Iterable[str],
    k: int,
    sketch_size: int,
) -> np.ndarray:
    counts = np.zeros(int(sketch_size), dtype=np.uint64)
    total = 0
    for sequence in sequences:
        s = sequence.strip().upper()
        if len(s) < k or "N" in s:
            continue
        b = s.encode("ascii", errors="ignore")
        for i in range(0, len(b) - k + 1):
            h = zlib.crc32(b[i : i + k]) & 0xFFFFFFFF
            counts[h % int(sketch_size)] += 1
            total += 1
    if total == 0:
        return np.full(int(sketch_size), 1.0 / float(sketch_size), dtype=np.float64)
    return (counts / float(total)).astype(np.float64)


def js_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    divergence = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(np.sqrt(max(float(divergence), 0.0)))


def pairwise_js_distance_matrix(
    signatures: Sequence[np.ndarray],
    names: Sequence[str],
) -> pd.DataFrame:
    n = len(signatures)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = js_distance(signatures[i], signatures[j])
            matrix[i, j] = d
            matrix[j, i] = d
    return pd.DataFrame(matrix, index=list(names), columns=list(names))


def build_symmetric_knn_graph(distance: pd.DataFrame, k: int) -> nx.Graph:
    names = distance.index.tolist()
    values = distance.to_numpy(dtype=float)
    graph = nx.Graph()
    graph.add_nodes_from(names)
    edges = set()
    for i, name in enumerate(names):
        order = np.argsort(values[i], kind="stable")
        neighbors = [j for j in order if j != i][: int(k)]
        for j in neighbors:
            edges.add(tuple(sorted((name, names[j]))))
    edge_distances = [float(distance.loc[a, b]) for a, b in edges]
    sigma = max(float(np.median(edge_distances)) if edge_distances else 1.0, 1e-12)
    for a, b in sorted(edges):
        d = float(distance.loc[a, b])
        affinity = float(np.exp(-(d * d) / (2.0 * sigma * sigma)))
        graph.add_edge(a, b, d=d, w=affinity)
    return graph


def wasserstein1_transport(cost: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    cost = np.asarray(cost, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m, n = cost.shape
    objective = cost.reshape(-1)
    constraints = []
    rhs = []
    for i in range(m):
        row = np.zeros(m * n, dtype=np.float64)
        row[i * n : (i + 1) * n] = 1.0
        constraints.append(row)
        rhs.append(a[i])
    for j in range(n):
        col = np.zeros(m * n, dtype=np.float64)
        col[j::n] = 1.0
        constraints.append(col)
        rhs.append(b[j])
    result = linprog(
        objective,
        A_eq=np.vstack(constraints),
        b_eq=np.asarray(rhs),
        bounds=[(0.0, None)] * (m * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Optimal transport failed: {result.message}")
    return float(result.fun)


def ollivier_ricci_curvature(
    graph: nx.Graph,
    distance: pd.DataFrame,
    alpha: float,
) -> Dict[Tuple[str, str], float]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    names = distance.index.tolist()
    index = {name: i for i, name in enumerate(names)}
    values = distance.to_numpy(dtype=float)
    curvature: Dict[Tuple[str, str], float] = {}
    for u, v in graph.edges():
        nu = list(graph.neighbors(u))
        nv = list(graph.neighbors(v))
        su = [u, *nu]
        sv = [v, *nv]
        a = np.zeros(len(su), dtype=float)
        b = np.zeros(len(sv), dtype=float)
        a[0] = 1.0 - alpha
        b[0] = 1.0 - alpha
        if nu:
            a[1:] = alpha / len(nu)
        else:
            a[0] = 1.0
        if nv:
            b[1:] = alpha / len(nv)
        else:
            b[0] = 1.0
        iu = [index[x] for x in su]
        iv = [index[x] for x in sv]
        cost = values[np.ix_(iu, iv)]
        d_uv = float(distance.loc[u, v])
        kappa = 0.0 if d_uv <= 0 else 1.0 - wasserstein1_transport(cost, a, b) / d_uv
        curvature[tuple(sorted((u, v)))] = float(kappa)
    return curvature


def zscore(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    sd = float(np.nanstd(x))
    if sd <= 1e-12:
        return np.zeros_like(x)
    return (x - float(np.nanmean(x))) / sd


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    best_f1 = -1.0
    best_threshold = float("nan")
    for threshold in np.unique(scores):
        prediction = (scores >= threshold).astype(int)
        value = float(f1_score(y_true, prediction, zero_division=0))
        if value > best_f1:
            best_f1 = value
            best_threshold = float(threshold)
    return best_f1, best_threshold


def evaluate_score(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return {
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "best_f1": float("nan"),
            "best_threshold": float("nan"),
        }
    best_f1, threshold = best_f1_threshold(y, s)
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "best_f1": best_f1,
        "best_threshold": threshold,
    }


def compute_node_scores(
    distance: pd.DataFrame,
    k: int,
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, nx.Graph]:
    graph = build_symmetric_knn_graph(distance, k=k)
    curvature = ollivier_ricci_curvature(graph, distance, alpha=alpha)
    for u, v in graph.edges():
        graph.edges[u, v]["orc"] = curvature[tuple(sorted((u, v)))]
    betweenness = nx.betweenness_centrality(graph, weight="d", normalized=True)
    negative_orc = {node: 0.0 for node in graph.nodes()}
    incident_distance = {node: [] for node in graph.nodes()}
    edge_rows = []
    for u, v, data in graph.edges(data=True):
        kappa = float(data["orc"])
        d = float(data["d"])
        incident_distance[u].append(d)
        incident_distance[v].append(d)
        if kappa < 0:
            negative_orc[u] += -kappa
            negative_orc[v] += -kappa
        edge_rows.append({"u": u, "v": v, "distance": d, "orc": kappa})
    nodes = list(graph.nodes())
    bet = np.asarray([betweenness[n] for n in nodes], dtype=float)
    neg = np.asarray([negative_orc[n] for n in nodes], dtype=float)
    mean_knn = np.asarray(
        [float(np.mean(incident_distance[n])) if incident_distance[n] else 0.0 for n in nodes],
        dtype=float,
    )
    tms = zscore(bet) + zscore(neg)

    # Simple unsupervised comparators.
    values = distance.loc[nodes, nodes].to_numpy(dtype=float)
    n_components = max(1, min(5, len(nodes) - 1, values.shape[1]))
    coords = PCA(n_components=n_components, svd_solver="full").fit_transform(values)
    pca_distance = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
    n_neighbors = max(2, min(int(k), len(nodes) - 1))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric="precomputed")
    lof.fit_predict(values)
    lof_score = -lof.negative_outlier_factor_

    node_df = pd.DataFrame(
        {
            "sample_id": nodes,
            "betweenness": bet,
            "negative_orc_incidence": neg,
            "mean_incident_distance": mean_knn,
            "tms": tms,
            "betweenness_z": zscore(bet),
            "negative_orc_z": zscore(neg),
            "mean_distance_z": zscore(mean_knn),
            "pca_distance": pca_distance,
            "lof_score": lof_score,
        }
    )
    edge_df = pd.DataFrame(edge_rows)
    return node_df, edge_df, graph


def spearman_safe(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3 or len(np.unique(a[mask])) < 2 or len(np.unique(b[mask])) < 2:
        return float("nan"), float("nan")
    rho, p = spearmanr(a[mask], b[mask])
    return float(rho), float(p)


def write_json(path: str | os.PathLike[str], payload: Mapping) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
