# -*- coding: utf-8 -*-
"""Shared utilities for the locked finger millet external benchmark."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, silhouette_score
from sklearn.neighbors import LocalOutlierFactor


@dataclass(frozen=True)
class FastqRecord:
    header: str
    sequence: str
    plus: str
    quality: str


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def read_tsv(path: str | os.PathLike[str]) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(
    path: str | os.PathLike[str],
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    target = Path(path)
    rows = list(rows)
    ensure_dir(target.parent)
    if fieldnames is None:
        fields: List[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: "" if row.get(key) is None else row.get(key) for key in fieldnames}
            )
    os.replace(temporary, target)


def sha256_file(path: str | os.PathLike[str], block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *[str(part) for part in parts]])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (
        2**32 - 1
    )


def stable_key(base_seed: int, *parts: object) -> str:
    payload = "|".join([str(base_seed), *[str(part) for part in parts]])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def parse_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def normalize_read_id(header: str) -> str:
    token = header[1:] if header.startswith("@") else header
    token = token.split()[0]
    token = re.sub(r"([/._-])?[12]$", "", token)
    return token


def open_fastq_text(path: str | os.PathLike[str]):
    text = str(path)
    if text.lower().endswith(".gz"):
        return gzip.open(text, "rt", encoding="ascii", errors="strict", newline="")
    return open(text, "rt", encoding="ascii", errors="strict", newline="")


def read_fastq_record(handle, path: str, pair_index: int) -> Optional[FastqRecord]:
    header = handle.readline()
    if not header:
        return None
    sequence = handle.readline()
    plus = handle.readline()
    quality = handle.readline()
    if not (sequence and plus and quality):
        raise ValueError(f"Truncated FASTQ record at pair {pair_index} in {path}")
    header = header.rstrip("\r\n")
    sequence = sequence.rstrip("\r\n")
    plus = plus.rstrip("\r\n")
    quality = quality.rstrip("\r\n")
    if not header.startswith("@"):
        raise ValueError(
            f"Invalid FASTQ header at pair {pair_index} in {path}: {header[:100]}"
        )
    if not plus.startswith("+"):
        raise ValueError(
            f"Invalid FASTQ plus line at pair {pair_index} in {path}: {plus[:100]}"
        )
    if len(sequence) != len(quality):
        raise ValueError(
            f"Sequence/quality mismatch at pair {pair_index} in {path}: "
            f"{len(sequence)} != {len(quality)}"
        )
    return FastqRecord(header, sequence, plus, quality)


def iter_paired_fastq(
    r1_path: str | os.PathLike[str],
    r2_path: str | os.PathLike[str],
) -> Iterator[Tuple[int, FastqRecord, FastqRecord]]:
    r1 = str(r1_path)
    r2 = str(r2_path)
    with open_fastq_text(r1) as handle1, open_fastq_text(r2) as handle2:
        pair_index = 0
        while True:
            record1 = read_fastq_record(handle1, r1, pair_index)
            record2 = read_fastq_record(handle2, r2, pair_index)
            if record1 is None and record2 is None:
                return
            if (record1 is None) != (record2 is None):
                raise ValueError(f"R1/R2 record counts differ: {r1} vs {r2}")
            assert record1 is not None and record2 is not None
            if normalize_read_id(record1.header) != normalize_read_id(record2.header):
                raise ValueError(
                    f"Paired identifiers differ at pair {pair_index}: "
                    f"{record1.header[:100]} vs {record2.header[:100]}"
                )
            yield pair_index, record1, record2
            pair_index += 1


def write_fastq_record(handle, record: FastqRecord, new_header: str) -> None:
    handle.write(new_header + "\n")
    handle.write(record.sequence + "\n")
    handle.write("+\n")
    handle.write(record.quality + "\n")


def deterministic_gzip_from_files(
    source_paths: Sequence[str | os.PathLike[str]],
    destination: str | os.PathLike[str],
    compresslevel: int = 6,
) -> None:
    target = Path(destination)
    ensure_dir(target.parent)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_output,
            compresslevel=int(compresslevel),
            mtime=0,
        ) as gzip_output:
            for source_path in source_paths:
                with Path(source_path).open("rb") as source:
                    shutil.copyfileobj(source, gzip_output, length=1024 * 1024)
    os.replace(temporary, target)


def kmer_count_update(counts: np.ndarray, sequence: str, k: int) -> int:
    sequence_bytes = sequence.strip().upper().encode("ascii", errors="ignore")
    if len(sequence_bytes) < int(k):
        return 0
    added = 0
    for index in range(0, len(sequence_bytes) - int(k) + 1):
        kmer = sequence_bytes[index : index + int(k)]
        if b"N" in kmer:
            continue
        hashed = zlib.crc32(kmer) & 0xFFFFFFFF
        counts[hashed % len(counts)] += 1
        added += 1
    return added


def counts_to_probability(counts: np.ndarray) -> np.ndarray:
    total = int(np.asarray(counts, dtype=np.uint64).sum())
    if total <= 0:
        return np.full(len(counts), 1.0 / float(len(counts)), dtype=np.float64)
    return np.asarray(counts, dtype=np.float64) / float(total)


def sketch_generated_pair(
    r1_path: str,
    r2_path: str,
    k: int,
    sketch_size: int,
    expected_pairs: int,
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    counts1 = np.zeros(int(sketch_size), dtype=np.uint64)
    counts2 = np.zeros(int(sketch_size), dtype=np.uint64)
    pair_count = 0
    kmer_count1 = 0
    kmer_count2 = 0
    for _, record1, record2 in iter_paired_fastq(r1_path, r2_path):
        kmer_count1 += kmer_count_update(counts1, record1.sequence, int(k))
        kmer_count2 += kmer_count_update(counts2, record2.sequence, int(k))
        pair_count += 1
    if pair_count != int(expected_pairs):
        raise ValueError(
            f"Generated pair count mismatch for {r1_path}: "
            f"observed={pair_count}, expected={expected_pairs}"
        )
    signature1 = counts_to_probability(counts1)
    signature2 = counts_to_probability(counts2)
    paired = 0.5 * (signature1 + signature2)
    paired = paired / paired.sum()
    return signature1, paired, pair_count, kmer_count1, kmer_count2


def js_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    divergence = 0.5 * np.sum(p * np.log(p / midpoint)) + 0.5 * np.sum(
        q * np.log(q / midpoint)
    )
    return float(np.sqrt(max(float(divergence), 0.0)))


def pairwise_js_distance_matrix(
    signatures: Sequence[np.ndarray],
    names: Sequence[str],
) -> pd.DataFrame:
    count = len(signatures)
    matrix = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            distance = js_distance(signatures[i], signatures[j])
            matrix[i, j] = distance
            matrix[j, i] = distance
    return pd.DataFrame(matrix, index=list(names), columns=list(names))


def build_symmetric_knn_graph(distance: pd.DataFrame, k: int) -> nx.Graph:
    names = distance.index.tolist()
    values = distance.to_numpy(dtype=float)
    graph = nx.Graph()
    graph.add_nodes_from(names)
    edges = set()
    for row_index, name in enumerate(names):
        order = np.argsort(values[row_index], kind="stable")
        neighbors = [index for index in order if index != row_index][: int(k)]
        for neighbor_index in neighbors:
            edges.add(tuple(sorted((name, names[neighbor_index]))))
    edge_distances = [float(distance.loc[a, b]) for a, b in edges]
    sigma = max(
        float(np.median(edge_distances)) if edge_distances else 1.0,
        1e-12,
    )
    for a, b in sorted(edges):
        edge_distance = float(distance.loc[a, b])
        affinity = float(
            np.exp(-(edge_distance * edge_distance) / (2.0 * sigma * sigma))
        )
        graph.add_edge(a, b, d=edge_distance, w=affinity)
    graph.graph["sigma"] = sigma
    return graph


def wasserstein1_transport(cost: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    cost = np.asarray(cost, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rows, columns = cost.shape
    objective = cost.reshape(-1)
    constraints = []
    right_hand = []
    for row_index in range(rows):
        row = np.zeros(rows * columns, dtype=np.float64)
        row[row_index * columns : (row_index + 1) * columns] = 1.0
        constraints.append(row)
        right_hand.append(a[row_index])
    for column_index in range(columns):
        column = np.zeros(rows * columns, dtype=np.float64)
        column[column_index::columns] = 1.0
        constraints.append(column)
        right_hand.append(b[column_index])
    result = linprog(
        objective,
        A_eq=np.vstack(constraints),
        b_eq=np.asarray(right_hand),
        bounds=[(0.0, None)] * (rows * columns),
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
    name_to_index = {name: index for index, name in enumerate(names)}
    values = distance.to_numpy(dtype=float)
    curvature: Dict[Tuple[str, str], float] = {}
    for u, v in graph.edges():
        neighbors_u = list(graph.neighbors(u))
        neighbors_v = list(graph.neighbors(v))
        support_u = [u, *neighbors_u]
        support_v = [v, *neighbors_v]
        measure_u = np.zeros(len(support_u), dtype=float)
        measure_v = np.zeros(len(support_v), dtype=float)
        measure_u[0] = 1.0 - alpha
        measure_v[0] = 1.0 - alpha
        if neighbors_u:
            measure_u[1:] = alpha / len(neighbors_u)
        else:
            measure_u[0] = 1.0
        if neighbors_v:
            measure_v[1:] = alpha / len(neighbors_v)
        else:
            measure_v[0] = 1.0
        indices_u = [name_to_index[name] for name in support_u]
        indices_v = [name_to_index[name] for name in support_v]
        cost = values[np.ix_(indices_u, indices_v)]
        edge_distance = float(distance.loc[u, v])
        if edge_distance <= 0:
            kappa = 0.0
        else:
            kappa = 1.0 - wasserstein1_transport(
                cost, measure_u, measure_v
            ) / edge_distance
        curvature[tuple(sorted((u, v)))] = float(kappa)
    return curvature


def zscore(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    standard_deviation = float(np.nanstd(array))
    if standard_deviation <= 1e-12:
        return np.zeros_like(array)
    return (array - float(np.nanmean(array))) / standard_deviation


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


def evaluate_score(y_true: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return {
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "best_f1": float("nan"),
            "best_threshold": float("nan"),
        }
    best_f1, threshold = best_f1_threshold(labels, values)
    return {
        "roc_auc": float(roc_auc_score(labels, values)),
        "average_precision": float(average_precision_score(labels, values)),
        "best_f1": best_f1,
        "best_threshold": threshold,
    }


def compute_node_scores(
    distance: pd.DataFrame,
    k: int,
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, nx.Graph]:
    graph = build_symmetric_knn_graph(distance, k=int(k))
    curvature = ollivier_ricci_curvature(graph, distance, alpha=float(alpha))
    for u, v in graph.edges():
        graph.edges[u, v]["orc"] = curvature[tuple(sorted((u, v)))]

    betweenness = nx.betweenness_centrality(graph, weight="d", normalized=True)
    negative_orc = {node: 0.0 for node in graph.nodes()}
    incident_distance: Dict[str, List[float]] = {node: [] for node in graph.nodes()}
    incident_orc: Dict[str, List[float]] = {node: [] for node in graph.nodes()}
    edge_rows: List[Dict[str, Any]] = []

    for u, v, data in graph.edges(data=True):
        kappa = float(data["orc"])
        edge_distance = float(data["d"])
        incident_distance[u].append(edge_distance)
        incident_distance[v].append(edge_distance)
        incident_orc[u].append(kappa)
        incident_orc[v].append(kappa)
        if kappa < 0:
            negative_orc[u] += -kappa
            negative_orc[v] += -kappa
        edge_rows.append(
            {
                "u": u,
                "v": v,
                "distance": edge_distance,
                "orc": kappa,
                "affinity": float(data["w"]),
            }
        )

    nodes = list(graph.nodes())
    bet = np.asarray([betweenness[node] for node in nodes], dtype=float)
    neg = np.asarray([negative_orc[node] for node in nodes], dtype=float)
    mean_distance = np.asarray(
        [
            float(np.mean(incident_distance[node]))
            if incident_distance[node]
            else 0.0
            for node in nodes
        ],
        dtype=float,
    )
    mean_orc = np.asarray(
        [
            float(np.mean(incident_orc[node])) if incident_orc[node] else 0.0
            for node in nodes
        ],
        dtype=float,
    )

    tms = zscore(bet) + zscore(neg)
    real_bridge_score = zscore(bet) + zscore(mean_distance) - zscore(mean_orc)

    values = distance.loc[nodes, nodes].to_numpy(dtype=float)
    component_count = max(1, min(5, len(nodes) - 1, values.shape[1]))
    coordinates = PCA(n_components=component_count, svd_solver="full").fit_transform(
        values
    )
    pca_distance = np.linalg.norm(coordinates - coordinates.mean(axis=0), axis=1)
    neighbor_count = max(2, min(int(k), len(nodes) - 1))
    lof = LocalOutlierFactor(n_neighbors=neighbor_count, metric="precomputed")
    lof.fit_predict(values)
    lof_score = -lof.negative_outlier_factor_

    node_table = pd.DataFrame(
        {
            "sample_id": nodes,
            "betweenness": bet,
            "negative_orc_incidence": neg,
            "mean_incident_distance": mean_distance,
            "mean_incident_orc": mean_orc,
            "tms": tms,
            "real_bridge_score": real_bridge_score,
            "betweenness_z": zscore(bet),
            "negative_orc_z": zscore(neg),
            "mean_distance_z": zscore(mean_distance),
            "mean_orc_z": zscore(mean_orc),
            "pca_distance": pca_distance,
            "lof_score": lof_score,
        }
    )
    return node_table, pd.DataFrame(edge_rows), graph


def spearman_safe(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    first = np.asarray(x, dtype=float)
    second = np.asarray(y, dtype=float)
    mask = np.isfinite(first) & np.isfinite(second)
    if (
        int(mask.sum()) < 3
        or len(np.unique(first[mask])) < 2
        or len(np.unique(second[mask])) < 2
    ):
        return float("nan"), float("nan")
    rho, p_value = spearmanr(first[mask], second[mask])
    return float(rho), float(p_value)


def exact_any_topk_chance(n: int, positives: int, k: int) -> float:
    k = min(int(k), int(n))
    positives = int(positives)
    n = int(n)
    if positives <= 0:
        return 0.0
    if n - positives < k:
        return 1.0
    return float(1.0 - math.comb(n - positives, k) / math.comb(n, k))


def classical_mds(distance: pd.DataFrame, dimensions: int = 5) -> pd.DataFrame:
    names = distance.index.tolist()
    values = distance.to_numpy(dtype=float)
    n = len(names)
    centering = np.eye(n) - np.ones((n, n), dtype=float) / n
    gram = -0.5 * centering @ (values**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > 1e-12
    eigenvalues = eigenvalues[positive][: int(dimensions)]
    eigenvectors = eigenvectors[:, positive][:, : len(eigenvalues)]
    coordinates = eigenvectors * np.sqrt(eigenvalues)
    table = pd.DataFrame({"sample_id": names})
    for index in range(int(dimensions)):
        if index < coordinates.shape[1]:
            table[f"MDS{index + 1}"] = coordinates[:, index]
            table[f"eigenvalue_{index + 1}"] = float(eigenvalues[index])
        else:
            table[f"MDS{index + 1}"] = 0.0
            table[f"eigenvalue_{index + 1}"] = 0.0
    return table


def diffusion_coordinates(graph: nx.Graph, dimensions: int = 3) -> pd.DataFrame:
    nodes = list(graph.nodes())
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight="w", dtype=float)
    degrees = adjacency.sum(axis=1)
    inverse_sqrt = np.zeros_like(degrees)
    mask = degrees > 0
    inverse_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
    symmetric = inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    table = pd.DataFrame({"sample_id": nodes})
    coordinate_index = 0
    for index in range(1, min(len(eigenvalues), int(dimensions) + 1)):
        coordinate_index += 1
        table[f"Diffusion{coordinate_index}"] = (
            eigenvectors[:, index] * eigenvalues[index]
        )
        table[f"diffusion_eigenvalue_{coordinate_index}"] = float(
            eigenvalues[index]
        )
    while coordinate_index < int(dimensions):
        coordinate_index += 1
        table[f"Diffusion{coordinate_index}"] = 0.0
        table[f"diffusion_eigenvalue_{coordinate_index}"] = 0.0
    return table


def distance_population_summary(
    distance: pd.DataFrame,
    populations: Mapping[str, str],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    names = distance.index.tolist()
    within_values: List[float] = []
    between_values: List[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            first = names[i]
            second = names[j]
            value = float(distance.loc[first, second])
            same = populations[first] == populations[second]
            rows.append(
                {
                    "sample_a": first,
                    "sample_b": second,
                    "population_a": populations[first],
                    "population_b": populations[second],
                    "same_population": int(same),
                    "js_distance": value,
                }
            )
            if same:
                within_values.append(value)
            else:
                between_values.append(value)
    summary = {
        "within_n": len(within_values),
        "within_mean": float(np.mean(within_values)),
        "within_median": float(np.median(within_values)),
        "between_n": len(between_values),
        "between_mean": float(np.mean(between_values)),
        "between_median": float(np.median(between_values)),
        "between_minus_within_mean": float(
            np.mean(between_values) - np.mean(within_values)
        ),
    }
    labels = np.asarray([populations[name] for name in names])
    try:
        summary["silhouette_precomputed"] = float(
            silhouette_score(distance.to_numpy(float), labels, metric="precomputed")
        )
    except Exception:
        summary["silhouette_precomputed"] = float("nan")
    return pd.DataFrame(rows), summary


def permanova_pseudo_f(
    distance: pd.DataFrame,
    labels: Sequence[str],
    permutations: int,
    seed: int,
) -> Dict[str, float]:
    values = distance.to_numpy(dtype=float)
    n = len(values)
    label_array = np.asarray(labels, dtype=object)
    unique_labels = np.unique(label_array)
    group_count = len(unique_labels)
    if group_count < 2 or n <= group_count:
        return {
            "pseudo_f": float("nan"),
            "permutation_p": float("nan"),
            "permutations": int(permutations),
        }
    centering = np.eye(n) - np.ones((n, n), dtype=float) / n
    gram = -0.5 * centering @ (values**2) @ centering
    total_ss = float(np.trace(gram))

    def statistic(current_labels: np.ndarray) -> float:
        within_ss = 0.0
        for label in np.unique(current_labels):
            indices = np.where(current_labels == label)[0]
            block = gram[np.ix_(indices, indices)]
            within_ss += float(np.trace(block) - block.sum() / len(indices))
        between_ss = total_ss - within_ss
        denominator_df = n - len(np.unique(current_labels))
        numerator_df = len(np.unique(current_labels)) - 1
        if within_ss <= 0 or denominator_df <= 0 or numerator_df <= 0:
            return float("nan")
        return float((between_ss / numerator_df) / (within_ss / denominator_df))

    observed = statistic(label_array)
    generator = np.random.default_rng(int(seed))
    exceed = 0
    valid = 0
    for _ in range(int(permutations)):
        permuted = generator.permutation(label_array)
        value = statistic(permuted)
        if np.isfinite(value):
            valid += 1
            if value >= observed - 1e-12:
                exceed += 1
    p_value = (exceed + 1) / (valid + 1) if valid >= 0 else float("nan")
    return {
        "pseudo_f": observed,
        "permutation_p": float(p_value),
        "permutations": int(valid),
    }


def summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
