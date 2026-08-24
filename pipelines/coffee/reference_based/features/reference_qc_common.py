# -*- coding: utf-8 -*-
"""
Shared functions for reference-based quality-control benchmarking.

This module uses Bowtie 2 SAM output directly and does not require BAM/SAM
files to be retained on disk. It is designed for paired-end reduced-
representation sequencing data and Windows execution.
"""
from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")
MD_TOKEN_RE = re.compile(r"(\d+|\^[A-Za-z]+|[A-Za-z])")


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | os.PathLike[str], block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_tsv(path: str | os.PathLike[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write_json(obj: object, path: str | os.PathLike[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def safe_float(x: object, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v


def parse_tags(fields: Sequence[str]) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for field in fields:
        parts = field.split(":", 2)
        if len(parts) == 3:
            tags[parts[0]] = parts[2]
    return tags


@dataclass
class SamRecord:
    qname: str
    flag: int
    rname: str
    pos1: int
    mapq: int
    cigar: str
    rnext: str
    pnext: int
    tlen: int
    seq: str
    qual: str
    tags: Dict[str, str]

    @property
    def is_unmapped(self) -> bool:
        return bool(self.flag & 0x4)

    @property
    def mate_unmapped(self) -> bool:
        return bool(self.flag & 0x8)

    @property
    def is_reverse(self) -> bool:
        return bool(self.flag & 0x10)

    @property
    def is_read1(self) -> bool:
        return bool(self.flag & 0x40)

    @property
    def is_read2(self) -> bool:
        return bool(self.flag & 0x80)

    @property
    def is_secondary(self) -> bool:
        return bool(self.flag & 0x100)

    @property
    def is_supplementary(self) -> bool:
        return bool(self.flag & 0x800)

    @property
    def is_primary(self) -> bool:
        return not self.is_secondary and not self.is_supplementary

    @property
    def is_proper_pair(self) -> bool:
        return bool(self.flag & 0x2)


def parse_sam_line(line: str) -> SamRecord:
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) < 11:
        raise ValueError(f"Malformed SAM record with {len(fields)} fields: {line[:200]}")
    return SamRecord(
        qname=fields[0],
        flag=int(fields[1]),
        rname=fields[2],
        pos1=int(fields[3]),
        mapq=int(fields[4]),
        cigar=fields[5],
        rnext=fields[6],
        pnext=int(fields[7]),
        tlen=int(fields[8]),
        seq=fields[9],
        qual=fields[10],
        tags=parse_tags(fields[11:]),
    )


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1]


def oriented_seq_qual(record: SamRecord) -> Tuple[str, str]:
    """
    Return sequence/quality in reference-forward orientation.

    SAM stores SEQ in the orientation in which the read was aligned. Bowtie 2
    follows the SAM specification; for reverse-strand records the stored SEQ is
    reverse-complemented relative to the original FASTQ. Therefore no further
    reverse-complement operation is required here. This helper exists to make
    that convention explicit and to permit future aligner-specific changes.
    """
    return record.seq, record.qual


def cigar_ops(cigar: str) -> List[Tuple[int, str]]:
    if cigar == "*" or not cigar:
        return []
    ops = [(int(n), op) for n, op in CIGAR_RE.findall(cigar)]
    if not ops or "".join(f"{n}{op}" for n, op in ops) != cigar:
        raise ValueError(f"Unsupported or malformed CIGAR: {cigar}")
    return ops


def aligned_query_bases(cigar: str) -> int:
    return int(sum(n for n, op in cigar_ops(cigar) if op in {"M", "=", "X", "I"}))


def reference_span(cigar: str) -> int:
    return int(sum(n for n, op in cigar_ops(cigar) if op in {"M", "=", "X", "D", "N"}))


def mismatch_events(record: SamRecord, min_baseq: int = 20) -> Iterator[Tuple[str, int, str, str]]:
    """
    Yield (contig, one-based position, reference base, observed base) for
    substitution mismatches represented by the SAM MD tag.

    Insertions and deletions are intentionally ignored. Reference coordinates
    are reconstructed from CIGAR and MD without loading the full reference
    sequence into memory.
    """
    if record.is_unmapped or not record.is_primary or record.cigar == "*":
        return
    md = record.tags.get("MD")
    if not md:
        return

    seq, qual = oriented_seq_qual(record)
    ref_to_query: Dict[int, int] = {}
    ref0 = record.pos1 - 1
    q0 = 0
    for length, op in cigar_ops(record.cigar):
        if op in {"M", "=", "X"}:
            for offset in range(length):
                ref_to_query[ref0 + offset] = q0 + offset
            ref0 += length
            q0 += length
        elif op in {"I", "S"}:
            q0 += length
        elif op in {"D", "N"}:
            ref0 += length
        elif op in {"H", "P"}:
            continue

    cursor = record.pos1 - 1
    for token in MD_TOKEN_RE.findall(md):
        if token.isdigit():
            cursor += int(token)
        elif token.startswith("^"):
            cursor += len(token) - 1
        else:
            for ref_base in token:
                q_idx = ref_to_query.get(cursor)
                if q_idx is not None and q_idx < len(seq) and q_idx < len(qual):
                    alt = seq[q_idx].upper()
                    qv = ord(qual[q_idx]) - 33
                    rb = ref_base.upper()
                    if qv >= min_baseq and alt in "ACGT" and rb in "ACGT" and alt != rb:
                        yield record.rname, cursor + 1, rb, alt
                cursor += 1


@dataclass
class MarkerIndex:
    positions: Dict[str, List[int]]
    marker_ids: Dict[Tuple[str, int], int]
    ref_alleles: np.ndarray
    alt_alleles: np.ndarray
    marker_table: pd.DataFrame


def load_marker_index(marker_path: str | os.PathLike[str]) -> MarkerIndex:
    panel = pd.read_csv(marker_path, sep="\t")
    required = {"marker_id", "contig", "position", "ref", "alt"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Marker panel missing columns: {sorted(missing)}")
    panel = panel.sort_values(["contig", "position"]).reset_index(drop=True)
    panel["marker_index"] = np.arange(len(panel), dtype=int)
    positions: Dict[str, List[int]] = {}
    marker_ids: Dict[Tuple[str, int], int] = {}
    for row in panel.itertuples(index=False):
        contig = str(row.contig)
        position = int(row.position)
        idx = int(row.marker_index)
        positions.setdefault(contig, []).append(position)
        marker_ids[(contig, position)] = idx
    return MarkerIndex(
        positions=positions,
        marker_ids=marker_ids,
        ref_alleles=panel["ref"].astype(str).str.upper().to_numpy(),
        alt_alleles=panel["alt"].astype(str).str.upper().to_numpy(),
        marker_table=panel,
    )


def count_marker_bases(
    record: SamRecord,
    marker_index: MarkerIndex,
    ref_counts: np.ndarray,
    alt_counts: np.ndarray,
    other_counts: np.ndarray,
    min_mapq: int = 20,
    min_baseq: int = 20,
) -> None:
    if (
        record.is_unmapped
        or not record.is_primary
        or record.mapq < min_mapq
        or record.cigar == "*"
        or record.rname not in marker_index.positions
    ):
        return

    marker_positions = marker_index.positions[record.rname]
    seq, qual = oriented_seq_qual(record)
    ref_pos = record.pos1
    query_pos = 0

    for length, op in cigar_ops(record.cigar):
        if op in {"M", "=", "X"}:
            left = bisect.bisect_left(marker_positions, ref_pos)
            right = bisect.bisect_left(marker_positions, ref_pos + length)
            for position in marker_positions[left:right]:
                q_idx = query_pos + (position - ref_pos)
                if q_idx < 0 or q_idx >= len(seq) or q_idx >= len(qual):
                    continue
                if ord(qual[q_idx]) - 33 < min_baseq:
                    continue
                idx = marker_index.marker_ids[(record.rname, position)]
                base = seq[q_idx].upper()
                if base == marker_index.ref_alleles[idx]:
                    ref_counts[idx] += 1
                elif base == marker_index.alt_alleles[idx]:
                    alt_counts[idx] += 1
                elif base in "ACGT":
                    other_counts[idx] += 1
            ref_pos += length
            query_pos += length
        elif op in {"I", "S"}:
            query_pos += length
        elif op in {"D", "N"}:
            ref_pos += length
        elif op in {"H", "P"}:
            continue


@dataclass
class AlignmentSummary:
    total_pairs: int = 0
    both_mapped_pairs: int = 0
    proper_pairs: int = 0
    discordant_pairs: int = 0
    primary_records: int = 0
    mapped_records: int = 0
    unique_records: int = 0
    mapq_sum: float = 0.0
    nm_sum: float = 0.0
    aligned_bases_sum: int = 0
    insert_sizes: List[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.insert_sizes is None:
            self.insert_sizes = []

    def update(self, record: SamRecord, unique_mapq: int = 20) -> None:
        if not record.is_primary:
            return
        self.primary_records += 1
        if not record.is_unmapped:
            self.mapped_records += 1
            self.mapq_sum += record.mapq
            if record.mapq >= unique_mapq:
                self.unique_records += 1
            nm = safe_float(record.tags.get("NM"), 0.0)
            self.nm_sum += nm
            self.aligned_bases_sum += aligned_query_bases(record.cigar)
        if record.is_read1:
            self.total_pairs += 1
            both_mapped = not record.is_unmapped and not record.mate_unmapped
            if both_mapped:
                self.both_mapped_pairs += 1
                if record.is_proper_pair:
                    self.proper_pairs += 1
                    if record.tlen:
                        self.insert_sizes.append(abs(record.tlen))
                else:
                    self.discordant_pairs += 1

    def as_dict(self) -> Dict[str, float | int]:
        insert_median = float(np.median(self.insert_sizes)) if self.insert_sizes else float("nan")
        insert_mad = (
            float(np.median(np.abs(np.asarray(self.insert_sizes, dtype=float) - insert_median)))
            if self.insert_sizes
            else float("nan")
        )
        return {
            "total_pairs": int(self.total_pairs),
            "primary_records": int(self.primary_records),
            "mapped_records": int(self.mapped_records),
            "mapping_rate": self.mapped_records / self.primary_records if self.primary_records else float("nan"),
            "both_mapped_pair_rate": self.both_mapped_pairs / self.total_pairs if self.total_pairs else float("nan"),
            "proper_pair_rate": self.proper_pairs / self.total_pairs if self.total_pairs else float("nan"),
            "discordant_pair_rate": self.discordant_pairs / self.total_pairs if self.total_pairs else float("nan"),
            "unique_mapping_rate": self.unique_records / self.primary_records if self.primary_records else float("nan"),
            "mean_mapq": self.mapq_sum / self.mapped_records if self.mapped_records else float("nan"),
            "mismatch_rate": self.nm_sum / self.aligned_bases_sum if self.aligned_bases_sum else float("nan"),
            "insert_size_median": insert_median,
            "insert_size_mad": insert_mad,
        }


def build_bowtie_command(
    launcher: str,
    index_prefix: str,
    r1_path: str,
    r2_path: str,
    threads: int,
    seed: int,
    upto_pairs: int = 0,
    skip_pairs: int = 0,
) -> str:
    args = [
        launcher,
        "--very-sensitive",
        "--reorder",
        "--seed",
        str(seed),
        "-p",
        str(max(1, threads)),
        "-x",
        index_prefix,
        "-1",
        r1_path,
        "-2",
        r2_path,
    ]
    # Bowtie 2 writes SAM to stdout by default. Do not pass ``-S -``:
    # the argument to ``-S`` is an output path and would redirect SAM away
    # from the stdout stream consumed by ``stream_bowtie_sam``.
    if skip_pairs > 0:
        args.extend(["-s", str(skip_pairs)])
    if upto_pairs > 0:
        args.extend(["-u", str(upto_pairs)])
    # `call` is required so cmd.exe returns control after a .cmd launcher.
    return "call " + subprocess.list2cmdline(args)


def stream_bowtie_sam(
    launcher: str,
    index_prefix: str,
    r1_path: str,
    r2_path: str,
    threads: int,
    seed: int,
    upto_pairs: int = 0,
    skip_pairs: int = 0,
    stderr_log: Optional[str] = None,
) -> Iterator[SamRecord]:
    command = build_bowtie_command(
        launcher=launcher,
        index_prefix=index_prefix,
        r1_path=r1_path,
        r2_path=r2_path,
        threads=threads,
        seed=seed,
        upto_pairs=upto_pairs,
        skip_pairs=skip_pairs,
    )
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    stderr_lines: List[str] = []
    emitted_records = 0
    rc = -999
    try:
        for line in proc.stdout:
            if not line or line.startswith("@"):
                continue
            record = parse_sam_line(line)
            emitted_records += 1
            yield record
        stderr_lines = proc.stderr.read().splitlines()
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    if stderr_log:
        p = Path(stderr_log)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(stderr_lines) + ("\n" if stderr_lines else ""), encoding="utf-8")

    tail = "\n".join(stderr_lines[-30:])
    if rc != 0:
        raise RuntimeError(
            f"Bowtie 2 failed with return code {rc}\n"
            f"R1: {r1_path}\nR2: {r2_path}\n"
            f"Command: {command}\n"
            f"stderr tail:\n{tail}"
        )
    if emitted_records == 0:
        raise RuntimeError(
            "Bowtie 2 returned success but emitted no SAM records to stdout.\n"
            f"R1: {r1_path}\nR2: {r2_path}\n"
            f"Command: {command}\n"
            "The input may be empty or SAM output may have been redirected away "
            "from stdout.\n"
            f"stderr tail:\n{tail}"
        )


def robust_location_scale(values: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 1.0
    center = float(np.median(x))
    mad = float(np.median(np.abs(x - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        scale = float(np.std(x))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return center, scale


def robust_z(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    center, scale = robust_location_scale(reference_values)
    return (np.asarray(values, dtype=float) - center) / scale


def binary_entropy(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return -(x * np.log(x) + (1.0 - x) * np.log(1.0 - x)) / math.log(2.0)


def evaluate_binary(y: Sequence[int], score: Sequence[float]) -> Dict[str, float]:
    y_arr = np.asarray(y, dtype=int)
    s_arr = np.asarray(score, dtype=float)
    mask = np.isfinite(s_arr)
    y_arr = y_arr[mask]
    s_arr = s_arr[mask]
    if len(np.unique(y_arr)) < 2:
        return {
            "n": int(len(y_arr)),
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "best_f1": float("nan"),
            "best_threshold": float("nan"),
            "prevalence": float(np.mean(y_arr)) if len(y_arr) else float("nan"),
        }

    roc_auc = float(roc_auc_score(y_arr, s_arr))
    ap = float(average_precision_score(y_arr, s_arr))
    thresholds = np.unique(s_arr)
    best_f1 = -1.0
    best_threshold = float("nan")
    for threshold in thresholds:
        pred = (s_arr >= threshold).astype(int)
        f1 = float(f1_score(y_arr, pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return {
        "n": int(len(y_arr)),
        "roc_auc": roc_auc,
        "average_precision": ap,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "prevalence": float(np.mean(y_arr)),
    }


def pairwise_overlap_distance(
    target: np.ndarray,
    candidate: np.ndarray,
    min_overlap: int = 20,
) -> Tuple[float, int]:
    mask = np.isfinite(target) & np.isfinite(candidate)
    n = int(mask.sum())
    if n < min_overlap:
        return float("nan"), n
    return float(np.sqrt(np.mean((target[mask] - candidate[mask]) ** 2))), n


def best_single_and_pair_reconstruction(
    target: np.ndarray,
    control_matrix: np.ndarray,
    control_names: Sequence[str],
    exclude_name: Optional[str] = None,
    min_overlap: int = 20,
) -> Dict[str, object]:
    eligible = [i for i, name in enumerate(control_names) if name != exclude_name]
    if not eligible:
        return {
            "best_single_error": float("nan"),
            "best_single_name": "",
            "best_pair_error": float("nan"),
            "best_pair_a": "",
            "best_pair_b": "",
            "best_pair_weight_a": float("nan"),
            "pair_gain": float("nan"),
            "pair_gain_fraction": float("nan"),
            "single_overlap": 0,
            "pair_overlap": 0,
        }

    best_single_error = float("inf")
    best_single_name = ""
    best_single_overlap = 0
    for idx in eligible:
        error, overlap = pairwise_overlap_distance(target, control_matrix[idx], min_overlap=min_overlap)
        if np.isfinite(error) and error < best_single_error:
            best_single_error = error
            best_single_name = str(control_names[idx])
            best_single_overlap = overlap

    best_pair_error = float("inf")
    best_pair_a = ""
    best_pair_b = ""
    best_weight = float("nan")
    best_pair_overlap = 0

    for ii, a_idx in enumerate(eligible):
        a = control_matrix[a_idx]
        for b_idx in eligible[ii + 1 :]:
            b = control_matrix[b_idx]
            mask = np.isfinite(target) & np.isfinite(a) & np.isfinite(b)
            overlap = int(mask.sum())
            if overlap < min_overlap:
                continue
            y = target[mask]
            av = a[mask]
            bv = b[mask]
            direction = av - bv
            denom = float(np.dot(direction, direction))
            if denom <= 1e-15:
                weight = 0.5
            else:
                weight = float(np.clip(np.dot(y - bv, direction) / denom, 0.0, 1.0))
            pred = weight * av + (1.0 - weight) * bv
            error = float(np.sqrt(np.mean((y - pred) ** 2)))
            if error < best_pair_error:
                best_pair_error = error
                best_pair_a = str(control_names[a_idx])
                best_pair_b = str(control_names[b_idx])
                best_weight = weight
                best_pair_overlap = overlap

    if not np.isfinite(best_single_error):
        best_single_error = float("nan")
    if not np.isfinite(best_pair_error):
        best_pair_error = float("nan")

    if np.isfinite(best_single_error) and np.isfinite(best_pair_error):
        gain = best_single_error - best_pair_error
        gain_fraction = gain / max(best_single_error, 1e-12)
    else:
        gain = float("nan")
        gain_fraction = float("nan")

    return {
        "best_single_error": best_single_error,
        "best_single_name": best_single_name,
        "best_pair_error": best_pair_error,
        "best_pair_a": best_pair_a,
        "best_pair_b": best_pair_b,
        "best_pair_weight_a": best_weight,
        "pair_gain": gain,
        "pair_gain_fraction": gain_fraction,
        "single_overlap": int(best_single_overlap),
        "pair_overlap": int(best_pair_overlap),
    }


def pca_reconstruction_error(
    matrix: np.ndarray,
    control_mask: np.ndarray,
    max_components: int = 5,
) -> np.ndarray:
    X = np.asarray(matrix, dtype=float)
    controls = X[control_mask]
    if controls.shape[0] < 3 or controls.shape[1] < 2:
        return np.full(X.shape[0], np.nan, dtype=float)

    medians = np.nanmedian(controls, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    X_imp = np.where(np.isfinite(X), X, medians)
    controls_imp = X_imp[control_mask]

    n_components = int(min(max_components, controls_imp.shape[0] - 1, controls_imp.shape[1]))
    if n_components < 1:
        return np.full(X.shape[0], np.nan, dtype=float)

    model = PCA(n_components=n_components, random_state=0)
    model.fit(controls_imp)
    transformed = model.transform(X_imp)
    reconstructed = model.inverse_transform(transformed)
    return np.sqrt(np.mean((X_imp - reconstructed) ** 2, axis=1))


def rank_percentile_high(values: Sequence[float]) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=float))
    return s.rank(method="average", pct=True).to_numpy(dtype=float)


def spearman_safe(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(mask.sum()) < 3 or np.nanstd(xa[mask]) == 0 or np.nanstd(ya[mask]) == 0:
        return float("nan"), float("nan")
    rho, p = spearmanr(xa[mask], ya[mask])
    return float(rho), float(p)
