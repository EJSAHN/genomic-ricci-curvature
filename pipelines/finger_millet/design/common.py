# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np


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


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def read_tsv(path: str | os.PathLike[str]) -> List[Dict[str, str]]:
    with Path(path).open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write_tsv(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    p = Path(path)
    rows = list(rows)
    ensure_dir(p.parent)
    if fieldnames is None:
        fields: List[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: '' if row.get(key) is None else row.get(key) for key in fieldnames})
    os.replace(tmp, p)


def sha256_file(path: str | os.PathLike[str], block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = '|'.join([str(base_seed), *[str(part) for part in parts]])
    return int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest()[:8], 'big') % (2**32 - 1)


def stable_key(base_seed: int, *parts: object) -> str:
    payload = '|'.join([str(base_seed), *[str(part) for part in parts]])
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def parse_float(value: Any, default: float = float('nan')) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def normalize_read_id(header: str) -> str:
    token = header[1:] if header.startswith('@') else header
    token = token.split()[0]
    token = re.sub(r'([/._-])?[12]$', '', token)
    return token


def open_fastq_text(path: str | os.PathLike[str]):
    p = str(path)
    if p.lower().endswith('.gz'):
        return gzip.open(p, 'rt', encoding='ascii', errors='strict', newline='')
    return open(p, 'rt', encoding='ascii', errors='strict', newline='')


def read_fastq_record(handle, path: str, pair_index: int) -> Optional[FastqRecord]:
    header = handle.readline()
    if not header:
        return None
    sequence = handle.readline()
    plus = handle.readline()
    quality = handle.readline()
    if not (sequence and plus and quality):
        raise ValueError(f'Truncated FASTQ record at pair {pair_index} in {path}')
    header = header.rstrip('\r\n')
    sequence = sequence.rstrip('\r\n')
    plus = plus.rstrip('\r\n')
    quality = quality.rstrip('\r\n')
    if not header.startswith('@'):
        raise ValueError(f'Invalid FASTQ header at pair {pair_index} in {path}: {header[:100]}')
    if not plus.startswith('+'):
        raise ValueError(f'Invalid FASTQ plus line at pair {pair_index} in {path}: {plus[:100]}')
    if len(sequence) != len(quality):
        raise ValueError(f'Sequence/quality mismatch at pair {pair_index} in {path}: {len(sequence)} != {len(quality)}')
    return FastqRecord(header, sequence, plus, quality)


def iter_paired_fastq(r1_path: str | os.PathLike[str], r2_path: str | os.PathLike[str]) -> Iterator[Tuple[int, FastqRecord, FastqRecord]]:
    r1 = str(r1_path)
    r2 = str(r2_path)
    with open_fastq_text(r1) as h1, open_fastq_text(r2) as h2:
        index = 0
        while True:
            a = read_fastq_record(h1, r1, index)
            b = read_fastq_record(h2, r2, index)
            if a is None and b is None:
                return
            if (a is None) != (b is None):
                raise ValueError(f'R1/R2 record counts differ: {r1} vs {r2}')
            assert a is not None and b is not None
            if normalize_read_id(a.header) != normalize_read_id(b.header):
                raise ValueError(f'Paired identifiers differ at pair {index}: {a.header[:100]} vs {b.header[:100]}')
            yield index, a, b
            index += 1


def iter_paired_range(r1_path: str, r2_path: str, offset: int, count: int) -> Iterator[Tuple[FastqRecord, FastqRecord]]:
    stop = int(offset) + int(count)
    emitted = 0
    for index, r1, r2 in iter_paired_fastq(r1_path, r2_path):
        if index < offset:
            continue
        if index >= stop:
            break
        emitted += 1
        yield r1, r2
    if emitted != count:
        raise ValueError(f'Requested {count} pairs at offset {offset}, but only {emitted} were available: {r1_path}')


def kmer_probability_from_sequences(sequences: Iterable[str], k: int, sketch_size: int) -> np.ndarray:
    counts = np.zeros(int(sketch_size), dtype=np.uint64)
    total = 0
    for sequence in sequences:
        seq = sequence.strip().upper()
        if len(seq) < k:
            continue
        data = seq.encode('ascii', errors='ignore')
        for index in range(0, len(data) - k + 1):
            kmer = data[index:index+k]
            if b'N' in kmer:
                continue
            hashed = zlib.crc32(kmer) & 0xFFFFFFFF
            counts[hashed % int(sketch_size)] += 1
            total += 1
    if total <= 0:
        return np.full(int(sketch_size), 1.0 / float(sketch_size), dtype=np.float64)
    return counts.astype(np.float64) / float(total)


def paired_kmer_sketch(r1_path: str, r2_path: str, offset: int, count: int, k: int, sketch_size: int) -> np.ndarray:
    seq1: List[str] = []
    seq2: List[str] = []
    for r1, r2 in iter_paired_range(r1_path, r2_path, offset, count):
        seq1.append(r1.sequence)
        seq2.append(r2.sequence)
    p1 = kmer_probability_from_sequences(seq1, k=k, sketch_size=sketch_size)
    p2 = kmer_probability_from_sequences(seq2, k=k, sketch_size=sketch_size)
    combined = 0.5 * (p1 + p2)
    return combined / combined.sum()


def js_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    divergence = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(np.sqrt(max(0.0, divergence)))


def pairwise_js(signatures: np.ndarray, names: Sequence[str]) -> np.ndarray:
    n = len(names)
    result = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            value = js_distance(signatures[i], signatures[j])
            result[i, j] = value
            result[j, i] = value
    return result


def largest_remainder_counts(weights: Sequence[float], total: int) -> List[int]:
    values = np.asarray(weights, dtype=float)
    if np.any(values < 0) or not np.isfinite(values).all() or values.sum() <= 0:
        raise ValueError(f'Invalid mixture weights: {weights}')
    values = values / values.sum()
    raw = values * int(total)
    counts = np.floor(raw).astype(int)
    remainder = int(total) - int(counts.sum())
    order = np.argsort(-(raw - counts), kind='stable')
    for index in order[:remainder]:
        counts[index] += 1
    return counts.tolist()


def normalized_entropy(weights: Sequence[float]) -> float:
    values = np.asarray(weights, dtype=float)
    values = values[values > 0]
    if len(values) <= 1:
        return 0.0
    values = values / values.sum()
    return float(-np.sum(values * np.log(values)) / np.log(len(values)))


def coprime_multiplier(modulus: int, seed: int) -> int:
    if modulus <= 2:
        return 1
    candidate = 1 + (seed % (modulus - 1))
    while math.gcd(candidate, modulus) != 1:
        candidate += 1
        if candidate >= modulus:
            candidate = 1
    return candidate
