# -*- coding: utf-8 -*-
"""
utils_fastq_qc.py

Heuristic QC tool to check if FASTQ files look "pre-trimmed".
It checks for:
- Adapter contamination (Illumina universal)
- Strong fixed prefixes (potential barcodes / restriction remnants)
- Read length distribution (variable vs fixed)
- 3' poly-G tails (Novaseq/NextSeq artifacts)

Dependencies: Standard Python libraries only (gzip, os, re, math, collections).

Usage Example:
  python utils_fastq_qc.py --fastq_dir "./data" --n_reads 50000 --kmer 17 --prefix_len 12

or single file:
  python utils_fastq_qc.py --fastq "./data/sample_R1.fastq.gz" --n_reads 50000
"""

import argparse
import gzip
import os
import re
import math
from collections import Counter, defaultdict

ILLUMINA_ADAPTERS = [
    # Common TruSeq / Nextera adapter motifs (partial)
    "AGATCGGAAGAGC",  # universal
    "CTGTCTCTTATACACATCT",  # Nextera transposase (partial)
    "GATCGGAAGAGCACACGTCTGAACTCCAGTCA",  # longer universal (partial)
    "AATGATACGGCGACCACCGA",  # P5 (partial)
    "CAAGCAGAAGACGGCATACGAGAT",  # index read primer (partial)
]

def open_maybe_gz(path: str):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")

def iter_fastq_sequences(path: str, max_reads: int):
    """
    Yields (seq) strings from FASTQ. Stops at max_reads.
    FASTQ format: 4 lines per record.
    """
    n = 0
    with open_maybe_gz(path) as fh:
        while True:
            h = fh.readline()
            if not h:
                break
            s = fh.readline().strip()
            _ = fh.readline()
            _q = fh.readline()
            if not s:
                continue
            yield s
            n += 1
            if n >= max_reads:
                break

def shannon_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent

def find_fastqs_in_dir(d: str):
    exts = (".fastq", ".fq", ".fastq.gz", ".fq.gz")
    out = []
    for root, _, files in os.walk(d):
        for fn in files:
            if fn.lower().endswith(exts):
                out.append(os.path.join(root, fn))
    out.sort()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq_dir", default=None, help="Directory to scan for FASTQ files")
    ap.add_argument("--fastq", nargs="*", default=None, help="One or more FASTQ(.gz) files")
    ap.add_argument("--n_reads", type=int, default=50000, help="Reads to sample per file (default: 50000)")
    ap.add_argument("--prefix_len", type=int, default=12, help="Prefix length to tabulate (default: 12)")
    ap.add_argument("--tail_len", type=int, default=12, help="Tail length to tabulate (default: 12)")
    ap.add_argument("--kmer", type=int, default=17, help="k-mer length for simple overrep test (default: 17)")
    ap.add_argument("--top", type=int, default=10, help="Top N items to show (default: 10)")
    args = ap.parse_args()

    files = []
    if args.fastq:
        files.extend(args.fastq)
    if args.fastq_dir:
        files.extend(find_fastqs_in_dir(args.fastq_dir))

    files = [f for f in files if f and os.path.exists(f)]
    if not files:
        raise SystemExit("[ERROR] No FASTQ files found. Use --fastq or --fastq_dir")

    print(f"[INFO] Files: {len(files)}")
    print(f"[INFO] Sample reads per file: {args.n_reads}\n")

    for fp in files:
        print("=" * 80)
        print(f"[FILE] {fp}")

        lens = []
        prefix_counts = Counter()
        tail_counts = Counter()
        first_base_counts = Counter()
        last_base_counts = Counter()
        adapter_hits = Counter()
        polyG_tail = 0
        total = 0

        # Optional: detect a few most common k-mers near ends
        end_kmers_5p = Counter()
        end_kmers_3p = Counter()

        for seq in iter_fastq_sequences(fp, args.n_reads):
            total += 1
            L = len(seq)
            lens.append(L)

            pre = seq[:args.prefix_len]
            suf = seq[-args.tail_len:] if L >= args.tail_len else seq
            prefix_counts[pre] += 1
            tail_counts[suf] += 1

            first_base_counts[seq[0]] += 1
            last_base_counts[seq[-1]] += 1

            # Poly-G tail heuristic
            if suf.upper().count("G") >= max(8, int(0.8 * len(suf))):
                polyG_tail += 1

            # Adapter motif presence (substring search; fast heuristic)
            up = seq.upper()
            for a in ILLUMINA_ADAPTERS:
                if a in up:
                    adapter_hits[a] += 1

            # End k-mers: first/last k bases (if long enough)
            k = args.kmer
            if L >= k:
                end_kmers_5p[up[:k]] += 1
                end_kmers_3p[up[-k:]] += 1

        if total == 0:
            print("[WARN] No reads sampled.")
            continue

        lens_sorted = sorted(lens)
        def pct(v): return 100.0 * v / total

        # Length stats
        minL = lens_sorted[0]
        maxL = lens_sorted[-1]
        medL = lens_sorted[len(lens_sorted)//2]
        meanL = sum(lens_sorted) / len(lens_sorted)

        # Entropy of first bases/prefixes
        H1 = shannon_entropy(first_base_counts)
        Hpre = shannon_entropy(prefix_counts)

        print(f"[STATS] reads_sampled={total:,}")
        print(f"[LEN] min={minL}  median={medL}  mean={meanL:.1f}  max={maxL}")

        # Show if lengths are highly variable (common in untrimmed/merged/mixed)
        unique_lens = len(set(lens_sorted))
        print(f"[LEN] unique_lengths={unique_lens}")

        # Prefix dominance
        top_pre = prefix_counts.most_common(args.top)
        top_suf = tail_counts.most_common(args.top)

        print(f"[PREFIX] first_base_entropy(H)={H1:.3f} bits  prefix_entropy(H)={Hpre:.3f} bits")
        print("[PREFIX] top prefixes:")
        for p, c in top_pre:
            print(f"  {p}\t{c}\t({pct(c):.2f}%)")

        print("[TAIL] top tails:")
        for t, c in top_suf:
            print(f"  {t}\t{c}\t({pct(c):.2f}%)")

        # Adapter hits
        if adapter_hits:
            print("[ADAPTER] detected motifs:")
            for a, c in adapter_hits.most_common():
                print(f"  {a}\t{c}\t({pct(c):.2f}%)")
        else:
            print("[ADAPTER] no common Illumina adapter motifs detected (heuristic)")

        # Poly-G tail
        print(f"[TAIL] poly-G-like tail (last {args.tail_len} bp mostly G): {polyG_tail} ({pct(polyG_tail):.2f}%)")

        # End k-mer dominance (can reveal untrimmed fixed sequences)
        print(f"[END-KMER] k={args.kmer} top 5' kmers:")
        for kmer, c in end_kmers_5p.most_common(args.top):
            print(f"  {kmer}\t{c}\t({pct(c):.2f}%)")

        print(f"[END-KMER] k={args.kmer} top 3' kmers:")
        for kmer, c in end_kmers_3p.most_common(args.top):
            print(f"  {kmer}\t{c}\t({pct(c):.2f}%)")

        # Quick interpretation hints
        print("\n[INTERPRETATION HINTS]")
        print("- If one prefix (length ~8-12) dominates (e.g., >20-30%), likely barcode/RE remnant/primer not removed.")
        print("- If adapter motifs show up (even 0.5-2%), trimming is incomplete or reads are too short to remove fully.")
        print("- If many unique read lengths or wide length range, you may have mixed/merged/untrimmed reads.")
        print("- Poly-G tail high can indicate platform/artifact; consider quality trimming.")
        print("=" * 80)
        print()

if __name__ == "__main__":
    main()
