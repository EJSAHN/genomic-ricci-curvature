# -*- coding: utf-8 -*-
"""
utils_kmer_matrix.py

Builds a hashed k-mer count matrix (feature matrix) from FASTQ files.
It maps k-mers to a fixed-dimension vector using MurmurHash3-style hashing,
allowing for alignment-free comparison of samples.

Features:
- Efficient bitwise rolling k-mer encoding.
- Canonical k-mers (treats forward/reverse-complement as same).
- Outputs a compressed .npz file containing the matrix X and sample IDs.

Usage Example:
  python utils_kmer_matrix.py \
    --fastq_dir "./data" \
    --out "./results/kmer_matrix.npz" \
    --k 17 \
    --dim 262144
"""
import os, re, glob, gzip, argparse
import numpy as np

# 256-entry table: A,C,G,T -> 0,1,2,3 ; others -> -1
BASEMAP = np.full(256, -1, dtype=np.int16)
BASEMAP[ord('A')] = 0
BASEMAP[ord('C')] = 1
BASEMAP[ord('G')] = 2
BASEMAP[ord('T')] = 3
BASEMAP[ord('a')] = 0
BASEMAP[ord('c')] = 1
BASEMAP[ord('g')] = 2
BASEMAP[ord('t')] = 3

def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0

def mix64(x: int) -> int:
    # MurmurHash3 fmix64 style finalizer (good bit diffusion)
    x &= 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 33)
    x = (x * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 33)
    x = (x * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 33)
    return x

def sample_id_from_filename(path: str) -> str:
    name = os.path.basename(path)
    # remove .gz, .fastq/.fq
    name = re.sub(r"\.gz$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\.(fastq|fq)$", "", name, flags=re.IGNORECASE)
    # remove read mate suffix: _1/_2, .1/.2, -1/-2
    name = re.sub(r"([._-])(1|2)$", "", name)
    return name

def iter_fastq_seqs(filepath: str, n_reads: int):
    opener = gzip.open if filepath.lower().endswith(".gz") else open
    with opener(filepath, "rt", encoding="utf-8", errors="ignore") as f:
        i = 0
        while True:
            h = f.readline()
            if not h:
                break
            seq = f.readline().strip()
            f.readline()
            f.readline()
            if seq:
                yield seq
                i += 1
                if n_reads > 0 and i >= n_reads:
                    break

def add_kmers_hashed(vec: np.ndarray, seq: str, k: int, dim: int, canonical: bool, trim5: int, trim3: int):
    if trim5 or trim3:
        if trim3 > 0:
            seq = seq[trim5:len(seq)-trim3]
        else:
            seq = seq[trim5:]
    if len(seq) < k:
        return

    mask = (1 << (2*k)) - 1
    fw = 0
    rv = 0
    valid = 0
    # for reverse-complement rolling: comp = 3 - base
    shift_rc = 2*(k-1)

    dim_mask = dim - 1 if is_power_of_two(dim) else None

    for ch in seq:
        b = BASEMAP[ord(ch) & 0xFF]
        if b < 0:
            valid = 0
            fw = 0
            rv = 0
            continue

        fw = ((fw << 2) | int(b)) & mask
        rv = (rv >> 2) | ((3 - int(b)) << shift_rc)
        valid += 1

        if valid >= k:
            val = fw
            if canonical:
                val = fw if fw < rv else rv
            h = mix64(val)
            idx = (h & dim_mask) if dim_mask is not None else (h % dim)
            vec[idx] += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq_dir", required=True)
    ap.add_argument("--pattern", default="*.fastq.gz", help="e.g., *.fastq.gz or *.fq.gz (can repeat by running twice)")
    ap.add_argument("--k", type=int, default=17)
    ap.add_argument("--dim", type=int, default=262144, help="hashed feature dimension (power of 2 recommended)")
    ap.add_argument("--n_reads", type=int, default=20000, help="reads per FASTQ file (0 = all)")
    ap.add_argument("--canonical", action="store_true", help="use canonical k-mers (min of k-mer and reverse complement)")
    ap.add_argument("--trim5", type=int, default=0, help="trim this many bases from 5' before k-mer counting")
    ap.add_argument("--trim3", type=int, default=0, help="trim this many bases from 3' before k-mer counting")
    ap.add_argument("--out", required=True, help="output .npz (will contain X, ids)")
    args = ap.parse_args()

    if not os.path.isdir(args.fastq_dir):
        raise SystemExit(f"[ERROR] fastq_dir not found: {args.fastq_dir}")

    files = sorted(glob.glob(os.path.join(args.fastq_dir, args.pattern)))
    if not files:
        raise SystemExit(f"[ERROR] no files matched: {os.path.join(args.fastq_dir, args.pattern)}")

    # group mates into sample_id
    groups = {}
    for fp in files:
        sid = sample_id_from_filename(fp)
        groups.setdefault(sid, []).append(fp)

    ids = sorted(groups.keys())
    X = np.zeros((len(ids), args.dim), dtype=np.uint32)

    print(f"[INFO] files={len(files)} samples={len(ids)} k={args.k} dim={args.dim} n_reads/file={args.n_reads}")
    print(f"[INFO] canonical={args.canonical} trim5={args.trim5} trim3={args.trim3}")

    for i, sid in enumerate(ids):
        vec = np.zeros(args.dim, dtype=np.uint32)
        fps = sorted(groups[sid])
        for fp in fps:
            n = 0
            for seq in iter_fastq_seqs(fp, args.n_reads):
                add_kmers_hashed(vec, seq, args.k, args.dim, args.canonical, args.trim5, args.trim3)
                n += 1
            print(f"[OK] {sid} <- {os.path.basename(fp)} reads_used={n}")
        X[i, :] = vec

    np.savez_compressed(args.out, X=X, ids=np.array(ids, dtype=object))
    print(f"[DONE] saved: {args.out} (X, ids)")

if __name__ == "__main__":
    main()
