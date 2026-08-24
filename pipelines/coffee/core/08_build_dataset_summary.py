# -*- coding: utf-8 -*-
"""
08_build_dataset_summary.py
Build a dataset summary table from FASTQ files.

Outputs:
- Dataset_Summary_Table.xlsx
- Dataset_Summary_Table.csv
"""
from __future__ import annotations

import argparse
import os
import glob
import gzip
from pathlib import Path
import pandas as pd

FASTQ_EXTS = (".fastq", ".fastq.gz", ".fq", ".fq.gz")


def iter_fastq_records(path: str):
    opener = gzip.open if path.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
        i = 0
        rec = []
        for line in fh:
            rec.append(line.rstrip("\n"))
            i += 1
            if i % 4 == 0:
                # header, seq, +, qual
                yield rec[0], rec[1], rec[3]
                rec = []


def find_pair_files(fastq_dir: str, sample: str):
    pat = os.path.join(fastq_dir, f"{sample}_*.fastq*")
    files = glob.glob(pat)
    files = [f for f in files if f.lower().endswith(FASTQ_EXTS)]
    return sorted(files)


def summarize_fastq(files, max_reads=None):
    total_reads = 0
    total_bases = 0
    total_qual = 0
    total_qual_bases = 0
    read_len_min = None
    read_len_max = None

    for fp in files:
        for _, seq, qual in iter_fastq_records(fp):
            L = len(seq)
            total_reads += 1
            total_bases += L
            # mean phred over bases
            # Phred+33
            qsum = sum((ord(c) - 33) for c in qual[:L])
            total_qual += qsum
            total_qual_bases += L
            read_len_min = L if read_len_min is None else min(read_len_min, L)
            read_len_max = L if read_len_max is None else max(read_len_max, L)

            if max_reads is not None and total_reads >= max_reads:
                break
        if max_reads is not None and total_reads >= max_reads:
            break

    mean_len = (total_bases / total_reads) if total_reads else 0.0
    mean_phred = (total_qual / total_qual_bases) if total_qual_bases else 0.0
    return {
        "total_reads": int(total_reads),
        "total_bases": int(total_bases),
        "mean_read_length": float(mean_len),
        "min_read_length": int(read_len_min or 0),
        "max_read_length": int(read_len_max or 0),
        "mean_phred": float(mean_phred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq_dir", required=True, help="Folder containing paired FASTQ files")
    ap.add_argument("--outdir", required=True, help="Output folder")
    ap.add_argument("--max_reads", type=int, default=None,
                    help="Optional cap for speed (e.g., 200000). Omit to scan full files.")
    ap.add_argument("--pattern", default="SRR*_1.fastq*", help="Pattern used to detect R1 files")
    args = ap.parse_args()

    fastq_dir = Path(args.fastq_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    r1_files = sorted(glob.glob(str(fastq_dir / args.pattern)))
    samples = sorted(set(Path(f).name.split("_")[0] for f in r1_files))

    rows = []
    for s in samples:
        files = find_pair_files(str(fastq_dir), s)
        if not files:
            continue
        d = summarize_fastq(files, max_reads=args.max_reads)
        d["sample"] = s
        d["n_fastq_files"] = len(files)
        rows.append(d)

    df = pd.DataFrame(rows).sort_values("sample")

    # Add cohort-level summary rows
    summary = {
        "sample": "SUMMARY",
        "n_fastq_files": "",
        "total_reads": int(df["total_reads"].sum()),
        "total_bases": int(df["total_bases"].sum()),
        "mean_read_length": float((df["total_bases"].sum() / df["total_reads"].sum()) if df["total_reads"].sum() else 0.0),
        "min_read_length": int(df["min_read_length"].min()),
        "max_read_length": int(df["max_read_length"].max()),
        "mean_phred": float((df["mean_phred"] * df["total_reads"]).sum() / df["total_reads"].sum()) if df["total_reads"].sum() else 0.0,
    }
    df_out = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    xlsx = outdir / "Dataset_Summary_Table.xlsx"
    csv = outdir / "Dataset_Summary_Table.csv"
    df_out.to_excel(xlsx, index=False)
    df_out.to_csv(csv, index=False)

    print("[DONE]", xlsx)
    print("[DONE]", csv)
    print("[INFO] Rows:", len(df), "samples")


if __name__ == "__main__":
    main()