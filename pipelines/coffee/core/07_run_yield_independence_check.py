# -*- coding: utf-8 -*-
"""
Check whether the real-sample mixture score is associated with sequencing yield.

What this script does
---------------------
1. Reads sample-level mixture scores from geometry analysis.
2. Calculates total FASTQ yield per sample (bases or reads) from FASTQ files.
3. Computes Spearman correlation and a simple linear fit.
4. Writes an Excel workbook with per-sample values and summary statistics.

Primary workbook
----------------
<outdir>/Supplementary_Data_S7_YieldIndependence.xlsx
"""
from __future__ import annotations

import argparse
import glob
import gzip
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FASTQ_EXTS = (".fastq", ".fastq.gz", ".fq", ".fq.gz")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_fastq_files(fastq_dir: str, sample_id: str) -> List[str]:
    fastq_dir = os.path.abspath(fastq_dir)
    patterns = [
        os.path.join(fastq_dir, "**", f"{sample_id}_*.fastq*"),
        os.path.join(fastq_dir, "**", f"{sample_id}*.fastq*"),
        os.path.join(fastq_dir, "**", f"{sample_id}_*.fq*"),
        os.path.join(fastq_dir, "**", f"{sample_id}*.fq*"),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))

    out = []
    seen = set()
    for fp in files:
        if fp.lower().endswith(FASTQ_EXTS) and os.path.isfile(fp) and fp not in seen:
            out.append(fp)
            seen.add(fp)
    return out


def fastq_bases_and_reads(path: str) -> Tuple[int, int]:
    opener = gzip.open if path.lower().endswith(".gz") else open
    total_bases = 0
    total_reads = 0
    with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
        line_idx = 0
        for line in fh:
            line_idx += 1
            if line_idx % 4 == 2:
                seq = line.strip()
                total_bases += len(seq)
                total_reads += 1
    return int(total_bases), int(total_reads)


def total_yield_for_sample(fastq_dir: str, sample_id: str) -> Tuple[int, int, int]:
    files = find_fastq_files(fastq_dir, sample_id)
    total_bases = 0
    total_reads = 0
    for fp in files:
        bases, reads = fastq_bases_and_reads(fp)
        total_bases += bases
        total_reads += reads
    return int(total_bases), int(total_reads), int(len(files))


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def load_scores(geometry_dir: Path) -> pd.DataFrame:
    xlsx = geometry_dir / "Supplementary_Data_S2_Geometry.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(f"Missing workbook: {xlsx}")
    for sheet_name in ["ranked_candidates", "node_metrics", "mixture_ranking"]:
        try:
            df = pd.read_excel(xlsx, sheet_name=sheet_name)
            if "sample" in df.columns and "mixture_score" in df.columns:
                return df[["sample", "mixture_score"]].copy()
        except Exception:
            continue
    raise ValueError("Could not find sample/mixture_score columns in the geometry workbook.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry_dir", required=True, help="Output directory from 02_run_geometry_analysis.py")
    ap.add_argument("--fastq_dir", required=True, help="Directory containing FASTQ files")
    ap.add_argument("--outdir", required=True, help="Output directory for workbook")
    ap.add_argument("--metric", default="bases", choices=["bases", "reads"], help="Yield metric used for statistics")
    args = ap.parse_args()

    geometry_dir = Path(args.geometry_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = load_scores(geometry_dir)
    df["sample"] = df["sample"].astype(str)

    bases_list = []
    reads_list = []
    nfiles_list = []
    for sample_id in df["sample"].tolist():
        bases, reads, nfiles = total_yield_for_sample(args.fastq_dir, sample_id)
        bases_list.append(bases)
        reads_list.append(reads)
        nfiles_list.append(nfiles)

    df["total_bases"] = bases_list
    df["total_reads"] = reads_list
    df["n_fastq_files"] = nfiles_list

    df = df[(df["n_fastq_files"] > 0) & ((df["total_bases"] > 0) | (df["total_reads"] > 0))].copy()
    if len(df) < 3:
        raise SystemExit("[ERR] Too few samples with FASTQ yield found. Check --fastq_dir and file naming.")

    x_raw = df["total_bases"].to_numpy(dtype=float) if args.metric == "bases" else df["total_reads"].to_numpy(dtype=float)
    y = df["mixture_score"].to_numpy(dtype=float)
    rho, p = spearmanr(x_raw, y)
    intercept, slope = fit_line(x_raw, y)

    stats_df = pd.DataFrame(
        [
            {
                "metric": args.metric,
                "n_samples": len(df),
                "spearman_rho": float(rho),
                "spearman_p": float(p),
                "linear_intercept": intercept,
                "linear_slope": slope,
                "min_fastq_files_per_sample": int(df["n_fastq_files"].min()),
                "median_fastq_files_per_sample": float(df["n_fastq_files"].median()),
                "max_fastq_files_per_sample": int(df["n_fastq_files"].max()),
            }
        ]
    )

    workbook = outdir / "Supplementary_Data_S7_YieldIndependence.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as xw:
        stats_df.to_excel(xw, sheet_name="stats_summary", index=False)
        df.to_excel(xw, sheet_name="sample_yield_scores", index=False)

    print("\n[DONE] Yield independence check complete.")
    print(f"  -> Workbook: {workbook}")
    print(f"  -> Spearman rho = {rho:.4f}, p = {p:.4g}, n = {len(df)}")


if __name__ == "__main__":
    main()
