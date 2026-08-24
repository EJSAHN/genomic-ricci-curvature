# -*- coding: utf-8 -*-
"""
09_profile_runtime_memory.py
Profile runtime and peak memory for key pipeline steps.

Outputs:
- Runtime_Memory_Profile.xlsx
- Runtime_Memory_Profile.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


def run_and_measure(cmd, cwd=None):
    t0 = time.time()
    peak_rss = None

    if HAS_PSUTIL:
        proc = subprocess.Popen(cmd, cwd=cwd)
        p = psutil.Process(proc.pid)
        peak = 0
        while proc.poll() is None:
            try:
                rss = p.memory_info().rss
                peak = max(peak, rss)
            except Exception:
                pass
            time.sleep(0.05)
        rc = proc.returncode
        peak_rss = peak
    else:
        rc = subprocess.call(cmd, cwd=cwd)

    dt = time.time() - t0
    return rc, dt, peak_rss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    scripts_dir = Path(__file__).parent.resolve()

    rows = []

    # Step 01 (distance matrix)
    cmd1 = [py, str(scripts_dir / "01_run_preprocessing.py"),
            "--fastq_dir", str(Path(args.fastq_dir).resolve()),
            "--outdir", str(Path(args.results_dir).resolve())]
    rc, dt, rss = run_and_measure(cmd1)
    rows.append({"step": "01_run_preprocessing", "return_code": rc, "wall_seconds": dt,
                 "peak_rss_gb": (rss / (1024**3)) if rss else ""})

    # Step 02 (geometry)
    cmd2 = [py, str(scripts_dir / "02_run_geometry_analysis.py"),
            "--results_dir", str(Path(args.results_dir).resolve()),
            "--outdir", str(Path(args.results_dir).resolve() / "gauss_euler")]
    rc, dt, rss = run_and_measure(cmd2)
    rows.append({"step": "02_run_geometry_analysis", "return_code": rc, "wall_seconds": dt,
                 "peak_rss_gb": (rss / (1024**3)) if rss else ""})

    # Step 03 (synthetic validation)
    cmd3 = [py, str(scripts_dir / "03_run_synthetic_validation.py"),
            "--fastq_dir", str(Path(args.fastq_dir).resolve()),
            "--outdir", str(Path(args.results_dir).resolve() / "synthetic"),
            "--seed", str(args.seed)]
    rc, dt, rss = run_and_measure(cmd3)
    rows.append({"step": "03_run_synthetic_validation", "return_code": rc, "wall_seconds": dt,
                 "peak_rss_gb": (rss / (1024**3)) if rss else ""})

    df = pd.DataFrame(rows)

    xlsx = outdir / "Runtime_Memory_Profile.xlsx"
    csv = outdir / "Runtime_Memory_Profile.csv"
    df.to_excel(xlsx, index=False)
    df.to_csv(csv, index=False)

    print("[DONE]", xlsx)
    print("[DONE]", csv)
    if not HAS_PSUTIL:
        print("[NOTE] psutil not installed; peak RSS not recorded. Install via: conda install -y psutil")


if __name__ == "__main__":
    main()