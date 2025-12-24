# -*- coding: utf-8 -*-
"""
Sensitivity + Reproducibility Sweeps (Bootstrapped)
===================================================

What this does
--------------
This script runs *parameter sweeps* with multiple random seeds ("bootstraps")
by repeatedly calling '03_run_synthetic_validation.py'.

1) Reads-per-sample sweep: 3000 / 6000 / 12000
   - For each setting, run N boot replicates
   - Collect ROC AUC & Spearman rho (monotonicity)
   - Produce ONE main figure with error bars (mean ± SD).

2) k-mer length sweep: k = 15 / 17 / 19 / 21
   - Same metrics + one main figure with error bars.

Outputs
-------
<outdir>/
  - Supplementary_Data_S5_SensitivitySweeps.xlsx
  - figures/main/
      ReadDepthSensitivity.(png|pdf)
      KmerSensitivity.(png|pdf)

Usage Example
-------------
python 04_run_sensitivity_sweeps.py \
  --fastq_dir "./data" \
  --outdir "./results/sensitivity_sweeps" \
  --reads_list "3000,6000,12000" \
  --kmer_list "15,17,19,21" \
  --n_boot 30
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_fig(path_png: str, path_pdf: str) -> None:
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.savefig(path_pdf)
    plt.close()


@dataclass
class RunMetrics:
    sweep_type: str
    param_name: str
    param_value: float
    seed: int
    run_dir: str
    roc_auc: float
    avg_precision: float
    best_f1: float
    best_threshold: float
    spearman_rho: float
    spearman_p: float
    pearson_r: float
    pearson_p: float


def _read_single_run_metrics(xlsx_path: str) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Returns:
      roc_auc, avg_precision, best_f1, best_threshold,
      spearman_rho, spearman_p, pearson_r, pearson_p
    """
    m = pd.read_excel(xlsx_path, sheet_name="metrics_summary")
    if m.empty:
        raise ValueError(f"metrics_summary sheet is empty in {xlsx_path}")

    roc_auc = float(m.loc[0, "roc_auc_synthetic"])
    avg_precision = float(m.loc[0, "avg_precision_synthetic"])
    best_f1 = float(m.loc[0, "best_f1_synthetic"])
    best_threshold = float(m.loc[0, "best_threshold"])

    syn = pd.read_excel(xlsx_path, sheet_name="synthetic_score_vs_truth")
    if syn.empty:
        raise ValueError(f"synthetic_score_vs_truth sheet is empty in {xlsx_path}")

    x = syn["entropy_norm"].astype(float).values
    y = syn["mixture_score_ot"].astype(float).values

    # Spearman for monotonicity
    sr = spearmanr(x, y, nan_policy="omit")
    spearman_rho = float(sr.correlation) if sr.correlation is not None else float("nan")
    spearman_p = float(sr.pvalue) if sr.pvalue is not None else float("nan")

    # Pearson for linear-ish trend
    pr = pearsonr(x, y)
    pearson_r = float(pr.statistic) if hasattr(pr, "statistic") else float(pr[0])
    pearson_p = float(pr.pvalue) if hasattr(pr, "pvalue") else float(pr[1])

    return roc_auc, avg_precision, best_f1, best_threshold, spearman_rho, spearman_p, pearson_r, pearson_p


def run_calibration_once(
    calib_script: str,
    fastq_dir: str,
    run_dir: str,
    reads_per_sample: int,
    kmer: int,
    sketch: int,
    n_synth: int,
    max_parents: int,
    min_minor: float,
    knn: int,
    alpha: float,
    seed: int,
    use_r2: bool,
    include_real_pools: bool,
    skip_existing: bool,
) -> str:
    """
    Runs 03_run_synthetic_validation.py once via subprocess.
    Returns path to its Excel output.
    """
    ensure_dir(run_dir)
    xlsx_path = os.path.join(run_dir, "Supplementary_Data_S4_SyntheticMixing.xlsx")
    if skip_existing and os.path.exists(xlsx_path):
        return xlsx_path

    cmd = [
        sys.executable,
        calib_script,
        "--fastq_dir", fastq_dir,
        "--outdir", run_dir,
        "--reads_per_sample", str(int(reads_per_sample)),
        "--kmer", str(int(kmer)),
        "--sketch", str(int(sketch)),
        "--n_synth", str(int(n_synth)),
        "--max_parents", str(int(max_parents)),
        "--min_minor", str(float(min_minor)),
        "--knn", str(int(knn)),
        "--alpha", str(float(alpha)),
        "--seed", str(int(seed)),
    ]
    if use_r2:
        cmd.append("--use_r2")
    if include_real_pools:
        cmd.append("--include_real_pools")

    # Run and wait
    subprocess.run(cmd, check=True)

    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Expected output Excel not found: {xlsx_path}")

    return xlsx_path


def summarize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    agg = df.groupby(group_col).agg(
        n_runs=("seed", "count"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_sd=("roc_auc", "std"),
        ap_mean=("avg_precision", "mean"),
        ap_sd=("avg_precision", "std"),
        f1_mean=("best_f1", "mean"),
        f1_sd=("best_f1", "std"),
        spearman_rho_mean=("spearman_rho", "mean"),
        spearman_rho_sd=("spearman_rho", "std"),
        pearson_r_mean=("pearson_r", "mean"),
        pearson_r_sd=("pearson_r", "std"),
    ).reset_index()
    # Replace NaN SD with 0 when n_runs==1
    for c in agg.columns:
        if c.endswith("_sd"):
            agg[c] = agg[c].fillna(0.0)
    return agg


def plot_sweep(summary: pd.DataFrame, xcol: str, title: str, out_png: str, out_pdf: str) -> None:
    """
    Single figure with two panels:
      A: ROC AUC vs parameter (mean ± SD)
      B: Spearman rho vs parameter (mean ± SD)
    """
    x = summary[xcol].astype(float).values
    auc_m = summary["roc_auc_mean"].values
    auc_s = summary["roc_auc_sd"].values
    rho_m = summary["spearman_rho_mean"].values
    rho_s = summary["spearman_rho_sd"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].errorbar(x, auc_m, yerr=auc_s, marker="o", linestyle="-", capsize=4)
    axes[0].set_title("Synthetic pool detection (ROC AUC)")
    axes[0].set_xlabel(title)
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].errorbar(x, rho_m, yerr=rho_s, marker="o", linestyle="-", capsize=4)
    axes[1].set_title("Monotonicity (Spearman ρ)")
    axes[1].set_xlabel(title)
    axes[1].set_ylabel("Spearman ρ (score vs entropy)")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"{title}: bootstrapped sensitivity (mean ± SD)", y=1.02, fontsize=12)
    save_fig(out_png, out_pdf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq_dir", required=True, help="Folder with SRR*_1.fastq(.gz) and SRR*_2.fastq(.gz)")
    ap.add_argument("--outdir", required=True, help="Output folder for sweeps")

    # IMPORTANT: Changed default to the new filename
    ap.add_argument("--calib_script", default="03_run_synthetic_validation.py",
                    help="Path to 03_run_synthetic_validation.py (default: in current folder)")

    ap.add_argument("--reads_list", default="3000,6000,12000", help="Comma-separated reads_per_sample values")
    ap.add_argument("--kmer_list", default="15,17,19,21", help="Comma-separated k-mer values")

    ap.add_argument("--n_boot", type=int, default=5, help="Boot replicates per setting (different seeds)")
    ap.add_argument("--base_seed", type=int, default=123, help="Base seed; replicate seeds = base_seed + i")
    ap.add_argument("--base_kmer", type=int, default=17, help="k-mer used for reads sweep")
    ap.add_argument("--base_reads", type=int, default=6000, help="reads_per_sample used for k-mer sweep")

    ap.add_argument("--sketch", type=int, default=16384)
    ap.add_argument("--n_synth", type=int, default=40)
    ap.add_argument("--max_parents", type=int, default=3)
    ap.add_argument("--min_minor", type=float, default=0.10)
    ap.add_argument("--knn", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--use_r2", action="store_true")
    ap.add_argument("--include_real_pools", action="store_true", help="(Slower) include real pools in each run")
    ap.add_argument("--skip_existing", action="store_true", help="Skip runs if their S4 Excel already exists")

    args = ap.parse_args()

    outdir = args.outdir
    ensure_dir(outdir)
    fig_main = os.path.join(outdir, "figures", "main")
    ensure_dir(fig_main)

    reads_list = [int(x.strip()) for x in args.reads_list.split(",") if x.strip()]
    kmer_list = [int(x.strip()) for x in args.kmer_list.split(",") if x.strip()]

    calib_script = args.calib_script
    if not os.path.exists(calib_script):
        # also try relative to this file
        here = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(here, calib_script)
        if os.path.exists(alt):
            calib_script = alt
        else:
            raise FileNotFoundError(f"Calibration script not found: {args.calib_script}\n"
                                    "Please make sure '03_run_synthetic_validation.py' is in the same folder.")

    all_runs: List[RunMetrics] = []

    # -------------------------
    # Sweep 1: reads_per_sample
    # -------------------------
    print("\n[SWEEP] Read depth sensitivity ...")
    for reads in reads_list:
        for b in range(args.n_boot):
            seed = int(args.base_seed + b)
            run_dir = os.path.join(outdir, f"runs_reads_{reads}", f"boot_{b+1:02d}")
            xlsx_path = run_calibration_once(
                calib_script=calib_script,
                fastq_dir=args.fastq_dir,
                run_dir=run_dir,
                reads_per_sample=reads,
                kmer=args.base_kmer,
                sketch=args.sketch,
                n_synth=args.n_synth,
                max_parents=args.max_parents,
                min_minor=args.min_minor,
                knn=args.knn,
                alpha=args.alpha,
                seed=seed,
                use_r2=args.use_r2,
                include_real_pools=args.include_real_pools,
                skip_existing=args.skip_existing,
            )
            roc_auc, ap, best_f1, best_th, srho, sp, pr, pp = _read_single_run_metrics(xlsx_path)
            all_runs.append(
                RunMetrics(
                    sweep_type="reads",
                    param_name="reads_per_sample",
                    param_value=float(reads),
                    seed=seed,
                    run_dir=run_dir,
                    roc_auc=roc_auc,
                    avg_precision=ap,
                    best_f1=best_f1,
                    best_threshold=best_th,
                    spearman_rho=srho,
                    spearman_p=sp,
                    pearson_r=pr,
                    pearson_p=pp,
                )
            )
            print(f"  [OK] reads={reads} seed={seed}  AUC={roc_auc:.3f}  Spearmanρ={srho:.3f}")

    # -------------------------
    # Sweep 2: k-mer length
    # -------------------------
    print("\n[SWEEP] k-mer sensitivity ...")
    for k in kmer_list:
        for b in range(args.n_boot):
            # shift seed block
            seed = int(args.base_seed + 1000 + b)
            run_dir = os.path.join(outdir, f"runs_kmer_{k}", f"boot_{b+1:02d}")
            xlsx_path = run_calibration_once(
                calib_script=calib_script,
                fastq_dir=args.fastq_dir,
                run_dir=run_dir,
                reads_per_sample=args.base_reads,
                kmer=k,
                sketch=args.sketch,
                n_synth=args.n_synth,
                max_parents=args.max_parents,
                min_minor=args.min_minor,
                knn=args.knn,
                alpha=args.alpha,
                seed=seed,
                use_r2=args.use_r2,
                include_real_pools=args.include_real_pools,
                skip_existing=args.skip_existing,
            )
            roc_auc, ap, best_f1, best_th, srho, sp, pr, pp = _read_single_run_metrics(xlsx_path)
            all_runs.append(
                RunMetrics(
                    sweep_type="kmer",
                    param_name="kmer",
                    param_value=float(k),
                    seed=seed,
                    run_dir=run_dir,
                    roc_auc=roc_auc,
                    avg_precision=ap,
                    best_f1=best_f1,
                    best_threshold=best_th,
                    spearman_rho=srho,
                    spearman_p=sp,
                    pearson_r=pr,
                    pearson_p=pp,
                )
            )
            print(f"  [OK] k={k} seed={seed}  AUC={roc_auc:.3f}  Spearmanρ={srho:.3f}")

    # -------------------------
    # Summaries + Figures + Excel
    # -------------------------
    df = pd.DataFrame([r.__dict__ for r in all_runs])

    reads_df = df[df["sweep_type"] == "reads"].copy()
    kmer_df = df[df["sweep_type"] == "kmer"].copy()

    reads_summary = summarize(reads_df, "param_value").rename(columns={"param_value": "reads_per_sample"})
    kmer_summary = summarize(kmer_df, "param_value").rename(columns={"param_value": "kmer"})

    # plot figures (Renamed)
    plot_sweep(
        reads_summary.sort_values("reads_per_sample"),
        xcol="reads_per_sample",
        title="Read depth (reads_per_sample)",
        # Renamed: FigH -> ReadDepthSensitivity
        out_png=os.path.join(fig_main, "ReadDepthSensitivity.png"),
        out_pdf=os.path.join(fig_main, "ReadDepthSensitivity.pdf"),
    )
    plot_sweep(
        kmer_summary.sort_values("kmer"),
        xcol="kmer",
        title="k-mer length",
        # Renamed: FigI -> KmerSensitivity
        out_png=os.path.join(fig_main, "KmerSensitivity.png"),
        out_pdf=os.path.join(fig_main, "KmerSensitivity.pdf"),
    )

    # write Excel
    xlsx_out = os.path.join(outdir, "Supplementary_Data_S5_SensitivitySweeps.xlsx")
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="run_metrics_raw", index=False)
        reads_summary.to_excel(xw, sheet_name="reads_summary", index=False)
        kmer_summary.to_excel(xw, sheet_name="kmer_summary", index=False)

    print("\n[DONE] Sweeps completed.")
    print(f"  -> Excel: {xlsx_out}")
    print(f"  -> Figures: {fig_main}")
    print("\nInterpretation quick guide:")
    print("  - You want ROC AUC to stay high across reads/k.")
    print("  - You want Spearman rho(score, entropy) to stay positive and strong across reads/k.")
    print("  - Error bars (SD) should shrink or remain reasonable as reads increase.")


if __name__ == "__main__":
    main()