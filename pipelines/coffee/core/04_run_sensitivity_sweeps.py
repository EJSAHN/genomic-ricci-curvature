# -*- coding: utf-8 -*-
"""
Bootstrapped sensitivity sweeps for synthetic validation.

What this script does
---------------------
1. Re-runs 03_run_synthetic_validation.py across read-depth settings.
2. Re-runs 03_run_synthetic_validation.py across k-mer settings.
3. Summarizes discrimination and monotonicity metrics into an Excel workbook.

Primary workbook
----------------
<outdir>/Supplementary_Data_S4_SensitivitySweeps.xlsx
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    metrics = pd.read_excel(xlsx_path, sheet_name="metrics_summary")
    if metrics.empty:
        raise ValueError(f"metrics_summary sheet is empty in {xlsx_path}")

    roc_auc = float(metrics.loc[0, "roc_auc_synthetic"])
    avg_precision = float(metrics.loc[0, "avg_precision_synthetic"])
    best_f1 = float(metrics.loc[0, "best_f1_synthetic"])
    best_threshold = float(metrics.loc[0, "best_threshold"])

    syn = pd.read_excel(xlsx_path, sheet_name="synthetic_score_vs_truth")
    if syn.empty:
        raise ValueError(f"synthetic_score_vs_truth sheet is empty in {xlsx_path}")

    x = syn["entropy_norm"].astype(float).values
    y = syn["mixture_score"].astype(float).values

    sr = spearmanr(x, y, nan_policy="omit")
    spearman_rho = float(sr.correlation) if sr.correlation is not None else float("nan")
    spearman_p = float(sr.pvalue) if sr.pvalue is not None else float("nan")

    pr = pearsonr(x, y)
    pearson_r = float(pr.statistic) if hasattr(pr, "statistic") else float(pr[0])
    pearson_p = float(pr.pvalue) if hasattr(pr, "pvalue") else float(pr[1])

    return roc_auc, avg_precision, best_f1, best_threshold, spearman_rho, spearman_p, pearson_r, pearson_p


def run_validation_once(
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
    real_pool_ids: str,
    skip_existing: bool,
) -> str:
    ensure_dir(run_dir)
    xlsx_path = os.path.join(run_dir, "Supplementary_Data_S3_SyntheticValidation.xlsx")
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
        cmd.extend(["--include_real_pools", "--real_pool_ids", real_pool_ids])

    subprocess.run(cmd, check=True)

    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Expected output workbook not found: {xlsx_path}")
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
    for c in agg.columns:
        if c.endswith("_sd"):
            agg[c] = agg[c].fillna(0.0)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq_dir", required=True, help="Folder with paired FASTQ files")
    ap.add_argument("--outdir", required=True, help="Output folder")
    ap.add_argument("--calib_script", default="03_run_synthetic_validation.py", help="Path to 03_run_synthetic_validation.py")
    ap.add_argument("--reads_list", default="3000,6000,12000", help="Comma-separated reads_per_sample values")
    ap.add_argument("--kmer_list", default="15,17,19,21", help="Comma-separated k-mer values")
    ap.add_argument("--n_boot", type=int, default=5, help="Boot replicates per setting")
    ap.add_argument("--base_seed", type=int, default=123, help="Base seed")
    ap.add_argument("--base_kmer", type=int, default=17, help="k-mer used for the read-depth sweep")
    ap.add_argument("--base_reads", type=int, default=6000, help="reads_per_sample used for the k-mer sweep")
    ap.add_argument("--sketch", type=int, default=16384)
    ap.add_argument("--n_synth", type=int, default=40)
    ap.add_argument("--max_parents", type=int, default=3)
    ap.add_argument("--min_minor", type=float, default=0.10)
    ap.add_argument("--knn", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--use_r2", action="store_true")
    ap.add_argument("--include_real_pools", action="store_true")
    ap.add_argument("--real_pool_ids", default="", help="Comma-separated IDs used only when --include_real_pools is set")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    outdir = args.outdir
    ensure_dir(outdir)

    reads_list = [int(x.strip()) for x in args.reads_list.split(",") if x.strip()]
    kmer_list = [int(x.strip()) for x in args.kmer_list.split(",") if x.strip()]

    calib_script = args.calib_script
    if not os.path.exists(calib_script):
        here = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(here, calib_script)
        if os.path.exists(alt):
            calib_script = alt
        else:
            raise FileNotFoundError(f"Calibration script not found: {args.calib_script}")

    if args.include_real_pools and not args.real_pool_ids.strip():
        raise SystemExit("[ERR] --include_real_pools requires --real_pool_ids.")

    all_runs: List[RunMetrics] = []

    print("\n[SWEEP] Read-depth sensitivity")
    for reads in reads_list:
        for b in range(args.n_boot):
            seed = int(args.base_seed + b)
            run_dir = os.path.join(outdir, f"runs_reads_{reads}", f"boot_{b+1:02d}")
            xlsx_path = run_validation_once(
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
                real_pool_ids=args.real_pool_ids,
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
            print(f"  [OK] reads={reads} seed={seed} AUC={roc_auc:.3f} Spearmanρ={srho:.3f}")

    print("\n[SWEEP] k-mer sensitivity")
    for k in kmer_list:
        for b in range(args.n_boot):
            seed = int(args.base_seed + 1000 + b)
            run_dir = os.path.join(outdir, f"runs_kmer_{k}", f"boot_{b+1:02d}")
            xlsx_path = run_validation_once(
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
                real_pool_ids=args.real_pool_ids,
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
            print(f"  [OK] k={k} seed={seed} AUC={roc_auc:.3f} Spearmanρ={srho:.3f}")

    df = pd.DataFrame([r.__dict__ for r in all_runs])
    reads_summary = summarize(df[df["sweep_type"] == "reads"].copy(), "param_value").rename(columns={"param_value": "reads_per_sample"})
    kmer_summary = summarize(df[df["sweep_type"] == "kmer"].copy(), "param_value").rename(columns={"param_value": "kmer"})

    params_df = pd.DataFrame(
        [
            {
                "reads_list": ",".join(map(str, reads_list)),
                "kmer_list": ",".join(map(str, kmer_list)),
                "n_boot": args.n_boot,
                "base_seed": args.base_seed,
                "base_kmer": args.base_kmer,
                "base_reads": args.base_reads,
                "sketch": args.sketch,
                "n_synth": args.n_synth,
                "max_parents": args.max_parents,
                "min_minor": args.min_minor,
                "knn": args.knn,
                "alpha": args.alpha,
                "use_r2": bool(args.use_r2),
                "include_real_pools": bool(args.include_real_pools),
            }
        ]
    )

    xlsx_out = os.path.join(outdir, "Supplementary_Data_S4_SensitivitySweeps.xlsx")
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as xw:
        params_df.to_excel(xw, sheet_name="run_params", index=False)
        df.to_excel(xw, sheet_name="run_metrics_raw", index=False)
        reads_summary.to_excel(xw, sheet_name="reads_summary", index=False)
        kmer_summary.to_excel(xw, sheet_name="kmer_summary", index=False)

    print("\n[DONE] Sensitivity sweeps completed.")
    print(f"  -> Workbook: {xlsx_out}")


if __name__ == "__main__":
    main()
