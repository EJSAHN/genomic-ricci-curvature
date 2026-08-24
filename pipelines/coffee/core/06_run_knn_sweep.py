# -*- coding: utf-8 -*-
"""
kNN sweep for geometry-analysis stability.

What this script does
---------------------
1. Re-runs geometry analysis across a user-defined kNN range.
2. Collects node-level mixture scores from each run.
3. Summarizes rank stability and top-N set overlap into an Excel workbook.

Primary workbook
----------------
<outdir>/Supplementary_Data_S6_KNNSweep.xlsx
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_geometry_step(
    geom_script: Path,
    results_dir: Path,
    outdir: Path,
    knn: int,
    alpha: float,
    n_thresholds: int,
) -> None:
    cmd = [
        sys.executable,
        str(geom_script),
        "--results_dir", str(results_dir),
        "--outdir", str(outdir),
        "--knn", str(knn),
        "--alpha", str(alpha),
        "--n_thresholds", str(n_thresholds),
    ]
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_node_table(geometry_outdir: Path) -> pd.DataFrame:
    xlsx = geometry_outdir / "Supplementary_Data_S2_Geometry.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(f"Missing expected workbook: {xlsx}")
    df = pd.read_excel(xlsx, sheet_name="node_metrics")
    if "sample" not in df.columns or "mixture_score" not in df.columns:
        raise ValueError("node_metrics sheet must contain 'sample' and 'mixture_score' columns.")
    return df.set_index("sample")


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna()
    b = b.dropna()
    idx = a.index.intersection(b.index)
    if len(idx) < 3:
        return float("nan")
    ra = a.loc[idx].rank(method="average")
    rb = b.loc[idx].rank(method="average")
    return float(np.corrcoef(ra.values, rb.values)[0, 1])


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="Folder containing kmer_js_distance.csv")
    ap.add_argument("--outdir", required=True, help="Output folder for sweep summary + per-k outputs")
    ap.add_argument("--geom_script", default="02_run_geometry_analysis.py", help="Path to geometry script")
    ap.add_argument("--knn_min", type=int, default=3, help="Minimum k to test")
    ap.add_argument("--knn_max", type=int, default=7, help="Maximum k to test")
    ap.add_argument("--top_n", type=int, default=5, help="Top-N candidates to evaluate for set overlap")
    ap.add_argument("--alpha", type=float, default=0.0, help="Idleness for ORC")
    ap.add_argument("--n_thresholds", type=int, default=21, help="Number of topology thresholds")
    ap.add_argument("--skip_run", action="store_true", help="Do not rerun geometry; only summarize existing outputs")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)

    geom_script = Path(args.geom_script)
    if not geom_script.exists():
        geom_script = (Path(__file__).parent / args.geom_script).resolve()
    if not geom_script.exists():
        raise FileNotFoundError(f"Cannot find geometry script: {args.geom_script}")

    knn_values = list(range(int(args.knn_min), int(args.knn_max) + 1))
    print(f"[OK] Sweep knn = {knn_values}")

    if not args.skip_run:
        for k in knn_values:
            k_out = outdir / f"k{k:02d}"
            ensure_dir(k_out)
            run_geometry_step(
                geom_script=geom_script,
                results_dir=results_dir,
                outdir=k_out,
                knn=int(k),
                alpha=float(args.alpha),
                n_thresholds=int(args.n_thresholds),
            )

    node_tables: Dict[int, pd.DataFrame] = {}
    for k in knn_values:
        node_tables[int(k)] = read_node_table(outdir / f"k{k:02d}")

    all_samples = sorted(set().union(*[set(df.index) for df in node_tables.values()]))
    score_mat = pd.DataFrame(index=all_samples)
    for k, df in node_tables.items():
        score_mat[f"k{k:02d}"] = df["mixture_score"]

    top_n = int(args.top_n)
    top_lists: Dict[int, List[str]] = {}
    for k in knn_values:
        s = score_mat[f"k{k:02d}"].sort_values(ascending=False)
        top_lists[int(k)] = [x for x in s.index[:top_n] if pd.notna(s.loc[x])]

    ks = [int(k) for k in knn_values]
    jac = pd.DataFrame(index=ks, columns=ks, dtype=float)
    rho = pd.DataFrame(index=ks, columns=ks, dtype=float)

    for i in ks:
        for j in ks:
            jac.loc[i, j] = jaccard(set(top_lists[i]), set(top_lists[j]))
            rho.loc[i, j] = spearman_corr(score_mat[f"k{i:02d}"], score_mat[f"k{j:02d}"])

    top_long_rows = []
    for k in ks:
        s = score_mat[f"k{k:02d}"].sort_values(ascending=False)
        for rank, sample in enumerate(s.index[:top_n], start=1):
            top_long_rows.append(
                {
                    "knn": k,
                    "rank": rank,
                    "sample": sample,
                    "mixture_score": float(s.loc[sample]) if pd.notna(s.loc[sample]) else np.nan,
                }
            )
    top_long = pd.DataFrame(top_long_rows)

    union_set = set().union(*[set(v) for v in top_lists.values()])
    inter_set = set(top_lists[ks[0]])
    for k in ks[1:]:
        inter_set &= set(top_lists[k])

    stability_summary = pd.DataFrame(
        [
            {
                "knn_values": ",".join(map(str, ks)),
                "top_n": top_n,
                "union_size": len(union_set),
                "intersection_size": len(inter_set),
                "intersection_samples": ",".join(sorted(inter_set)) if inter_set else "",
                "spearman_mean": float(np.nanmean(rho.values)),
                "spearman_min": float(np.nanmin(rho.values)),
                "spearman_max": float(np.nanmax(rho.values)),
                "jaccard_mean": float(np.nanmean(jac.values)),
                "jaccard_min": float(np.nanmin(jac.values)),
                "jaccard_max": float(np.nanmax(jac.values)),
            }
        ]
    )

    params_df = pd.DataFrame(
        [
            {
                "results_dir": str(results_dir),
                "outdir": str(outdir),
                "knn_values": ",".join(map(str, ks)),
                "top_n": top_n,
                "alpha": float(args.alpha),
                "n_thresholds": int(args.n_thresholds),
                "skip_run": bool(args.skip_run),
            }
        ]
    )

    out_xlsx = outdir / "Supplementary_Data_S6_KNNSweep.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        params_df.to_excel(xw, sheet_name="run_params", index=False)
        stability_summary.to_excel(xw, sheet_name="stability_summary", index=False)
        top_long.to_excel(xw, sheet_name="topN_per_k", index=False)
        score_mat.reset_index().rename(columns={"index": "sample"}).to_excel(xw, sheet_name="mixture_scores", index=False)
        jac.to_excel(xw, sheet_name="jaccard_topN")
        rho.to_excel(xw, sheet_name="spearman_all")

    print("\n[DONE] kNN sweep complete.")
    print(f"  -> Workbook: {out_xlsx}")
    print(f"  -> Intersection across all k: {sorted(inter_set)}")


if __name__ == "__main__":
    main()
