# -*- coding: utf-8 -*-
"""
Metadata Validation Script
==========================

GOAL:
Validate whether an *unsupervised* score (e.g., mixture_score from geometry analysis)
actually corresponds to *known experimental labels* (pool-GBS vs individual-GBS).

INPUTS (Expected):
1) results_dir/
   - kmer_js_distance.csv              (distance matrix; defines sample list)
   - (optional) Supplementary_Data_S1.xlsx (kmer_entropy etc.)

2) gauss_dir/  (output of 02_run_geometry_analysis.py)
   - Supplementary_Data_S2_GaussEuler.xlsx (node_metrics with mixture_score)

GROUND TRUTH LABELS:
You can provide ground truth in 3 ways:
A) --pools "SRR...,SRR...,..."                    (manual pool list)
B) --pools_file pools.txt                         (one SRR per line)
C) --infer_from_ncbi                              (requires internet; parses SRA runinfo 'Title')

OUTPUTS:
- Excel: Supplementary_Data_S3_MetadataValidation.xlsx
- Figures (PNG + PDF):
    figures/main/
      ScoreVsGroundTruth
      ROC_MixtureScore
      ConfusionMatrix
    figures/supplementary/
      PR_MixtureScore
      LabelCounts

USAGE EXAMPLE:
  python 05_check_metadata.py \
    --results_dir "./results" \
    --gauss_dir "./results/gauss_euler" \
    --outdir "./results/validation" \
    --infer_from_ncbi \
    --topk 5
"""

from __future__ import annotations

import argparse
import io
import sys
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
)

from scipy.stats import fisher_exact, hypergeom

import urllib.parse
import urllib.request


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_distance_matrix(results_dir: Path) -> pd.DataFrame:
    fp = results_dir / "kmer_js_distance.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing distance matrix: {fp}")
    df = pd.read_csv(fp, index_col=0)
    # basic sanity
    if df.shape[0] != df.shape[1]:
        raise ValueError("Distance matrix is not square.")
    if list(df.index) != list(df.columns):
        # try to reorder columns
        df = df.loc[df.index, df.index]
    return df


def load_gauss_node_metrics(gauss_dir: Path) -> pd.DataFrame:
    xlsx = gauss_dir / "Supplementary_Data_S2_GaussEuler.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(f"Missing: {xlsx}")
    df = pd.read_excel(xlsx, sheet_name="node_metrics")
    # In our pipeline, 'sample' is an index column name in Excel.
    # Make it robust:
    if "sample" in df.columns:
        df = df.set_index("sample")
    else:
        # If first unnamed column was index:
        if df.columns[0].startswith("Unnamed"):
            df = df.set_index(df.columns[0])
            df.index.name = "sample"
    df.index = df.index.astype(str)
    return df


def load_kmer_entropy(results_dir: Path) -> Optional[pd.Series]:
    # optional; only if S1 exists and has sheet "entropy"
    xlsx = results_dir / "Supplementary_Data_S1.xlsx"
    if not xlsx.exists():
        return None
    try:
        ent = pd.read_excel(xlsx, sheet_name="entropy")
        if "sample" not in ent.columns or "kmer_entropy" not in ent.columns:
            return None
        s = ent.set_index("sample")["kmer_entropy"].astype(float)
        return s
    except Exception:
        return None


# -----------------------------
# Ground truth label acquisition
# -----------------------------
RUNINFO_URL = "https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi"


def fetch_runinfo(srr_list: List[str], timeout: int = 30) -> pd.DataFrame:
    """
    Fetch SRA runinfo as CSV from NCBI.

    NOTE: Requires internet.
    """
    # One request for all SRRs (faster, less rate-limit risk)
    term = " OR ".join(srr_list)
    params = {
        "save": "efetch",
        "db": "sra",
        "rettype": "runinfo",
        "term": term,
    }
    url = RUNINFO_URL + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        raw = r.read().decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(raw))
    if "Run" not in df.columns:
        raise ValueError("NCBI runinfo did not include a 'Run' column. Response format may have changed.")
    df["Run"] = df["Run"].astype(str)
    return df


def infer_pool_labels_from_runinfo(runinfo: pd.DataFrame) -> pd.Series:
    """
    Infer labels from runinfo text fields. Looks for 'pool-GBS' vs 'individual-GBS'.
    Returns a Series indexed by SRR (Run) with values in {'pool','individual','unknown'}.
    """
    text_cols = [c for c in ["Title", "Experiment", "SampleName", "LibraryName", "LibraryStrategy", "LibrarySelection"] if c in runinfo.columns]
    if not text_cols:
        # fallback: all columns string-join (heavy but ok for n=18)
        text_cols = list(runinfo.columns)

    def label_row(row: pd.Series) -> str:
        txt = " ".join([str(row.get(c, "")) for c in text_cols])
        t = txt.lower()
        if "pool-gbs" in t or "pool gbs" in t:
            return "pool"
        if "individual-gbs" in t or "individual gbs" in t:
            return "individual"
        # fallback: still allow pool detection if both 'pool' and 'gbs' appear
        if ("pool" in t) and ("gbs" in t):
            return "pool"
        if ("individual" in t) and ("gbs" in t):
            return "individual"
        return "unknown"

    lab = runinfo.apply(label_row, axis=1)
    out = pd.Series(lab.values, index=runinfo["Run"].astype(str), name="ground_truth")
    # If duplicates exist, keep the first non-unknown
    out = out.groupby(level=0).agg(lambda x: "pool" if "pool" in set(x) else ("individual" if "individual" in set(x) else "unknown"))
    return out


def read_pools_file(p: Path) -> List[str]:
    srrs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        srrs.append(line)
    return srrs


def build_ground_truth(samples: List[str],
                       pools_list: Optional[List[str]] = None,
                       runinfo_labels: Optional[pd.Series] = None) -> pd.Series:
    """
    Returns Series indexed by sample with values in {'pool','individual','unknown'}.
    Priority:
      1) runinfo_labels (if provided)
      2) pools_list (manual)
      3) all unknown
    """
    if runinfo_labels is not None:
        gt = pd.Series(index=samples, dtype=object)
        for s in samples:
            gt.loc[s] = runinfo_labels.get(s, "unknown")
        gt.name = "ground_truth"
        return gt

    if pools_list is not None and len(pools_list) > 0:
        pools_set = set([x.strip() for x in pools_list])
        gt = pd.Series(["pool" if s in pools_set else "individual" for s in samples], index=samples, name="ground_truth")
        return gt

    return pd.Series(["unknown"] * len(samples), index=samples, name="ground_truth")


# -----------------------------
# Evaluation + plotting
# -----------------------------
def evaluate_scores(scores: pd.Series, gt: pd.Series, topk: Optional[int] = None) -> Dict[str, float]:
    """
    Evaluate continuous score vs binary truth (pool=1, individual=0).
    Drops 'unknown' labels.
    """
    df = pd.DataFrame({"score": scores, "gt": gt}).dropna()
    df = df[df["gt"].isin(["pool", "individual"])].copy()
    if df.empty or df["gt"].nunique() < 2:
        return {"n_used": float(len(df))}

    y_true = (df["gt"] == "pool").astype(int).values
    y_score = df["score"].astype(float).values

    # Some metrics can fail if all y_score identical.
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")

    try:
        ap = average_precision_score(y_true, y_score)
    except Exception:
        ap = float("nan")

    # decision rule: topK predicted pools by score
    P = int(y_true.sum())
    if topk is None:
        topk = P if P > 0 else max(1, len(df) // 3)

    order = np.argsort(-y_score)  # descending
    y_pred = np.zeros_like(y_true)
    y_pred[order[:topk]] = 1

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # [[TP,FN],[FP,TN]] if reorder? careful
    # Let's compute explicitly:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    # Fisher exact enrichment in topK
    # table: [[TP, FN],[FP, TN]]
    _, p_fisher = fisher_exact([[tp, fn], [fp, tn]], alternative="greater")

    # Hypergeometric: probability of >=TP pools in topK draws
    N = len(y_true)
    M = P
    n = int(topk)
    k = tp
    # survival function: P[X >= k] = sf(k-1)
    p_hyper = float(hypergeom.sf(k - 1, N, M, n)) if (N > 0 and M >= 0 and n >= 0) else float("nan")

    out = {
        "n_used": float(N),
        "n_pools": float(P),
        "topk": float(topk),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(auc),
        "avg_precision": float(ap),
        "p_fisher_enrichment": float(p_fisher),
        "p_hypergeom_enrichment": float(p_hyper),
    }
    return out


def plot_score_vs_truth(scores: pd.Series, gt: pd.Series, out_base: Path, title: str) -> None:
    df = pd.DataFrame({"score": scores, "gt": gt}).copy()
    df = df.sort_values("score", ascending=False)

    # keep unknowns at bottom
    gt_order = {"pool": 0, "individual": 1, "unknown": 2}
    df["_gt_order"] = df["gt"].map(lambda x: gt_order.get(x, 2))
    df = df.sort_values(["_gt_order", "score"], ascending=[True, False])
    df = df.drop(columns=["_gt_order"])

    xs = np.arange(len(df))
    plt.figure(figsize=(12, 6))
    plt.bar(xs, df["score"].values)
    plt.xticks(xs, df.index.tolist(), rotation=90, fontsize=7)
    plt.ylabel("Score (higher = more 'pool-like' by this method)")
    plt.title(title)
    plt.grid(True, axis="y", linestyle=":", alpha=0.5)

    # marker overlay to indicate ground truth
    # pool: star marker at bar top; individual: circle; unknown: x
    y = df["score"].values
    for i, lab in enumerate(df["gt"].values):
        if lab == "pool":
            plt.scatter([i], [y[i]], marker="*", s=80)
        elif lab == "individual":
            plt.scatter([i], [y[i]], marker="o", s=25)
        else:
            plt.scatter([i], [y[i]], marker="x", s=35)

    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".png"), dpi=300)
    plt.savefig(out_base.with_suffix(".pdf"))
    plt.close()


def plot_roc(scores: pd.Series, gt: pd.Series, out_base: Path, title: str) -> None:
    df = pd.DataFrame({"score": scores, "gt": gt}).dropna()
    df = df[df["gt"].isin(["pool", "individual"])]
    if df.empty or df["gt"].nunique() < 2:
        return
    y_true = (df["gt"] == "pool").astype(int).values
    y_score = df["score"].astype(float).values
    fpr, tpr, _ = roc_curve(y_true, y_score)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title}\nAUC = {auc:.3f}")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".png"), dpi=300)
    plt.savefig(out_base.with_suffix(".pdf"))
    plt.close()


def plot_pr(scores: pd.Series, gt: pd.Series, out_base: Path, title: str) -> None:
    df = pd.DataFrame({"score": scores, "gt": gt}).dropna()
    df = df[df["gt"].isin(["pool", "individual"])]
    if df.empty or df["gt"].nunique() < 2:
        return
    y_true = (df["gt"] == "pool").astype(int).values
    y_score = df["score"].astype(float).values
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    try:
        ap = average_precision_score(y_true, y_score)
    except Exception:
        ap = float("nan")

    plt.figure(figsize=(7, 6))
    plt.plot(rec, prec, marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title}\nAverage precision = {ap:.3f}")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".png"), dpi=300)
    plt.savefig(out_base.with_suffix(".pdf"))
    plt.close()


def plot_confusion_at_topk(scores: pd.Series, gt: pd.Series, out_base: Path, topk: int, title: str) -> pd.DataFrame:
    df = pd.DataFrame({"score": scores, "gt": gt}).dropna()
    df = df[df["gt"].isin(["pool", "individual"])]
    df = df.sort_values("score", ascending=False)
    if df.empty:
        return pd.DataFrame()

    y_true = (df["gt"] == "pool").astype(int).values
    y_pred = np.zeros_like(y_true)
    topk = min(topk, len(df))
    y_pred[:topk] = 1

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    cm = np.array([[tp, fn], [fp, tn]], dtype=int)
    cm_df = pd.DataFrame(cm, index=["TruePool", "TrueIndividual"], columns=["PredPool", "PredIndividual"])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, aspect="auto")
    plt.xticks([0, 1], ["Pred Pool", "Pred Individual"])
    plt.yticks([0, 1], ["True Pool", "True Individual"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.title(f"{title} (topK={topk})")
    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".png"), dpi=300)
    plt.savefig(out_base.with_suffix(".pdf"))
    plt.close()

    return cm_df


def plot_label_counts(gt: pd.Series, out_base: Path, title: str) -> None:
    counts = gt.value_counts(dropna=False)
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".png"), dpi=300)
    plt.savefig(out_base.with_suffix(".pdf"))
    plt.close()


# -----------------------------
# Main
# -----------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__),
    )
    ap.add_argument("--results_dir", required=True, help="Path to base results directory that contains kmer_js_distance.csv")
    ap.add_argument("--gauss_dir", required=True, help="Path to gauss_euler output dir containing Supplementary_Data_S2_GaussEuler.xlsx")
    ap.add_argument("--outdir", required=True, help="Output directory for validation results")

    ap.add_argument("--pools", default="", help="Comma-separated SRR list to treat as pools (ground truth), e.g., SRR1,SRR2,...")
    ap.add_argument("--pools_file", default="", help="Text file with one SRR per line (ground truth pools)")
    ap.add_argument("--infer_from_ncbi", action="store_true", help="Fetch NCBI runinfo and infer pool/individual from titles (requires internet)")

    ap.add_argument("--topk", type=int, default=0, help="Decision threshold: topK by score called 'pool'. Default: K=#true pools.")
    ap.add_argument("--also_entropy", action="store_true", help="Also evaluate kmer_entropy (if Supplementary_Data_S1.xlsx exists)")

    return ap.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    gauss_dir = Path(args.gauss_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    fig_main = outdir / "figures" / "main"
    fig_sup = outdir / "figures" / "supplementary"
    ensure_dir(fig_main)
    ensure_dir(fig_sup)

    dist = read_distance_matrix(results_dir)
    samples = list(dist.index.astype(str))

    # Load mixture_score
    node_metrics = load_gauss_node_metrics(gauss_dir)
    if "mixture_score" not in node_metrics.columns:
        raise ValueError("node_metrics sheet does not include 'mixture_score'.")
    mixture_score = node_metrics["mixture_score"].reindex(samples).astype(float)

    # Ground truth
    pools_list = []
    if args.pools_file:
        pools_list = read_pools_file(Path(args.pools_file))
    elif args.pools.strip():
        pools_list = [x.strip() for x in args.pools.split(",") if x.strip()]

    runinfo_df = None
    runinfo_labels = None
    if args.infer_from_ncbi:
        try:
            runinfo_df = fetch_runinfo(samples)
            runinfo_labels = infer_pool_labels_from_runinfo(runinfo_df)
        except Exception as e:
            print(f"[WARN] NCBI runinfo fetch/inference failed: {e}", file=sys.stderr)
            print("[WARN] Falling back to manual pools list (if provided) or 'unknown' labels.", file=sys.stderr)

    gt = build_ground_truth(samples, pools_list=pools_list if pools_list else None, runinfo_labels=runinfo_labels)

    # Evaluation on mixture_score
    topk = args.topk if args.topk and args.topk > 0 else None
    metrics_mix = evaluate_scores(mixture_score, gt, topk=topk)

    # Figures (mixture_score)
    # Renamed: Fig11 -> MixtureScoreVsGroundTruth
    plot_score_vs_truth(
        mixture_score, gt,
        fig_main / "MixtureScoreVsGroundTruth",
        "Mixture score (bridge score) vs ground-truth labels"
    )
    # Renamed: Fig12 -> ROC_MixtureScore
    plot_roc(
        mixture_score, gt,
        fig_main / "ROC_MixtureScore",
        "ROC: mixture_score predicting pool label"
    )
    # Renamed: FigS11 -> PR_MixtureScore
    plot_pr(
        mixture_score, gt,
        fig_sup / "PR_MixtureScore",
        "Precision–Recall: mixture_score predicting pool label"
    )

    # Confusion matrix at chosen topK
    if topk is None:
        # if ground truth has P pools, use that as topk
        P = int((gt == "pool").sum())
        topk_used = P if P > 0 else min(5, len(samples))
    else:
        topk_used = int(topk)

    # Renamed: Fig13 -> ConfusionMatrix_TopK
    cm_df = plot_confusion_at_topk(
        mixture_score, gt,
        fig_main / "ConfusionMatrix_TopK",
        topk=topk_used,
        title="Confusion matrix (topK rule)"
    )

    # Renamed: FigS12 -> LabelCounts
    plot_label_counts(gt, fig_sup / "LabelCounts", "Ground-truth label counts")

    # Optional: evaluate kmer_entropy
    metrics_ent = None
    ent = None
    if args.also_entropy:
        ent = load_kmer_entropy(results_dir)
        if ent is not None:
            ent = ent.reindex(samples).astype(float)
            metrics_ent = evaluate_scores(ent, gt, topk=topk)
            # Renamed: FigS13 -> kmerEntropyVsGroundTruth
            plot_score_vs_truth(
                ent, gt,
                fig_sup / "kmerEntropyVsGroundTruth",
                "k-mer entropy vs ground-truth pool labels"
            )
            # Renamed: FigS14 -> ROC_kmerEntropy
            plot_roc(
                ent, gt,
                fig_sup / "ROC_kmerEntropy",
                "ROC: kmer_entropy predicting pool label"
            )

    # Write Excel
    excel_path = outdir / "Supplementary_Data_S3_MetadataValidation.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as xw:
        # sample table
        sample_table = pd.DataFrame({
            "sample": samples,
            "ground_truth": gt.values,
            "mixture_score": mixture_score.values,
        }).set_index("sample")
        sample_table.to_excel(xw, sheet_name="sample_labels_scores")

        pd.DataFrame([metrics_mix]).to_excel(xw, sheet_name="metrics_mixture_score", index=False)
        if metrics_ent is not None:
            pd.DataFrame([metrics_ent]).to_excel(xw, sheet_name="metrics_kmer_entropy", index=False)
        if cm_df is not None and not cm_df.empty:
            cm_df.to_excel(xw, sheet_name="confusion_topk")

        # include runinfo (if used)
        if runinfo_df is not None:
            runinfo_df.to_excel(xw, sheet_name="ncbi_runinfo", index=False)

    print("\n[DONE] Metadata matching validation complete.")
    print(f"  -> Excel: {excel_path}")
    print(f"  -> Figures: {fig_main} and {fig_sup}")
    print("\nKey numbers (mixture_score):")
    for k in ["n_used","n_pools","topk","roc_auc","avg_precision","f1","p_fisher_enrichment","p_hypergeom_enrichment"]:
        if k in metrics_mix:
            print(f"  {k}: {metrics_mix[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))