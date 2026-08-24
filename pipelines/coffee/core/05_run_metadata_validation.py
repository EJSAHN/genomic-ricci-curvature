# -*- coding: utf-8 -*-
"""
Metadata-based validation of the unsupervised mixture score.

What this script does
---------------------
1. Loads the real-sample mixture score from geometry analysis.
2. Builds ground-truth labels from a manual list, a text file, or optional NCBI runinfo.
3. Evaluates ranking performance against those labels.
4. Writes the results to an Excel workbook.

Primary workbook
----------------
<outdir>/Supplementary_Data_S5_MetadataValidation.xlsx
"""
from __future__ import annotations

import argparse
import io
import sys
import textwrap
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_distance_matrix(results_dir: Path) -> pd.DataFrame:
    fp = results_dir / "kmer_js_distance.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing distance matrix: {fp}")
    df = pd.read_csv(fp, index_col=0)
    if df.shape[0] != df.shape[1]:
        raise ValueError("Distance matrix is not square.")
    if list(df.index) != list(df.columns):
        df = df.loc[df.index, df.index]
    return df


def load_geometry_node_metrics(geometry_dir: Path) -> pd.DataFrame:
    xlsx = geometry_dir / "Supplementary_Data_S2_Geometry.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(f"Missing workbook: {xlsx}")
    df = pd.read_excel(xlsx, sheet_name="node_metrics")
    if "sample" in df.columns:
        df = df.set_index("sample")
    elif str(df.columns[0]).startswith("Unnamed"):
        df = df.set_index(df.columns[0])
        df.index.name = "sample"
    df.index = df.index.astype(str)
    return df


def load_kmer_entropy(results_dir: Path) -> Optional[pd.Series]:
    xlsx = results_dir / "Supplementary_Data_S1_Preprocessing.xlsx"
    if not xlsx.exists():
        return None
    try:
        ent = pd.read_excel(xlsx, sheet_name="entropy")
        if "sample" not in ent.columns or "kmer_entropy" not in ent.columns:
            return None
        return ent.set_index("sample")["kmer_entropy"].astype(float)
    except Exception:
        return None


RUNINFO_URL = "https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi"


def fetch_runinfo(sample_ids: List[str], timeout: int = 30) -> pd.DataFrame:
    term = " OR ".join(sample_ids)
    params = {
        "save": "efetch",
        "db": "sra",
        "rettype": "runinfo",
        "term": term,
    }
    url = RUNINFO_URL + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(raw))
    if "Run" not in df.columns:
        raise ValueError("NCBI runinfo response did not include a 'Run' column.")
    df["Run"] = df["Run"].astype(str)
    return df


def infer_labels_from_runinfo(runinfo: pd.DataFrame) -> pd.Series:
    text_cols = [
        c for c in ["Title", "Experiment", "SampleName", "LibraryName", "LibraryStrategy", "LibrarySelection"]
        if c in runinfo.columns
    ]
    if not text_cols:
        text_cols = list(runinfo.columns)

    def label_row(row: pd.Series) -> str:
        txt = " ".join([str(row.get(c, "")) for c in text_cols]).lower()
        if "pool-gbs" in txt or "pool gbs" in txt:
            return "pool"
        if "individual-gbs" in txt or "individual gbs" in txt:
            return "individual"
        if ("pool" in txt) and ("gbs" in txt):
            return "pool"
        if ("individual" in txt) and ("gbs" in txt):
            return "individual"
        return "unknown"

    out = pd.Series(runinfo.apply(label_row, axis=1).values, index=runinfo["Run"].astype(str), name="ground_truth")
    out = out.groupby(level=0).agg(
        lambda x: "pool" if "pool" in set(x) else ("individual" if "individual" in set(x) else "unknown")
    )
    return out


def read_pools_file(path: Path) -> List[str]:
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def build_ground_truth(
    samples: List[str],
    pools_list: Optional[List[str]] = None,
    runinfo_labels: Optional[pd.Series] = None,
) -> pd.Series:
    if runinfo_labels is not None:
        gt = pd.Series(index=samples, dtype=object)
        for sample in samples:
            gt.loc[sample] = runinfo_labels.get(sample, "unknown")
        gt.name = "ground_truth"
        return gt
    if pools_list:
        pools_set = {x.strip() for x in pools_list}
        return pd.Series(["pool" if s in pools_set else "individual" for s in samples], index=samples, name="ground_truth")
    return pd.Series(["unknown"] * len(samples), index=samples, name="ground_truth")


def evaluate_scores(scores: pd.Series, gt: pd.Series, topk: Optional[int] = None) -> Dict[str, float]:
    df = pd.DataFrame({"score": scores, "gt": gt}).dropna()
    df = df[df["gt"].isin(["pool", "individual"])].copy()
    if df.empty or df["gt"].nunique() < 2:
        return {"n_used": float(len(df))}

    y_true = (df["gt"] == "pool").astype(int).values
    y_score = df["score"].astype(float).values

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")

    try:
        ap = average_precision_score(y_true, y_score)
    except Exception:
        ap = float("nan")

    P = int(y_true.sum())
    if topk is None:
        topk = P if P > 0 else max(1, len(df) // 3)

    order = np.argsort(-y_score)
    y_pred = np.zeros_like(y_true)
    y_pred[order[:topk]] = 1

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    _, p_fisher = fisher_exact([[tp, fn], [fp, tn]], alternative="greater")
    N = len(y_true)
    M = P
    n = int(topk)
    k = tp
    p_hyper = float(hypergeom.sf(k - 1, N, M, n)) if (N > 0 and M >= 0 and n >= 0) else float("nan")

    return {
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


def build_confusion_at_topk(scores: pd.Series, gt: pd.Series, topk: int) -> pd.DataFrame:
    df = pd.DataFrame({"score": scores, "gt": gt}).dropna()
    df = df[df["gt"].isin(["pool", "individual"])].sort_values("score", ascending=False)
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
    return pd.DataFrame(cm, index=["TruePool", "TrueIndividual"], columns=["PredPool", "PredIndividual"])


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__),
    )
    ap.add_argument("--results_dir", required=True, help="Base results directory from preprocessing")
    ap.add_argument("--geometry_dir", required=True, help="Output directory from 02_run_geometry_analysis.py")
    ap.add_argument("--outdir", required=True, help="Output directory for validation workbook")
    ap.add_argument("--pools", default="", help="Comma-separated sample IDs to treat as positive labels")
    ap.add_argument("--pools_file", default="", help="Text file with one positive sample ID per line")
    ap.add_argument("--infer_from_ncbi", action="store_true", help="Query NCBI runinfo and infer labels from titles")
    ap.add_argument("--topk", type=int, default=0, help="Decision threshold: topK by score called positive")
    ap.add_argument("--also_entropy", action="store_true", help="Also evaluate k-mer entropy if available")
    return ap.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    geometry_dir = Path(args.geometry_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    dist = read_distance_matrix(results_dir)
    samples = list(dist.index.astype(str))

    node_metrics = load_geometry_node_metrics(geometry_dir)
    if "mixture_score" not in node_metrics.columns:
        raise ValueError("node_metrics sheet does not include 'mixture_score'.")
    mixture_score = node_metrics["mixture_score"].reindex(samples).astype(float)

    pools_list: List[str] = []
    if args.pools_file:
        pools_list = read_pools_file(Path(args.pools_file))
    elif args.pools.strip():
        pools_list = [x.strip() for x in args.pools.split(",") if x.strip()]

    runinfo_df = None
    runinfo_labels = None
    if args.infer_from_ncbi:
        try:
            runinfo_df = fetch_runinfo(samples)
            runinfo_labels = infer_labels_from_runinfo(runinfo_df)
        except Exception as exc:
            print(f"[WARN] NCBI runinfo inference failed: {exc}", file=sys.stderr)
            print("[WARN] Falling back to manual labels or unknown labels.", file=sys.stderr)

    gt = build_ground_truth(samples, pools_list=pools_list if pools_list else None, runinfo_labels=runinfo_labels)

    topk = args.topk if args.topk and args.topk > 0 else None
    metrics_mix = evaluate_scores(mixture_score, gt, topk=topk)

    if topk is None:
        P = int((gt == "pool").sum())
        topk_used = P if P > 0 else min(5, len(samples))
    else:
        topk_used = int(topk)

    cm_df = build_confusion_at_topk(mixture_score, gt, topk=topk_used)

    metrics_ent = None
    ent = None
    if args.also_entropy:
        ent = load_kmer_entropy(results_dir)
        if ent is not None:
            ent = ent.reindex(samples).astype(float)
            metrics_ent = evaluate_scores(ent, gt, topk=topk)

    sample_table = pd.DataFrame(
        {
            "sample": samples,
            "ground_truth": gt.values,
            "mixture_score": mixture_score.values,
        }
    )
    if ent is not None:
        sample_table["kmer_entropy"] = ent.values

    params_df = pd.DataFrame(
        [
            {
                "results_dir": str(results_dir),
                "geometry_dir": str(geometry_dir),
                "outdir": str(outdir),
                "topk_used": topk_used,
                "infer_from_ncbi": bool(args.infer_from_ncbi),
                "manual_positive_count": len(pools_list),
                "also_entropy": bool(args.also_entropy),
            }
        ]
    )

    excel_path = outdir / "Supplementary_Data_S5_MetadataValidation.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as xw:
        params_df.to_excel(xw, sheet_name="run_params", index=False)
        sample_table.to_excel(xw, sheet_name="sample_labels_scores", index=False)
        pd.DataFrame([metrics_mix]).to_excel(xw, sheet_name="metrics_mixture_score", index=False)
        if metrics_ent is not None:
            pd.DataFrame([metrics_ent]).to_excel(xw, sheet_name="metrics_kmer_entropy", index=False)
        if cm_df is not None and not cm_df.empty:
            cm_df.to_excel(xw, sheet_name="confusion_topk")
        if runinfo_df is not None:
            runinfo_df.to_excel(xw, sheet_name="ncbi_runinfo", index=False)

    print("\n[DONE] Metadata validation complete.")
    print(f"  -> Workbook: {excel_path}")
    if "roc_auc" in metrics_mix:
        print(f"  -> Mixture score ROC AUC: {metrics_mix['roc_auc']}")
        print(f"  -> Mixture score AP: {metrics_mix['avg_precision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
