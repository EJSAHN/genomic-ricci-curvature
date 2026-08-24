# -*- coding: utf-8 -*-
"""
10_summarize_score_components.py
Summarize score-component performance on synthetic mixtures.

Uses Supplementary_Data_S1_Submission.xlsx:
  - Synthetic_Node_Scores: expects sample, label, betweenness, neg_orc_incidence, mixture_score_ot
  - Synthetic_Truth: expects sample, entropy_norm (optional but recommended)

Outputs:
  - Score_Component_Summary.xlsx
  - Score_Component_Summary.csv

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.stats import spearmanr


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd == 0.0:
        return x * 0.0
    return (x - mu) / sd


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan"), float("nan")
    uniq = np.unique(scores)
    best_f1 = -1.0
    best_t = float("nan")
    for t in uniq:
        y_pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_f1, best_t


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, entropy: np.ndarray | None) -> Dict[str, float]:
    m = np.isfinite(scores) & np.isfinite(y_true)
    y = y_true[m].astype(int)
    s = scores[m].astype(float)
    out: Dict[str, float] = {"n": int(len(y))}
    if len(np.unique(y)) < 2 or len(y) < 3:
        out.update({"auc": float("nan"), "ap": float("nan"), "best_f1": float("nan"), "best_thr": float("nan"),
                    "spearman_rho_entropy": float("nan"), "spearman_p_entropy": float("nan")})
        return out

    out["auc"] = float(roc_auc_score(y, s))
    out["ap"] = float(average_precision_score(y, s))
    bf1, thr = best_f1_threshold(y, s)
    out["best_f1"] = float(bf1)
    out["best_thr"] = float(thr)

    if entropy is not None and np.isfinite(entropy[m]).sum() >= 3:
        rho, p = spearmanr(s, entropy[m].astype(float))
        out["spearman_rho_entropy"] = float(rho)
        out["spearman_p_entropy"] = float(p)
    else:
        out["spearman_rho_entropy"] = float("nan")
        out["spearman_p_entropy"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True, help="Path to Supplementary_Data_S1_Submission.xlsx")
    ap.add_argument("--scores_sheet", default="Synthetic_Node_Scores")
    ap.add_argument("--truth_sheet", default="Synthetic_Truth")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    s1 = Path(args.s1).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_excel(s1, sheet_name=args.scores_sheet)
    truth = pd.read_excel(s1, sheet_name=args.truth_sheet)

    # Required columns
    req_scores = {"sample", "label", "betweenness", "neg_orc_incidence", "mixture_score_ot"}
    missing = req_scores - set(scores.columns)
    if missing:
        raise SystemExit(f"[ERR] Synthetic_Node_Scores missing columns: {sorted(missing)}")

    # entropy optional
    entropy = None
    if "entropy_norm" in truth.columns:
        # merge entropy onto scores by sample
        merged = scores.merge(truth[["sample", "entropy_norm"]], on="sample", how="left")
        entropy = pd.to_numeric(merged["entropy_norm"], errors="coerce").to_numpy()
    else:
        merged = scores.copy()
        entropy = None

    # Label handling: use scores['label'] (already provided in your S1)
    lab = merged["label"]
    if lab.dtype == bool:
        y = lab.astype(int).to_numpy()
    elif lab.dtype == object:
        y = lab.astype(str).str.lower().str.contains("mix|pool|true|1").astype(int).to_numpy()
    else:
        y = (pd.to_numeric(lab, errors="coerce").fillna(0) > 0).astype(int).to_numpy()

    # Components
    betw = pd.to_numeric(merged["betweenness"], errors="coerce")
    negorc = pd.to_numeric(merged["neg_orc_incidence"], errors="coerce")
    full = pd.to_numeric(merged["mixture_score_ot"], errors="coerce")  # this is your shipped synthetic score

    rows = []

    def add(model_name: str, s: pd.Series):
        m = compute_metrics(y, s.to_numpy(dtype=float), entropy)
        rows.append({"model": model_name, **m})

    # Ablation set
    add("Full synthetic score (mixture_score_ot)", full)
    add("Betweenness only (Z)", zscore(betw))
    add("NegORC only (Z)", zscore(negorc))
    add("Z(Betweenness) + Z(NegORC)", zscore(betw) + zscore(negorc))
    add("Betweenness + NegORC (raw sum)", betw + negorc)

    out = pd.DataFrame(rows)

    diag = pd.DataFrame([{
        "n_rows": int(len(merged)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "has_entropy_norm": bool(entropy is not None),
        "scores_sheet": args.scores_sheet,
        "truth_sheet": args.truth_sheet
    }])

    out_xlsx = outdir / "Score_Component_Summary.xlsx"
    out_csv = outdir / "Score_Component_Summary.csv"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="ablation", index=False)
        diag.to_excel(xw, sheet_name="diagnostics", index=False)
        preview_cols = ["sample", "label", "betweenness", "neg_orc_incidence", "mixture_score_ot"]
        if entropy is not None:
            merged["entropy_norm"] = pd.to_numeric(merged.get("entropy_norm"), errors="coerce")
            preview_cols.append("entropy_norm")
        merged[preview_cols].head(50).to_excel(xw, sheet_name="preview_head50", index=False)

    out.to_csv(out_csv, index=False)

    print("[DONE]", out_xlsx)
    print("[DONE]", out_csv)
    print("[INFO] rows:", len(merged), "pos:", int((y == 1).sum()), "neg:", int((y == 0).sum()))


if __name__ == "__main__":
    main()