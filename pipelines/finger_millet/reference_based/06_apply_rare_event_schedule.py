# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from crossfit_common import SCORE_VARIANTS, ensure_dir, write_json
from scipy.stats import rankdata


def split_ids(value: object) -> list[str]:
    return [x.strip() for x in str(value).split(";") if x.strip()]


def any_top_k_chance(n: int, m: int, k: int) -> float:
    if m <= 0 or n <= 0:
        return float("nan")
    k = min(k, n)
    if n - m < k:
        return 1.0
    return float(1.0 - math.comb(n - m, k) / math.comb(n, k))


def rank_percentile_high(values: Sequence[float]) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).to_numpy(float)



def evaluate_binary_fast(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    prevalence = float(np.mean(y)) if len(y) else float("nan")
    if len(np.unique(y)) < 2:
        return {
            "n": int(len(y)),
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "best_f1": float("nan"),
            "best_threshold": float("nan"),
            "prevalence": prevalence,
        }
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    ss = score[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    total_pos = int(tp[-1])
    # A threshold changes only at the last observation in a tied score group.
    boundaries = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    tpb = tp[boundaries].astype(float)
    fpb = fp[boundaries].astype(float)
    fn = total_pos - tpb
    denom = 2.0 * tpb + fpb + fn
    f1 = np.divide(2.0 * tpb, denom, out=np.zeros_like(tpb), where=denom > 0)
    best_idx = int(np.argmax(f1))
    # Exact ROC AUC from average ranks (Mann-Whitney U), including ties.
    average_ranks = rankdata(score, method="average")
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    auc = (float(average_ranks[y == 1].sum()) - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)

    # Non-interpolated average precision at distinct score thresholds.
    precision = tpb / np.maximum(tpb + fpb, 1.0)
    recall = tpb / float(total_pos)
    recall_previous = np.r_[0.0, recall[:-1]]
    average_precision = float(np.sum((recall - recall_previous) * precision))

    return {
        "n": int(len(y)),
        "roc_auc": float(auc),
        "average_precision": average_precision,
        "best_f1": float(f1[best_idx]),
        "best_threshold": float(ss[boundaries[best_idx]]),
        "prevalence": prevalence,
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--crossfit_scores", required=True)
    ap.add_argument("--crossfit_pass", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    if not Path(args.crossfit_pass).is_file():
        raise SystemExit("[ERROR] Cross-fit scoring PASS marker is absent")
    outdir = ensure_dir(args.outdir)
    schedule = pd.read_csv(args.schedule, sep="\t")
    scores = pd.read_csv(args.crossfit_scores, sep="\t")
    if len(schedule) != 735 or len(scores) != 560:
        raise SystemExit(f"Expected 735 schedule rows and 560 score rows; got {len(schedule)} and {len(scores)}")
    if scores["sample_id"].duplicated().any():
        raise SystemExit("Cross-fit score table contains duplicated sample IDs")
    score_index = scores.set_index("sample_id", drop=False)

    graph_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for graph_index, row in enumerate(schedule.itertuples(index=False), start=1):
        controls = split_ids(row.control_sample_ids)
        mixtures = split_ids(row.mixture_sample_ids)
        ids = controls + mixtures
        missing = [sample_id for sample_id in ids if sample_id not in score_index.index]
        if missing:
            raise KeyError(f"Graph {row.graph_id} contains samples missing from cross-fit scores: {missing[:5]}")
        frame = score_index.loc[ids].copy().reset_index(drop=True)
        y = frame["sample_id"].isin(mixtures).astype(int).to_numpy()
        n = len(frame)
        m = len(mixtures)
        for variant in SCORE_VARIANTS:
            values = pd.to_numeric(frame[variant], errors="coerce").to_numpy(float)
            metrics = evaluate_binary_fast(y, values)
            ranks = rankdata(values, method="average") / float(len(values))
            descending_ranks = rankdata(-values, method="min").astype(int)
            order = np.argsort(-values, kind="mergesort")
            top1_ids = set(frame.iloc[order[:1]]["sample_id"].astype(str))
            top3_ids = set(frame.iloc[order[:min(3,n)]]["sample_id"].astype(str))
            top5_ids = set(frame.iloc[order[:min(5,n)]]["sample_id"].astype(str))
            graph_rows.append({
                "replicate": int(row.replicate),
                "injection_count": int(row.injection_count),
                "graph_id": str(row.graph_id),
                "graph_n": int(n),
                "score_name": variant,
                **metrics,
                "ap_lift_over_prevalence": float(metrics["average_precision"] / metrics["prevalence"]) if metrics["prevalence"] > 0 else float("nan"),
                "any_mixture_top1": float(bool(top1_ids & set(mixtures))),
                "any_mixture_top3": float(bool(top3_ids & set(mixtures))),
                "any_mixture_top5": float(bool(top5_ids & set(mixtures))),
                "chance_top1": float(m / n),
                "chance_any_top3": any_top_k_chance(n,m,3),
                "chance_any_top5": any_top_k_chance(n,m,5),
            })
            for i, sample_id in enumerate(frame["sample_id"].astype(str)):
                if sample_id in mixtures:
                    rank_rows.append({
                        "replicate": int(row.replicate),
                        "injection_count": int(row.injection_count),
                        "graph_id": str(row.graph_id),
                        "score_name": variant,
                        "sample_id": sample_id,
                        "score": float(values[i]),
                        "rank_percentile": float(ranks[i]),
                        "rank_descending": int(descending_ranks[i]),
                    })
        if graph_index == 1 or graph_index % 50 == 0 or graph_index == len(schedule):
            print(f"[REFERENCE RARE {graph_index}/{len(schedule)}] {row.graph_id}", flush=True)

    graph_df = pd.DataFrame(graph_rows)
    rank_df = pd.DataFrame(rank_rows)
    graph_df.to_csv(outdir / "reference_rare_event_graph_metrics.tsv", sep="\t", index=False)
    rank_df.to_csv(outdir / "reference_rare_event_mixture_rank_metrics.tsv", sep="\t", index=False)

    replicate_rows: list[dict[str, Any]] = []
    for (replicate, injection_count, score_name), group in graph_df.groupby(["replicate","injection_count","score_name"], sort=True):
        rank_group = rank_df[
            (rank_df["replicate"] == replicate)
            & (rank_df["injection_count"] == injection_count)
            & (rank_df["score_name"] == score_name)
        ]
        replicate_rows.append({
            "replicate": int(replicate),
            "injection_count": int(injection_count),
            "score_name": str(score_name),
            "graph_count": int(len(group)),
            "roc_auc_mean": float(group["roc_auc"].mean()),
            "average_precision_mean": float(group["average_precision"].mean()),
            "prevalence_mean": float(group["prevalence"].mean()),
            "ap_lift_over_prevalence_mean": float(group["ap_lift_over_prevalence"].mean()),
            "top1_capture_rate": float(group["any_mixture_top1"].mean()),
            "top3_capture_rate": float(group["any_mixture_top3"].mean()),
            "top5_capture_rate": float(group["any_mixture_top5"].mean()),
            "chance_top1": float(group["chance_top1"].mean()),
            "chance_any_top3": float(group["chance_any_top3"].mean()),
            "chance_any_top5": float(group["chance_any_top5"].mean()),
            "mean_mixture_rank_percentile": float(rank_group["rank_percentile"].mean()),
        })
    replicate_df = pd.DataFrame(replicate_rows)
    replicate_df.to_csv(outdir / "reference_rare_event_replicate_summary.tsv", sep="\t", index=False)

    summary_rows: list[dict[str, Any]] = []
    for (injection_count, score_name), group in replicate_df.groupby(["injection_count","score_name"], sort=True):
        summary_rows.append({
            "injection_count": int(injection_count),
            "score_name": str(score_name),
            "replicate_count": int(group["replicate"].nunique()),
            "roc_auc_mean": float(group["roc_auc_mean"].mean()),
            "roc_auc_sd": float(group["roc_auc_mean"].std(ddof=0)),
            "average_precision_mean": float(group["average_precision_mean"].mean()),
            "ap_lift_over_prevalence_mean": float(group["ap_lift_over_prevalence_mean"].mean()),
            "top1_capture_rate": float(group["top1_capture_rate"].mean()),
            "top3_capture_rate": float(group["top3_capture_rate"].mean()),
            "top5_capture_rate": float(group["top5_capture_rate"].mean()),
            "chance_top1": float(group["chance_top1"].mean()),
            "chance_any_top3": float(group["chance_any_top3"].mean()),
            "chance_any_top5": float(group["chance_any_top5"].mean()),
            "mean_mixture_rank_percentile": float(group["mean_mixture_rank_percentile"].mean()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "reference_rare_event_summary.tsv", sep="\t", index=False)
    write_json({
        "status":"PASS",
        "schedule_rows":int(len(schedule)),
        "graph_metric_rows":int(len(graph_df)),
        "mixture_rank_rows":int(len(rank_df)),
        "score_variants":SCORE_VARIANTS,
    }, outdir / "reference_rare_event_audit.json")
    (outdir / "REFERENCE_RARE_EVENT_PASS.txt").write_text("PASS\n", encoding="utf-8")
    print(f"[DONE] Reference-based rare-event schedule applied: graphs={len(schedule)}, score variants={len(SCORE_VARIANTS)}")


if __name__ == "__main__":
    main()
