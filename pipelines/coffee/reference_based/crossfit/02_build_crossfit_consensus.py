# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from crossfit_common import (
    SCORE_VARIANTS,
    ensure_dir,
    evaluate_binary,
    jaccard_top_n,
    spearman_safe,
    write_json,
)


GENERATED_META = [
    "sample_id",
    "replicate",
    "class_label",
    "scenario",
    "pattern_id",
    "n_parents",
    "parents",
    "actual_entropy_norm",
    "actual_minor_fraction",
    "is_mixture",
]
REAL_META = [
    "sample_id",
    "class_label",
    "source_role",
    "label_source",
    "include_primary",
    "is_metadata_pool_candidate",
]


def consensus_table(
    frame: pd.DataFrame,
    meta_columns: List[str],
    variants: List[str],
) -> pd.DataFrame:
    available_meta = [c for c in meta_columns if c in frame.columns]
    first = (
        frame.sort_values("reference_id")
        .drop_duplicates(subset=["sample_id"] + (["replicate"] if "replicate" in available_meta else []))
        [available_meta]
        .copy()
    )
    index_cols = ["sample_id"] + (["replicate"] if "replicate" in available_meta else [])
    result = first.set_index(index_cols)

    for variant in variants:
        pivot = frame.pivot_table(
            index=index_cols,
            columns="reference_id",
            values=variant,
            aggfunc="first",
        )
        pivot.columns = [f"{variant}_{c}" for c in pivot.columns]
        reference_columns = list(pivot.columns)
        pivot[f"{variant}_consensus"] = pivot[reference_columns].mean(axis=1)
        result = result.join(pivot, how="left")
    return result.reset_index()


def load_real_tms(path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    candidates = [
        name for name in xls.sheet_names if "mixture" in name.lower() or "ranking" in name.lower()
    ]
    if not candidates:
        raise ValueError(f"No mixture/ranking sheet found in {path}")
    frame = pd.read_excel(path, sheet_name=candidates[0])
    sample_col = next((c for c in frame.columns if str(c).lower() in {"sample", "sample_id"}), None)
    score_col = next(
        (
            c
            for c in frame.columns
            if str(c).lower() in {"mixture_score", "tms", "score"}
        ),
        None,
    )
    if sample_col is None or score_col is None:
        raise ValueError(f"Cannot identify sample/score columns in {path}")
    return frame[[sample_col, score_col]].rename(
        columns={sample_col: "sample_id", score_col: "real_tms"}
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--tms_node_scores", required=True)
    ap.add_argument("--baseline_s2", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--nominal_generated_consensus", default="")
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    outdir = ensure_dir(args.outdir)

    generated = pd.read_csv(
        scores_dir / "crossfit_generated_reference_scores.tsv", sep="\t"
    )
    real = pd.read_csv(scores_dir / "crossfit_real_reference_scores.tsv", sep="\t")

    generated_consensus = consensus_table(generated, GENERATED_META, SCORE_VARIANTS)
    real_consensus = consensus_table(real, REAL_META, SCORE_VARIANTS)

    generated_consensus.to_csv(
        outdir / "crossfit_generated_consensus_scores.tsv", sep="\t", index=False
    )
    real_consensus.to_csv(
        outdir / "crossfit_real_consensus_scores.tsv", sep="\t", index=False
    )

    metric_rows: list[dict[str, object]] = []
    for replicate, group in generated_consensus.groupby("replicate"):
        y = group["is_mixture"].astype(int).to_numpy()
        for variant in SCORE_VARIANTS:
            score_col = variant + "_consensus"
            metric_rows.append(
                {
                    "evaluation_scope": "per_replicate",
                    "replicate": int(replicate),
                    "score_variant": variant,
                    **evaluate_binary(y, group[score_col].to_numpy(dtype=float)),
                }
            )
    y_all = generated_consensus["is_mixture"].astype(int).to_numpy()
    for variant in SCORE_VARIANTS:
        metric_rows.append(
            {
                "evaluation_scope": "pooled_descriptive",
                "replicate": 0,
                "score_variant": variant,
                **evaluate_binary(
                    y_all,
                    generated_consensus[variant + "_consensus"].to_numpy(dtype=float),
                ),
            }
        )
    consensus_metrics = pd.DataFrame(metric_rows)
    consensus_metrics.to_csv(
        outdir / "crossfit_generated_consensus_metrics.tsv", sep="\t", index=False
    )

    # Descriptive evaluation against source metadata labels. These labels are
    # not used in model fitting and are not treated as definitive ground truth.
    real_metric_rows: list[dict[str, object]] = []
    y_real = real_consensus["is_metadata_pool_candidate"].astype(int).to_numpy()
    for variant in SCORE_VARIANTS:
        real_metric_rows.append(
            {
                "score_variant": variant,
                "label_interpretation": "descriptive_metadata_holdout",
                **evaluate_binary(
                    y_real, real_consensus[variant + "_consensus"].to_numpy(dtype=float)
                ),
            }
        )
    pd.DataFrame(real_metric_rows).to_csv(
        outdir / "crossfit_real_metadata_metrics_descriptive.tsv",
        sep="\t",
        index=False,
    )

    # Compare generated scores with the paired-read TMS on exactly the same
    # held-out replicate/sample rows.
    tms = pd.read_csv(args.tms_node_scores, sep="\t")
    tms = tms[
        (tms["scenario"].astype(str) == "primary")
        & (tms["analysis_mode"].astype(str) == "paired")
    ].copy()
    tms = tms[
        [
            "sample_id",
            "replicate",
            "tms",
            "betweenness",
            "negative_orc_incidence",
        ]
    ]
    generated_vs_tms = generated_consensus.merge(
        tms, on=["sample_id", "replicate"], how="inner", validate="one_to_one"
    )
    generated_vs_tms.to_csv(
        outdir / "crossfit_generated_consensus_and_tms.tsv", sep="\t", index=False
    )
    concordance_rows: list[dict[str, object]] = []
    for replicate, group in generated_vs_tms.groupby("replicate"):
        for variant in [
            "reference_qc_crossfit",
            "reference_qc_crossfit_no_pca",
            "reference_mapping_crossfit",
            "pca_only_crossfit",
        ]:
            score_col = variant + "_consensus"
            rho, p = spearman_safe(group[score_col], group["tms"])
            jac = jaccard_top_n(group, group, score_col, "tms", n=5)
            concordance_rows.append(
                {
                    "replicate": int(replicate),
                    "score_variant": variant,
                    "spearman_rho": rho,
                    "spearman_p": p,
                    "top5_jaccard": jac,
                    "n": int(len(group)),
                }
            )
    pd.DataFrame(concordance_rows).to_csv(
        outdir / "crossfit_tms_generated_concordance.tsv", sep="\t", index=False
    )

    # Real-data ranking comparison with the submitted reference-free TMS.
    real_tms = load_real_tms(args.baseline_s2)
    real_vs_tms = real_consensus.merge(
        real_tms, on="sample_id", how="inner", validate="one_to_one"
    )
    real_vs_tms.to_csv(
        outdir / "crossfit_real_consensus_and_tms.tsv", sep="\t", index=False
    )
    real_concordance: dict[str, object] = {"n": int(len(real_vs_tms))}
    for variant in [
        "reference_qc_crossfit",
        "reference_qc_crossfit_no_pca",
        "reference_mapping_crossfit",
        "pca_only_crossfit",
    ]:
        score_col = variant + "_consensus"
        rho, p = spearman_safe(real_vs_tms[score_col], real_vs_tms["real_tms"])
        jac = jaccard_top_n(real_vs_tms, real_vs_tms, score_col, "real_tms", n=5)
        real_concordance[variant] = {
            "spearman_rho": rho,
            "spearman_p": p,
            "top5_jaccard": jac,
            "top5_reference": real_vs_tms.nlargest(5, score_col)["sample_id"].tolist(),
            "top5_tms": real_vs_tms.nlargest(5, "real_tms")["sample_id"].tolist(),
        }
    write_json(real_concordance, outdir / "crossfit_real_rank_concordance.json")

    # Optional direct comparison with the previous nominal reference-QC score.
    nominal_rows: list[dict[str, object]] = []
    if args.nominal_generated_consensus:
        nominal_path = Path(args.nominal_generated_consensus)
        if nominal_path.is_file():
            nominal = pd.read_csv(nominal_path, sep="\t")
            nominal_col = next(
                (
                    c
                    for c in [
                        "reference_qc_consensus",
                        "reference_qc_score",
                    ]
                    if c in nominal.columns
                ),
                None,
            )
            if nominal_col:
                merged = generated_consensus.merge(
                    nominal[["sample_id", "replicate", nominal_col]],
                    on=["sample_id", "replicate"],
                    how="inner",
                    validate="one_to_one",
                )
                for replicate, group in merged.groupby("replicate"):
                    y = group["is_mixture"].astype(int).to_numpy()
                    nominal_metric = evaluate_binary(y, group[nominal_col])
                    full_metric = evaluate_binary(
                        y, group["reference_qc_crossfit_consensus"]
                    )
                    no_pca_metric = evaluate_binary(
                        y, group["reference_qc_crossfit_no_pca_consensus"]
                    )
                    nominal_rows.append(
                        {
                            "replicate": int(replicate),
                            "nominal_auc": nominal_metric["roc_auc"],
                            "crossfit_full_auc": full_metric["roc_auc"],
                            "crossfit_no_pca_auc": no_pca_metric["roc_auc"],
                            "nominal_minus_crossfit_full": nominal_metric["roc_auc"]
                            - full_metric["roc_auc"],
                        }
                    )
                pd.DataFrame(nominal_rows).to_csv(
                    outdir / "nominal_vs_crossfit_auc.tsv",
                    sep="\t",
                    index=False,
                )

    write_json(
        {
            "status": "PASS",
            "generated_consensus_rows": int(len(generated_consensus)),
            "real_consensus_rows": int(len(real_consensus)),
            "generated_tms_overlap_rows": int(len(generated_vs_tms)),
            "real_tms_overlap_rows": int(len(real_vs_tms)),
            "reference_count": int(generated["reference_id"].nunique()),
            "replicate_count": int(generated_consensus["replicate"].nunique()),
            "real_rank_concordance": real_concordance,
            "nominal_comparison_rows": nominal_rows,
        },
        outdir / "crossfit_comparison_master.json",
    )
    print("[DONE] Cross-fitted consensus and comparison tables complete.")


if __name__ == "__main__":
    main()
