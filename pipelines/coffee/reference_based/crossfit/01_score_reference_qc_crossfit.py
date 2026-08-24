# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from crossfit_common import (
    SCORE_VARIANTS,
    calibrate_and_score,
    ensure_dir,
    evaluate_binary,
    fit_marker_model,
    load_matrix,
    score_raw_features,
    safe_source_ids,
    stable_hash,
    write_json,
)


def control_mask(samples: pd.DataFrame) -> np.ndarray:
    return samples["class_label"].astype(str).to_numpy() == "single_source_control"


def build_inner_reference(
    *,
    X: np.ndarray,
    samples: pd.DataFrame,
    mapping: pd.DataFrame,
    training_replicates: List[int],
    min_control_call_rate: float,
    max_features: int,
    max_pca_components: int,
    min_overlap: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    reps = sorted(int(x) for x in training_replicates)
    controls = control_mask(samples)
    rep_values = samples["replicate"].astype(int).to_numpy()

    for calibration_rep in reps:
        inner_fit_reps = [r for r in reps if r != calibration_rep]
        fit_mask = controls & np.isin(rep_values, inner_fit_reps)
        calibration_mask = controls & (rep_values == calibration_rep)
        if int(fit_mask.sum()) == 0 or int(calibration_mask.sum()) == 0:
            raise RuntimeError(
                f"Invalid inner split: fit={int(fit_mask.sum())}, "
                f"calibration={int(calibration_mask.sum())}"
            )
        model = fit_marker_model(
            training_matrix_full=X[fit_mask],
            training_samples=samples.loc[fit_mask].reset_index(drop=True),
            min_control_call_rate=min_control_call_rate,
            max_features=max_features,
            max_pca_components=max_pca_components,
            min_overlap=min_overlap,
        )
        raw = score_raw_features(
            model=model,
            target_matrix_full=X[calibration_mask],
            target_samples=samples.loc[calibration_mask].reset_index(drop=True),
            target_mapping=mapping,
            min_overlap=min_overlap,
        )
        raw["inner_calibration_replicate"] = int(calibration_rep)
        raw["inner_fit_replicates"] = ",".join(str(x) for x in inner_fit_reps)
        raw["inner_selected_marker_count"] = int(len(model.selected))
        raw["inner_selected_marker_hash"] = model.selected_hash
        parts.append(raw)

        fit_ids = samples.loc[fit_mask, "sample_id"].astype(str).tolist()
        cal_ids = samples.loc[calibration_mask, "sample_id"].astype(str).tolist()
        overlap = sorted(set(fit_ids) & set(cal_ids))
        audit_rows.append(
            {
                "inner_calibration_replicate": int(calibration_rep),
                "inner_fit_replicates": ",".join(str(x) for x in inner_fit_reps),
                "inner_fit_control_count": int(fit_mask.sum()),
                "inner_calibration_control_count": int(calibration_mask.sum()),
                "inner_train_test_overlap_count": len(overlap),
                "inner_training_sample_hash": stable_hash(fit_ids),
                "inner_calibration_sample_hash": stable_hash(cal_ids),
                "inner_selected_marker_count": int(len(model.selected)),
                "inner_selected_marker_hash": model.selected_hash,
                "inner_pca_components": int(model.pca_components),
                "inner_source_centroid_count": int(len(model.source_names)),
            }
        )

    reference = pd.concat(parts, ignore_index=True)
    return reference, audit_rows


def parent_recovery_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    any_recovery: list[float] = []
    exact_recovery: list[float] = []
    for row in out.itertuples(index=False):
        true_parents = {
            p.strip()
            for p in str(row.parents).replace(",", ";").split(";")
            if p.strip() and p.strip().lower() != "nan"
        }
        inferred = {
            str(row.best_pair_a).strip(),
            str(row.best_pair_b).strip(),
        } - {"", "nan"}
        is_mix = str(row.class_label) == "synthetic_mixture"
        if is_mix and true_parents:
            any_recovery.append(float(bool(true_parents & inferred)))
            exact_recovery.append(
                float(len(true_parents) == 2 and true_parents == inferred)
            )
        else:
            any_recovery.append(float("nan"))
            exact_recovery.append(float("nan"))
    out["best_pair_any_parent_recovered"] = any_recovery
    out["best_pair_exact_two_parent_recovery"] = exact_recovery
    return out


def score_generated_reference(
    *,
    reference_id: str,
    X: np.ndarray,
    samples: pd.DataFrame,
    mapping: pd.DataFrame,
    min_control_call_rate: float,
    max_features: int,
    max_pca_components: int,
    min_overlap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_scores: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    reps = sorted(samples["replicate"].astype(int).unique().tolist())
    controls = control_mask(samples)
    rep_values = samples["replicate"].astype(int).to_numpy()

    for test_rep in reps:
        train_reps = [int(r) for r in reps if int(r) != int(test_rep)]
        train_mask = controls & np.isin(rep_values, train_reps)
        test_mask = rep_values == int(test_rep)

        train_ids = samples.loc[train_mask, "sample_id"].astype(str).tolist()
        test_ids = samples.loc[test_mask, "sample_id"].astype(str).tolist()
        overlap = sorted(set(train_ids) & set(test_ids))
        if overlap:
            raise RuntimeError(
                f"{reference_id} fold {test_rep}: train/test overlap: {overlap[:5]}"
            )

        inner_reference, inner_audit = build_inner_reference(
            X=X,
            samples=samples,
            mapping=mapping,
            training_replicates=train_reps,
            min_control_call_rate=min_control_call_rate,
            max_features=max_features,
            max_pca_components=max_pca_components,
            min_overlap=min_overlap,
        )

        final_model = fit_marker_model(
            training_matrix_full=X[train_mask],
            training_samples=samples.loc[train_mask].reset_index(drop=True),
            min_control_call_rate=min_control_call_rate,
            max_features=max_features,
            max_pca_components=max_pca_components,
            min_overlap=min_overlap,
        )
        raw_test = score_raw_features(
            model=final_model,
            target_matrix_full=X[test_mask],
            target_samples=samples.loc[test_mask].reset_index(drop=True),
            target_mapping=mapping,
            min_overlap=min_overlap,
        )
        scored, scaling_meta = calibrate_and_score(inner_reference, raw_test)
        scored = parent_recovery_columns(scored)
        scored["reference_id"] = reference_id
        scored["outer_test_replicate"] = int(test_rep)
        scored["outer_training_replicates"] = ",".join(str(x) for x in train_reps)
        scored["outer_selected_marker_count"] = int(len(final_model.selected))
        scored["outer_selected_marker_hash"] = final_model.selected_hash
        scored["outer_pca_components"] = int(final_model.pca_components)
        scored["outer_source_centroid_count"] = int(len(final_model.source_names))
        scored["is_mixture"] = (
            scored["class_label"].astype(str) == "synthetic_mixture"
        ).astype(int)
        all_scores.append(scored)

        y = scored["is_mixture"].to_numpy(dtype=int)
        for variant in SCORE_VARIANTS:
            metric_rows.append(
                {
                    "reference_id": reference_id,
                    "replicate": int(test_rep),
                    "score_variant": variant,
                    **evaluate_binary(y, scored[variant].to_numpy(dtype=float)),
                }
            )

        provenance_rows.append(
            {
                "reference_id": reference_id,
                "outer_test_replicate": int(test_rep),
                "outer_training_replicates": ",".join(str(x) for x in train_reps),
                "outer_training_control_count": int(train_mask.sum()),
                "outer_test_sample_count": int(test_mask.sum()),
                "outer_test_control_count": int(
                    (
                        samples.loc[test_mask, "class_label"].astype(str)
                        == "single_source_control"
                    ).sum()
                ),
                "outer_test_mixture_count": int(
                    (
                        samples.loc[test_mask, "class_label"].astype(str)
                        == "synthetic_mixture"
                    ).sum()
                ),
                "outer_train_test_overlap_count": len(overlap),
                "outer_training_sample_hash": stable_hash(train_ids),
                "outer_test_sample_hash": stable_hash(test_ids),
                "outer_selected_marker_count": int(len(final_model.selected)),
                "outer_selected_marker_hash": final_model.selected_hash,
                "outer_pca_components": int(final_model.pca_components),
                "outer_source_centroid_count": int(len(final_model.source_names)),
                "inner_calibration_row_count": int(len(inner_reference)),
                "inner_fold_count": int(len(inner_audit)),
                "inner_min_selected_markers": int(
                    min(x["inner_selected_marker_count"] for x in inner_audit)
                ),
                "inner_max_selected_markers": int(
                    max(x["inner_selected_marker_count"] for x in inner_audit)
                ),
                "marker_selection_used_outer_test_rows": 0,
                "pca_fit_used_outer_test_rows": 0,
                "scaling_used_outer_test_rows": 0,
                "outer_test_labels_used_for_model_fitting": False,
                "training_control_role_used_for_model_fitting": True,
                "mixture_labels_used_for_parameter_tuning": False,
                "scaling_reference_metadata": json.dumps(
                    scaling_meta, sort_keys=True
                ),
                "inner_split_audit": json.dumps(inner_audit, sort_keys=True),
            }
        )
        full_auc = metric_rows[-len(SCORE_VARIANTS)]["roc_auc"]
        no_pca_auc = metric_rows[-len(SCORE_VARIANTS) + 1]["roc_auc"]
        print(
            f"[{reference_id} fold {test_rep}] "
            f"full AUC={full_auc:.3f}; no-PCA AUC={no_pca_auc:.3f}; "
            f"markers={len(final_model.selected)}"
        )

    return (
        pd.concat(all_scores, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.DataFrame(provenance_rows),
    )


def score_real_reference(
    *,
    reference_id: str,
    X_generated: np.ndarray,
    generated_samples: pd.DataFrame,
    generated_mapping: pd.DataFrame,
    X_real: np.ndarray,
    real_samples: pd.DataFrame,
    real_mapping: pd.DataFrame,
    min_control_call_rate: float,
    max_features: int,
    max_pca_components: int,
    min_overlap: int,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    controls = control_mask(generated_samples)
    all_reps = sorted(generated_samples["replicate"].astype(int).unique().tolist())
    train_ids = generated_samples.loc[controls, "sample_id"].astype(str).tolist()
    training_sources = sorted(
        set(safe_source_ids(generated_samples.loc[controls].reset_index(drop=True)))
    )
    real_ids = real_samples["sample_id"].astype(str).tolist()
    exact_overlap = sorted(set(train_ids) & set(real_ids))
    source_identity_overlap = sorted(set(training_sources) & set(real_ids))

    inner_reference, inner_audit = build_inner_reference(
        X=X_generated,
        samples=generated_samples,
        mapping=generated_mapping,
        training_replicates=[int(x) for x in all_reps],
        min_control_call_rate=min_control_call_rate,
        max_features=max_features,
        max_pca_components=max_pca_components,
        min_overlap=min_overlap,
    )
    final_model = fit_marker_model(
        training_matrix_full=X_generated[controls],
        training_samples=generated_samples.loc[controls].reset_index(drop=True),
        min_control_call_rate=min_control_call_rate,
        max_features=max_features,
        max_pca_components=max_pca_components,
        min_overlap=min_overlap,
    )
    raw_real = score_raw_features(
        model=final_model,
        target_matrix_full=X_real,
        target_samples=real_samples.reset_index(drop=True),
        target_mapping=real_mapping,
        min_overlap=min_overlap,
    )
    scored, scaling_meta = calibrate_and_score(inner_reference, raw_real)
    scored["reference_id"] = reference_id

    extra_meta = real_samples[
        [
            c
            for c in [
                "sample_id",
                "source_role",
                "label_source",
                "include_primary",
                "class_label",
            ]
            if c in real_samples.columns
        ]
    ].copy()
    extra_meta["sample_id"] = extra_meta["sample_id"].astype(str)
    # class_label already exists in scored; avoid a duplicate suffix.
    extra_meta = extra_meta.drop(columns=["class_label"], errors="ignore")
    scored = scored.merge(extra_meta, on="sample_id", how="left", validate="one_to_one")
    scored["is_metadata_pool_candidate"] = (
        pd.to_numeric(scored["include_primary"], errors="coerce").fillna(1).astype(int)
        == 0
    ).astype(int)

    for variant in SCORE_VARIANTS:
        scored[variant + "_real_rank"] = (
            scored[variant].rank(method="min", ascending=False).astype(int)
        )

    descriptive_metrics: dict[str, object] = {}
    y = scored["is_metadata_pool_candidate"].to_numpy(dtype=int)
    for variant in SCORE_VARIANTS:
        descriptive_metrics[variant] = evaluate_binary(
            y, scored[variant].to_numpy(dtype=float)
        )

    provenance = pd.DataFrame(
        [
            {
                "reference_id": reference_id,
                "real_training_generated_control_count": int(controls.sum()),
                "real_test_sample_count": int(len(real_samples)),
                "real_train_test_exact_id_overlap_count": len(exact_overlap),
                "real_training_source_identity_overlap_count": len(source_identity_overlap),
                "real_training_source_identities": ",".join(training_sources),
                "real_overlapping_source_identities": ",".join(source_identity_overlap),
                "real_training_sample_hash": stable_hash(train_ids),
                "real_test_sample_hash": stable_hash(real_ids),
                "real_selected_marker_count": int(len(final_model.selected)),
                "real_selected_marker_hash": final_model.selected_hash,
                "real_pca_components": int(final_model.pca_components),
                "real_source_centroid_count": int(len(final_model.source_names)),
                "real_inner_calibration_row_count": int(len(inner_reference)),
                "real_inner_fold_count": int(len(inner_audit)),
                "marker_selection_used_real_test_rows": 0,
                "pca_fit_used_real_test_rows": 0,
                "scaling_used_real_test_rows": 0,
                "metadata_pool_labels_used_for_model_fitting": False,
                "training_control_role_used_for_model_fitting": True,
                "metadata_pool_labels_used_only_for_descriptive_evaluation": True,
                "scaling_reference_metadata": json.dumps(
                    scaling_meta, sort_keys=True
                ),
                "inner_split_audit": json.dumps(inner_audit, sort_keys=True),
            }
        ]
    )
    summary = {
        "reference_id": reference_id,
        "selected_marker_count": int(len(final_model.selected)),
        "top5_full": scored.nlargest(5, "reference_qc_crossfit")[
            "sample_id"
        ].astype(str).tolist(),
        "top5_no_pca": scored.nlargest(5, "reference_qc_crossfit_no_pca")[
            "sample_id"
        ].astype(str).tolist(),
        "metadata_holdout_metrics_descriptive": descriptive_metrics,
    }
    return scored, summary, provenance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_manifest", required=True)
    ap.add_argument("--features_root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min_control_call_rate", type=float, default=0.30)
    ap.add_argument("--max_features", type=int, default=5000)
    ap.add_argument("--max_pca_components", type=int, default=5)
    ap.add_argument("--min_overlap", type=int, default=10)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    refs = pd.read_csv(args.reference_manifest, sep="\t")
    features_root = Path(args.features_root)

    generated_scores_all: list[pd.DataFrame] = []
    generated_metrics_all: list[pd.DataFrame] = []
    generated_provenance_all: list[pd.DataFrame] = []
    real_scores_all: list[pd.DataFrame] = []
    real_summaries: list[dict[str, object]] = []
    real_provenance_all: list[pd.DataFrame] = []

    for ref_row in refs.itertuples(index=False):
        reference_id = str(ref_row.reference_id)
        ref_features = features_root / reference_id
        print(f"\n=== Cross-fitted scoring: {reference_id} ===")
        Xg, dg, ids_g, sg = load_matrix(ref_features / "generated")
        Xr, dr, ids_r, sr = load_matrix(ref_features / "real")
        generated_mapping = pd.read_csv(
            ref_features / "generated_mapping_and_marker_metrics.tsv", sep="\t"
        )
        real_mapping = pd.read_csv(
            ref_features / "real_mapping_and_marker_metrics.tsv", sep="\t"
        )

        g_scores, g_metrics, g_provenance = score_generated_reference(
            reference_id=reference_id,
            X=Xg,
            samples=sg,
            mapping=generated_mapping,
            min_control_call_rate=args.min_control_call_rate,
            max_features=args.max_features,
            max_pca_components=args.max_pca_components,
            min_overlap=args.min_overlap,
        )
        generated_scores_all.append(g_scores)
        generated_metrics_all.append(g_metrics)
        generated_provenance_all.append(g_provenance)

        r_scores, r_summary, r_provenance = score_real_reference(
            reference_id=reference_id,
            X_generated=Xg,
            generated_samples=sg,
            generated_mapping=generated_mapping,
            X_real=Xr,
            real_samples=sr,
            real_mapping=real_mapping,
            min_control_call_rate=args.min_control_call_rate,
            max_features=args.max_features,
            max_pca_components=args.max_pca_components,
            min_overlap=args.min_overlap,
        )
        real_scores_all.append(r_scores)
        real_summaries.append(r_summary)
        real_provenance_all.append(r_provenance)
        print(
            f"[{reference_id} real] full top 5: {r_summary['top5_full']}"
        )
        print(
            f"[{reference_id} real] no-PCA top 5: {r_summary['top5_no_pca']}"
        )

    generated_scores = pd.concat(generated_scores_all, ignore_index=True)
    generated_metrics = pd.concat(generated_metrics_all, ignore_index=True)
    generated_provenance = pd.concat(generated_provenance_all, ignore_index=True)
    real_scores = pd.concat(real_scores_all, ignore_index=True)
    real_provenance = pd.concat(real_provenance_all, ignore_index=True)

    generated_scores.to_csv(
        outdir / "crossfit_generated_reference_scores.tsv", sep="\t", index=False
    )
    generated_metrics.to_csv(
        outdir / "crossfit_generated_reference_metrics.tsv", sep="\t", index=False
    )
    generated_provenance.to_csv(
        outdir / "crossfit_generated_fold_provenance.tsv", sep="\t", index=False
    )
    real_scores.to_csv(
        outdir / "crossfit_real_reference_scores.tsv", sep="\t", index=False
    )
    real_provenance.to_csv(
        outdir / "crossfit_real_provenance.tsv", sep="\t", index=False
    )
    pd.DataFrame(real_summaries).to_csv(
        outdir / "crossfit_real_reference_summary.tsv", sep="\t", index=False
    )
    write_json(
        {
            "status": "PASS",
            "reference_ids": refs["reference_id"].astype(str).tolist(),
            "generated_score_rows": int(len(generated_scores)),
            "generated_metric_rows": int(len(generated_metrics)),
            "generated_fold_rows": int(len(generated_provenance)),
            "real_score_rows": int(len(real_scores)),
            "real_summaries": real_summaries,
            "configuration": {
                "min_control_call_rate": args.min_control_call_rate,
                "max_features": args.max_features,
                "max_pca_components": args.max_pca_components,
                "min_overlap": args.min_overlap,
                "outer_crossfit": "leave-one-generated-replicate-out",
                "inner_calibration": "leave-one-training-replicate-out",
                "full_score_weight_marker": 0.75,
                "full_score_weight_mapping": 0.25,
            },
        },
        outdir / "crossfit_scoring_master.json",
    )
    print("\n[DONE] Leakage-controlled cross-fitted scoring complete.")


if __name__ == "__main__":
    main()
