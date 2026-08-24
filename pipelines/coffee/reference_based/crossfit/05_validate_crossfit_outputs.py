# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crossfit_common import SCORE_VARIANTS, ensure_dir, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--comparison_dir", required=True)
    ap.add_argument("--summary_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--expected_references", type=int, default=2)
    ap.add_argument("--expected_replicates", type=int, default=5)
    ap.add_argument("--expected_generated_per_reference", type=int, default=185)
    ap.add_argument("--expected_real_per_reference", type=int, default=18)
    ap.add_argument("--min_markers", type=int, default=10)
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    comparison_dir = Path(args.comparison_dir)
    summary_dir = Path(args.summary_dir)
    outdir = ensure_dir(args.outdir)

    generated = pd.read_csv(
        scores_dir / "crossfit_generated_reference_scores.tsv", sep="\t"
    )
    generated_metrics = pd.read_csv(
        scores_dir / "crossfit_generated_reference_metrics.tsv", sep="\t"
    )
    provenance = pd.read_csv(
        scores_dir / "crossfit_generated_fold_provenance.tsv", sep="\t"
    )
    real = pd.read_csv(
        scores_dir / "crossfit_real_reference_scores.tsv", sep="\t"
    )
    real_provenance = pd.read_csv(
        scores_dir / "crossfit_real_provenance.tsv", sep="\t"
    )
    consensus = pd.read_csv(
        comparison_dir / "crossfit_generated_consensus_scores.tsv", sep="\t"
    )
    consensus_metrics = pd.read_csv(
        comparison_dir / "crossfit_generated_consensus_metrics.tsv", sep="\t"
    )
    real_consensus = pd.read_csv(
        comparison_dir / "crossfit_real_consensus_scores.tsv", sep="\t"
    )
    master = json.loads(
        (summary_dir / "reference_qc_crossfit_master_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    checks: list[dict[str, object]] = []

    def check(name: str, observed: object, expected: object, ok: bool) -> None:
        checks.append(
            {
                "name": name,
                "observed": observed,
                "expected": expected,
                "ok": bool(ok),
            }
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] {name}: "
            f"observed={observed}; expected={expected}"
        )

    expected_generated_rows = (
        args.expected_references * args.expected_generated_per_reference
    )
    expected_real_rows = args.expected_references * args.expected_real_per_reference
    expected_folds = args.expected_references * args.expected_replicates

    check(
        "reference count",
        int(generated["reference_id"].nunique()),
        args.expected_references,
        generated["reference_id"].nunique() == args.expected_references,
    )
    check(
        "generated score rows",
        int(len(generated)),
        expected_generated_rows,
        len(generated) == expected_generated_rows,
    )
    check(
        "generated fold count",
        int(len(provenance)),
        expected_folds,
        len(provenance) == expected_folds,
    )
    check(
        "generated metric rows",
        int(len(generated_metrics)),
        expected_folds * len(SCORE_VARIANTS),
        len(generated_metrics) == expected_folds * len(SCORE_VARIANTS),
    )
    check(
        "replicate set",
        sorted(generated["replicate"].astype(int).unique().tolist()),
        list(range(1, args.expected_replicates + 1)),
        sorted(generated["replicate"].astype(int).unique().tolist())
        == list(range(1, args.expected_replicates + 1)),
    )
    check(
        "outer train/test overlap",
        int(provenance["outer_train_test_overlap_count"].max()),
        0,
        int(provenance["outer_train_test_overlap_count"].max()) == 0,
    )
    check(
        "test rows used in marker selection",
        int(provenance["marker_selection_used_outer_test_rows"].max()),
        0,
        int(provenance["marker_selection_used_outer_test_rows"].max()) == 0,
    )
    check(
        "test rows used in PCA fitting",
        int(provenance["pca_fit_used_outer_test_rows"].max()),
        0,
        int(provenance["pca_fit_used_outer_test_rows"].max()) == 0,
    )
    check(
        "test rows used in scaling",
        int(provenance["scaling_used_outer_test_rows"].max()),
        0,
        int(provenance["scaling_used_outer_test_rows"].max()) == 0,
    )
    check(
        "outer marker minimum",
        int(provenance["outer_selected_marker_count"].min()),
        f">={args.min_markers}",
        int(provenance["outer_selected_marker_count"].min()) >= args.min_markers,
    )
    check(
        "inner marker minimum",
        int(provenance["inner_min_selected_markers"].min()),
        f">={args.min_markers}",
        int(provenance["inner_min_selected_markers"].min()) >= args.min_markers,
    )
    check(
        "inner fold count",
        sorted(provenance["inner_fold_count"].astype(int).unique().tolist()),
        [args.expected_replicates - 1],
        sorted(provenance["inner_fold_count"].astype(int).unique().tolist())
        == [args.expected_replicates - 1],
    )
    check(
        "outer-test labels used for fitting",
        bool(provenance["outer_test_labels_used_for_model_fitting"].astype(bool).any()),
        False,
        not bool(
            provenance["outer_test_labels_used_for_model_fitting"].astype(bool).any()
        ),
    )
    check(
        "mixture labels used for parameter tuning",
        bool(provenance["mixture_labels_used_for_parameter_tuning"].astype(bool).any()),
        False,
        not bool(
            provenance["mixture_labels_used_for_parameter_tuning"].astype(bool).any()
        ),
    )
    check(
        "training control role declared",
        bool(provenance["training_control_role_used_for_model_fitting"].astype(bool).all()),
        True,
        bool(provenance["training_control_role_used_for_model_fitting"].astype(bool).all()),
    )

    finite_failures: dict[str, int] = {}
    for variant in SCORE_VARIANTS:
        count = int((~np.isfinite(generated[variant].to_numpy(dtype=float))).sum())
        if count:
            finite_failures[variant] = count
    check("finite generated scores", finite_failures, {}, not finite_failures)

    check(
        "real score rows",
        int(len(real)),
        expected_real_rows,
        len(real) == expected_real_rows,
    )
    real_fit_max = int(
        real_provenance[
            [
                "marker_selection_used_real_test_rows",
                "pca_fit_used_real_test_rows",
                "scaling_used_real_test_rows",
            ]
        ].to_numpy(dtype=int).max()
    )
    check("real test rows used in fitting", real_fit_max, 0, real_fit_max == 0)
    check(
        "metadata pool labels used for fitting",
        bool(
            real_provenance[
                "metadata_pool_labels_used_for_model_fitting"
            ].astype(bool).any()
        ),
        False,
        not bool(
            real_provenance[
                "metadata_pool_labels_used_for_model_fitting"
            ].astype(bool).any()
        ),
    )

    check(
        "generated consensus rows",
        int(len(consensus)),
        args.expected_generated_per_reference,
        len(consensus) == args.expected_generated_per_reference,
    )
    check(
        "real consensus rows",
        int(len(real_consensus)),
        args.expected_real_per_reference,
        len(real_consensus) == args.expected_real_per_reference,
    )
    per_rep_metrics = consensus_metrics[
        consensus_metrics["evaluation_scope"] == "per_replicate"
    ]
    check(
        "consensus per-replicate metric rows",
        int(len(per_rep_metrics)),
        args.expected_replicates * len(SCORE_VARIANTS),
        len(per_rep_metrics) == args.expected_replicates * len(SCORE_VARIANTS),
    )
    check(
        "required full score",
        "reference_qc_crossfit" in set(per_rep_metrics["score_variant"]),
        True,
        "reference_qc_crossfit" in set(per_rep_metrics["score_variant"]),
    )
    check(
        "required no-PCA score",
        "reference_qc_crossfit_no_pca" in set(
            per_rep_metrics["score_variant"]
        ),
        True,
        "reference_qc_crossfit_no_pca"
        in set(per_rep_metrics["score_variant"]),
    )
    check(
        "finite consensus AUC",
        int(np.isfinite(per_rep_metrics["roc_auc"]).sum()),
        int(len(per_rep_metrics)),
        int(np.isfinite(per_rep_metrics["roc_auc"]).sum())
        == int(len(per_rep_metrics)),
    )
    check(
        "master status",
        master.get("status"),
        "COMPLETE",
        master.get("status") == "COMPLETE",
    )
    check(
        "primary endpoint method",
        master.get("primary_endpoint", {}).get("score_variant"),
        "reference_qc_crossfit",
        master.get("primary_endpoint", {}).get("score_variant")
        == "reference_qc_crossfit",
    )
    check(
        "primary endpoint AUC finite",
        master.get("primary_endpoint", {}).get("roc_auc_mean"),
        "finite",
        np.isfinite(
            float(master.get("primary_endpoint", {}).get("roc_auc_mean"))
        ),
    )

    status = "PASS" if all(bool(x["ok"]) for x in checks) else "FAIL"
    payload = {"status": status, "checks": checks}
    write_json(payload, outdir / "reference_qc_crossfit_audit.json")
    lines = [f"STATUS: {status}"] + [
        f"[{'PASS' if x['ok'] else 'FAIL'}] {x['name']}: "
        f"observed={x['observed']}; expected={x['expected']}"
        for x in checks
    ]
    (outdir / "reference_qc_crossfit_audit.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if status == "PASS":
        (outdir / "REFERENCE_QC_CROSSFIT_AUDIT_PASS.txt").write_text(
            "STATUS: PASS\n", encoding="utf-8"
        )
    print(f"STATUS: {status}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
