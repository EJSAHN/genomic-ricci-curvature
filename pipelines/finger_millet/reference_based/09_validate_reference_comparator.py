# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from crossfit_common import SCORE_VARIANTS, ensure_dir, write_json
from reference_common import sha256_file


EXPECTED_LOCK = "e9117c96aa765bc4cd619e8b66bedc42c88fead82083d566ab330b5b4a503101"
PRIMARY_SCORE = "reference_qc_crossfit"


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    observed: Any,
    expected: Any,
    *,
    critical: bool = True,
    note: str = "",
) -> None:
    if isinstance(expected, str) and expected.startswith(">="):
        ok = float(observed) >= float(expected[2:])
    elif expected == "finite":
        try:
            ok = math.isfinite(float(observed))
        except Exception:
            ok = False
    elif expected == "reported":
        ok = True
    else:
        ok = observed == expected
    checks.append(
        {
            "name": name,
            "observed": observed,
            "expected": expected,
            "ok": bool(ok),
            "critical": bool(critical),
            "note": note,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_failures(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, int]:
    failures: dict[str, int] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        count = int((~np.isfinite(values)).sum())
        if count:
            failures[column] = count
    return failures


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def collect_tree(
    root: Path,
    arc_prefix: str,
    *,
    exclude_parts: Sequence[str] = (),
    exclude_suffixes: Sequence[str] = (),
) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    if not root.exists():
        return entries
    excluded = set(exclude_parts)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts):
            continue
        if any(path.name.endswith(suffix) for suffix in exclude_suffixes):
            continue
        entries.append((path, str(Path(arc_prefix) / relative)))
    return entries


def zip_files(destination: Path, entries: Iterable[tuple[Path, str]]) -> None:
    if destination.exists():
        destination.unlink()
    unique: dict[str, Path] = {}
    for source, arcname in entries:
        if source.is_file():
            unique[arcname] = source
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for arcname, source in sorted(unique.items()):
            archive.write(source, arcname)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module_root", required=True)
    ap.add_argument("--preanalysis_root", required=True)
    ap.add_argument("--external_root", required=True)
    ap.add_argument("--work_root", required=True)
    ap.add_argument("--code_root", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    module_root = Path(args.module_root)
    preanalysis_root = Path(args.preanalysis_root)
    external_root = Path(args.external_root)
    work_root = Path(args.work_root)
    code_root = Path(args.code_root)
    outdir = ensure_dir(args.outdir)

    preflight = work_root / "preflight"
    reference_dir = work_root / "reference"
    panel_dir = work_root / "panel"
    features_dir = work_root / "features"
    scores_dir = work_root / "scores"
    comparison_dir = work_root / "comparison"
    rare_dir = work_root / "rare_event"
    summary_dir = work_root / "summary"
    figures_dir = work_root / "figures"
    logs_dir = work_root / "logs"

    required_markers = [
        preflight / "REFERENCE_COMPARATOR_PREFLIGHT_PASS.txt",
        reference_dir / "REFERENCE_PREPARATION_PASS.txt",
        panel_dir / "MARKER_PANEL_PASS.txt",
        features_dir / "FEATURE_EXTRACTION_PASS.txt",
        scores_dir / "CROSSFIT_SCORING_PASS.txt",
        comparison_dir / "BATCH_COMPARISON_PASS.txt",
        rare_dir / "REFERENCE_RARE_EVENT_PASS.txt",
    ]
    checks: list[dict[str, Any]] = []
    missing = [str(path) for path in required_markers if not path.exists()]
    add_check(checks, "required PASS markers missing", len(missing), 0)
    if missing:
        write_tsv(outdir / "missing_pass_markers.tsv", [{"path": value} for value in missing])
        raise SystemExit("[ERROR] Required PASS markers are absent")

    preflight_json = load_json(preflight / "reference_comparator_preflight.json")
    marker = load_json(panel_dir / "marker_panel_summary.json")
    feature = load_json(features_dir / "feature_extraction_master.json")
    scoring = load_json(scores_dir / "crossfit_scoring_summary.json")
    comparison = load_json(comparison_dir / "batch_comparison_audit.json")
    rare_audit = load_json(rare_dir / "reference_rare_event_audit.json")
    master = load_json(summary_dir / "external_reference_comparator_master_metrics.json")
    external_master = load_json(external_root / "summary" / "external_reference_free_master_metrics.json")

    add_check(checks, "preflight status", preflight_json.get("status"), "PASS")
    add_check(checks, "preflight master lock", preflight_json.get("expected_master_lock"), EXPECTED_LOCK)
    add_check(checks, "external master lock", external_master.get("master_lock_sha256"), EXPECTED_LOCK)
    add_check(checks, "marker-panel status", marker.get("status"), "PASS")
    add_check(checks, "feature-extraction status", feature.get("status"), "PASS")
    add_check(checks, "cross-fit scoring status", scoring.get("status"), "PASS")
    add_check(checks, "batch comparison status", comparison.get("status"), "PASS")
    add_check(checks, "rare-event comparator status", rare_audit.get("status"), "PASS")
    add_check(checks, "master summary status", master.get("status"), "COMPLETE")

    refs = pd.read_csv(reference_dir / "reference_manifest.tsv", sep="\t")
    add_check(checks, "reference count", len(refs), 1)
    add_check(checks, "reference accession", str(refs.iloc[0]["accession"]) if len(refs) else "", "GCA_032690845.1")
    reference_failures: list[dict[str, Any]] = []
    if len(refs):
        ref = refs.iloc[0]
        fasta = Path(str(ref["fasta_path"]))
        if not fasta.is_file() or sha256_file(fasta) != str(ref["fasta_sha256"]):
            reference_failures.append({"path": str(fasta), "reason": "missing_or_sha256_mismatch"})
        prefix = str(ref["index_prefix"])
        index_files = list(Path(prefix).parent.glob(Path(prefix).name + ".*.bt2*"))
        if len(index_files) < 6:
            reference_failures.append({"path": prefix, "reason": f"incomplete_index_files={len(index_files)}"})
    write_tsv(outdir / "reference_inventory_failures.tsv", reference_failures)
    add_check(checks, "reference/index inventory failures", len(reference_failures), 0)

    marker_panel = pd.read_csv(panel_dir / "marker_panel.tsv", sep="\t")
    independence = pd.read_csv(panel_dir / "marker_discovery_independence.tsv", sep="\t")
    add_check(checks, "marker source library count", independence["source_run_accession"].nunique(), 28)
    add_check(checks, "marker panel row count", len(marker_panel), int(marker.get("marker_count", -1)))
    add_check(checks, "marker panel minimum size", len(marker_panel), ">=100")
    add_check(checks, "marker-discovery generated-read overlap", int(independence["overlap_with_generated_pairs"].sum()), 0)
    add_check(checks, "marker-discovery geometry-prefix overlap", int(independence["overlap_with_reserved_prefix"].sum()), 0)
    add_check(checks, "independent pairs per source minimum", int(independence["selected_pairs"].min()), 100000)

    feature_summary = pd.read_csv(features_dir / "feature_extraction_summary.tsv", sep="\t")
    feature_metrics = pd.read_csv(features_dir / str(refs.iloc[0]["reference_id"]) / "generated_mapping_and_marker_metrics.tsv", sep="\t")
    add_check(checks, "feature summary rows", len(feature_summary), 1)
    add_check(checks, "generated feature rows", len(feature_metrics), 560)
    add_check(checks, "generated feature sample uniqueness", feature_metrics["sample_id"].nunique(), 560)
    add_check(checks, "generated mapping metrics finite", finite_failures(feature_metrics, ["mapping_rate", "unique_mapping_rate", "mean_mapq", "callable_fraction_depth1"]), {})
    matrix = np.load(features_dir / str(refs.iloc[0]["reference_id"]) / "generated_matrix.npz", allow_pickle=False)
    add_check(checks, "generated feature matrix rows", int(matrix["alt_fraction"].shape[0]), 560)
    add_check(checks, "generated feature matrix columns", int(matrix["alt_fraction"].shape[1]), len(marker_panel))
    add_check(checks, "finite marker entries", int(np.isfinite(matrix["alt_fraction"]).sum()), ">=1")

    scores = pd.read_csv(scores_dir / "generated_crossfit_scores.tsv", sep="\t")
    metrics = pd.read_csv(scores_dir / "generated_crossfit_metrics.tsv", sep="\t")
    provenance = pd.read_csv(scores_dir / "crossfit_fold_provenance.tsv", sep="\t")
    add_check(checks, "cross-fit generated score rows", len(scores), 560)
    add_check(checks, "cross-fit sample uniqueness", scores["sample_id"].nunique(), 560)
    add_check(checks, "cross-fit metric rows", len(metrics), 35)
    add_check(checks, "cross-fit fold rows", len(provenance), 5)
    add_check(checks, "cross-fit replicate set", sorted(scores["replicate"].astype(int).unique().tolist()), [1, 2, 3, 4, 5])
    add_check(checks, "cross-fit score variants", sorted(metrics["score_variant"].astype(str).unique().tolist()), sorted(SCORE_VARIANTS))
    add_check(checks, "finite cross-fit scores", finite_failures(scores, SCORE_VARIANTS), {})
    add_check(checks, "outer train/test overlap", int(pd.to_numeric(provenance["outer_train_test_overlap_count"], errors="coerce").max()), 0)
    add_check(checks, "test rows used for marker selection", int(pd.to_numeric(provenance["marker_selection_used_outer_test_rows"], errors="coerce").sum()), 0)
    add_check(checks, "test rows used for PCA fitting", int(pd.to_numeric(provenance["pca_fit_used_outer_test_rows"], errors="coerce").sum()), 0)
    add_check(checks, "test rows used for scaling", int(pd.to_numeric(provenance["scaling_used_outer_test_rows"], errors="coerce").sum()), 0)
    add_check(checks, "mixture labels used for parameter tuning", bool(provenance["mixture_labels_used_for_parameter_tuning"].astype(str).str.lower().isin(["true", "1", "yes"]).any()), False)
    add_check(checks, "outer-test labels used for fitting", bool(provenance["outer_test_labels_used_for_model_fitting"].astype(str).str.lower().isin(["true", "1", "yes"]).any()), False)
    add_check(checks, "outer marker minimum", int(pd.to_numeric(provenance["outer_selected_marker_count"], errors="coerce").min()), ">=10")
    add_check(checks, "inner marker minimum", int(pd.to_numeric(provenance["inner_min_selected_markers"], errors="coerce").min()), ">=10")

    batch_metrics = pd.read_csv(comparison_dir / "batch_reference_and_tms_metrics.tsv", sep="\t")
    merged = pd.read_csv(comparison_dir / "generated_reference_vs_reference_free_scores.tsv", sep="\t")
    add_check(checks, "batch merged rows", len(merged), 560)
    add_check(checks, "batch metric rows", len(batch_metrics), 40)
    add_check(checks, "batch metric score names", batch_metrics["score_name"].nunique(), 8)
    add_check(checks, "batch finite metrics", finite_failures(batch_metrics, ["roc_auc", "average_precision", "best_f1"]), {})

    rare_graph = pd.read_csv(rare_dir / "reference_rare_event_graph_metrics.tsv", sep="\t")
    rare_rank = pd.read_csv(rare_dir / "reference_rare_event_mixture_rank_metrics.tsv", sep="\t")
    rare_rep = pd.read_csv(rare_dir / "reference_rare_event_replicate_summary.tsv", sep="\t")
    rare_summary = pd.read_csv(rare_dir / "reference_rare_event_summary.tsv", sep="\t")
    add_check(checks, "rare-event graph metric rows", len(rare_graph), 735 * len(SCORE_VARIANTS))
    add_check(checks, "rare-event mixture-rank rows", len(rare_rank), 1260 * len(SCORE_VARIANTS))
    add_check(checks, "rare-event replicate rows", len(rare_rep), 5 * 3 * len(SCORE_VARIANTS))
    add_check(checks, "rare-event summary rows", len(rare_summary), 3 * len(SCORE_VARIANTS))
    add_check(checks, "rare-event injection set", sorted(rare_summary["injection_count"].astype(int).unique().tolist()), [1, 2, 4])
    add_check(checks, "rare-event finite graph metrics", finite_failures(rare_graph, ["roc_auc", "average_precision", "any_mixture_top1", "any_mixture_top3", "any_mixture_top5"]), {})
    add_check(checks, "rare-event finite rank metrics", finite_failures(rare_rank, ["score", "rank_percentile", "rank_descending"]), {})

    primary = master.get("primary_endpoint", {})
    add_check(checks, "primary injection count", int(primary.get("injection_count", -1)), 1)
    add_check(checks, "primary score name", primary.get("score_name"), PRIMARY_SCORE)
    add_check(checks, "primary replicate count", int(primary.get("replicate_count", -1)), 5)
    add_check(checks, "primary ROC AUC finite", primary.get("roc_auc_mean"), "finite")
    auc = float(primary.get("roc_auc_mean", float("nan")))
    expected_status = "NOT_EVALUABLE"
    if math.isfinite(auc):
        expected_status = "SUPPORTED" if auc >= 0.70 else ("WEAK_TO_MODERATE" if auc >= 0.60 else "NOT_SUPPORTED")
    add_check(checks, "empirical status follows thresholds", master.get("reference_detection_status"), expected_status)

    failed = [row for row in checks if row["critical"] and not row["ok"]]
    status = "PASS" if not failed else "FAIL"
    write_tsv(outdir / "reference_comparator_audit_checks.tsv", checks)

    canonical = [
        preflight / "reference_comparator_preflight.json",
        reference_dir / "reference_manifest.tsv",
        panel_dir / "marker_panel.tsv",
        panel_dir / "marker_panel_summary.json",
        features_dir / "feature_extraction_master.json",
        scores_dir / "generated_crossfit_scores.tsv",
        scores_dir / "generated_crossfit_metrics.tsv",
        scores_dir / "crossfit_fold_provenance.tsv",
        comparison_dir / "batch_reference_and_tms_summary.tsv",
        rare_dir / "reference_rare_event_summary.tsv",
        summary_dir / "external_reference_comparator_master_metrics.json",
        summary_dir / "REFERENCE_COMPARATOR_RESULTS_SUMMARY.txt",
    ]
    checksum_rows = [
        {
            "relative_path": str(path.relative_to(work_root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in canonical
        if path.is_file()
    ]
    write_tsv(outdir / "canonical_reference_comparator_sha256.tsv", checksum_rows)

    audit = {
        "status": status,
        "dataset": "finger_millet_PRJNA791522",
        "master_lock_sha256": EXPECTED_LOCK,
        "critical_failure_count": len(failed),
        "check_count": len(checks),
        "primary_endpoint": primary,
        "reference_detection_status": master.get("reference_detection_status"),
        "counts": {
            "reference_count": len(refs),
            "marker_count": len(marker_panel),
            "generated_feature_rows": len(feature_metrics),
            "crossfit_score_rows": len(scores),
            "rare_event_graph_metric_rows": len(rare_graph),
        },
    }
    write_json(audit, outdir / "reference_comparator_audit.json")

    lines = [
        "Finger millet external reference-based comparator audit",
        "=======================================================",
        "",
        f"STATUS: {status}",
        f"Critical failures: {len(failed)}",
        f"Master lock SHA-256: {EXPECTED_LOCK}",
        f"Reference assembly: {refs.iloc[0]['accession'] if len(refs) else 'missing'}",
        f"Marker panel: {len(marker_panel):,}",
        f"Generated libraries scored: {len(scores)}",
        f"Primary ROC AUC: {auc:.6f}" if math.isfinite(auc) else "Primary ROC AUC: non-finite",
        f"Performance status: {master.get('reference_detection_status')}",
        "",
    ]
    for row in checks:
        prefix = "PASS" if row["ok"] else ("INFO" if not row["critical"] else "FAIL")
        lines.append(f"[{prefix}] {row['name']}: observed={row['observed']}; expected={row['expected']}")
    audit_text = "\n".join(lines) + "\n"
    (outdir / "REFERENCE_COMPARATOR_AUDIT.txt").write_text(audit_text, encoding="utf-8")
    print(audit_text)

    review_entries: list[tuple[Path, str]] = []
    review_entries += collect_tree(preflight, "preflight")
    review_entries += collect_tree(reference_dir, "reference", exclude_parts=["index", "ncbi_extract"], exclude_suffixes=[".fna", ".bt2", ".bt2l", ".exe", ".zip"])
    review_entries += collect_tree(panel_dir, "panel", exclude_parts=["cache", "temp"])
    review_entries += collect_tree(features_dir, "features", exclude_parts=["cache_generated"], exclude_suffixes=["_matrix.npz"])
    review_entries += collect_tree(scores_dir, "scores")
    review_entries += collect_tree(comparison_dir, "comparison")
    review_entries += collect_tree(rare_dir, "rare_event")
    review_entries += collect_tree(summary_dir, "summary")

    review_entries += collect_tree(outdir, "audit")
    review_entries += collect_tree(logs_dir, "logs", exclude_suffixes=[".sam", ".bam"])
    review_entries += collect_tree(preanalysis_root / "design_lock", "preanalysis_lock/design_lock")
    review_entries += collect_tree(external_root / "summary", "reference_free_context/summary")

    review_zip = work_root / "finger_millet_reference_comparator_results.zip"
    zip_files(review_zip, review_entries)

    code_entries = collect_tree(code_root, "code", exclude_parts=["__pycache__"])
    code_entries += collect_tree(work_root / "config", "config")
    code_entries += collect_tree(work_root / "run", "run")
    for name in ["README.txt", "VERSION.txt", "install_FINGER_MILLET_REFERENCE_COMPARATOR.ps1", "prepare_tools_and_reference.ps1"]:
        path = work_root / name
        if path.is_file():
            code_entries.append((path, name))
    code_zip = work_root / "finger_millet_reference_comparator_source.zip"
    zip_files(code_zip, code_entries)

    if status == "PASS":
        (outdir / "REFERENCE_COMPARATOR_PASS.txt").write_text("PASS\n", encoding="utf-8")
        print(f"Results archive: {review_zip}")
        print(f"Source archive: {code_zip}")
    else:
        marker_path = outdir / "REFERENCE_COMPARATOR_PASS.txt"
        if marker_path.exists():
            marker_path.unlink()
        raise SystemExit("[ERROR] Reference-comparator audit failed")


if __name__ == "__main__":
    main()
