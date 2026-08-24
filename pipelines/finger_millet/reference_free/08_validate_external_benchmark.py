# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from external_common import ensure_dir, read_json, sha256_file, write_json, write_tsv


EXPECTED_MASTER_LOCK = "e9117c96aa765bc4cd619e8b66bedc42c88fead82083d566ab330b5b4a503101"
EXPECTED_SCORE_COUNT = 8


def add_check(
    checks: List[Dict[str, Any]],
    name: str,
    observed: Any,
    expected: Any,
    *,
    critical: bool = True,
    note: str = "",
) -> None:
    if isinstance(expected, str) and expected.startswith(">="):
        threshold = float(expected[2:])
        ok = float(observed) >= threshold
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


def finite_table_failures(table: pd.DataFrame, columns: Sequence[str]) -> Dict[str, int]:
    failures: Dict[str, int] = {}
    for column in columns:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
        count = int((~np.isfinite(values)).sum())
        if count:
            failures[column] = count
    return failures


def zip_files(
    destination: Path,
    entries: Iterable[tuple[Path, str]],
) -> None:
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, arcname in sorted(entries, key=lambda item: item[1]):
            if source.exists() and source.is_file():
                archive.write(source, arcname)


def collect_tree(
    root: Path,
    arc_prefix: str,
    *,
    exclude_parts: Sequence[str] = (),
) -> List[tuple[Path, str]]:
    entries: List[tuple[Path, str]] = []
    excluded = set(exclude_parts)
    if not root.exists():
        return entries
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts):
            continue
        entries.append((path, str(Path(arc_prefix) / relative)))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_root", required=True)
    parser.add_argument("--preanalysis_root", required=True)
    parser.add_argument("--work_root", required=True)
    parser.add_argument("--code_root", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    module_root = Path(args.module_root)
    preanalysis_root = Path(args.preanalysis_root)
    work_root = Path(args.work_root)
    code_root = Path(args.code_root)
    outdir = ensure_dir(args.outdir)

    preflight = work_root / "preflight"
    full_dir = work_root / "full_cohort"
    generated = work_root / "generated"
    sketch_dir = work_root / "sketches"
    batch_dir = work_root / "batch"
    rare_dir = work_root / "rare_event"
    summary_dir = work_root / "summary"
    figures_dir = work_root / "figures"

    required_markers = [
        preanalysis_root / "audit" / "PREANALYSIS_QC_AND_DESIGN_LOCK_PASS.txt",
        preflight / "EXTERNAL_BENCHMARK_PREFLIGHT_PASS.txt",
        full_dir / "FULL83_GEOMETRY_PASS.txt",
        generated / "manifests" / "GENERATED_FASTQ_BUILD_PASS.txt",
        sketch_dir / "GENERATED_SKETCH_PASS.txt",
        batch_dir / "BATCH_BENCHMARK_PASS.txt",
        rare_dir / "RARE_EVENT_BENCHMARK_PASS.txt",
    ]
    checks: List[Dict[str, Any]] = []
    missing_markers = [str(path) for path in required_markers if not path.exists()]
    add_check(checks, "required PASS markers missing", len(missing_markers), 0)
    if missing_markers:
        write_tsv(outdir / "missing_pass_markers.tsv", [{"path": path} for path in missing_markers])
        raise SystemExit("[ERROR] Required PASS markers are missing.")

    lock = read_json(preanalysis_root / "design_lock" / "analysis_design_lock.json")
    pre_audit = read_json(preanalysis_root / "audit" / "preanalysis_lock_audit.json")
    preflight_json = read_json(preflight / "external_benchmark_preflight.json")
    full_summary = read_json(full_dir / "full83_geometry_summary.json")
    build_audit = read_json(generated / "manifests" / "generated_fastq_build_audit.json")
    sketch_summary = read_json(sketch_dir / "generated_sketch_summary.json")
    batch_parameters = read_json(batch_dir / "batch_analysis_parameters.json")
    rare_parameters = read_json(rare_dir / "rare_event_analysis_summary.json")
    master = read_json(summary_dir / "external_reference_free_master_metrics.json")

    add_check(checks, "preanalysis master lock", pre_audit.get("master_lock_sha256"), EXPECTED_MASTER_LOCK)
    add_check(checks, "preflight status", preflight_json.get("status"), "PASS")
    add_check(checks, "full cohort status", full_summary.get("status"), "PASS")
    add_check(checks, "generated FASTQ build status", build_audit.get("status"), "PASS")
    add_check(checks, "generated sketch status", sketch_summary.get("status"), "PASS")
    add_check(checks, "batch parameter status", batch_parameters.get("status"), "PASS")
    add_check(checks, "rare-event analysis status", rare_parameters.get("status"), "PASS")
    add_check(checks, "master summary status", master.get("status"), "COMPLETE")

    full_nodes = pd.read_csv(full_dir / "full83_node_scores.tsv", sep="\t")
    full_edges = pd.read_csv(full_dir / "full83_edge_scores.tsv", sep="\t")
    add_check(checks, "full cohort node rows", len(full_nodes), 83)
    add_check(checks, "full cohort population count", full_nodes["population"].nunique(), 7)
    add_check(checks, "full cohort graph connected", full_summary.get("graph_connected"), True)
    add_check(checks, "full cohort edge rows", len(full_edges), full_summary.get("graph_edges"))

    generated_manifest = pd.read_csv(generated / "manifests" / "generated_fastq_manifest.tsv", sep="\t")
    truth_manifest = pd.read_csv(generated / "manifests" / "truth_manifest.tsv", sep="\t")
    add_check(checks, "generated manifest rows", len(generated_manifest), 560)
    add_check(checks, "truth manifest rows", len(truth_manifest), 560)
    add_check(checks, "generated controls", int((generated_manifest["class_label"] == "single_source_control").sum()), 140)
    add_check(checks, "generated mixtures", int((generated_manifest["class_label"] == "synthetic_mixture").sum()), 420)
    add_check(checks, "generated read pairs", int(generated_manifest["read_pairs"].sum()), 3_360_000)
    add_check(checks, "source-pair reuse detected", bool(build_audit.get("source_pair_reuse_detected")), False)
    add_check(checks, "unique FASTQ checksum pairs", build_audit.get("generated_checksum_pair_unique_count"), 560)

    fastq_failures: List[Dict[str, Any]] = []
    for row in generated_manifest.itertuples(index=False):
        for mate in (1, 2):
            path = Path(getattr(row, f"r{mate}_path"))
            expected_size = int(getattr(row, f"r{mate}_size_bytes"))
            if not path.exists() or path.stat().st_size != expected_size:
                fastq_failures.append(
                    {
                        "sample_id": row.sample_id,
                        "mate": mate,
                        "path": str(path),
                        "exists": path.exists(),
                        "observed_size": path.stat().st_size if path.exists() else 0,
                        "expected_size": expected_size,
                    }
                )
    write_tsv(outdir / "generated_fastq_inventory_failures.tsv", fastq_failures)
    add_check(checks, "generated FASTQ path/size failures", len(fastq_failures), 0)

    sketch_manifest = pd.read_csv(sketch_dir / "generated_sketch_manifest.tsv", sep="\t")
    add_check(checks, "generated sketch rows", len(sketch_manifest), 560)
    add_check(checks, "generated sketch pair-count failures", sketch_summary.get("pair_count_failures"), 0)
    missing_sketches = sum(not Path(path).exists() for path in sketch_manifest["sketch_cache"].astype(str))
    add_check(checks, "generated sketch cache files missing", int(missing_sketches), 0)

    batch_runs = pd.read_csv(batch_dir / "batch_run_metrics.tsv", sep="\t")
    batch_nodes = pd.read_csv(batch_dir / "batch_node_scores_all.tsv", sep="\t")
    batch_comparators = pd.read_csv(batch_dir / "batch_comparator_metrics.tsv", sep="\t")
    batch_categories = pd.read_csv(batch_dir / "batch_category_metrics.tsv", sep="\t")
    batch_patterns = pd.read_csv(batch_dir / "batch_pattern_metrics.tsv", sep="\t")
    add_check(checks, "batch run rows", len(batch_runs), 10)
    add_check(checks, "batch node rows", len(batch_nodes), 1120)
    add_check(checks, "batch comparator rows", len(batch_comparators), 10 * EXPECTED_SCORE_COUNT)
    add_check(checks, "batch category rows", len(batch_categories), 40)
    add_check(checks, "batch pattern rows", len(batch_patterns), 60)
    add_check(checks, "batch replicate set", sorted(batch_runs["replicate"].astype(int).unique().tolist()), [1, 2, 3, 4, 5])
    add_check(checks, "batch mode set", sorted(batch_runs["analysis_mode"].astype(str).unique().tolist()), ["paired", "r1"])
    add_check(checks, "batch finite primary metrics", finite_table_failures(batch_runs, ["roc_auc", "average_precision", "best_f1"]), {})
    add_check(checks, "batch graph connectivity", int(batch_runs["graph_connected"].astype(bool).sum()), len(batch_runs), critical=False, note="Reported; locked kNN was chosen on control-only design sketches.")

    rare_graphs = pd.read_csv(rare_dir / "rare_event_graph_metrics.tsv", sep="\t")
    rare_mixtures = pd.read_csv(rare_dir / "rare_event_mixture_rank_metrics.tsv", sep="\t")
    rare_comparators = pd.read_csv(rare_dir / "rare_event_comparator_metrics.tsv", sep="\t")
    rare_edges = pd.read_csv(rare_dir / "rare_event_edge_diagnostics.tsv", sep="\t")
    add_check(checks, "rare-event graph rows", len(rare_graphs), 1470)
    add_check(checks, "rare-event mixture rows", len(rare_mixtures), 2520)
    add_check(checks, "rare-event comparator rows", len(rare_comparators), 1470 * EXPECTED_SCORE_COUNT)
    add_check(checks, "rare-event edge diagnostic rows", len(rare_edges), 1470)
    expected_graph_counts = {1: 840, 2: 420, 4: 210}
    observed_graph_counts = Counter(rare_graphs["injection_count"].astype(int))
    add_check(checks, "rare-event graph counts by injection", dict(sorted(observed_graph_counts.items())), expected_graph_counts)
    expected_mixture_counts = {1: 840, 2: 840, 4: 840}
    observed_mixture_counts = Counter(rare_mixtures["injection_count"].astype(int))
    add_check(checks, "rare-event mixture rows by injection", dict(sorted(observed_mixture_counts.items())), expected_mixture_counts)
    add_check(checks, "rare-event finite graph metrics", finite_table_failures(rare_graphs, ["roc_auc", "average_precision", "mean_mixture_percentile"]), {})
    add_check(checks, "rare-event finite rank metrics", finite_table_failures(rare_mixtures, ["rank_percentile", "tms"]), {})
    add_check(checks, "rare-event connected graphs", int(rare_graphs["graph_connected"].astype(bool).sum()), len(rare_graphs), critical=False, note="Reported rather than enforced.")

    primary = master.get("primary_endpoint", {})
    add_check(checks, "primary analysis mode", primary.get("analysis_mode"), lock.get("primary_analysis_mode"))
    add_check(checks, "primary injection count", primary.get("injection_count"), 1)
    add_check(checks, "primary replicate count", primary.get("replicate_count"), 5)
    add_check(checks, "primary graph count", primary.get("graph_count"), 420)
    add_check(checks, "primary ROC AUC finite", primary.get("roc_auc_mean"), "finite")
    add_check(
        checks,
        "empirical status label",
        master.get("external_reference_free_detection_status"),
        master.get("external_reference_free_detection_status"),
        note="Status is checked below against the prespecified thresholds.",
    )
    auc = float(primary.get("roc_auc_mean", float("nan")))
    expected_status = "UNDETERMINED"
    if math.isfinite(auc):
        expected_status = "SUPPORTED" if auc >= 0.70 else ("WEAK_TO_MODERATE" if auc >= 0.60 else "NOT_SUPPORTED")
    add_check(checks, "empirical status follows locked thresholds", master.get("external_reference_free_detection_status"), expected_status)

    failed = [row for row in checks if row["critical"] and not row["ok"]]
    status = "PASS" if not failed else "FAIL"
    write_tsv(outdir / "external_reference_free_audit_checks.tsv", checks)

    canonical_files = [
        preanalysis_root / "design_lock" / "analysis_design_lock.json",
        preanalysis_root / "audit" / "preanalysis_lock_audit.json",
        preflight / "external_benchmark_preflight.json",
        full_dir / "full83_geometry_summary.json",
        generated / "manifests" / "generated_fastq_manifest.tsv",
        generated / "manifests" / "truth_manifest.tsv",
        generated / "manifests" / "generated_fastq_build_audit.json",
        sketch_dir / "generated_sketch_manifest.tsv",
        sketch_dir / "generated_sketch_summary.json",
        batch_dir / "batch_run_metrics.tsv",
        batch_dir / "batch_comparator_metrics.tsv",
        rare_dir / "rare_event_graph_metrics.tsv",
        rare_dir / "rare_event_mixture_rank_metrics.tsv",
        summary_dir / "external_reference_free_master_metrics.json",
        summary_dir / "EXTERNAL_REFERENCE_FREE_RESULTS_SUMMARY.txt",
    ]
    checksum_rows = [
        {
            "path": str(path),
            "relative_path": (
                str(path.relative_to(work_root)) if path.is_relative_to(work_root)
                else str(Path("preanalysis_lock") / path.relative_to(preanalysis_root))
            ),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in canonical_files
        if path.exists()
    ]
    write_tsv(outdir / "canonical_external_benchmark_sha256.tsv", checksum_rows)

    audit = {
        "status": status,
        "dataset": lock.get("dataset"),
        "master_lock_sha256": EXPECTED_MASTER_LOCK,
        "critical_failure_count": len(failed),
        "check_count": len(checks),
        "primary_endpoint": primary,
        "external_reference_free_detection_status": master.get("external_reference_free_detection_status"),
        "counts": {
            "full_cohort_nodes": len(full_nodes),
            "generated_libraries": len(generated_manifest),
            "batch_runs": len(batch_runs),
            "rare_event_graphs": len(rare_graphs),
            "rare_event_mixture_rows": len(rare_mixtures),
        },
    }
    write_json(outdir / "external_reference_free_audit.json", audit)

    lines = [
        "Finger millet external reference-free benchmark audit",
        "=====================================================",
        "",
        f"STATUS: {status}",
        f"Critical failures: {len(failed)}",
        f"Master lock SHA-256: {EXPECTED_MASTER_LOCK}",
        f"Generated libraries: {len(generated_manifest)}",
        f"Batch runs: {len(batch_runs)}",
        f"Rare-event graphs: {len(rare_graphs)}",
        f"Primary ROC AUC: {auc:.6f}" if math.isfinite(auc) else "Primary ROC AUC: non-finite",
        f"Performance status: {master.get('external_reference_free_detection_status')}",
        "",
    ]
    for row in checks:
        prefix = "PASS" if row["ok"] else ("INFO" if not row["critical"] else "FAIL")
        lines.append(f"[{prefix}] {row['name']}: observed={row['observed']}; expected={row['expected']}")
    audit_text = "\n".join(lines) + "\n"
    (outdir / "EXTERNAL_REFERENCE_FREE_AUDIT.txt").write_text(audit_text, encoding="utf-8")
    print(audit_text)

    review_entries: List[tuple[Path, str]] = []
    review_entries += collect_tree(preflight, "preflight")
    review_entries += collect_tree(full_dir, "full_cohort")
    review_entries += collect_tree(generated / "manifests", "generated/manifests")
    review_entries += collect_tree(sketch_dir, "sketches", exclude_parts=["cache"])
    review_entries += collect_tree(batch_dir, "batch", exclude_parts=["runs"])
    # Include batch per-run metrics and node/edge scores, but omit repeated distance matrices.
    for path in (batch_dir / "runs").rglob("*") if (batch_dir / "runs").exists() else []:
        if path.is_file() and path.name != "js_distance.csv":
            review_entries.append((path, str(Path("batch/runs") / path.relative_to(batch_dir / "runs"))))
    review_entries += collect_tree(rare_dir, "rare_event", exclude_parts=["parts"])
    review_entries += collect_tree(summary_dir, "summary")

    review_entries += collect_tree(outdir, "audit")
    review_entries += collect_tree(preanalysis_root / "design_lock", "preanalysis_lock/design_lock")
    review_entries += collect_tree(preanalysis_root / "source_selection", "preanalysis_lock/source_selection")
    review_entries += collect_tree(preanalysis_root / "audit", "preanalysis_lock/audit")

    review_zip = work_root / "finger_millet_reference_free_results.zip"
    zip_files(review_zip, review_entries)

    code_entries = collect_tree(code_root, "code")
    code_entries += collect_tree(code_root.parent / "config", "config")
    code_entries += collect_tree(code_root.parent / "run", "run")
    for name in ["README.txt", "VERSION.txt", "install_FINGER_MILLET_EXTERNAL_BENCHMARK.ps1"]:
        path = code_root.parent / name
        if path.exists():
            code_entries.append((path, name))
    code_zip = work_root / "finger_millet_reference_free_source.zip"
    zip_files(code_zip, code_entries)

    if status == "PASS":
        (outdir / "EXTERNAL_REFERENCE_FREE_BENCHMARK_PASS.txt").write_text("PASS\n", encoding="utf-8")
        print(f"Results archive: {review_zip}")
        print(f"Source archive: {code_zip}")
    else:
        marker = outdir / "EXTERNAL_REFERENCE_FREE_BENCHMARK_PASS.txt"
        if marker.exists():
            marker.unlink()
        raise SystemExit(8)


if __name__ == "__main__":
    main()
