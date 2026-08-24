# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from external_common import (
    ensure_dir,
    parse_int,
    read_json,
    read_tsv,
    sha256_file,
    write_json,
    write_tsv,
)


EXPECTED_LOCK = {
    "status": "LOCKED",
    "dataset": "finger_millet_PRJNA791522",
    "panel_samples": 83,
    "benchmark_sources": 28,
    "generated_libraries_total": 560,
    "rare_event_graphs_total": 735,
    "replicates": 5,
    "generated_controls_per_replicate": 28,
    "generated_mixtures_per_replicate": 84,
    "synthetic_read_pairs_per_library": 6000,
    "kmer": 17,
    "sketch_dimension": 16384,
    "locked_knn_full_cohort": 4,
    "locked_knn_synthetic": 2,
    "orc_alpha": 0.5,
    "primary_analysis_mode": "paired",
    "secondary_analysis_mode": "r1",
}


def check(
    rows: List[Dict[str, Any]],
    name: str,
    observed: Any,
    expected: Any,
    *,
    note: str = "",
) -> None:
    ok = observed == expected
    rows.append(
        {
            "name": name,
            "observed": observed,
            "expected": expected,
            "ok": ok,
            "note": note,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_root", required=True)
    parser.add_argument("--preanalysis_root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--expected_master_lock",
        default="e9117c96aa765bc4cd619e8b66bedc42c88fead82083d566ab330b5b4a503101",
    )
    parser.add_argument("--minimum_free_gb", type=float, default=15.0)
    args = parser.parse_args()

    module_root = Path(args.module_root)
    preanalysis_root = Path(args.preanalysis_root)
    outdir = ensure_dir(args.outdir)

    lock_json_path = preanalysis_root / "design_lock" / "analysis_design_lock.json"
    preanalysis_audit_path = preanalysis_root / "audit" / "preanalysis_lock_audit.json"
    preanalysis_pass = (
        preanalysis_root / "audit" / "PREANALYSIS_QC_AND_DESIGN_LOCK_PASS.txt"
    )
    download_pass = module_root / "06_download_audit" / "FASTQ_DOWNLOAD_COMPLETE.txt"

    required_files = [
        lock_json_path,
        preanalysis_audit_path,
        preanalysis_pass,
        download_pass,
        preanalysis_root / "source_selection" / "full_cohort_geometry_manifest_83.tsv",
        preanalysis_root / "source_selection" / "benchmark_source_panel_28.tsv",
        preanalysis_root / "sketches" / "full83_js_distance.csv",
        preanalysis_root / "sketches" / "source28_js_distance.csv",
        preanalysis_root / "design_lock" / "locked_parent_sets.tsv",
        preanalysis_root / "design_lock" / "locked_mixture_definitions_84.tsv",
        preanalysis_root / "design_lock" / "locked_generated_library_design_560.tsv",
        preanalysis_root / "design_lock" / "locked_read_allocations.tsv",
        preanalysis_root / "design_lock" / "locked_rare_event_schedule_735.tsv",
        preanalysis_root / "audit" / "canonical_lock_file_sha256.txt",
    ]

    checks: List[Dict[str, Any]] = []
    missing_files = [str(path) for path in required_files if not path.exists()]
    check(checks, "required input files missing", len(missing_files), 0)

    if missing_files:
        write_tsv(outdir / "missing_inputs.tsv", [{"path": path} for path in missing_files])
        raise SystemExit("[ERROR] Required locked inputs are missing.")

    lock = read_json(lock_json_path)
    audit = read_json(preanalysis_audit_path)
    check(checks, "preanalysis audit status", audit.get("status"), "PASS")
    check(
        checks,
        "master lock SHA-256",
        audit.get("master_lock_sha256"),
        args.expected_master_lock,
    )
    for key, expected in EXPECTED_LOCK.items():
        check(checks, f"lock field: {key}", lock.get(key), expected)

    full_manifest = read_tsv(
        preanalysis_root / "source_selection" / "full_cohort_geometry_manifest_83.tsv"
    )
    source_manifest = read_tsv(
        preanalysis_root / "source_selection" / "benchmark_source_panel_28.tsv"
    )
    generated_design = read_tsv(
        preanalysis_root / "design_lock" / "locked_generated_library_design_560.tsv"
    )
    allocations = read_tsv(
        preanalysis_root / "design_lock" / "locked_read_allocations.tsv"
    )
    rare_schedule = read_tsv(
        preanalysis_root / "design_lock" / "locked_rare_event_schedule_735.tsv"
    )
    parent_sets = read_tsv(
        preanalysis_root / "design_lock" / "locked_parent_sets.tsv"
    )
    mixture_definitions = read_tsv(
        preanalysis_root / "design_lock" / "locked_mixture_definitions_84.tsv"
    )

    check(checks, "full cohort manifest rows", len(full_manifest), 83)
    check(checks, "benchmark source rows", len(source_manifest), 28)
    check(checks, "parent-set rows", len(parent_sets), 28)
    check(checks, "mixture-definition rows", len(mixture_definitions), 84)
    check(checks, "generated design rows", len(generated_design), 560)
    check(checks, "rare-event schedule rows", len(rare_schedule), 735)

    class_counts = Counter(row.get("class_label", "") for row in generated_design)
    check(
        checks,
        "generated control rows",
        class_counts.get("single_source_control", 0),
        140,
    )
    check(
        checks,
        "generated mixture rows",
        class_counts.get("synthetic_mixture", 0),
        420,
    )
    injection_counts = Counter(parse_int(row.get("injection_count")) for row in rare_schedule)
    check(checks, "one-mixture graphs", injection_counts.get(1, 0), 420)
    check(checks, "two-mixture graphs", injection_counts.get(2, 0), 210)
    check(checks, "four-mixture graphs", injection_counts.get(4, 0), 105)

    allocation_total = sum(parse_int(row.get("read_pairs")) for row in allocations)
    check(checks, "total allocated source pairs", allocation_total, 3_360_000)
    allocation_by_library: Dict[str, int] = Counter()
    for row in allocations:
        allocation_by_library[row["sample_id"]] += parse_int(row.get("read_pairs"))
    bad_library_allocations = [
        sample_id
        for sample_id, count in allocation_by_library.items()
        if count != 6000
    ]
    check(
        checks,
        "libraries with allocation total not equal to 6000",
        len(bad_library_allocations),
        0,
    )

    manifest_paths: List[Dict[str, str]] = []
    path_failures: List[Dict[str, str]] = []
    for context, rows in (("full83", full_manifest), ("source28", source_manifest)):
        for row in rows:
            for mate_field in ("r1_path", "r2_path"):
                path = Path(row[mate_field])
                exists = path.exists()
                manifest_paths.append(
                    {
                        "context": context,
                        "sample_accession": row.get("sample_accession", ""),
                        "run_accession": row.get("run_accession", ""),
                        "mate_field": mate_field,
                        "path": str(path),
                        "exists": str(exists),
                        "bytes": str(path.stat().st_size if exists else 0),
                    }
                )
                if not exists:
                    path_failures.append(
                        {
                            "context": context,
                            "sample_accession": row.get("sample_accession", ""),
                            "run_accession": row.get("run_accession", ""),
                            "path": str(path),
                        }
                    )
    check(checks, "manifest FASTQ paths missing", len(path_failures), 0)
    write_tsv(outdir / "manifest_path_inventory.tsv", manifest_paths)
    write_tsv(outdir / "manifest_path_failures.tsv", path_failures)

    unique_source_paths = {
        Path(row[field])
        for row in source_manifest
        for field in ("r1_path", "r2_path")
    }
    source_total_bytes = sum(path.stat().st_size for path in unique_source_paths if path.exists())

    free_bytes = shutil.disk_usage(str(module_root)).free
    minimum_bytes = int(float(args.minimum_free_gb) * 1024**3)
    check(
        checks,
        "minimum free disk space",
        free_bytes >= minimum_bytes,
        True,
        note=f"free_bytes={free_bytes}; minimum_bytes={minimum_bytes}",
    )
    temp_value = os.environ.get("TEMP", "")
    tmp_value = os.environ.get("TMP", "")
    target_drive = module_root.drive.upper()
    temp_on_target = (
        Path(temp_value).drive.upper() == target_drive
        and Path(tmp_value).drive.upper() == target_drive
    )
    check(checks, "TEMP and TMP use module drive", temp_on_target, True)

    canonical_inputs = [
        preanalysis_root / "preflight" / "paired_input_manifest.tsv",
        preanalysis_root / "qc" / "fastq_qc_per_sample.tsv",
        preanalysis_root / "source_selection" / "benchmark_source_panel_28.tsv",
        preanalysis_root / "sketches" / "full83_js_distance.csv",
        preanalysis_root / "sketches" / "source28_js_distance.csv",
        preanalysis_root / "design_lock" / "locked_parent_sets.tsv",
        preanalysis_root / "design_lock" / "locked_mixture_definitions_84.tsv",
        preanalysis_root / "design_lock" / "locked_generated_library_design_560.tsv",
        preanalysis_root / "design_lock" / "locked_read_allocations.tsv",
        preanalysis_root / "design_lock" / "locked_rare_event_schedule_735.tsv",
        preanalysis_root / "design_lock" / "analysis_design_lock.json",
    ]
    hash_rows = [
        {
            "path": str(path),
            "relative_to_preanalysis": str(path.relative_to(preanalysis_root)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in canonical_inputs
    ]
    write_tsv(outdir / "locked_input_sha256.tsv", hash_rows)

    failed = [row for row in checks if not bool(row["ok"])]
    status = "PASS" if not failed else "FAIL"
    report = {
        "status": status,
        "expected_master_lock_sha256": args.expected_master_lock,
        "observed_master_lock_sha256": audit.get("master_lock_sha256"),
        "module_root": str(module_root),
        "preanalysis_root": str(preanalysis_root),
        "source_fastq_bytes": source_total_bytes,
        "free_disk_bytes": free_bytes,
        "checks": checks,
    }
    write_json(outdir / "external_benchmark_preflight.json", report)
    write_tsv(outdir / "external_benchmark_preflight_checks.tsv", checks)

    text_lines = [
        "Finger millet external benchmark preflight",
        "===========================================",
        "",
        f"Status: {status}",
        f"Master lock SHA-256: {audit.get('master_lock_sha256')}",
        f"Panel samples: {len(full_manifest)}",
        f"Benchmark sources: {len(source_manifest)}",
        f"Generated libraries planned: {len(generated_design)}",
        f"Rare-event graphs planned: {len(rare_schedule)}",
        f"Locked kNN: full cohort={lock.get('locked_knn_full_cohort')}; synthetic={lock.get('locked_knn_synthetic')}",
        f"Source paired FASTQ bytes: {source_total_bytes}",
        f"Free disk bytes: {free_bytes}",
        "",
        "No performance result was used to alter the locked sources, parent sets,",
        "read allocations, graph resolutions, or score parameters.",
    ]
    (outdir / "EXTERNAL_BENCHMARK_PREFLIGHT.txt").write_text(
        "\n".join(text_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(text_lines))

    marker = outdir / "EXTERNAL_BENCHMARK_PREFLIGHT_PASS.txt"
    if status == "PASS":
        marker.write_text("PASS\n", encoding="utf-8")
    else:
        if marker.exists():
            marker.unlink()
        print(f"[DETAIL] Failed checks: {len(failed)}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
