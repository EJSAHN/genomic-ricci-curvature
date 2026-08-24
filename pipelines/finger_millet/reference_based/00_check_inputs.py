# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from reference_common import ensure_dir, sha256_file, write_json

EXPECTED_LOCK = "e9117c96aa765bc4cd619e8b66bedc42c88fead82083d566ab330b5b4a503101"


def add_check(rows: list[dict[str, Any]], name: str, observed: Any, expected: Any, ok: bool, critical: bool = True, note: str = "") -> None:
    rows.append({
        "name": name,
        "observed": observed,
        "expected": expected,
        "ok": bool(ok),
        "critical": bool(critical),
        "note": note,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module_root", required=True)
    ap.add_argument("--preanalysis_root", required=True)
    ap.add_argument("--external_root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--reference_config", required=True)
    ap.add_argument("--minimum_free_gb", type=float, default=25.0)
    args = ap.parse_args()

    module_root = Path(args.module_root)
    preanalysis = Path(args.preanalysis_root)
    external = Path(args.external_root)
    outdir = ensure_dir(args.outdir)
    checks: list[dict[str, Any]] = []

    paths = {
        "module_root": module_root,
        "preanalysis_root": preanalysis,
        "external_root": external,
        "preanalysis_pass": preanalysis / "audit" / "PREANALYSIS_QC_AND_DESIGN_LOCK_PASS.txt",
        "external_audit": external / "audit" / "external_reference_free_audit.json",
        "external_pass": external / "audit" / "EXTERNAL_REFERENCE_FREE_AUDIT.txt",
        "lock_json": preanalysis / "design_lock" / "analysis_design_lock.json",
        "source_manifest": preanalysis / "source_selection" / "benchmark_source_panel_28.tsv",
        "read_allocations": preanalysis / "design_lock" / "locked_read_allocations.tsv",
        "rare_schedule": preanalysis / "design_lock" / "locked_rare_event_schedule_735.tsv",
        "generated_manifest": external / "generated" / "manifests" / "generated_fastq_manifest.tsv",
        "truth_manifest": external / "generated" / "manifests" / "truth_manifest.tsv",
        "batch_nodes": external / "batch" / "batch_node_scores_all.tsv",
        "rare_comparators": external / "rare_event" / "rare_event_comparator_metrics.tsv",
        "external_master": external / "summary" / "external_reference_free_master_metrics.json",
        "reference_config": Path(args.reference_config),
    }
    for name, path in paths.items():
        exists = path.is_dir() if name.endswith("root") else path.is_file()
        add_check(checks, name, str(path), "existing", exists)

    critical_missing = [row for row in checks if row["critical"] and not row["ok"]]
    if critical_missing:
        status = "FAIL"
    else:
        lock = json.loads(paths["lock_json"].read_text(encoding="utf-8"))
        external_audit = json.loads(paths["external_audit"].read_text(encoding="utf-8"))
        external_master = json.loads(paths["external_master"].read_text(encoding="utf-8"))
        source = pd.read_csv(paths["source_manifest"], sep="\t")
        allocations = pd.read_csv(paths["read_allocations"], sep="\t")
        schedule = pd.read_csv(paths["rare_schedule"], sep="\t")
        generated = pd.read_csv(paths["generated_manifest"], sep="\t")
        truth = pd.read_csv(paths["truth_manifest"], sep="\t")
        batch = pd.read_csv(paths["batch_nodes"], sep="\t")
        refs = pd.read_csv(paths["reference_config"], sep="\t")

        # The lock JSON itself does not store the master digest; the audited external run does.
        observed_lock = str(external_audit.get("master_lock_sha256", ""))
        add_check(checks, "external audit master lock", observed_lock, EXPECTED_LOCK, observed_lock == EXPECTED_LOCK)
        add_check(checks, "external audit status", external_audit.get("status"), "PASS", external_audit.get("status") == "PASS")
        add_check(checks, "external master status", external_master.get("status"), "COMPLETE", external_master.get("status") == "COMPLETE")
        add_check(checks, "benchmark sources", len(source), 28, len(source) == 28)
        add_check(checks, "source populations", source["population"].nunique(), 7, source["population"].nunique() == 7)
        add_check(checks, "locked allocation rows", len(allocations), 1085, len(allocations) == 1085)
        add_check(checks, "rare-event schedule rows", len(schedule), 735, len(schedule) == 735)
        add_check(checks, "generated libraries", len(generated), 560, len(generated) == 560)
        add_check(checks, "truth rows", len(truth), 560, len(truth) == 560)
        add_check(checks, "generated controls", int((truth["class_label"].astype(str) == "single_source_control").sum()), 140, int((truth["class_label"].astype(str) == "single_source_control").sum()) == 140)
        add_check(checks, "generated mixtures", int((truth["class_label"].astype(str) == "synthetic_mixture").sum()), 420, int((truth["class_label"].astype(str) == "synthetic_mixture").sum()) == 420)
        add_check(checks, "batch node rows", len(batch), 1120, len(batch) == 1120)
        add_check(checks, "batch paired rows", int((batch["analysis_mode"].astype(str) == "paired").sum()), 560, int((batch["analysis_mode"].astype(str) == "paired").sum()) == 560)
        add_check(checks, "replicate set", sorted(truth["replicate"].astype(int).unique().tolist()), [1,2,3,4,5], sorted(truth["replicate"].astype(int).unique().tolist()) == [1,2,3,4,5])

        missing_paths = 0
        size_mismatch = 0
        for row in generated.itertuples(index=False):
            for path_field, size_field in (("r1_path", "r1_size_bytes"), ("r2_path", "r2_size_bytes")):
                path = Path(str(getattr(row, path_field)))
                if not path.is_file():
                    missing_paths += 1
                elif path.stat().st_size != int(getattr(row, size_field)):
                    size_mismatch += 1
        add_check(checks, "generated FASTQ missing paths", missing_paths, 0, missing_paths == 0)
        add_check(checks, "generated FASTQ size mismatches", size_mismatch, 0, size_mismatch == 0)

        source_missing = 0
        for row in source.itertuples(index=False):
            for field in ("r1_path", "r2_path"):
                if not Path(str(getattr(row, field))).is_file():
                    source_missing += 1
        add_check(checks, "source FASTQ missing paths", source_missing, 0, source_missing == 0)

        disk = shutil.disk_usage(str(module_root))
        free_gb = disk.free / (1024 ** 3)
        add_check(checks, "free disk GB", round(free_gb, 2), f">={args.minimum_free_gb}", free_gb >= args.minimum_free_gb)

        enabled_refs = refs[refs["enabled"].astype(str).str.lower().isin(["1","true","yes"])]
        add_check(checks, "enabled reference count", len(enabled_refs), 1, len(enabled_refs) == 1)
        add_check(checks, "reference accession", enabled_refs.iloc[0]["accession"] if len(enabled_refs) else "", "GCA_032690845.1", len(enabled_refs) == 1 and str(enabled_refs.iloc[0]["accession"]) == "GCA_032690845.1")

        status = "PASS" if all(row["ok"] or not row["critical"] for row in checks) else "FAIL"

    pd.DataFrame(checks).to_csv(outdir / "reference_comparator_preflight_checks.tsv", sep="\t", index=False)
    report = {
        "status": status,
        "module_root": str(module_root),
        "preanalysis_root": str(preanalysis),
        "external_root": str(external),
        "expected_master_lock": EXPECTED_LOCK,
        "critical_failures": sum(1 for row in checks if row["critical"] and not row["ok"]),
        "checks": checks,
    }
    write_json(report, outdir / "reference_comparator_preflight.json")

    lines = [
        "Finger millet external reference-comparator preflight",
        "======================================================",
        "",
        f"Status: {status}",
        f"Expected master lock: {EXPECTED_LOCK}",
        f"Critical failures: {report['critical_failures']}",
        "",
        "This comparator reuses the locked 28-source/560-library benchmark and",
        "does not alter parent sets, generated reads, mixture labels, or graph results.",
        "Marker discovery uses independent source read pairs only.",
    ]
    (outdir / "REFERENCE_COMPARATOR_PREFLIGHT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    marker = outdir / "REFERENCE_COMPARATOR_PREFLIGHT_PASS.txt"
    if status == "PASS":
        marker.write_text("PASS\n", encoding="utf-8")
    else:
        if marker.exists(): marker.unlink()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
