from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List

from common import md5_file, parse_int, read_json, read_tsv, sha256_file, write_json, write_tsv


def add_tree_to_zip(archive: zipfile.ZipFile, root: Path, arc_prefix: str, exclude_names: set[str] | None = None) -> None:
    exclude_names = exclude_names or set()
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in exclude_names:
            continue
        archive.write(path, str(Path(arc_prefix) / path.relative_to(root)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_root", required=True)
    parser.add_argument("--fastq_manifest", required=True)
    parser.add_argument("--download_root", required=True)
    parser.add_argument("--selection_summary", required=True)
    parser.add_argument("--metadata_audit", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    module_root = Path(args.module_root)
    manifest = read_tsv(Path(args.fastq_manifest))
    download_root = Path(args.download_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inventory: List[Dict[str, str]] = []
    failures: List[Dict[str, str]] = []
    population_counts: Counter[str] = Counter()
    run_accessions = set()
    sample_accessions = set()

    total_expected = 0
    total_observed = 0
    for row in manifest:
        population = row.get("population", "")
        filename = row.get("filename", "")
        path = download_root / population / filename
        expected_bytes = parse_int(row.get("bytes", 0))
        expected_md5 = row.get("md5", "").lower()
        total_expected += expected_bytes
        observed_bytes = path.stat().st_size if path.exists() else 0
        observed_md5 = md5_file(path) if path.exists() and observed_bytes == expected_bytes else ""
        ok = (
            path.exists()
            and observed_bytes == expected_bytes
            and observed_md5.lower() == expected_md5
        )
        if ok:
            total_observed += observed_bytes
            population_counts[population] += 1
            run_accessions.add(row.get("run_accession", ""))
            sample_accessions.add(row.get("sample_accession", ""))
        else:
            failures.append(
                {
                    "population": population,
                    "sample_accession": row.get("sample_accession", ""),
                    "run_accession": row.get("run_accession", ""),
                    "mate": row.get("mate", ""),
                    "filename": filename,
                    "path": str(path),
                    "expected_bytes": str(expected_bytes),
                    "observed_bytes": str(observed_bytes),
                    "expected_md5": expected_md5,
                    "observed_md5": observed_md5,
                    "reason": (
                        "missing"
                        if not path.exists()
                        else "size_mismatch"
                        if observed_bytes != expected_bytes
                        else "md5_mismatch"
                    ),
                }
            )
        inventory.append(
            {
                **row,
                "local_path": str(path),
                "observed_bytes": str(observed_bytes),
                "observed_md5": observed_md5,
                "verified": str(ok),
            }
        )

    selection_summary = read_json(Path(args.selection_summary))
    metadata_audit = read_json(Path(args.metadata_audit))
    expected_files = len(manifest)
    status = (
        "PASS"
        if not failures
        and len(inventory) == expected_files
        and total_observed == total_expected
        and selection_summary.get("status") == "PASS"
        and metadata_audit.get("status") == "PASS"
        else "FAIL"
    )

    write_tsv(outdir / "fastq_inventory_verified.tsv", inventory)
    write_tsv(outdir / "fastq_validation_failures.tsv", failures)

    audit = {
        "status": status,
        "expected_fastq_files": expected_files,
        "verified_fastq_files": sum(1 for row in inventory if row["verified"] == "True"),
        "failure_count": len(failures),
        "selected_sample_count": len(sample_accessions),
        "selected_run_count": len(run_accessions),
        "total_expected_bytes": total_expected,
        "total_observed_bytes": total_observed,
        "population_fastq_file_counts": dict(population_counts),
        "selection_manifest_sha256": selection_summary.get("selection_manifest_sha256", ""),
        "metadata_audit_status": metadata_audit.get("status", ""),
        "selection_status": selection_summary.get("status", ""),
    }
    write_json(outdir / "fastq_download_audit.json", audit)
    lines = [
        "Finger millet FASTQ download audit",
        "==================================",
        "",
        f"Status: {status}",
        f"Selected biological samples: {len(sample_accessions)}",
        f"Selected runs: {len(run_accessions)}",
        f"Verified FASTQ files: {audit['verified_fastq_files']} / {expected_files}",
        f"Validation failures: {len(failures)}",
        f"Verified bytes: {total_observed} / {total_expected}",
        f"Selection manifest SHA-256: {audit['selection_manifest_sha256']}",
    ]
    (outdir / "fastq_download_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    pass_marker = outdir / "FASTQ_DOWNLOAD_COMPLETE.txt"
    if status == "PASS":
        pass_marker.write_text("PASS\n", encoding="utf-8")
    elif pass_marker.exists():
        pass_marker.unlink()

    # Compact results archive: metadata, selection, audits, logs, and code; no FASTQ files.
    review_zip = module_root / "finger_millet_metadata_panel_and_download_audit.zip"
    code_zip = module_root / "finger_millet_metadata_downloader_source_archive.zip"
    with zipfile.ZipFile(review_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for folder, prefix in [
            (module_root / "00_metadata", "00_metadata"),
            (module_root / "01_metadata_audit", "01_metadata_audit"),
            (module_root / "02_subset_manifest", "02_subset_manifest"),
            (module_root / "04_disk_preflight", "04_disk_preflight"),
            (module_root / "05_download_status", "05_download_status"),
            (module_root / "06_download_audit", "06_download_audit"),
            (module_root / "_logs", "_logs"),
            (module_root / "config", "config"),
        ]:
            add_tree_to_zip(
                archive,
                folder,
                prefix,
                exclude_names={"PMC9090224.tar.gz"},
            )
        add_tree_to_zip(archive, module_root / "code", "code")
        add_tree_to_zip(archive, module_root / "run", "run")

    with zipfile.ZipFile(code_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        add_tree_to_zip(archive, module_root / "code", "code")
        add_tree_to_zip(archive, module_root / "config", "config")
        add_tree_to_zip(archive, module_root / "run", "run")
        for name in ("README.txt", "VERSION.txt"):
            path = module_root / name
            if path.exists():
                archive.write(path, name)

    checksums = [
        {"file": review_zip.name, "sha256": sha256_file(review_zip), "bytes": str(review_zip.stat().st_size)},
        {"file": code_zip.name, "sha256": sha256_file(code_zip), "bytes": str(code_zip.stat().st_size)},
    ]
    write_tsv(module_root / "package_checksums.tsv", checksums)

    if status != "PASS":
        print("")
        print("Some files are missing or invalid. Re-run the same one-click BAT to resume.")
        raise SystemExit(5)

    print("")
    print("[DONE] Metadata audit, 83-sample panel selection, and FASTQ download are complete.")
    print(f"Results archive: {review_zip}")
    print(f"Source archive: {code_zip}")


if __name__ == "__main__":
    main()
