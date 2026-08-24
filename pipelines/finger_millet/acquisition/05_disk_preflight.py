from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from common import format_gib, parse_int, read_tsv, write_json, write_tsv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastq_manifest", required=True)
    parser.add_argument("--download_root", required=True)
    parser.add_argument("--temp_root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--minimum_free_gb", type=float, default=80.0)
    parser.add_argument("--reserve_gb", type=float, default=40.0)
    args = parser.parse_args()

    fastq_manifest = Path(args.fastq_manifest)
    download_root = Path(args.download_root)
    temp_root = Path(args.temp_root)
    outdir = Path(args.outdir)
    download_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_tsv(fastq_manifest)
    total_expected = sum(parse_int(row.get("bytes", 0)) for row in rows)
    verified_or_present = 0
    partial_bytes = 0
    existing_rows = []

    for row in rows:
        population = row.get("population", "unassigned")
        run_accession = row.get("run_accession", "")
        filename = row.get("filename", "")
        final_path = download_root / population / filename
        part_path = final_path.with_suffix(final_path.suffix + ".part")
        expected = parse_int(row.get("bytes", 0))
        final_size = final_path.stat().st_size if final_path.exists() else 0
        part_size = part_path.stat().st_size if part_path.exists() else 0
        if final_size == expected and expected > 0:
            verified_or_present += final_size
        partial_bytes += part_size
        existing_rows.append(
            {
                "population": population,
                "run_accession": run_accession,
                "filename": filename,
                "expected_bytes": str(expected),
                "final_bytes": str(final_size),
                "partial_bytes": str(part_size),
            }
        )

    remaining_download = max(0, total_expected - verified_or_present - partial_bytes)
    usage = shutil.disk_usage(str(download_root))
    free_bytes = usage.free
    minimum_free_bytes = int(args.minimum_free_gb * (1024 ** 3))
    reserve_bytes = int(args.reserve_gb * (1024 ** 3))
    required_now = remaining_download + reserve_bytes
    required_threshold = max(minimum_free_bytes, required_now)

    same_drive = download_root.drive.upper() == temp_root.drive.upper()
    environment_temp = os.environ.get("TEMP", "")
    environment_tmp = os.environ.get("TMP", "")
    temp_on_target_drive = (
        Path(environment_temp).drive.upper() == download_root.drive.upper()
        and Path(environment_tmp).drive.upper() == download_root.drive.upper()
    )

    checks = {
        "manifest_rows": len(rows),
        "expected_fastq_files": len(rows),
        "total_expected_bytes": total_expected,
        "already_complete_bytes": verified_or_present,
        "partial_bytes": partial_bytes,
        "remaining_download_bytes": remaining_download,
        "reserve_bytes": reserve_bytes,
        "free_bytes": free_bytes,
        "required_threshold_bytes": required_threshold,
        "download_temp_same_drive": same_drive,
        "process_temp_on_target_drive": temp_on_target_drive,
        "TEMP": environment_temp,
        "TMP": environment_tmp,
    }

    status = (
        "PASS"
        if len(rows) > 0
        and total_expected > 0
        and free_bytes >= required_threshold
        and same_drive
        and temp_on_target_drive
        else "FAIL"
    )

    write_tsv(outdir / "existing_download_inventory.tsv", existing_rows)
    write_json(outdir / "disk_preflight.json", {"status": status, "checks": checks})

    lines = [
        "Finger millet FASTQ disk preflight",
        "===================================",
        "",
        f"Status: {status}",
        f"Target drive: {download_root.drive}",
        f"FASTQ files planned: {len(rows)}",
        f"Expected compressed download: {format_gib(total_expected)} GiB",
        f"Already complete: {format_gib(verified_or_present)} GiB",
        f"Existing partial files: {format_gib(partial_bytes)} GiB",
        f"Remaining download: {format_gib(remaining_download)} GiB",
        f"Reserved downstream workspace: {args.reserve_gb:.1f} GiB",
        f"Free space: {format_gib(free_bytes)} GiB",
        f"Required free-space threshold: {format_gib(required_threshold)} GiB",
        f"Download and temp roots share drive: {same_drive}",
        f"TEMP/TMP redirected to target drive: {temp_on_target_drive}",
        "",
        f"TEMP={environment_temp}",
        f"TMP={environment_tmp}",
    ]
    (outdir / "disk_preflight.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    marker = outdir / "DISK_PREFLIGHT_PASS.txt"
    if status == "PASS":
        marker.write_text("PASS\n", encoding="utf-8")
    else:
        if marker.exists():
            marker.unlink()
        raise SystemExit(3)


if __name__ == "__main__":
    main()
