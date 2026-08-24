from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from common import md5_file, parse_int, read_tsv, write_json, write_tsv


PRINT_LOCK = threading.Lock()
STATUS_LOCK = threading.Lock()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def quarantine(path: Path, quarantine_root: Path, reason: str) -> Path:
    quarantine_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = quarantine_root / f"{path.name}.{stamp}.{reason}"
    shutil.move(str(path), str(target))
    return target


def validate_file(path: Path, expected_bytes: int, expected_md5: str) -> Tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    observed_bytes = path.stat().st_size
    if expected_bytes > 0 and observed_bytes != expected_bytes:
        return False, f"size_mismatch:{observed_bytes}!={expected_bytes}"
    observed_md5 = md5_file(path)
    if expected_md5 and observed_md5.lower() != expected_md5.lower():
        return False, f"md5_mismatch:{observed_md5}!={expected_md5}"
    return True, "verified"


def run_curl(url: str, part_path: Path, log_path: Path, retries: int) -> int:
    command = [
        "curl.exe",
        "--fail",
        "--location",
        "--retry",
        str(retries),
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--speed-time",
        "180",
        "--speed-limit",
        "1024",
        "--continue-at",
        "-",
        "--output",
        str(part_path),
        url,
    ]
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write("\nCOMMAND: " + " ".join(command) + "\n")
        process = subprocess.run(command, stdout=handle, stderr=handle, check=False)
    return int(process.returncode)


def download_one(
    row: Dict[str, str],
    download_root: Path,
    log_root: Path,
    quarantine_root: Path,
    retries: int,
) -> Dict[str, str]:
    population = row.get("population", "unassigned")
    run_accession = row.get("run_accession", "")
    filename = row.get("filename", "")
    url = row.get("url", "")
    expected_md5 = row.get("md5", "").lower()
    expected_bytes = parse_int(row.get("bytes", 0))

    final_dir = download_root / population
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / filename
    part_path = final_path.with_suffix(final_path.suffix + ".part")
    log_path = log_root / f"{run_accession}_{row.get('mate', '')}.log"
    log_root.mkdir(parents=True, exist_ok=True)

    result = {
        "population": population,
        "sample_accession": row.get("sample_accession", ""),
        "run_accession": run_accession,
        "mate": row.get("mate", ""),
        "filename": filename,
        "url": url,
        "expected_bytes": str(expected_bytes),
        "expected_md5": expected_md5,
        "final_path": str(final_path),
        "status": "",
        "message": "",
        "observed_bytes": "",
        "observed_md5": "",
    }

    if final_path.exists():
        ok, message = validate_file(final_path, expected_bytes, expected_md5)
        if ok:
            result["status"] = "VERIFIED_EXISTING"
            result["message"] = message
            result["observed_bytes"] = str(final_path.stat().st_size)
            result["observed_md5"] = expected_md5
            log(f"[SKIP] {filename} already verified")
            return result
        moved = quarantine(final_path, quarantine_root, "invalid_final")
        log(f"[QUARANTINE] {filename} -> {moved.name} ({message})")

    if part_path.exists() and expected_bytes > 0 and part_path.stat().st_size > expected_bytes:
        moved = quarantine(part_path, quarantine_root, "oversize_partial")
        log(f"[QUARANTINE] oversize partial {filename} -> {moved.name}")

    current = part_path.stat().st_size if part_path.exists() else 0
    log(
        f"[DOWNLOAD] {filename} "
        f"({current / (1024**2):.1f}/{expected_bytes / (1024**2):.1f} MiB)"
    )
    return_code = run_curl(url, part_path, log_path, retries)

    # curl code 33 means the server rejected resume. Restart once from zero.
    if return_code == 33 and part_path.exists():
        moved = quarantine(part_path, quarantine_root, "resume_rejected")
        log(f"[RESTART] Server rejected resume for {filename}; archived {moved.name}")
        return_code = run_curl(url, part_path, log_path, retries)

    if return_code != 0:
        result["status"] = "DOWNLOAD_FAILED"
        result["message"] = f"curl_return_code={return_code}"
        result["observed_bytes"] = str(part_path.stat().st_size if part_path.exists() else 0)
        log(f"[FAIL] {filename}: curl return code {return_code}")
        return result

    ok, message = validate_file(part_path, expected_bytes, expected_md5)
    if not ok:
        result["status"] = "VALIDATION_FAILED"
        result["message"] = message
        result["observed_bytes"] = str(part_path.stat().st_size if part_path.exists() else 0)
        if part_path.exists():
            moved = quarantine(part_path, quarantine_root, "validation_failed")
            log(f"[FAIL] {filename}: {message}; archived {moved.name}")
        return result

    observed_md5 = md5_file(part_path)
    os.replace(part_path, final_path)
    result["status"] = "DOWNLOADED_AND_VERIFIED"
    result["message"] = "verified"
    result["observed_bytes"] = str(final_path.stat().st_size)
    result["observed_md5"] = observed_md5
    log(f"[OK] {filename}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastq_manifest", required=True)
    parser.add_argument("--disk_preflight_pass", required=True)
    parser.add_argument("--download_root", required=True)
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--status_dir", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=12)
    args = parser.parse_args()

    if not Path(args.disk_preflight_pass).exists():
        raise SystemExit("[ERROR] Disk preflight PASS marker is absent; download is blocked.")
    if shutil.which("curl.exe") is None and shutil.which("curl") is None:
        raise SystemExit("[ERROR] curl.exe is unavailable on PATH.")

    manifest = read_tsv(Path(args.fastq_manifest))
    download_root = Path(args.download_root)
    log_root = Path(args.log_root)
    status_dir = Path(args.status_dir)
    quarantine_root = status_dir / "quarantine"
    download_root.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("Finger millet resumable FASTQ download")
    print("======================================")
    print(f"Files: {len(manifest)}")
    print(f"Workers: {max(1, args.workers)}")
    print(f"Target: {download_root}")
    print("Re-running this command resumes .part files and skips verified files.")
    print("")

    results: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                download_one,
                row,
                download_root,
                log_root,
                quarantine_root,
                args.retries,
            )
            for row in manifest
        ]
        total = len(futures)
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            with STATUS_LOCK:
                write_tsv(status_dir / "download_status.tsv", sorted(results, key=lambda r: (r["run_accession"], r["mate"])))
                with (status_dir / "download_events.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, sort_keys=True) + "\n")
            if completed == 1 or completed % 10 == 0 or completed == total:
                log(f"[PROGRESS] {completed}/{total}")

    status_counts: Dict[str, int] = {}
    for row in results:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    failures = [
        row for row in results
        if row["status"] not in {"VERIFIED_EXISTING", "DOWNLOADED_AND_VERIFIED"}
    ]
    summary = {
        "status": "COMPLETE" if not failures and len(results) == len(manifest) else "INCOMPLETE",
        "manifest_files": len(manifest),
        "processed_files": len(results),
        "status_counts": status_counts,
        "failure_count": len(failures),
        "download_root": str(download_root),
    }
    write_json(status_dir / "download_summary.json", summary)
    write_tsv(status_dir / "download_failures.tsv", failures)

    print("")
    print(f"Download status: {summary['status']}")
    print(f"Failures: {len(failures)}")
    for key in sorted(status_counts):
        print(f"  {key}: {status_counts[key]}")
    if failures:
        print("")
        print("Partial files were retained. Run the same BAT again to resume.")
        raise SystemExit(4)


if __name__ == "__main__":
    main()
