from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List

from common import (
    normalize_token,
    parse_int,
    read_tsv,
    split_semicolon,
    stable_selection_key,
    url_https_from_ena,
    write_json,
    write_tsv,
)


TARGET_COUNTS = {
    "Pop-1": 12,
    "Pop-2": 12,
    "Pop-3": 12,
    "Pop-4": 12,
    "Pop-5": 9,
    "Pop-6": 14,
    "Pop-7": 12,
}



MATE1_PATTERNS = (
    re.compile(r"(?:^|[_\.-])1\.(?:fastq|fq)\.gz$", re.IGNORECASE),
    re.compile(r"(?:^|[_\.-])R1(?:[_\.-]|$).*\.(?:fastq|fq)\.gz$", re.IGNORECASE),
)
MATE2_PATTERNS = (
    re.compile(r"(?:^|[_\.-])2\.(?:fastq|fq)\.gz$", re.IGNORECASE),
    re.compile(r"(?:^|[_\.-])R2(?:[_\.-]|$).*\.(?:fastq|fq)\.gz$", re.IGNORECASE),
)


def url_filename(url: str) -> str:
    return Path(urlparse(url).path).name


def classify_mate(filename: str):
    for pattern in MATE1_PATTERNS:
        if pattern.search(filename):
            return 1
    for pattern in MATE2_PATTERNS:
        if pattern.search(filename):
            return 2
    return None


def parse_generated_fastqs(run):
    ftp = split_semicolon(run.get("fastq_ftp", ""))
    md5 = split_semicolon(run.get("fastq_md5", ""))
    sizes = split_semicolon(run.get("fastq_bytes", ""))

    if not ftp:
        return None, [], "generated_fastq_urls_missing"
    if len(ftp) != len(md5) or len(ftp) != len(sizes):
        return (
            None,
            [],
            f"generated_fastq_metadata_length_mismatch:"
            f"urls={len(ftp)};md5={len(md5)};bytes={len(sizes)}",
        )

    paired = {}
    ignored = []
    for url_value, md5_value, size_value in zip(ftp, md5, sizes):
        url = url_https_from_ena(url_value)
        filename = url_filename(url)
        mate = classify_mate(filename)
        info = {
            "filename": filename,
            "url": url,
            "md5": md5_value.lower(),
            "bytes": str(parse_int(size_value)),
        }
        if mate in (1, 2):
            if mate in paired:
                return None, ignored, f"duplicate_mate_{mate}:{filename}"
            paired[mate] = info
        else:
            ignored.append(info)

    if set(paired) != {1, 2}:
        all_names = ",".join(
            url_filename(url_https_from_ena(value)) for value in ftp
        )
        return (
            None,
            ignored,
            f"paired_mates_not_found:"
            f"mate1={1 in paired};mate2={2 in paired};files={all_names}",
        )

    for mate, info in paired.items():
        if not info["md5"]:
            return None, ignored, f"mate_{mate}_md5_missing:{info['filename']}"
        if parse_int(info["bytes"]) <= 0:
            return None, ignored, f"mate_{mate}_byte_count_missing:{info['filename']}"

    return paired, ignored, ""

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population_assignment", required=True)
    parser.add_argument("--run_metadata", required=True)
    parser.add_argument("--metadata_audit_pass", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=791522)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pass_marker = Path(args.metadata_audit_pass)
    if not pass_marker.exists():
        raise SystemExit("[ERROR] Metadata audit PASS marker is absent; panel selection is blocked.")

    assignments = read_tsv(Path(args.population_assignment))
    runs = read_tsv(Path(args.run_metadata))
    assignment_by_sample = {row["sample_accession"]: row for row in assignments}
    runs_by_sample: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for run in runs:
        runs_by_sample[run.get("sample_accession", "")].append(run)

    selected_samples: List[Dict[str, str]] = []
    for population, target in TARGET_COUNTS.items():
        candidates = [row for row in assignments if row.get("population") == population]
        if len(candidates) < target:
            raise SystemExit(f"[ERROR] {population} contains {len(candidates)} samples, fewer than target {target}.")
        ranked = sorted(
            candidates,
            key=lambda row: (
                stable_selection_key(args.seed, population, row.get("sample_accession", "")),
                row.get("sample_accession", ""),
            ),
        )
        chosen = ranked if len(candidates) == target else ranked[:target]
        for rank, row in enumerate(chosen, start=1):
            copy = dict(row)
            copy.update(
                {
                    "selection_seed": str(args.seed),
                    "selection_target": str(target),
                    "within_population_selection_rank": str(rank),
                    "selection_key_sha256": stable_selection_key(
                        args.seed, population, row.get("sample_accession", "")
                    ),
                    "selection_rule": (
                        "all_samples_retained"
                        if len(candidates) == target
                        else "lowest_sha256_keys_within_population"
                    ),
                }
            )
            selected_samples.append(copy)

    selected_samples.sort(
        key=lambda row: (
            int(row["population"].split("-")[1]),
            int(row["within_population_selection_rank"]),
        )
    )
    for index, row in enumerate(selected_samples, start=1):
        row["panel_order"] = str(index)

    selected_sample_ids = {row["sample_accession"] for row in selected_samples}
    selected_runs: List[Dict[str, str]] = []
    fastq_files: List[Dict[str, str]] = []
    ignored_files: List[Dict[str, str]] = []
    failures: List[Dict[str, str]] = []

    sample_lookup = {row["sample_accession"]: row for row in selected_samples}
    for sample_id in sorted(selected_sample_ids):
        sample_runs = sorted(runs_by_sample.get(sample_id, []), key=lambda r: r.get("run_accession", ""))
        if not sample_runs:
            failures.append({"sample_accession": sample_id, "reason": "no_run_metadata"})
            continue
        for run in sample_runs:
            run_accession = run.get("run_accession", "")
            layout = (run.get("library_layout", "") or "").upper()
            if layout != "PAIRED":
                failures.append(
                    {
                        "sample_accession": sample_id,
                        "run_accession": run_accession,
                        "reason": f"library_layout={layout}",
                    }
                )
                continue

            paired, ignored, error = parse_generated_fastqs(run)
            sample = sample_lookup[sample_id]

            for info in ignored:
                ignored_files.append(
                    {
                        "panel_order": sample["panel_order"],
                        "population": sample["population"],
                        "sample_accession": sample_id,
                        "run_accession": run_accession,
                        **info,
                        "reason": (
                            "ENA archive-generated orphan/unpaired FASTQ; "
                            "excluded from paired-read benchmark"
                        ),
                    }
                )

            if paired is None:
                failures.append(
                    {
                        "sample_accession": sample_id,
                        "run_accession": run_accession,
                        "reason": error,
                        "library_layout": layout,
                        "fastq_ftp": run.get("fastq_ftp", ""),
                        "fastq_md5": run.get("fastq_md5", ""),
                        "fastq_bytes": run.get("fastq_bytes", ""),
                        "submitted_ftp": run.get("submitted_ftp", ""),
                    }
                )
                continue

            selected_run = dict(run)
            selected_run.update(
                {
                    "population": sample["population"],
                    "panel_order": sample["panel_order"],
                    "selection_key_sha256": sample["selection_key_sha256"],
                    "paired_fastq_1": paired[1]["filename"],
                    "paired_fastq_2": paired[2]["filename"],
                    "ignored_orphan_fastq_count": str(len(ignored)),
                }
            )
            selected_runs.append(selected_run)

            for mate in (1, 2):
                info = paired[mate]
                fastq_files.append(
                    {
                        "panel_order": sample["panel_order"],
                        "population": sample["population"],
                        "sample_accession": sample_id,
                        "sample_alias": sample.get("sample_alias", ""),
                        "run_accession": run_accession,
                        "mate": str(mate),
                        "filename": info["filename"],
                        "url": info["url"],
                        "md5": info["md5"],
                        "bytes": info["bytes"],
                    }
                )

    write_tsv(outdir / "finger_millet_panel_83_samples.tsv", selected_samples)
    write_tsv(outdir / "finger_millet_panel_83_runs.tsv", selected_runs)
    write_tsv(outdir / "finger_millet_panel_83_fastq_files.tsv", fastq_files)
    write_tsv(outdir / "ignored_ena_orphan_fastq_files.tsv", ignored_files)
    write_tsv(outdir / "panel_selection_failures.tsv", failures)
    (outdir / "selected_sample_accessions.txt").write_text(
        "\n".join(row["sample_accession"] for row in selected_samples) + "\n",
        encoding="utf-8",
    )
    (outdir / "selected_run_accessions.txt").write_text(
        "\n".join(row.get("run_accession", "") for row in selected_runs) + "\n",
        encoding="utf-8",
    )

    population_counts = Counter(row["population"] for row in selected_samples)
    total_bytes = sum(parse_int(row["bytes"]) for row in fastq_files)
    status = (
        "PASS"
        if len(selected_samples) == 83
        and len(selected_runs) == 83
        and len(fastq_files) == 166
        and not failures
        and all(
            population_counts.get(pop, 0) == count
            for pop, count in TARGET_COUNTS.items()
        )
        else "FAIL"
    )
    manifest_hash = hashlib.sha256(
        "\n".join(
            f"{row['population']}\t{row['sample_accession']}\t{row['selection_key_sha256']}"
            for row in selected_samples
        ).encode("utf-8")
    ).hexdigest()

    summary = {
        "status": status,
        "selection_seed": args.seed,
        "selected_sample_count": len(selected_samples),
        "selected_run_count": len(selected_runs),
        "selected_fastq_file_count": len(fastq_files),
        "expected_fastq_file_count": 166,
        "ignored_orphan_fastq_file_count": len(ignored_files),
        "failure_count": len(failures),
        "population_counts": dict(population_counts),
        "target_counts": TARGET_COUNTS,
        "total_fastq_bytes": total_bytes,
        "selection_manifest_sha256": manifest_hash,
        "selection_uses_depth_or_qc": False,
    }
    write_json(outdir / "panel_selection_summary.json", summary)
    (outdir / "panel_selection_summary.txt").write_text(
        "\n".join(
            [
                "Finger millet population-stratified panel",
                "========================================",
                "",
                f"Status: {status}",
                f"Seed: {args.seed}",
                f"Selected biological samples: {len(selected_samples)}",
                f"Selected runs: {len(selected_runs)}",
                f"Paired FASTQ files retained: {len(fastq_files)}",
                f"ENA orphan/unpaired FASTQ files ignored: {len(ignored_files)}",
                f"Selection failures: {len(failures)}",
                f"Compressed FASTQ bytes: {total_bytes}",
                f"Selection manifest SHA-256: {manifest_hash}",
                "",
                "Population counts:",
            ]
            + [f"  {pop}: {population_counts.get(pop, 0)} (target {target})" for pop, target in TARGET_COUNTS.items()]
            + [
                "",
                "Selection was performed only within published population labels using a fixed SHA-256 ordering.",
                "Read depth, FASTQ size, QC metrics, TMS, PCA, and mapping results were not used.",
                "For paired ENA runs with a third orphan/unpaired FASTQ, only the _1 and _2 mate files were retained.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if status == "PASS":
        (outdir / "PANEL_SELECTION_PASS.txt").write_text("PASS\n", encoding="utf-8")
    else:
        print((outdir / "panel_selection_summary.txt").read_text(encoding="utf-8"))
        print(f"[DETAIL] Review: {outdir / 'panel_selection_failures.tsv'}")
        raise SystemExit("[ERROR] Panel selection audit failed.")
    print((outdir / "panel_selection_summary.txt").read_text(encoding="utf-8"))


def urllib_path(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).path


if __name__ == "__main__":
    main()
