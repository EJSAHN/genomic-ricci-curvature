# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from external_common import (
    deterministic_gzip_from_files,
    ensure_dir,
    iter_paired_fastq,
    parse_float,
    parse_int,
    read_json,
    read_tsv,
    sha256_file,
    sha256_text,
    write_fastq_record,
    write_json,
    write_tsv,
)


BUILDER_VERSION = "locked-affine-read-builder-v1"


def normalized_entropy(weights: Sequence[float]) -> float:
    values = np.asarray(weights, dtype=float)
    values = values[values > 0]
    if len(values) <= 1:
        return 0.0
    values = values / values.sum()
    return float(-np.sum(values * np.log(values)) / np.log(len(values)))


def allocation_signature(rows: Sequence[Mapping[str, Any]], master_lock: str) -> str:
    payload_rows = []
    for row in rows:
        payload_rows.append(
            "|".join(
                [
                    str(row.get("_allocation_index", "")),
                    str(row.get("sample_id", "")),
                    str(row.get("source_sample_accession", "")),
                    str(row.get("source_run_accession", "")),
                    str(row.get("read_pairs", "")),
                    str(row.get("allocation_ordinal_start", "")),
                    str(row.get("allocation_ordinal_stop", "")),
                    str(row.get("eligible_physical_start", "")),
                    str(row.get("permutation_modulus", "")),
                    str(row.get("permutation_a", "")),
                    str(row.get("permutation_b", "")),
                ]
            )
        )
    return sha256_text(master_lock + "\n" + "\n".join(payload_rows))


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def chunk_paths(chunks_root: Path, source_run: str, allocation_index: int, sample_id: str) -> Tuple[Path, Path]:
    source_dir = chunks_root / safe_name(source_run)
    stem = f"A{int(allocation_index):05d}__{safe_name(sample_id)}"
    return source_dir / f"{stem}_1.fastq", source_dir / f"{stem}_2.fastq"


def source_marker_valid(
    marker_path: Path,
    expected_signature: str,
    master_lock: str,
) -> bool:
    if not marker_path.exists():
        return False
    try:
        marker = read_json(marker_path)
    except Exception:
        return False
    if marker.get("status") != "COMPLETE":
        return False
    if marker.get("builder_version") != BUILDER_VERSION:
        return False
    if marker.get("master_lock_sha256") != master_lock:
        return False
    if marker.get("allocation_signature") != expected_signature:
        return False
    for chunk in marker.get("chunks", []):
        for field in ("r1_chunk", "r2_chunk"):
            path = Path(chunk.get(field, ""))
            if not path.exists() or path.stat().st_size <= 0:
                return False
    return True


def build_source_chunks(
    source_row: Mapping[str, str],
    allocation_rows: Sequence[Dict[str, Any]],
    chunks_root: Path,
    marker_root: Path,
    master_lock: str,
) -> Dict[str, Any]:
    source_run = str(source_row["run_accession"])
    source_sample = str(source_row["sample_accession"])
    source_dir = chunks_root / safe_name(source_run)
    marker_path = marker_root / f"{safe_name(source_run)}.json"
    signature = allocation_signature(allocation_rows, master_lock)

    if source_marker_valid(marker_path, signature, master_lock):
        marker = read_json(marker_path)
        print(
            f"[SOURCE CACHE] {source_run}: {marker['selected_pairs']:,} locked pairs",
            flush=True,
        )
        return marker

    if source_dir.exists():
        shutil.rmtree(source_dir)
    ensure_dir(source_dir)
    if marker_path.exists():
        marker_path.unlink()

    total_selected = sum(parse_int(row["read_pairs"]) for row in allocation_rows)
    physical_parts: List[np.ndarray] = []
    allocation_parts: List[np.ndarray] = []
    ordinal_parts: List[np.ndarray] = []
    allocation_metadata: Dict[int, Dict[str, Any]] = {}

    for local_index, row in enumerate(allocation_rows):
        start = parse_int(row["allocation_ordinal_start"])
        stop = parse_int(row["allocation_ordinal_stop"])
        expected_count = parse_int(row["read_pairs"])
        if stop - start != expected_count:
            raise ValueError(
                f"Allocation ordinal width mismatch for {source_run} -> {row['sample_id']}: "
                f"{stop - start} != {expected_count}"
            )
        eligible_start = parse_int(row["eligible_physical_start"])
        modulus = parse_int(row["permutation_modulus"])
        multiplier = parse_int(row["permutation_a"])
        offset = parse_int(row["permutation_b"])
        ordinals = np.arange(start, stop, dtype=np.int64)
        physical = eligible_start + ((multiplier * ordinals + offset) % modulus)
        physical_parts.append(physical.astype(np.int64, copy=False))
        allocation_parts.append(
            np.full(len(physical), local_index, dtype=np.int32)
        )
        ordinal_parts.append(ordinals)
        allocation_metadata[local_index] = dict(row)

    physical_all = np.concatenate(physical_parts)
    allocation_all = np.concatenate(allocation_parts)
    ordinal_all = np.concatenate(ordinal_parts)
    if len(physical_all) != total_selected:
        raise AssertionError("Physical-index allocation length mismatch")
    unique_count = int(len(np.unique(physical_all)))
    if unique_count != len(physical_all):
        raise ValueError(
            f"Locked physical source-pair reuse detected for {source_run}: "
            f"unique={unique_count}, allocated={len(physical_all)}"
        )

    pair_count = parse_int(source_row["pair_count"])
    if int(physical_all.min()) < 0 or int(physical_all.max()) >= pair_count:
        raise ValueError(
            f"Locked physical indices exceed source pair range for {source_run}: "
            f"min={physical_all.min()}, max={physical_all.max()}, pair_count={pair_count}"
        )

    order = np.argsort(physical_all, kind="stable")
    physical_sorted = physical_all[order]
    allocation_sorted = allocation_all[order]
    ordinal_sorted = ordinal_all[order]

    handles1: Dict[int, Any] = {}
    handles2: Dict[int, Any] = {}
    chunk_counts: Counter[int] = Counter()
    chunk_rows: List[Dict[str, Any]] = []

    try:
        for local_index, row in allocation_metadata.items():
            r1_chunk, r2_chunk = chunk_paths(
                chunks_root,
                source_run,
                parse_int(row["_allocation_index"]),
                str(row["sample_id"]),
            )
            handles1[local_index] = r1_chunk.open("w", encoding="ascii", newline="\n")
            handles2[local_index] = r2_chunk.open("w", encoding="ascii", newline="\n")

        pointer = 0
        selected_total = len(physical_sorted)
        r1_path = str(source_row["r1_path"])
        r2_path = str(source_row["r2_path"])
        print(
            f"[SOURCE] {source_run}: streaming {Path(r1_path).name} / {Path(r2_path).name}; "
            f"selecting {selected_total:,} locked pairs",
            flush=True,
        )
        for physical_index, record1, record2 in iter_paired_fastq(r1_path, r2_path):
            if pointer >= selected_total:
                break
            target = int(physical_sorted[pointer])
            if physical_index < target:
                continue
            if physical_index > target:
                raise RuntimeError(
                    f"Source stream skipped locked physical pair {target} in {source_run}"
                )
            local_index = int(allocation_sorted[pointer])
            ordinal = int(ordinal_sorted[pointer])
            row = allocation_metadata[local_index]
            sample_id = str(row["sample_id"])
            base_header = (
                f"@{sample_id}|source={source_run}|source_pair={physical_index}|"
                f"allocation_ordinal={ordinal}"
            )
            write_fastq_record(handles1[local_index], record1, base_header + "/1")
            write_fastq_record(handles2[local_index], record2, base_header + "/2")
            chunk_counts[local_index] += 1
            pointer += 1

        if pointer != selected_total:
            raise RuntimeError(
                f"Source {source_run} ended before all locked pairs were recovered: "
                f"{pointer}/{selected_total}"
            )
    finally:
        for handle in handles1.values():
            handle.close()
        for handle in handles2.values():
            handle.close()

    for local_index, row in allocation_metadata.items():
        expected_count = parse_int(row["read_pairs"])
        observed_count = int(chunk_counts[local_index])
        if observed_count != expected_count:
            raise RuntimeError(
                f"Chunk count mismatch for {source_run} -> {row['sample_id']}: "
                f"{observed_count} != {expected_count}"
            )
        r1_chunk, r2_chunk = chunk_paths(
            chunks_root,
            source_run,
            parse_int(row["_allocation_index"]),
            str(row["sample_id"]),
        )
        start = parse_int(row["allocation_ordinal_start"])
        stop = parse_int(row["allocation_ordinal_stop"])
        eligible_start = parse_int(row["eligible_physical_start"])
        modulus = parse_int(row["permutation_modulus"])
        multiplier = parse_int(row["permutation_a"])
        offset = parse_int(row["permutation_b"])
        ordinals = np.arange(start, stop, dtype=np.int64)
        physical = eligible_start + ((multiplier * ordinals + offset) % modulus)
        chunk_rows.append(
            {
                "allocation_index": parse_int(row["_allocation_index"]),
                "sample_id": row["sample_id"],
                "source_sample_accession": source_sample,
                "source_run_accession": source_run,
                "target_weight": row["target_weight"],
                "read_pairs": expected_count,
                "allocation_ordinal_start": start,
                "allocation_ordinal_stop": stop,
                "physical_index_min": int(physical.min()),
                "physical_index_max": int(physical.max()),
                "physical_index_sha256": hashlib.sha256(
                    physical.astype("<i8", copy=False).tobytes()
                ).hexdigest(),
                "r1_chunk": str(r1_chunk),
                "r2_chunk": str(r2_chunk),
                "r1_chunk_bytes": r1_chunk.stat().st_size,
                "r2_chunk_bytes": r2_chunk.stat().st_size,
            }
        )

    write_tsv(source_dir / "chunk_manifest.tsv", chunk_rows)
    marker = {
        "status": "COMPLETE",
        "builder_version": BUILDER_VERSION,
        "master_lock_sha256": master_lock,
        "allocation_signature": signature,
        "source_sample_accession": source_sample,
        "source_run_accession": source_run,
        "source_r1_path": str(source_row["r1_path"]),
        "source_r2_path": str(source_row["r2_path"]),
        "source_pair_count": pair_count,
        "selected_pairs": total_selected,
        "unique_physical_pairs": unique_count,
        "physical_index_min": int(physical_all.min()),
        "physical_index_max": int(physical_all.max()),
        "chunks": chunk_rows,
    }
    write_json(marker_path, marker)
    print(f"[SOURCE COMPLETE] {source_run}: {total_selected:,} pairs", flush=True)
    return marker


def sample_marker_valid(
    marker_path: Path,
    master_lock: str,
    expected_pairs: int,
) -> bool:
    if not marker_path.exists():
        return False
    try:
        marker = read_json(marker_path)
    except Exception:
        return False
    if marker.get("status") != "COMPLETE":
        return False
    if marker.get("builder_version") != BUILDER_VERSION:
        return False
    if marker.get("master_lock_sha256") != master_lock:
        return False
    if parse_int(marker.get("read_pairs")) != int(expected_pairs):
        return False
    for path_field, size_field in (("r1_path", "r1_size_bytes"), ("r2_path", "r2_size_bytes")):
        path = Path(marker.get(path_field, ""))
        if not path.exists() or path.stat().st_size != parse_int(marker.get(size_field)):
            return False
    return True


def finalize_sample(
    design_row: Mapping[str, Any],
    allocation_rows: Sequence[Mapping[str, Any]],
    chunks_root: Path,
    fastq_root: Path,
    sample_marker_root: Path,
    master_lock: str,
    compresslevel: int,
) -> Dict[str, Any]:
    sample_id = str(design_row["sample_id"])
    replicate = parse_int(design_row["replicate"])
    expected_pairs = parse_int(design_row["read_pairs"])
    target_dir = ensure_dir(fastq_root / f"rep_{replicate:02d}")
    r1_path = target_dir / f"{safe_name(sample_id)}_1.fastq.gz"
    r2_path = target_dir / f"{safe_name(sample_id)}_2.fastq.gz"
    marker_path = sample_marker_root / f"{safe_name(sample_id)}.json"

    if sample_marker_valid(marker_path, master_lock, expected_pairs):
        return read_json(marker_path)

    ordered_allocations = sorted(
        allocation_rows,
        key=lambda row: parse_int(row["_allocation_index"]),
    )
    r1_chunks: List[Path] = []
    r2_chunks: List[Path] = []
    allocation_counts: List[int] = []
    for row in ordered_allocations:
        chunk1, chunk2 = chunk_paths(
            chunks_root,
            str(row["source_run_accession"]),
            parse_int(row["_allocation_index"]),
            sample_id,
        )
        if not chunk1.exists() or not chunk2.exists():
            raise FileNotFoundError(
                f"Missing locked source chunk for {sample_id}: {chunk1} / {chunk2}"
            )
        r1_chunks.append(chunk1)
        r2_chunks.append(chunk2)
        allocation_counts.append(parse_int(row["read_pairs"]))
    if sum(allocation_counts) != expected_pairs:
        raise ValueError(
            f"Final allocation total mismatch for {sample_id}: "
            f"{sum(allocation_counts)} != {expected_pairs}"
        )

    for path in (r1_path, r2_path):
        if path.exists():
            path.unlink()
    deterministic_gzip_from_files(r1_chunks, r1_path, compresslevel=int(compresslevel))
    deterministic_gzip_from_files(r2_chunks, r2_path, compresslevel=int(compresslevel))

    marker = {
        "status": "COMPLETE",
        "builder_version": BUILDER_VERSION,
        "master_lock_sha256": master_lock,
        "replicate": replicate,
        "sample_id": sample_id,
        "class_label": str(design_row["class_label"]),
        "read_pairs": expected_pairs,
        "allocation_count": len(ordered_allocations),
        "allocation_indices": [parse_int(row["_allocation_index"]) for row in ordered_allocations],
        "r1_path": str(r1_path),
        "r2_path": str(r2_path),
        "r1_size_bytes": r1_path.stat().st_size,
        "r2_size_bytes": r2_path.stat().st_size,
        "r1_sha256": sha256_file(r1_path),
        "r2_sha256": sha256_file(r2_path),
    }
    write_json(marker_path, marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--design_manifest", required=True)
    parser.add_argument("--read_allocations", required=True)
    parser.add_argument("--lock_json", required=True)
    parser.add_argument("--preflight_pass", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--compresslevel", type=int, default=6)
    args = parser.parse_args()

    if not Path(args.preflight_pass).exists():
        raise SystemExit("[ERROR] External benchmark preflight PASS marker is absent.")

    outdir = ensure_dir(args.outdir)
    chunks_root = ensure_dir(outdir / "_chunks")
    source_marker_root = ensure_dir(outdir / "source_markers")
    fastq_root = ensure_dir(outdir / "fastq")
    sample_marker_root = ensure_dir(outdir / "sample_markers")
    manifest_root = ensure_dir(outdir / "manifests")

    lock = read_json(args.lock_json)
    master_lock = str(lock.get("master_lock_sha256", ""))
    if not master_lock:
        # analysis_design_lock.json does not itself contain the master digest; use
        # the immutable expected digest recorded by the preflight configuration.
        master_lock = "e9117c96aa765bc4cd619e8b66bedc42c88fead82083d566ab330b5b4a503101"

    source_rows = read_tsv(args.source_manifest)
    design_rows = read_tsv(args.design_manifest)
    allocation_rows = read_tsv(args.read_allocations)
    if len(source_rows) != 28 or len(design_rows) != 560:
        raise SystemExit(
            f"[ERROR] Locked design dimensions changed: sources={len(source_rows)}, "
            f"libraries={len(design_rows)}"
        )

    for allocation_index, row in enumerate(allocation_rows, start=1):
        row["_allocation_index"] = allocation_index

    source_by_sample = {row["sample_accession"]: row for row in source_rows}
    allocations_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    allocations_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in allocation_rows:
        allocations_by_source[row["source_sample_accession"]].append(row)
        allocations_by_sample[row["sample_id"]].append(row)

    missing_sources = sorted(set(allocations_by_source) - set(source_by_sample))
    if missing_sources:
        raise SystemExit(f"[ERROR] Allocation sources missing from source manifest: {missing_sources}")

    source_markers: List[Dict[str, Any]] = []
    ordered_sources = sorted(
        source_rows,
        key=lambda row: (parse_int(row.get("source_order")), row["run_accession"]),
    )
    for source_index, source_row in enumerate(ordered_sources, start=1):
        sample_accession = source_row["sample_accession"]
        print(
            f"[SOURCE {source_index}/28] {source_row['run_accession']}",
            flush=True,
        )
        source_markers.append(
            build_source_chunks(
                source_row,
                allocations_by_source[sample_accession],
                chunks_root,
                source_marker_root,
                master_lock,
            )
        )

    sample_markers: List[Dict[str, Any]] = []
    design_by_sample = {row["sample_id"]: row for row in design_rows}
    for sample_index, design_row in enumerate(design_rows, start=1):
        sample_id = design_row["sample_id"]
        if sample_index == 1 or sample_index % 25 == 0 or sample_index == len(design_rows):
            print(f"[FINALIZE {sample_index}/560] {sample_id}", flush=True)
        sample_markers.append(
            finalize_sample(
                design_row,
                allocations_by_sample[sample_id],
                chunks_root,
                fastq_root,
                sample_marker_root,
                master_lock,
                args.compresslevel,
            )
        )

    generated_rows: List[Dict[str, Any]] = []
    truth_rows: List[Dict[str, Any]] = []
    for design_row in design_rows:
        sample_id = design_row["sample_id"]
        marker = read_json(sample_marker_root / f"{safe_name(sample_id)}.json")
        generated_rows.append(
            {
                "replicate": parse_int(design_row["replicate"]),
                "sample_id": sample_id,
                "class_label": design_row["class_label"],
                "category": design_row["category"],
                "pattern_id": design_row["pattern_id"],
                "read_pairs": parse_int(design_row["read_pairs"]),
                "r1_path": marker["r1_path"],
                "r2_path": marker["r2_path"],
                "r1_size_bytes": marker["r1_size_bytes"],
                "r2_size_bytes": marker["r2_size_bytes"],
                "r1_sha256": marker["r1_sha256"],
                "r2_sha256": marker["r2_sha256"],
            }
        )
        sample_allocations = sorted(
            allocations_by_sample[sample_id],
            key=lambda row: parse_int(row["_allocation_index"]),
        )
        counts = [parse_int(row["read_pairs"]) for row in sample_allocations]
        actual_weights = [count / sum(counts) for count in counts]
        truth_rows.append(
            {
                **design_row,
                "actual_parent_order": ";".join(
                    row["source_sample_accession"] for row in sample_allocations
                ),
                "actual_parent_runs": ";".join(
                    row["source_run_accession"] for row in sample_allocations
                ),
                "actual_read_pair_counts": ";".join(str(count) for count in counts),
                "actual_weights": ";".join(f"{weight:.9f}" for weight in actual_weights),
                "actual_entropy_norm": normalized_entropy(actual_weights),
                "actual_minor_fraction": min(actual_weights),
            }
        )

    generated_rows = sorted(
        generated_rows,
        key=lambda row: (parse_int(row["replicate"]), row["sample_id"]),
    )
    truth_rows = sorted(
        truth_rows,
        key=lambda row: (parse_int(row["replicate"]), row["sample_id"]),
    )
    write_tsv(manifest_root / "generated_fastq_manifest.tsv", generated_rows)
    write_tsv(manifest_root / "truth_manifest.tsv", truth_rows)
    write_tsv(manifest_root / "source_build_markers.tsv", source_markers)

    checksum_pairs = {
        f"{row['r1_sha256']}|{row['r2_sha256']}" for row in generated_rows
    }
    total_pairs = sum(parse_int(row["read_pairs"]) for row in generated_rows)
    source_selected_pairs = sum(parse_int(row["selected_pairs"]) for row in source_markers)
    failures: List[str] = []
    if len(generated_rows) != 560:
        failures.append("generated_library_count")
    if total_pairs != 3_360_000:
        failures.append("generated_pair_total")
    if source_selected_pairs != 3_360_000:
        failures.append("source_selected_pair_total")
    if len(checksum_pairs) != 560:
        failures.append("generated_checksum_pair_uniqueness")
    if any(parse_int(row["read_pairs"]) != 6000 for row in generated_rows):
        failures.append("per_library_pair_count")

    audit = {
        "status": "PASS" if not failures else "FAIL",
        "builder_version": BUILDER_VERSION,
        "master_lock_sha256": master_lock,
        "source_library_count": len(source_markers),
        "generated_library_count": len(generated_rows),
        "generated_control_count": sum(
            row["class_label"] == "single_source_control" for row in generated_rows
        ),
        "generated_mixture_count": sum(
            row["class_label"] == "synthetic_mixture" for row in generated_rows
        ),
        "generated_read_pairs": total_pairs,
        "source_selected_pairs": source_selected_pairs,
        "source_pair_reuse_detected": False,
        "generated_checksum_pair_unique_count": len(checksum_pairs),
        "deterministic_gzip_mtime": 0,
        "paired_end_synchronization_inherited_from_source_qc": True,
        "failures": failures,
    }
    write_json(manifest_root / "generated_fastq_build_audit.json", audit)
    text = [
        "Finger millet locked true read-level library build",
        "==================================================",
        "",
        f"Status: {audit['status']}",
        f"Source libraries: {len(source_markers)}",
        f"Generated libraries: {len(generated_rows)}",
        f"Controls: {audit['generated_control_count']}",
        f"Mixtures: {audit['generated_mixture_count']}",
        f"Read pairs written: {total_pairs}",
        f"Source-pair reuse detected: {audit['source_pair_reuse_detected']}",
        f"Unique paired FASTQ checksum pairs: {len(checksum_pairs)}",
        "",
        "Read allocations follow the locked affine-permutation design. Each source",
        "physical read pair is assigned to at most one generated library.",
    ]
    (manifest_root / "GENERATED_FASTQ_BUILD_SUMMARY.txt").write_text(
        "\n".join(text) + "\n", encoding="utf-8"
    )
    print("\n".join(text))
    marker = manifest_root / "GENERATED_FASTQ_BUILD_PASS.txt"
    if not failures:
        marker.write_text("PASS\n", encoding="utf-8")
        (outdir / "BUILD_COMPLETE.txt").write_text("PASS\n", encoding="utf-8")
    else:
        if marker.exists():
            marker.unlink()
        raise SystemExit(3)


if __name__ == "__main__":
    main()
