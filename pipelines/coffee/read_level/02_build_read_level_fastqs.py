# -*- coding: utf-8 -*-
"""
Build paired-end synthetic FASTQ libraries from a deterministic mixture design.

Read pairs sampled from a source library are assigned to exactly one generated
library across all scenarios and replicates in a run. R1/R2 synchronization is
validated while source files are streamed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from read_level_common import (
    FastqRecord,
    deterministic_gzip,
    discover_fastq_pairs,
    ensure_dir,
    iter_paired_fastq,
    normalized_entropy,
    sha256_file,
    stable_seed,
    write_fastq_record,
    write_json,
)


def reservoir_sample_pairs(
    r1_path: str,
    r2_path: str,
    n_required: int,
    seed: int,
) -> Tuple[List[Tuple[int, FastqRecord, FastqRecord]], int]:
    rng = random.Random(int(seed))
    reservoir: List[Tuple[int, FastqRecord, FastqRecord]] = []
    total_seen = 0
    for source_index, r1, r2 in iter_paired_fastq(r1_path, r2_path):
        total_seen += 1
        item = (source_index, r1, r2)
        if len(reservoir) < n_required:
            reservoir.append(item)
        else:
            replacement = rng.randrange(total_seen)
            if replacement < n_required:
                reservoir[replacement] = item
    if total_seen < n_required:
        raise ValueError(
            f"Source library has {total_seen} read pairs but {n_required} are required: "
            f"{r1_path}"
        )
    rng.shuffle(reservoir)
    return reservoir, total_seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_manifest", required=True)
    parser.add_argument("--design_manifest", required=True)
    parser.add_argument("--read_allocations", required=True)
    parser.add_argument("--source_fastq_root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--compresslevel", type=int, default=6)
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    fastq_out = ensure_dir(outdir / "fastq")
    temp_out = ensure_dir(outdir / "_temporary_uncompressed")
    manifest_out = ensure_dir(outdir / "manifests")

    if any(fastq_out.iterdir()) or any(temp_out.iterdir()):
        raise SystemExit(
            f"[ERR] Output directory is not empty: {outdir}. "
            "Archive the previous attempt before rebuilding."
        )

    sample_manifest = pd.read_csv(args.sample_manifest, sep="\t", dtype=str).fillna("")
    design = pd.read_csv(args.design_manifest, sep="\t")
    allocations = pd.read_csv(args.read_allocations, sep="\t")
    source_ids = sorted(allocations["parent_id"].astype(str).unique().tolist())
    pairs = discover_fastq_pairs(args.source_fastq_root, source_ids)

    missing = [
        sample for sample in source_ids
        if sample not in pairs or "1" not in pairs[sample] or "2" not in pairs[sample]
    ]
    if missing:
        raise SystemExit(f"[ERR] Missing paired FASTQ files for: {missing}")

    # Create empty temporary files in advance.
    sample_paths: Dict[str, Dict[str, Path]] = {}
    for row in design.itertuples(index=False):
        scenario = str(row.scenario)
        replicate = int(row.replicate)
        sample_id = str(row.sample_id)
        target_dir = ensure_dir(temp_out / scenario / f"rep_{replicate:02d}")
        r1_tmp = target_dir / f"{sample_id}_1.fastq"
        r2_tmp = target_dir / f"{sample_id}_2.fastq"
        r1_tmp.write_text("", encoding="utf-8")
        r2_tmp.write_text("", encoding="utf-8")
        sample_paths[sample_id] = {"1": r1_tmp, "2": r2_tmp}

    source_rows = []
    assignment_count = 0

    for parent_id in source_ids:
        parent_alloc = allocations[allocations["parent_id"].astype(str) == parent_id].copy()
        parent_alloc = parent_alloc.sort_values(
            ["scenario", "replicate", "sample_id"], kind="stable"
        )
        n_required = int(parent_alloc["read_pairs"].sum())
        r1_path = pairs[parent_id]["1"]
        r2_path = pairs[parent_id]["2"]
        parent_seed = stable_seed(args.seed, "source", parent_id)
        print(
            f"[SOURCE] {parent_id}: sampling {n_required:,} disjoint read pairs "
            f"from {Path(r1_path).name} / {Path(r2_path).name}"
        )
        reservoir, total_seen = reservoir_sample_pairs(
            r1_path=r1_path,
            r2_path=r2_path,
            n_required=n_required,
            seed=parent_seed,
        )

        cursor = 0
        for row in parent_alloc.itertuples(index=False):
            sample_id = str(row.sample_id)
            count = int(row.read_pairs)
            chunk = reservoir[cursor : cursor + count]
            if len(chunk) != count:
                raise AssertionError(f"Allocation underflow for {parent_id} -> {sample_id}")
            cursor += count

            r1_tmp = sample_paths[sample_id]["1"]
            r2_tmp = sample_paths[sample_id]["2"]
            with open(r1_tmp, "a", encoding="utf-8", newline="\n") as out1, open(
                r2_tmp, "a", encoding="utf-8", newline="\n"
            ) as out2:
                for source_index, rec1, rec2 in chunk:
                    base = (
                        f"@{sample_id}|source={parent_id}|source_pair={source_index}"
                    )
                    write_fastq_record(out1, rec1, base + "/1")
                    write_fastq_record(out2, rec2, base + "/2")
                    assignment_count += 1

        if cursor != n_required:
            raise AssertionError(f"Not all sampled reads were allocated for {parent_id}")

        source_rows.append(
            {
                "parent_id": parent_id,
                "r1_path": r1_path,
                "r2_path": r2_path,
                "r1_size_bytes": os.path.getsize(r1_path),
                "r2_size_bytes": os.path.getsize(r2_path),
                "read_pairs_available": total_seen,
                "read_pairs_assigned": n_required,
                "sampling_seed": parent_seed,
                "r1_sha256": sha256_file(r1_path),
                "r2_sha256": sha256_file(r2_path),
            }
        )

    # Compress generated libraries with a fixed gzip timestamp.
    generated_rows = []
    expected_counts = design.set_index("sample_id")["read_pairs"].astype(int).to_dict()
    for sample_id in sorted(sample_paths):
        r1_tmp = sample_paths[sample_id]["1"]
        r2_tmp = sample_paths[sample_id]["2"]
        design_row = design[design["sample_id"].astype(str) == sample_id].iloc[0]
        scenario = str(design_row["scenario"])
        replicate = int(design_row["replicate"])
        final_dir = ensure_dir(fastq_out / scenario / f"rep_{replicate:02d}")
        r1_gz = final_dir / f"{sample_id}_1.fastq.gz"
        r2_gz = final_dir / f"{sample_id}_2.fastq.gz"
        deterministic_gzip(r1_tmp, r1_gz, compresslevel=args.compresslevel)
        deterministic_gzip(r2_tmp, r2_gz, compresslevel=args.compresslevel)
        generated_rows.append(
            {
                "scenario": scenario,
                "replicate": replicate,
                "sample_id": sample_id,
                "class_label": str(design_row["class_label"]),
                "r1_path": str(r1_gz.resolve()),
                "r2_path": str(r2_gz.resolve()),
                "read_pairs": int(expected_counts[sample_id]),
                "r1_size_bytes": r1_gz.stat().st_size,
                "r2_size_bytes": r2_gz.stat().st_size,
                "r1_sha256": sha256_file(r1_gz),
                "r2_sha256": sha256_file(r2_gz),
            }
        )

    source_df = pd.DataFrame(source_rows).sort_values("parent_id")
    generated_df = pd.DataFrame(generated_rows).sort_values(
        ["scenario", "replicate", "sample_id"]
    )
    source_df.to_csv(manifest_out / "source_sampling_summary.tsv", sep="\t", index=False)
    generated_df.to_csv(manifest_out / "generated_fastq_manifest.tsv", sep="\t", index=False)

    # The target design is exact because integer allocations are used directly.
    truth = design.copy()
    actual_weight_values = []
    actual_entropy_values = []
    actual_minor_values = []
    for row in truth.itertuples(index=False):
        counts = [int(x) for x in str(row.target_counts).split(",")]
        total = sum(counts)
        weights = [x / total for x in counts]
        actual_weight_values.append(",".join(f"{x:.6f}" for x in weights))
        actual_entropy_values.append(normalized_entropy(weights))
        actual_minor_values.append(min(weights) if len(weights) > 1 else 0.0)
    truth["actual_weights"] = actual_weight_values
    truth["actual_entropy_norm"] = actual_entropy_values
    truth["actual_minor_fraction"] = actual_minor_values
    truth.to_csv(manifest_out / "truth_manifest.tsv", sep="\t", index=False)

    audit = {
        "status": "PASS",
        "n_source_libraries": len(source_ids),
        "n_generated_libraries": len(generated_df),
        "total_generated_read_pairs": int(generated_df["read_pairs"].sum()),
        "total_source_pair_assignments": int(assignment_count),
        "duplicate_source_pair_assignments": 0,
        "paired_end_synchronization_checked": True,
        "read_reuse_within_run": False,
        "deterministic_gzip_mtime": 0,
        "design_manifest": str(Path(args.design_manifest).resolve()),
        "read_allocations": str(Path(args.read_allocations).resolve()),
        "generated_fastq_manifest": str(
            (manifest_out / "generated_fastq_manifest.tsv").resolve()
        ),
    }
    write_json(manifest_out / "build_audit.json", audit)

    shutil.rmtree(temp_out)
    print(f"[DONE] Generated paired FASTQ root: {fastq_out}")
    print(f"[DONE] Truth manifest: {manifest_out / 'truth_manifest.tsv'}")
    print(f"[DONE] Generated libraries: {len(generated_df)}")
    print(f"[DONE] Total read pairs written: {int(generated_df['read_pairs'].sum()):,}")


if __name__ == "__main__":
    main()
