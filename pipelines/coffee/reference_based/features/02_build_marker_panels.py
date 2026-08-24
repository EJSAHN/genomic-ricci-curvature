# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import random
import re
from pathlib import Path
from typing import Dict, Iterator, Tuple

import pandas as pd

from reference_qc_common import (
    AlignmentSummary,
    ensure_dir,
    mismatch_events,
    stream_bowtie_sam,
    write_json,
)

GENERATED_HEADER_RE = re.compile(r"\|source=([^|]+)\|source_pair=(\d+)(?:/1)?$")


def find_fastq(root: Path, sample_id: str, mate: int) -> Path:
    candidates = [
        root / f"{sample_id}_{mate}.fastq.gz",
        root / f"{sample_id}_{mate}.fq.gz",
        root / f"{sample_id}_{mate}.fastq",
        root / f"{sample_id}_{mate}.fq",
        root / f"{sample_id}_R{mate}.fastq.gz",
        root / f"{sample_id}_R{mate}.fq.gz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"FASTQ mate {mate} not found for {sample_id} in {root}")


def open_fastq(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.name.lower().endswith(".gz") else open(path, "rt", encoding="utf-8", errors="replace")


def normalize_read_id(header: str) -> str:
    token = header.strip().split()[0]
    if token.startswith("@"):
        token = token[1:]
    return re.sub(r"(?:/1|/2)$", "", token)


def iter_paired_fastq(r1_path: Path, r2_path: Path) -> Iterator[tuple[int, tuple[str, str, str, str], tuple[str, str, str, str]]]:
    with open_fastq(r1_path) as f1, open_fastq(r2_path) as f2:
        idx = 0
        while True:
            a = [f1.readline() for _ in range(4)]
            b = [f2.readline() for _ in range(4)]
            end_a = not a[0]
            end_b = not b[0]
            if end_a and end_b:
                return
            if end_a != end_b or any(x == "" for x in a + b):
                raise ValueError(f"Malformed or unequal paired FASTQ files: {r1_path} / {r2_path}")
            ra = tuple(x.rstrip("\r\n") for x in a)
            rb = tuple(x.rstrip("\r\n") for x in b)
            if normalize_read_id(ra[0]) != normalize_read_id(rb[0]):
                raise ValueError(f"R1/R2 identifiers differ at pair {idx}: {ra[0]} vs {rb[0]}")
            yield idx, ra, rb
            idx += 1


def stable_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def build_exclusion_masks(
    generated_manifest_path: Path,
    source_summary_path: Path,
    source_samples: list[str],
    real_skip_pairs: int,
    real_pairs: int,
) -> tuple[dict[str, bytearray], dict[str, dict[str, int]]]:
    source_summary = pd.read_csv(source_summary_path, sep="\t")
    available = {
        str(row.parent_id): int(row.read_pairs_available)
        for row in source_summary.itertuples(index=False)
    }
    missing = sorted(set(source_samples) - set(available))
    if missing:
        raise ValueError(f"Source sampling summary is missing libraries: {missing}")

    masks = {sample: bytearray(available[sample]) for sample in source_samples}
    counts = {
        sample: {
            "read_pairs_available": available[sample],
            "generated_pairs_excluded": 0,
            "real_segment_pairs_reserved": 0,
        }
        for sample in source_samples
    }

    generated = pd.read_csv(generated_manifest_path, sep="\t", dtype=str).fillna("")
    for path_text in generated["r1_path"].astype(str):
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"Generated R1 FASTQ not found: {path}")
        with open_fastq(path) as fh:
            while True:
                header = fh.readline()
                if not header:
                    break
                seq = fh.readline(); plus = fh.readline(); qual = fh.readline()
                if not seq or not plus or not qual:
                    raise ValueError(f"Malformed generated FASTQ: {path}")
                token = header.strip().split()[0]
                match = GENERATED_HEADER_RE.search(token)
                if not match:
                    raise ValueError(f"Generated header lacks source metadata: {token}")
                source = match.group(1)
                pair_index = int(match.group(2))
                if source not in masks:
                    continue
                if pair_index < 0 or pair_index >= len(masks[source]):
                    raise IndexError(f"Source pair index out of range: {source}:{pair_index}")
                if masks[source][pair_index] == 0:
                    masks[source][pair_index] = 1
                    counts[source]["generated_pairs_excluded"] += 1

    # Reserve the exact raw-read segment later used for descriptive real-data
    # feature extraction. This keeps marker discovery independent of both the
    # generated benchmark libraries and the real-data scoring segment.
    if real_pairs > 0:
        start = max(0, int(real_skip_pairs))
        for source in source_samples:
            stop = min(len(masks[source]), start + int(real_pairs))
            newly_reserved = 0
            for idx in range(start, stop):
                if masks[source][idx] == 0:
                    masks[source][idx] = 1
                    newly_reserved += 1
            counts[source]["real_segment_pairs_reserved"] = newly_reserved

    return masks, counts


def sample_independent_pairs(
    r1_path: Path,
    r2_path: Path,
    excluded: bytearray,
    n_required: int,
    seed: int,
) -> tuple[list[tuple[tuple[str, str, str, str], tuple[str, str, str, str]]], int, int]:
    rng = random.Random(int(seed))
    reservoir: list[tuple[tuple[str, str, str, str], tuple[str, str, str, str]]] = []
    eligible_seen = 0
    total_seen = 0
    for pair_index, r1, r2 in iter_paired_fastq(r1_path, r2_path):
        total_seen += 1
        if pair_index < len(excluded) and excluded[pair_index]:
            continue
        eligible_seen += 1
        item = (r1, r2)
        if len(reservoir) < n_required:
            reservoir.append(item)
        else:
            replacement = rng.randrange(eligible_seen)
            if replacement < n_required:
                reservoir[replacement] = item
    if eligible_seen < n_required:
        raise ValueError(
            f"Only {eligible_seen} independent read pairs remain but {n_required} are required: {r1_path}"
        )
    rng.shuffle(reservoir)
    return reservoir, eligible_seen, total_seen


def write_panel_fastqs(
    pairs: list[tuple[tuple[str, str, str, str], tuple[str, str, str, str]]],
    r1_out: Path,
    r2_out: Path,
) -> None:
    with open(r1_out, "w", encoding="utf-8", newline="\n") as out1, open(
        r2_out, "w", encoding="utf-8", newline="\n"
    ) as out2:
        for r1, r2 in pairs:
            out1.write("\n".join(r1) + "\n")
            out2.write("\n".join(r2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_manifest", required=True)
    ap.add_argument("--sample_manifest", required=True)
    ap.add_argument("--source_fastq_root", required=True)
    ap.add_argument("--generated_fastq_manifest", required=True)
    ap.add_argument("--source_sampling_summary", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--source_pairs", type=int, default=100000)
    ap.add_argument("--reserve_real_skip_pairs", type=int, default=100000)
    ap.add_argument("--reserve_real_pairs", type=int, default=50000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=52001)
    ap.add_argument("--min_mapq", type=int, default=20)
    ap.add_argument("--min_baseq", type=int, default=20)
    ap.add_argument("--min_alt_total", type=int, default=4)
    ap.add_argument("--min_alt_samples", type=int, default=2)
    ap.add_argument("--min_alt_in_one_sample", type=int, default=4)
    ap.add_argument("--max_markers", type=int, default=50000)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    logs_dir = ensure_dir(args.logs_dir)
    source_root = Path(args.source_fastq_root)
    temp_root = ensure_dir(outdir / "_independent_panel_fastq")

    refs = pd.read_csv(args.reference_manifest, sep="\t")
    samples = pd.read_csv(args.sample_manifest, sep="\t")
    source_samples = (
        samples[samples["include_primary"].astype(int) == 1]["sample_id"]
        .astype(str)
        .tolist()
    )
    if len(source_samples) != 13:
        raise SystemExit(f"Expected 13 primary source libraries, found {len(source_samples)}")

    print("[PREP] Identifying generated read pairs and reserving the real-data scoring segment.")
    exclusion_masks, exclusion_counts = build_exclusion_masks(
        generated_manifest_path=Path(args.generated_fastq_manifest),
        source_summary_path=Path(args.source_sampling_summary),
        source_samples=source_samples,
        real_skip_pairs=args.reserve_real_skip_pairs,
        real_pairs=args.reserve_real_pairs,
    )

    # Select one independent marker-discovery read set per source library and
    # reuse it for both reference assemblies. This removes read-level leakage
    # between marker discovery and the paired-read benchmark.
    independent_fastqs: dict[str, tuple[Path, Path]] = {}
    independence_rows: list[dict[str, object]] = []
    for sample_id in source_samples:
        r1 = find_fastq(source_root, sample_id, 1)
        r2 = find_fastq(source_root, sample_id, 2)
        selected, eligible_seen, total_seen = sample_independent_pairs(
            r1_path=r1,
            r2_path=r2,
            excluded=exclusion_masks[sample_id],
            n_required=args.source_pairs,
            seed=stable_seed(args.seed, "independent_panel", sample_id),
        )
        r1_temp = temp_root / f"{sample_id}_1.fastq"
        r2_temp = temp_root / f"{sample_id}_2.fastq"
        write_panel_fastqs(selected, r1_temp, r2_temp)
        independent_fastqs[sample_id] = (r1_temp, r2_temp)
        independence_rows.append(
            {
                "sample_id": sample_id,
                **exclusion_counts[sample_id],
                "total_pairs_seen": total_seen,
                "eligible_independent_pairs": eligible_seen,
                "panel_pairs_selected": len(selected),
                "overlap_with_generated_pairs": 0,
                "overlap_with_reserved_real_segment": 0,
                "selection_seed": stable_seed(args.seed, "independent_panel", sample_id),
            }
        )
        print(
            f"[INDEPENDENT] {sample_id}: selected {len(selected):,} of "
            f"{eligible_seen:,} eligible pairs"
        )

    pd.DataFrame(independence_rows).to_csv(
        outdir / "marker_discovery_independence.tsv", sep="\t", index=False
    )

    all_panel_summaries: list[dict[str, object]] = []
    all_mapping_rows: list[dict[str, object]] = []

    try:
        for ref_row in refs.itertuples(index=False):
            ref_id = str(ref_row.reference_id)
            print(f"\n=== Marker discovery: {ref_id} ===")
            ref_out = ensure_dir(outdir / ref_id)

            per_site_sample_counts: Dict[Tuple[str, int, str, str], Dict[str, int]] = {}
            aggregate_counts: collections.Counter[Tuple[str, int, str, str]] = collections.Counter()

            for sample_index, sample_id in enumerate(source_samples, start=1):
                print(f"[SOURCE {sample_index}/{len(source_samples)}] {sample_id}")
                r1, r2 = independent_fastqs[sample_id]
                stats = AlignmentSummary()
                local_counts: collections.Counter[Tuple[str, int, str, str]] = collections.Counter()
                stderr_log = logs_dir / f"panel_{ref_id}_{sample_id}.bowtie2.log"

                for record in stream_bowtie_sam(
                    launcher=str(ref_row.align_launcher),
                    index_prefix=str(ref_row.index_prefix),
                    r1_path=str(r1),
                    r2_path=str(r2),
                    threads=args.threads,
                    seed=args.seed,
                    upto_pairs=args.source_pairs,
                    skip_pairs=0,
                    stderr_log=str(stderr_log),
                ):
                    stats.update(record, unique_mapq=args.min_mapq)
                    if record.is_unmapped or not record.is_primary or record.mapq < args.min_mapq:
                        continue
                    for contig, position, ref_base, alt_base in mismatch_events(
                        record, min_baseq=args.min_baseq
                    ):
                        local_counts[(contig, position, ref_base, alt_base)] += 1

                if stats.primary_records == 0 or stats.total_pairs == 0:
                    raise RuntimeError(
                        f"{ref_id} {sample_id}: Bowtie 2 emitted no primary paired SAM "
                        f"records. See {stderr_log}. Marker discovery was stopped to "
                        "prevent creation of an invalid empty panel."
                    )

                for key, count in local_counts.items():
                    aggregate_counts[key] += int(count)
                    per_site_sample_counts.setdefault(key, {})[sample_id] = int(count)

                all_mapping_rows.append(
                    {
                        "reference_id": ref_id,
                        "sample_id": sample_id,
                        "read_segment": "independent_nonbenchmark_pairs",
                        "pairs_requested": args.source_pairs,
                        **stats.as_dict(),
                        "n_distinct_mismatch_alleles": len(local_counts),
                        "n_mismatch_observations": int(sum(local_counts.values())),
                    }
                )

            position_candidates: Dict[Tuple[str, int], list[Tuple[Tuple[str, int, str, str], int]]] = {}
            for key, total in aggregate_counts.items():
                position_candidates.setdefault((key[0], key[1]), []).append((key, int(total)))

            panel_rows: list[dict[str, object]] = []
            rejected_rows: list[dict[str, object]] = []
            for (contig, position), candidates in position_candidates.items():
                candidates.sort(
                    key=lambda item: (
                        len(per_site_sample_counts.get(item[0], {})),
                        item[1],
                        max(per_site_sample_counts.get(item[0], {}).values() or [0]),
                    ),
                    reverse=True,
                )
                key, total = candidates[0]
                _, _, ref_base, alt_base = key
                sample_counts = per_site_sample_counts.get(key, {})
                n_samples_alt = int(sum(1 for c in sample_counts.values() if c > 0))
                max_sample_alt = int(max(sample_counts.values() or [0]))
                eligible = (
                    total >= args.min_alt_total
                    and (
                        n_samples_alt >= args.min_alt_samples
                        or max_sample_alt >= args.min_alt_in_one_sample
                    )
                )
                row = {
                    "reference_id": ref_id,
                    "contig": contig,
                    "position": int(position),
                    "ref": ref_base,
                    "alt": alt_base,
                    "total_alt_observations": int(total),
                    "n_source_libraries_alt": n_samples_alt,
                    "max_alt_observations_in_one_library": max_sample_alt,
                    "n_competing_alt_alleles_at_position": len(candidates),
                    "source_library_counts": ";".join(
                        f"{s}:{c}" for s, c in sorted(sample_counts.items())
                    ),
                    "selection_tier": "strict" if eligible else "rejected",
                }
                (panel_rows if eligible else rejected_rows).append(row)

            if len(panel_rows) < 20:
                fallback_candidates = [
                    dict(row, selection_tier="fallback_low_marker_count")
                    for row in rejected_rows
                    if int(row["total_alt_observations"]) >= 2
                    and int(row["max_alt_observations_in_one_library"]) >= 2
                ]
                fallback_candidates.sort(
                    key=lambda row: (
                        int(row["n_source_libraries_alt"]),
                        int(row["total_alt_observations"]),
                        int(row["max_alt_observations_in_one_library"]),
                    ),
                    reverse=True,
                )
                needed = max(0, min(args.max_markers, 5000) - len(panel_rows))
                panel_rows.extend(fallback_candidates[:needed])

            panel = pd.DataFrame(panel_rows)
            if len(panel):
                panel = panel.sort_values(
                    [
                        "n_source_libraries_alt",
                        "total_alt_observations",
                        "max_alt_observations_in_one_library",
                    ],
                    ascending=[False, False, False],
                ).head(args.max_markers)
                panel = panel.sort_values(["contig", "position"]).reset_index(drop=True)
                panel.insert(0, "marker_id", [f"{ref_id}_M{i+1:06d}" for i in range(len(panel))])
            else:
                panel = pd.DataFrame(
                    columns=[
                        "marker_id", "reference_id", "contig", "position", "ref", "alt",
                        "total_alt_observations", "n_source_libraries_alt",
                        "max_alt_observations_in_one_library",
                        "n_competing_alt_alleles_at_position", "source_library_counts",
                        "selection_tier",
                    ]
                )

            panel_path = ref_out / "marker_panel.tsv"
            panel.to_csv(panel_path, sep="\t", index=False)
            pd.DataFrame(rejected_rows).to_csv(
                ref_out / "rejected_candidates.tsv", sep="\t", index=False
            )

            strict_count = int((panel["selection_tier"] == "strict").sum()) if len(panel) else 0
            summary = {
                "reference_id": ref_id,
                "accession": str(ref_row.accession),
                "source_pairs_per_library": args.source_pairs,
                "n_source_libraries": len(source_samples),
                "marker_discovery_independent_of_generated_reads": True,
                "marker_discovery_independent_of_real_scoring_segment": True,
                "n_raw_mismatch_alleles": len(aggregate_counts),
                "n_raw_positions": len(position_candidates),
                "n_markers": len(panel),
                "n_strict_markers": strict_count,
                "min_alt_total": args.min_alt_total,
                "min_alt_samples": args.min_alt_samples,
                "min_alt_in_one_sample": args.min_alt_in_one_sample,
                "max_markers": args.max_markers,
                "marker_panel": str(panel_path),
                "status": (
                    "PASS_STRICT"
                    if len(panel) >= 20 and strict_count >= 20
                    else ("PASS_WITH_FALLBACK" if len(panel) >= 20 else "LOW_MARKER_COUNT")
                ),
            }
            write_json(summary, ref_out / "marker_panel_summary.json")
            all_panel_summaries.append(summary)
            print(f"[DONE] {ref_id}: retained {len(panel):,} markers")
    finally:
        # The independent discovery FASTQs are temporary and can be recreated
        # deterministically from manifests and raw data.
        for r1, r2 in independent_fastqs.values():
            r1.unlink(missing_ok=True)
            r2.unlink(missing_ok=True)
        try:
            temp_root.rmdir()
        except OSError:
            pass

    pd.DataFrame(all_mapping_rows).to_csv(
        outdir / "source_mapping_metrics.tsv", sep="\t", index=False
    )
    pd.DataFrame(all_panel_summaries).to_csv(
        outdir / "marker_panel_summary.tsv", sep="\t", index=False
    )
    write_json(
        {
            "references": all_panel_summaries,
            "status": "PASS" if all(s["n_markers"] >= 20 for s in all_panel_summaries) else "LOW_MARKER_COUNT",
            "independence_enforced": True,
        },
        outdir / "marker_panel_master.json",
    )
    print("\n[DONE] Marker-panel discovery complete.")


if __name__ == "__main__":
    main()
