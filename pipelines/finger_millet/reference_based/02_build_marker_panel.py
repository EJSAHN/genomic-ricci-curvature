# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from reference_common import (
    AlignmentSummary,
    ensure_dir,
    mismatch_events,
    sha256_file,
    stream_bowtie_sam,
    write_json,
)

CACHE_VERSION = "finger-millet-independent-marker-panel-v2-bowtie-stdout"


def open_fastq(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def normalize_read_id(header: str) -> str:
    token = header.strip().split()[0]
    if token.startswith("@"):
        token = token[1:]
    return re.sub(r"(?:/1|/2)$", "", token)


def iter_paired_fastq(r1_path: Path, r2_path: Path) -> Iterator[tuple[int, tuple[str,str,str,str], tuple[str,str,str,str]]]:
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


def physical_indices_for_allocations(rows: pd.DataFrame) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for row in rows.itertuples(index=False):
        start = int(row.allocation_ordinal_start)
        stop = int(row.allocation_ordinal_stop)
        expected = int(row.read_pairs)
        if stop - start != expected:
            raise ValueError(f"Allocation width mismatch for {row.sample_id}: {stop-start} != {expected}")
        ordinals = np.arange(start, stop, dtype=np.int64)
        physical = int(row.eligible_physical_start) + (
            (int(row.permutation_a) * ordinals + int(row.permutation_b))
            % int(row.permutation_modulus)
        )
        chunks.append(physical)
    if not chunks:
        return np.asarray([], dtype=np.int64)
    values = np.concatenate(chunks)
    if len(np.unique(values)) != len(values):
        raise ValueError("Locked allocation contains duplicated physical source pairs")
    return values


def select_independent_indices(
    pair_count: int,
    allocation_rows: pd.DataFrame,
    reserved_prefix_pairs: int,
    n_required: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    reserved = np.zeros(int(pair_count), dtype=bool)
    prefix_stop = min(int(pair_count), int(reserved_prefix_pairs))
    reserved[:prefix_stop] = True
    generated = physical_indices_for_allocations(allocation_rows)
    if len(generated):
        if int(generated.min()) < 0 or int(generated.max()) >= int(pair_count):
            raise ValueError("Generated physical index outside source range")
        reserved[generated] = True
    eligible = np.flatnonzero(~reserved)
    if len(eligible) < int(n_required):
        raise ValueError(f"Only {len(eligible)} independent pairs available; {n_required} required")
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(eligible, size=int(n_required), replace=False).astype(np.int64))
    if np.intersect1d(selected, generated, assume_unique=False).size:
        raise AssertionError("Independent marker-discovery selection overlaps generated pairs")
    if len(selected) and int(selected.min()) < prefix_stop:
        raise AssertionError("Independent selection overlaps reserved geometry/design prefix")
    return selected, {
        "pair_count": int(pair_count),
        "reserved_prefix_pairs": int(prefix_stop),
        "generated_pairs_excluded": int(len(generated)),
        "eligible_independent_pairs": int(len(eligible)),
        "selected_pairs": int(len(selected)),
    }


def write_selected_fastqs(
    r1_path: Path,
    r2_path: Path,
    selected: np.ndarray,
    r1_out: Path,
    r2_out: Path,
) -> None:
    selected = np.asarray(selected, dtype=np.int64)
    pointer = 0
    with r1_out.open("w", encoding="ascii", newline="\n") as out1, r2_out.open("w", encoding="ascii", newline="\n") as out2:
        for idx, r1, r2 in iter_paired_fastq(r1_path, r2_path):
            if pointer >= len(selected):
                break
            target = int(selected[pointer])
            if idx < target:
                continue
            if idx > target:
                raise RuntimeError(f"Source stream skipped selected pair {target}")
            out1.write("\n".join(r1) + "\n")
            out2.write("\n".join(r2) + "\n")
            pointer += 1
    if pointer != len(selected):
        raise RuntimeError(f"Recovered {pointer}/{len(selected)} selected source pairs")


def cache_valid(marker_path: Path, expected: Mapping[str, Any]) -> bool:
    if not marker_path.is_file():
        return False
    try:
        observed = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for key, value in expected.items():
        if observed.get(key) != value:
            return False
    counts_path = Path(observed.get("counts_path", ""))
    return observed.get("status") == "COMPLETE" and counts_path.is_file()


def load_counts(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["contig","position","ref","alt","count"])
    return pd.read_csv(path, sep="\t")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_manifest", required=True)
    ap.add_argument("--source_manifest", required=True)
    ap.add_argument("--read_allocations", required=True)
    ap.add_argument("--preflight_pass", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--source_pairs", type=int, default=100000)
    ap.add_argument("--reserved_prefix_pairs", type=int, default=100000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=932690846)
    ap.add_argument("--min_mapq", type=int, default=20)
    ap.add_argument("--min_baseq", type=int, default=20)
    ap.add_argument("--max_markers", type=int, default=50000)
    ap.add_argument("--minimum_viable_markers", type=int, default=100)
    args = ap.parse_args()

    if not Path(args.preflight_pass).is_file():
        raise SystemExit("[ERROR] Comparator preflight PASS marker is absent")

    outdir = ensure_dir(args.outdir)
    logs = ensure_dir(args.logs_dir)
    cache_root = ensure_dir(outdir / "cache")
    temp_root = ensure_dir(outdir / "temp")

    refs = pd.read_csv(args.reference_manifest, sep="\t")
    if len(refs) != 1:
        raise SystemExit(f"Expected one reference, found {len(refs)}")
    ref = refs.iloc[0]
    ref_id = str(ref.reference_id)
    ref_sha = str(ref.fasta_sha256)
    source = pd.read_csv(args.source_manifest, sep="\t")
    allocations = pd.read_csv(args.read_allocations, sep="\t")
    if len(source) != 28:
        raise SystemExit(f"Expected 28 source libraries, found {len(source)}")

    independence_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    per_source_counts: list[pd.DataFrame] = []

    for source_index, row in enumerate(source.itertuples(index=False), start=1):
        run = str(row.run_accession)
        sample = str(row.sample_accession)
        print(f"[MARKER SOURCE {source_index}/28] {run}", flush=True)
        alloc = allocations[allocations["source_run_accession"].astype(str) == run].copy()
        selected_seed = stable_seed(args.seed, ref_id, run)
        selected, counts = select_independent_indices(
            pair_count=int(row.pair_count),
            allocation_rows=alloc,
            reserved_prefix_pairs=args.reserved_prefix_pairs,
            n_required=args.source_pairs,
            seed=selected_seed,
        )
        selected_sha = hashlib.sha256(selected.astype("<i8", copy=False).tobytes()).hexdigest()
        counts_path = cache_root / f"{run}.mismatches.tsv.gz"
        marker_path = cache_root / f"{run}.cache.json"
        expected = {
            "cache_version": CACHE_VERSION,
            "status": "COMPLETE",
            "reference_id": ref_id,
            "reference_fasta_sha256": ref_sha,
            "source_run_accession": run,
            "selected_pairs": int(args.source_pairs),
            "selected_index_sha256": selected_sha,
            "min_mapq": int(args.min_mapq),
            "min_baseq": int(args.min_baseq),
        }

        if cache_valid(marker_path, expected):
            cache = json.loads(marker_path.read_text(encoding="utf-8"))
            print(f"[CACHE] {run}: independent marker-discovery alignment reused", flush=True)
        else:
            r1_temp = temp_root / f"{run}_1.fastq"
            r2_temp = temp_root / f"{run}_2.fastq"
            write_selected_fastqs(Path(str(row.r1_path)), Path(str(row.r2_path)), selected, r1_temp, r2_temp)
            stats = AlignmentSummary()
            local: collections.Counter[Tuple[str,int,str,str]] = collections.Counter()
            md_records = 0
            nm_records = 0
            stderr_log = logs / f"marker_{ref_id}_{run}.bowtie2.log"
            for record in stream_bowtie_sam(
                launcher=str(ref.align_launcher),
                index_prefix=str(ref.index_prefix),
                r1_path=str(r1_temp),
                r2_path=str(r2_temp),
                threads=args.threads,
                seed=args.seed,
                upto_pairs=args.source_pairs,
                skip_pairs=0,
                stderr_log=str(stderr_log),
            ):
                stats.update(record, unique_mapq=args.min_mapq)
                md_records += int("MD" in record.tags)
                nm_records += int("NM" in record.tags)
                if record.is_unmapped or not record.is_primary or record.mapq < args.min_mapq:
                    continue
                for contig, position, ref_base, alt_base in mismatch_events(record, min_baseq=args.min_baseq):
                    local[(contig, int(position), ref_base, alt_base)] += 1

            expected_primary_records = 2 * int(args.source_pairs)
            if stats.total_pairs != int(args.source_pairs):
                raise RuntimeError(
                    f"{ref_id} {run}: expected {args.source_pairs} primary read pairs "
                    f"in SAM, observed {stats.total_pairs}. See {stderr_log}."
                )
            if stats.primary_records != expected_primary_records:
                raise RuntimeError(
                    f"{ref_id} {run}: expected {expected_primary_records} primary SAM "
                    f"records, observed {stats.primary_records}. See {stderr_log}."
                )
            if md_records == 0 or nm_records == 0:
                raise RuntimeError(
                    f"{ref_id} {run}: SAM records were emitted but MD/NM tags were "
                    f"absent. See {stderr_log}."
                )

            local_df = pd.DataFrame(
                [
                    {"contig": k[0], "position": k[1], "ref": k[2], "alt": k[3], "count": int(v)}
                    for k, v in local.items()
                ]
            )
            if local_df.empty:
                local_df = pd.DataFrame(columns=["contig","position","ref","alt","count"])
            local_df.to_csv(counts_path, sep="\t", index=False, compression="gzip")
            cache = {
                **expected,
                "counts_path": str(counts_path),
                "mismatch_site_count": int(len(local_df)),
                "mismatch_observation_count": int(local_df["count"].sum()) if len(local_df) else 0,
                "records_with_MD": int(md_records),
                "records_with_NM": int(nm_records),
                "mapping": stats.as_dict(),
                "selection_seed": int(selected_seed),
            }
            write_json(cache, marker_path)
            r1_temp.unlink(missing_ok=True)
            r2_temp.unlink(missing_ok=True)

        local_df = load_counts(counts_path)
        mapping_preview = dict(cache.get("mapping", {}))
        print(
            f"[MARKER SOURCE COMPLETE] {run}: "
            f"pairs={int(mapping_preview.get('total_pairs', 0)):,}; "
            f"primary={int(mapping_preview.get('primary_records', 0)):,}; "
            f"mapping_rate={float(mapping_preview.get('mapping_rate', float('nan'))):.4f}; "
            f"unique_rate={float(mapping_preview.get('unique_mapping_rate', float('nan'))):.4f}; "
            f"mismatch_sites={int(cache.get('mismatch_site_count', len(local_df))):,}",
            flush=True,
        )
        local_df["source_run_accession"] = run
        per_source_counts.append(local_df)
        mapping = dict(cache.get("mapping", {}))
        mapping_rows.append({"source_sample_accession": sample, "source_run_accession": run, "population": str(row.population), **mapping})
        independence_rows.append({
            "source_sample_accession": sample,
            "source_run_accession": run,
            "population": str(row.population),
            **counts,
            "selected_index_sha256": selected_sha,
            "selection_seed": int(selected_seed),
            "overlap_with_reserved_prefix": 0,
            "overlap_with_generated_pairs": 0,
        })

    all_counts = pd.concat(per_source_counts, ignore_index=True)
    if all_counts.empty:
        raise SystemExit("[ERROR] No mismatch evidence was observed in independent source reads")
    grouped = (
        all_counts.groupby(["contig","position","ref","alt"], as_index=False)
        .agg(total_alt_count=("count","sum"), alt_sample_count=("source_run_accession","nunique"), max_alt_in_one_sample=("count","max"))
    )

    tiers = [
        {"tier": 1, "min_alt_total": 4, "min_alt_samples": 2, "min_alt_in_one_sample": 4},
        {"tier": 2, "min_alt_total": 3, "min_alt_samples": 2, "min_alt_in_one_sample": 3},
        {"tier": 3, "min_alt_total": 2, "min_alt_samples": 2, "min_alt_in_one_sample": 2},
    ]
    chosen = None
    selected_panel = pd.DataFrame()
    tier_rows = []
    for tier in tiers:
        candidate = grouped[
            (grouped["total_alt_count"] >= tier["min_alt_total"])
            & (grouped["alt_sample_count"] >= tier["min_alt_samples"])
            & (grouped["max_alt_in_one_sample"] >= tier["min_alt_in_one_sample"])
        ].copy()
        tier_rows.append({**tier, "candidate_markers": int(len(candidate))})
        if chosen is None and len(candidate) >= args.minimum_viable_markers:
            chosen = tier
            selected_panel = candidate
    if chosen is None:
        raise SystemExit(f"[ERROR] No predeclared marker tier produced at least {args.minimum_viable_markers} markers")

    selected_panel = selected_panel.sort_values(
        ["alt_sample_count","total_alt_count","max_alt_in_one_sample","contig","position","ref","alt"],
        ascending=[False,False,False,True,True,True,True],
        kind="mergesort",
    )
    candidate_rows_before_position_deduplication = int(len(selected_panel))
    # The downstream marker index is position-based; retain only the strongest
    # alternate allele at any contig/position to prevent ambiguous overwrite.
    selected_panel = (
        selected_panel.drop_duplicates(["contig", "position"], keep="first")
        .head(args.max_markers)
        .reset_index(drop=True)
    )
    duplicate_position_rows_removed = (
        candidate_rows_before_position_deduplication - int(len(selected_panel))
    )
    selected_panel.insert(0, "marker_id", [f"M{i+1:06d}" for i in range(len(selected_panel))])
    selected_panel["selection_tier"] = int(chosen["tier"])
    selected_panel.to_csv(outdir / "marker_panel.tsv", sep="\t", index=False)
    pd.DataFrame(independence_rows).to_csv(outdir / "marker_discovery_independence.tsv", sep="\t", index=False)
    pd.DataFrame(mapping_rows).to_csv(outdir / "marker_discovery_mapping_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(tier_rows).to_csv(outdir / "marker_threshold_tiers.tsv", sep="\t", index=False)

    summary = {
        "status": "PASS",
        "reference_id": ref_id,
        "reference_accession": str(ref.accession),
        "reference_fasta_sha256": ref_sha,
        "source_library_count": int(len(source)),
        "source_pairs_per_library": int(args.source_pairs),
        "reserved_prefix_pairs": int(args.reserved_prefix_pairs),
        "marker_count": int(len(selected_panel)),
        "marker_panel_sha256": sha256_file(outdir / "marker_panel.tsv"),
        "selection_tier": chosen,
        "candidate_rows_before_position_deduplication": candidate_rows_before_position_deduplication,
        "duplicate_position_rows_removed": duplicate_position_rows_removed,
        "marker_threshold_tiers": tier_rows,
        "generated_overlap_count": 0,
        "reserved_prefix_overlap_count": 0,
    }
    write_json(summary, outdir / "marker_panel_summary.json")
    lines = [
        "Finger millet independent reference marker panel",
        "=================================================",
        "",
        "Status: PASS",
        f"Reference: {ref_id} ({ref.accession})",
        f"Independent source libraries: {len(source)}",
        f"Independent read pairs per source: {args.source_pairs:,}",
        f"Selected markers: {len(selected_panel):,}",
        f"Duplicate position/alternate rows removed: {duplicate_position_rows_removed:,}",
        f"Threshold tier used: {chosen['tier']}",
        "Overlap with locked generated reads: 0",
        "Overlap with reserved geometry/design prefix: 0",
    ]
    (outdir / "MARKER_PANEL_SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "MARKER_PANEL_PASS.txt").write_text("PASS\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
