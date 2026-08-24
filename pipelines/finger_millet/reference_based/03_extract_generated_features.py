# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from reference_common import (
    AlignmentSummary,
    MarkerIndex,
    count_marker_bases,
    ensure_dir,
    load_marker_index,
    sha256_file,
    stream_bowtie_sam,
    write_json,
)

CACHE_VERSION = "finger-millet-generated-reference-features-v2-bowtie-stdout"


def process_sample(
    *,
    sample_id: str,
    r1_path: str,
    r2_path: str,
    marker_index: MarkerIndex,
    marker_panel_sha256: str,
    reference_fasta_sha256: str,
    align_launcher: str,
    index_prefix: str,
    threads: int,
    seed: int,
    upto_pairs: int,
    min_mapq: int,
    min_baseq: int,
    cache_path: Path,
    stderr_log: Path,
) -> Dict[str, Any]:
    n_markers = len(marker_index.marker_table)
    metrics_json = cache_path.with_suffix(".metrics.json")
    if cache_path.is_file() and metrics_json.is_file():
        try:
            payload = np.load(cache_path, allow_pickle=False)
            metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
            if (
                str(payload["cache_version"].item()) == CACHE_VERSION
                and str(payload["marker_panel_sha256"].item()) == marker_panel_sha256
                and str(payload["reference_fasta_sha256"].item()) == reference_fasta_sha256
                and int(payload["n_markers"]) == n_markers
                and int(metrics.get("pairs_requested", -1)) == int(upto_pairs)
                and int(metrics.get("min_mapq", -1)) == int(min_mapq)
                and int(metrics.get("min_baseq", -1)) == int(min_baseq)
            ):
                return metrics
        except Exception:
            pass

    ref_counts = np.zeros(n_markers, dtype=np.uint32)
    alt_counts = np.zeros(n_markers, dtype=np.uint32)
    other_counts = np.zeros(n_markers, dtype=np.uint32)
    stats = AlignmentSummary()

    for record in stream_bowtie_sam(
        launcher=align_launcher,
        index_prefix=index_prefix,
        r1_path=r1_path,
        r2_path=r2_path,
        threads=threads,
        seed=seed,
        upto_pairs=upto_pairs,
        skip_pairs=0,
        stderr_log=str(stderr_log),
    ):
        stats.update(record, unique_mapq=min_mapq)
        count_marker_bases(
            record,
            marker_index,
            ref_counts,
            alt_counts,
            other_counts,
            min_mapq=min_mapq,
            min_baseq=min_baseq,
        )

    expected_primary_records = 2 * int(upto_pairs)
    if stats.total_pairs != int(upto_pairs):
        raise RuntimeError(
            f"{sample_id}: expected {upto_pairs} primary read pairs in SAM, "
            f"observed {stats.total_pairs}. See {stderr_log}."
        )
    if stats.primary_records != expected_primary_records:
        raise RuntimeError(
            f"{sample_id}: expected {expected_primary_records} primary SAM records, "
            f"observed {stats.primary_records}. See {stderr_log}."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        cache_version=np.asarray(CACHE_VERSION, dtype="U80"),
        n_markers=np.asarray(n_markers, dtype=np.int64),
        marker_panel_sha256=np.asarray(marker_panel_sha256, dtype="U64"),
        reference_fasta_sha256=np.asarray(reference_fasta_sha256, dtype="U64"),
        ref_counts=ref_counts,
        alt_counts=alt_counts,
        other_counts=other_counts,
    )

    biallelic_depth = ref_counts + alt_counts
    total_depth = biallelic_depth + other_counts
    callable1 = biallelic_depth >= 1
    callable2 = biallelic_depth >= 2
    alt_fraction = np.full(n_markers, np.nan, dtype=float)
    alt_fraction[callable1] = alt_counts[callable1] / biallelic_depth[callable1]
    het_mask = callable2 & (alt_fraction >= 0.2) & (alt_fraction <= 0.8)
    minor_fraction = np.minimum(alt_fraction, 1.0 - alt_fraction)
    entropy = np.full(n_markers, np.nan, dtype=float)
    valid = callable1
    p = np.clip(alt_fraction[valid], 1e-8, 1.0 - 1e-8)
    entropy[valid] = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / math.log(2.0)

    metrics: Dict[str, Any] = {
        "sample_id": sample_id,
        "r1_path": r1_path,
        "r2_path": r2_path,
        "pairs_requested": int(upto_pairs),
        "min_mapq": int(min_mapq),
        "min_baseq": int(min_baseq),
        "n_markers": int(n_markers),
        "callable_markers_depth1": int(callable1.sum()),
        "callable_markers_depth2": int(callable2.sum()),
        "callable_fraction_depth1": float(callable1.mean()) if n_markers else float("nan"),
        "callable_fraction_depth2": float(callable2.mean()) if n_markers else float("nan"),
        "mean_biallelic_depth_callable": float(np.mean(biallelic_depth[callable1])) if callable1.any() else float("nan"),
        "heterozygous_marker_fraction": float(het_mask.sum() / max(int(callable2.sum()), 1)),
        "mean_minor_allele_fraction": float(np.nanmean(minor_fraction)) if np.isfinite(minor_fraction).any() else float("nan"),
        "mean_allele_entropy": float(np.nanmean(entropy)) if np.isfinite(entropy).any() else float("nan"),
        "other_base_fraction": float(other_counts.sum() / max(int(total_depth.sum()), 1)),
        **stats.as_dict(),
    }
    write_json(metrics, metrics_json)
    return metrics


def combine_cache(cache_dir: Path, metadata: pd.DataFrame, marker_index: MarkerIndex, out_prefix: Path) -> None:
    n_markers = len(marker_index.marker_table)
    sample_ids = metadata["sample_id"].astype(str).tolist()
    alt_matrix = np.full((len(sample_ids), n_markers), np.nan, dtype=np.float32)
    depth_matrix = np.zeros((len(sample_ids), n_markers), dtype=np.uint16)
    for i, sample_id in enumerate(sample_ids):
        payload = np.load(cache_dir / f"{sample_id}.npz", allow_pickle=False)
        ref_counts = payload["ref_counts"].astype(np.uint32)
        alt_counts = payload["alt_counts"].astype(np.uint32)
        depth = ref_counts + alt_counts
        mask = depth >= 1
        alt_matrix[i, mask] = (alt_counts[mask] / depth[mask]).astype(np.float32)
        depth_matrix[i, :] = np.minimum(depth, np.iinfo(np.uint16).max).astype(np.uint16)
    np.savez_compressed(
        str(out_prefix) + "_matrix.npz",
        sample_ids=np.asarray(sample_ids, dtype="U200"),
        alt_fraction=alt_matrix,
        depth=depth_matrix,
    )
    metadata.to_csv(str(out_prefix) + "_samples.tsv", sep="\t", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_manifest", required=True)
    ap.add_argument("--marker_panel", required=True)
    ap.add_argument("--marker_pass", required=True)
    ap.add_argument("--generated_fastq_manifest", required=True)
    ap.add_argument("--truth_manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--generated_pairs", type=int, default=6000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=932690847)
    ap.add_argument("--min_mapq", type=int, default=20)
    ap.add_argument("--min_baseq", type=int, default=20)
    args = ap.parse_args()

    if not Path(args.marker_pass).is_file():
        raise SystemExit("[ERROR] Marker-panel PASS marker is absent")
    outdir = ensure_dir(args.outdir)
    logs = ensure_dir(args.logs_dir)
    refs = pd.read_csv(args.reference_manifest, sep="\t")
    if len(refs) != 1:
        raise SystemExit(f"Expected one reference, found {len(refs)}")
    ref = refs.iloc[0]
    ref_id = str(ref.reference_id)
    marker_path = Path(args.marker_panel)
    marker_index = load_marker_index(marker_path)
    marker_sha = sha256_file(marker_path)

    generated = pd.read_csv(args.generated_fastq_manifest, sep="\t")
    truth = pd.read_csv(args.truth_manifest, sep="\t")
    meta_columns = [
        "replicate","sample_id","class_label","mixture_definition_id","category","pattern_id",
        "n_parents","parents","parent_runs","weights","read_pairs","entropy_norm","minor_fraction",
        "actual_parent_order","actual_parent_runs","actual_read_pair_counts","actual_weights",
        "actual_entropy_norm","actual_minor_fraction",
    ]
    metadata = generated.merge(
        truth[meta_columns],
        on=["replicate","sample_id","class_label","category","pattern_id","read_pairs"],
        how="left",
        validate="one_to_one",
    ).sort_values(["replicate","class_label","sample_id"]).reset_index(drop=True)
    if len(metadata) != 560:
        raise SystemExit(f"Expected 560 generated libraries, found {len(metadata)}")

    ref_out = ensure_dir(outdir / ref_id)
    cache = ensure_dir(ref_out / "cache_generated")
    metrics_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(metadata.itertuples(index=False), start=1):
        sample_id = str(row.sample_id)
        metrics_path = cache / f"{sample_id}.metrics.json"
        cached = metrics_path.is_file()
        print(f"[FEATURE {idx}/560] {sample_id} ({'cache' if cached else 'map'})", flush=True)
        metrics = process_sample(
            sample_id=sample_id,
            r1_path=str(row.r1_path),
            r2_path=str(row.r2_path),
            marker_index=marker_index,
            marker_panel_sha256=marker_sha,
            reference_fasta_sha256=str(ref.fasta_sha256),
            align_launcher=str(ref.align_launcher),
            index_prefix=str(ref.index_prefix),
            threads=args.threads,
            seed=args.seed,
            upto_pairs=args.generated_pairs,
            min_mapq=args.min_mapq,
            min_baseq=args.min_baseq,
            cache_path=cache / f"{sample_id}.npz",
            stderr_log=logs / f"generated_{ref_id}_{sample_id}.bowtie2.log",
        )
        metrics.update({
            "reference_id": ref_id,
            "replicate": int(row.replicate),
            "class_label": str(row.class_label),
            "mixture_definition_id": str(row.mixture_definition_id),
            "category": str(row.category),
            "pattern_id": str(row.pattern_id),
            "n_parents": int(row.n_parents),
            "parents": str(row.parents),
            "parent_runs": str(row.parent_runs),
            "actual_entropy_norm": float(row.actual_entropy_norm),
            "actual_minor_fraction": float(row.actual_minor_fraction),
        })
        metrics_rows.append(metrics)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(ref_out / "generated_mapping_and_marker_metrics.tsv", sep="\t", index=False)
    metadata_for_matrix = metadata[[
        "sample_id","replicate","class_label","mixture_definition_id","category","pattern_id",
        "n_parents","parents","parent_runs","actual_entropy_norm","actual_minor_fraction",
    ]].copy()
    combine_cache(cache, metadata_for_matrix, marker_index, ref_out / "generated")

    summary = {
        "status": "PASS",
        "reference_id": ref_id,
        "generated_sample_count": int(len(metadata)),
        "marker_count": int(len(marker_index.marker_table)),
        "marker_panel_sha256": marker_sha,
        "mapping_rate_mean": float(pd.to_numeric(metrics_df["mapping_rate"], errors="coerce").mean()),
        "unique_mapping_rate_mean": float(pd.to_numeric(metrics_df["unique_mapping_rate"], errors="coerce").mean()),
        "callable_markers_mean": float(pd.to_numeric(metrics_df["callable_markers_depth1"], errors="coerce").mean()),
        "finite_alt_fraction_entries": int(np.isfinite(np.load(str(ref_out / "generated_matrix.npz"), allow_pickle=False)["alt_fraction"]).sum()),
    }
    write_json(summary, ref_out / "feature_extraction_summary.json")
    write_json({"status":"PASS","references":[summary]}, outdir / "feature_extraction_master.json")
    pd.DataFrame([summary]).to_csv(outdir / "feature_extraction_summary.tsv", sep="\t", index=False)
    (outdir / "FEATURE_EXTRACTION_PASS.txt").write_text("PASS\n", encoding="utf-8")

    lines = [
        "Finger millet generated-library reference feature extraction",
        "============================================================",
        "",
        "Status: PASS",
        f"Reference: {ref_id}",
        f"Generated libraries: {len(metadata)}",
        f"Markers: {len(marker_index.marker_table):,}",
        f"Mean mapping rate: {summary['mapping_rate_mean']:.4f}",
        f"Mean callable markers: {summary['callable_markers_mean']:.1f}",
    ]
    (outdir / "FEATURE_EXTRACTION_SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
