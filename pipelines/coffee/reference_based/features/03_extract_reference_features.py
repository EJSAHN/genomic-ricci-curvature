# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from reference_qc_common import (
    AlignmentSummary,
    MarkerIndex,
    count_marker_bases,
    ensure_dir,
    load_marker_index,
    sha256_file,
    stream_bowtie_sam,
    write_json,
)


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


def process_sample(
    *,
    sample_id: str,
    r1_path: str,
    r2_path: str,
    marker_index: MarkerIndex,
    marker_panel_sha256: str,
    align_launcher: str,
    index_prefix: str,
    threads: int,
    seed: int,
    upto_pairs: int,
    skip_pairs: int,
    min_mapq: int,
    min_baseq: int,
    cache_path: Path,
    stderr_log: Path,
) -> Dict[str, object]:
    n_markers = len(marker_index.marker_table)

    if cache_path.is_file():
        payload = np.load(cache_path, allow_pickle=False)
        cached_sha = str(payload["marker_panel_sha256"].item()) if "marker_panel_sha256" in payload else ""
        if int(payload["n_markers"]) == n_markers and cached_sha == marker_panel_sha256:
            metrics_json = cache_path.with_suffix(".metrics.json")
            if metrics_json.is_file():
                return json.loads(metrics_json.read_text(encoding="utf-8"))

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
        skip_pairs=skip_pairs,
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

    if stats.primary_records == 0 or stats.total_pairs == 0:
        raise RuntimeError(
            f"{sample_id}: Bowtie 2 emitted no primary paired SAM records. "
            f"See {stderr_log}. Feature extraction was stopped before an invalid "
            "zero-filled cache could be written."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        n_markers=np.asarray(n_markers, dtype=np.int64),
        marker_panel_sha256=np.asarray(marker_panel_sha256, dtype="U64"),
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

    metrics: Dict[str, object] = {
        "sample_id": sample_id,
        "r1_path": r1_path,
        "r2_path": r2_path,
        "pairs_requested": upto_pairs,
        "pairs_skipped": skip_pairs,
        "n_markers": n_markers,
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
    metrics_json = cache_path.with_suffix(".metrics.json")
    write_json(metrics, metrics_json)
    return metrics


def combine_cache(
    *,
    cache_dir: Path,
    metadata: pd.DataFrame,
    marker_index: MarkerIndex,
    out_prefix: Path,
) -> None:
    n_markers = len(marker_index.marker_table)
    sample_ids = metadata["sample_id"].astype(str).tolist()
    alt_matrix = np.full((len(sample_ids), n_markers), np.nan, dtype=np.float32)
    depth_matrix = np.zeros((len(sample_ids), n_markers), dtype=np.uint16)

    for i, sample_id in enumerate(sample_ids):
        cache_path = cache_dir / f"{sample_id}.npz"
        payload = np.load(cache_path, allow_pickle=False)
        ref_counts = payload["ref_counts"].astype(np.uint32)
        alt_counts = payload["alt_counts"].astype(np.uint32)
        depth = ref_counts + alt_counts
        callable_mask = depth >= 1
        alt_matrix[i, callable_mask] = (
            alt_counts[callable_mask] / depth[callable_mask]
        ).astype(np.float32)
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
    ap.add_argument("--marker_root", required=True)
    ap.add_argument("--generated_fastq_manifest", required=True)
    ap.add_argument("--truth_manifest", required=True)
    ap.add_argument("--sample_manifest", required=True)
    ap.add_argument("--source_fastq_root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--scenario", default="primary")
    ap.add_argument("--generated_pairs", type=int, default=6000)
    ap.add_argument("--real_pairs", type=int, default=50000)
    ap.add_argument("--real_skip_pairs", type=int, default=100000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=53001)
    ap.add_argument("--min_mapq", type=int, default=20)
    ap.add_argument("--min_baseq", type=int, default=20)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    logs_dir = ensure_dir(args.logs_dir)
    marker_root = Path(args.marker_root)
    source_root = Path(args.source_fastq_root)

    refs = pd.read_csv(args.reference_manifest, sep="\t")
    generated = pd.read_csv(args.generated_fastq_manifest, sep="\t")
    truth = pd.read_csv(args.truth_manifest, sep="\t")
    sample_manifest = pd.read_csv(args.sample_manifest, sep="\t")

    generated = generated[generated["scenario"].astype(str) == args.scenario].copy()
    generated = generated.merge(
        truth[
            [
                "scenario",
                "replicate",
                "sample_id",
                "class_label",
                "pattern_id",
                "n_parents",
                "parents",
                "actual_entropy_norm",
                "actual_minor_fraction",
            ]
        ],
        on=["scenario", "replicate", "sample_id", "class_label"],
        how="left",
        validate="one_to_one",
    )
    generated = generated.sort_values(["replicate", "class_label", "sample_id"]).reset_index(drop=True)

    real_meta = sample_manifest.copy()
    real_meta["class_label"] = np.where(
        real_meta["include_primary"].astype(int) == 1,
        "reference_library",
        "metadata_pool_candidate",
    )
    real_meta["scenario"] = "real_data"
    real_meta["replicate"] = 0

    master_summary: list[dict[str, object]] = []

    for ref_row in refs.itertuples(index=False):
        ref_id = str(ref_row.reference_id)
        print(f"\n=== Feature extraction: {ref_id} ===")
        ref_out = ensure_dir(outdir / ref_id)
        marker_path = marker_root / ref_id / "marker_panel.tsv"
        if not marker_path.is_file():
            raise FileNotFoundError(f"Marker panel not found: {marker_path}")
        marker_index = load_marker_index(marker_path)
        marker_panel_sha256 = sha256_file(marker_path)

        cache_generated = ensure_dir(ref_out / "cache_generated")
        cache_real = ensure_dir(ref_out / "cache_real")
        generated_metrics: list[dict[str, object]] = []

        for idx, row in enumerate(generated.itertuples(index=False), start=1):
            sample_id = str(row.sample_id)
            print(f"[GENERATED {idx}/{len(generated)}] {sample_id}")
            metrics = process_sample(
                sample_id=sample_id,
                r1_path=str(row.r1_path),
                r2_path=str(row.r2_path),
                marker_index=marker_index,
                marker_panel_sha256=marker_panel_sha256,
                align_launcher=str(ref_row.align_launcher),
                index_prefix=str(ref_row.index_prefix),
                threads=args.threads,
                seed=args.seed,
                upto_pairs=args.generated_pairs,
                skip_pairs=0,
                min_mapq=args.min_mapq,
                min_baseq=args.min_baseq,
                cache_path=cache_generated / f"{sample_id}.npz",
                stderr_log=logs_dir / f"generated_{ref_id}_{sample_id}.bowtie2.log",
            )
            metrics.update(
                {
                    "reference_id": ref_id,
                    "scenario": str(row.scenario),
                    "replicate": int(row.replicate),
                    "class_label": str(row.class_label),
                    "pattern_id": str(row.pattern_id),
                    "n_parents": int(row.n_parents),
                    "parents": str(row.parents),
                    "actual_entropy_norm": float(row.actual_entropy_norm),
                    "actual_minor_fraction": float(row.actual_minor_fraction),
                }
            )
            generated_metrics.append(metrics)

        generated_metrics_df = pd.DataFrame(generated_metrics)
        generated_metrics_df.to_csv(
            ref_out / "generated_mapping_and_marker_metrics.tsv", sep="\t", index=False
        )
        combine_cache(
            cache_dir=cache_generated,
            metadata=generated[
                [
                    "sample_id",
                    "scenario",
                    "replicate",
                    "class_label",
                    "pattern_id",
                    "n_parents",
                    "parents",
                    "actual_entropy_norm",
                    "actual_minor_fraction",
                ]
            ].copy(),
            marker_index=marker_index,
            out_prefix=ref_out / "generated",
        )

        real_metrics: list[dict[str, object]] = []
        for idx, row in enumerate(real_meta.itertuples(index=False), start=1):
            sample_id = str(row.sample_id)
            print(f"[REAL {idx}/{len(real_meta)}] {sample_id}")
            r1 = find_fastq(source_root, sample_id, 1)
            r2 = find_fastq(source_root, sample_id, 2)
            metrics = process_sample(
                sample_id=sample_id,
                r1_path=str(r1),
                r2_path=str(r2),
                marker_index=marker_index,
                marker_panel_sha256=marker_panel_sha256,
                align_launcher=str(ref_row.align_launcher),
                index_prefix=str(ref_row.index_prefix),
                threads=args.threads,
                seed=args.seed + 1,
                upto_pairs=args.real_pairs,
                skip_pairs=args.real_skip_pairs,
                min_mapq=args.min_mapq,
                min_baseq=args.min_baseq,
                cache_path=cache_real / f"{sample_id}.npz",
                stderr_log=logs_dir / f"real_{ref_id}_{sample_id}.bowtie2.log",
            )
            metrics.update(
                {
                    "reference_id": ref_id,
                    "scenario": "real_data",
                    "replicate": 0,
                    "class_label": str(row.class_label),
                    "source_role": str(row.source_role),
                    "label_source": str(row.label_source),
                    "include_primary": int(row.include_primary),
                }
            )
            real_metrics.append(metrics)

        real_metrics_df = pd.DataFrame(real_metrics)
        real_metrics_df.to_csv(
            ref_out / "real_mapping_and_marker_metrics.tsv", sep="\t", index=False
        )
        combine_cache(
            cache_dir=cache_real,
            metadata=real_meta[
                [
                    "sample_id",
                    "scenario",
                    "replicate",
                    "class_label",
                    "source_role",
                    "label_source",
                    "include_primary",
                ]
            ].copy(),
            marker_index=marker_index,
            out_prefix=ref_out / "real",
        )

        summary = {
            "reference_id": ref_id,
            "n_generated_samples": len(generated_metrics_df),
            "n_real_samples": len(real_metrics_df),
            "n_markers": int(len(pd.read_csv(marker_path, sep="\t"))),
            "generated_mapping_rate_mean": float(generated_metrics_df["mapping_rate"].mean()),
            "generated_callable_markers_mean": float(
                generated_metrics_df["callable_markers_depth1"].mean()
            ),
            "real_mapping_rate_mean": float(real_metrics_df["mapping_rate"].mean()),
            "real_callable_markers_mean": float(
                real_metrics_df["callable_markers_depth1"].mean()
            ),
        }
        write_json(summary, ref_out / "feature_extraction_summary.json")
        master_summary.append(summary)

    pd.DataFrame(master_summary).to_csv(
        outdir / "feature_extraction_summary.tsv", sep="\t", index=False
    )
    write_json({"references": master_summary, "status": "PASS"}, outdir / "feature_extraction_master.json")
    print("\n[DONE] Reference-based feature extraction complete.")


if __name__ == "__main__":
    main()
