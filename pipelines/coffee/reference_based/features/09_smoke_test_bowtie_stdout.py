# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from reference_qc_common import AlignmentSummary, stream_bowtie_sam


def find_fastq(root: Path, sample_id: str, mate: int) -> Path:
    candidates = (
        root / f"{sample_id}_{mate}.fastq.gz",
        root / f"{sample_id}_{mate}.fq.gz",
        root / f"{sample_id}_{mate}.fastq",
        root / f"{sample_id}_{mate}.fq",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"FASTQ mate {mate} not found for {sample_id} in {root}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_manifest", required=True)
    ap.add_argument("--fastq_root", required=True)
    ap.add_argument("--sample_id", default="SRR17037610")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=54001)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    refs = pd.read_csv(args.reference_manifest, sep="\t")
    root = Path(args.fastq_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    r1 = find_fastq(root, args.sample_id, 1)
    r2 = find_fastq(root, args.sample_id, 2)

    rows = []
    for row in refs.itertuples(index=False):
        ref_id = str(row.reference_id)
        stats = AlignmentSummary()
        md_records = 0
        nm_records = 0
        stderr_log = outdir / f"smoke_{ref_id}.bowtie2.log"

        for record in stream_bowtie_sam(
            launcher=str(row.align_launcher),
            index_prefix=str(row.index_prefix),
            r1_path=str(r1),
            r2_path=str(r2),
            threads=args.threads,
            seed=args.seed,
            upto_pairs=args.pairs,
            skip_pairs=0,
            stderr_log=str(stderr_log),
        ):
            stats.update(record, unique_mapq=20)
            md_records += int("MD" in record.tags)
            nm_records += int("NM" in record.tags)

        metrics = stats.as_dict()
        rows.append(
            {
                "reference_id": ref_id,
                "sample_id": args.sample_id,
                "pairs_requested": args.pairs,
                **metrics,
                "records_with_MD": md_records,
                "records_with_NM": nm_records,
                "stderr_log": str(stderr_log),
            }
        )

        expected_records = 2 * args.pairs
        if int(metrics["total_pairs"]) != args.pairs:
            raise RuntimeError(
                f"{ref_id}: expected {args.pairs} read pairs in SAM, "
                f"observed {metrics['total_pairs']}."
            )
        if int(metrics["primary_records"]) != expected_records:
            raise RuntimeError(
                f"{ref_id}: expected {expected_records} primary SAM records, "
                f"observed {metrics['primary_records']}."
            )
        if nm_records == 0 or md_records == 0:
            raise RuntimeError(
                f"{ref_id}: SAM records were emitted but MD/NM tags were absent; "
                "marker discovery cannot proceed."
            )
        print(
            f"[PASS] {ref_id}: pairs={metrics['total_pairs']}, "
            f"primary_records={metrics['primary_records']}, "
            f"mapping_rate={metrics['mapping_rate']:.4f}, "
            f"MD_records={md_records}, NM_records={nm_records}"
        )

    table = pd.DataFrame(rows)
    path = outdir / "bowtie_stdout_smoke_test.tsv"
    table.to_csv(path, sep="\t", index=False)
    print("STATUS: PASS")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
