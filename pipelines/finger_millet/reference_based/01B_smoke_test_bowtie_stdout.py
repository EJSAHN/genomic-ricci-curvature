# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reference_common import AlignmentSummary, mismatch_events, stream_bowtie_sam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_manifest", required=True)
    ap.add_argument("--source_manifest", required=True)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=932690845)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    refs = pd.read_csv(args.reference_manifest, sep="\t")
    sources = pd.read_csv(args.source_manifest, sep="\t")
    if len(refs) != 1:
        raise SystemExit(f"Expected one reference, found {len(refs)}")
    if len(sources) < 1:
        raise SystemExit("Source manifest is empty")

    ref = refs.iloc[0]
    source = sources.iloc[0]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stats = AlignmentSummary()
    md_records = 0
    nm_records = 0
    mismatch_events_q0 = 0
    mismatch_events_q20 = 0
    stderr_log = outdir / f"smoke_{ref.reference_id}_{source.run_accession}.bowtie2.log"

    for record in stream_bowtie_sam(
        launcher=str(ref.align_launcher),
        index_prefix=str(ref.index_prefix),
        r1_path=str(source.r1_path),
        r2_path=str(source.r2_path),
        threads=args.threads,
        seed=args.seed,
        upto_pairs=args.pairs,
        skip_pairs=0,
        stderr_log=str(stderr_log),
    ):
        stats.update(record, unique_mapq=20)
        md_records += int("MD" in record.tags)
        nm_records += int("NM" in record.tags)
        if record.is_unmapped or not record.is_primary:
            continue
        mismatch_events_q0 += sum(1 for _ in mismatch_events(record, min_baseq=0))
        mismatch_events_q20 += sum(1 for _ in mismatch_events(record, min_baseq=20))

    metrics = stats.as_dict()
    expected_primary = 2 * int(args.pairs)
    row = {
        "reference_id": str(ref.reference_id),
        "reference_accession": str(ref.accession),
        "source_run_accession": str(source.run_accession),
        "pairs_requested": int(args.pairs),
        **metrics,
        "records_with_MD": int(md_records),
        "records_with_NM": int(nm_records),
        "mismatch_events_baseq0": int(mismatch_events_q0),
        "mismatch_events_baseq20": int(mismatch_events_q20),
        "stderr_log": str(stderr_log),
    }
    pd.DataFrame([row]).to_csv(
        outdir / "bowtie_stdout_smoke_test.tsv", sep="\t", index=False
    )

    if int(metrics["total_pairs"]) != int(args.pairs):
        raise RuntimeError(
            f"Expected {args.pairs} read pairs in SAM, observed "
            f"{metrics['total_pairs']}."
        )
    if int(metrics["primary_records"]) != expected_primary:
        raise RuntimeError(
            f"Expected {expected_primary} primary SAM records, observed "
            f"{metrics['primary_records']}."
        )
    if md_records == 0 or nm_records == 0:
        raise RuntimeError(
            "SAM records were emitted but MD/NM tags were absent; marker "
            "discovery cannot proceed."
        )

    (outdir / "BOWTIE_STDOUT_SMOKE_PASS.txt").write_text("PASS\n", encoding="utf-8")
    print(
        f"[PASS] reference={ref.reference_id}; source={source.run_accession}; "
        f"pairs={metrics['total_pairs']}; primary={metrics['primary_records']}; "
        f"mapping_rate={metrics['mapping_rate']:.4f}; "
        f"unique_rate={metrics['unique_mapping_rate']:.4f}; "
        f"MD_records={md_records}; NM_records={nm_records}; "
        f"mismatches_Q20={mismatch_events_q20}"
    )
    print(f"Report: {outdir / 'bowtie_stdout_smoke_test.tsv'}")


if __name__ == "__main__":
    main()
