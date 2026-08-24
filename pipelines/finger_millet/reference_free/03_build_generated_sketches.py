# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from external_common import (
    ensure_dir,
    parse_int,
    read_tsv,
    sketch_generated_pair,
    write_json,
    write_tsv,
)


SKETCH_BUILDER_VERSION = "generated-paired-sketch-v1"


def sketch_task(
    row: Dict[str, str],
    kmer: int,
    sketch_dimension: int,
    cache_path: str,
) -> Tuple[str, str, int, int, int, bool]:
    path = Path(cache_path)
    cache_key = (
        f"{SKETCH_BUILDER_VERSION}|{row['r1_sha256']}|{row['r2_sha256']}|"
        f"{row['read_pairs']}|{kmer}|{sketch_dimension}"
    )
    if path.exists():
        try:
            data = np.load(path, allow_pickle=False)
            if str(data["cache_key"].item()) == cache_key:
                return (
                    row["sample_id"],
                    str(path),
                    int(data["pair_count"].item()),
                    int(data["r1_kmer_count"].item()),
                    int(data["r2_kmer_count"].item()),
                    True,
                )
        except Exception:
            pass

    r1_signature, paired_signature, pair_count, r1_kmers, r2_kmers = sketch_generated_pair(
        row["r1_path"],
        row["r2_path"],
        k=int(kmer),
        sketch_size=int(sketch_dimension),
        expected_pairs=parse_int(row["read_pairs"]),
    )
    ensure_dir(path.parent)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        cache_key=np.asarray(cache_key),
        sample_id=np.asarray(row["sample_id"]),
        r1_signature=r1_signature,
        paired_signature=paired_signature,
        pair_count=np.asarray(pair_count),
        r1_kmer_count=np.asarray(r1_kmers),
        r2_kmer_count=np.asarray(r2_kmers),
        kmer=np.asarray(int(kmer)),
        sketch_dimension=np.asarray(int(sketch_dimension)),
    )
    temporary.replace(path)
    return row["sample_id"], str(path), pair_count, r1_kmers, r2_kmers, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_manifest", required=True)
    parser.add_argument("--build_pass", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--kmer", type=int, default=17)
    parser.add_argument("--sketch", type=int, default=16384)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if not Path(args.build_pass).exists():
        raise SystemExit("[ERROR] Generated FASTQ build PASS marker is absent.")

    outdir = ensure_dir(args.outdir)
    cache_dir = ensure_dir(args.cache_dir)
    rows = read_tsv(args.generated_manifest)
    if len(rows) != 560:
        raise SystemExit(f"[ERROR] Generated manifest has {len(rows)} rows, expected 560.")

    results: Dict[str, Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {}
        for row in rows:
            sample_id = row["sample_id"]
            cache_path = cache_dir / f"{sample_id}_k{args.kmer}_d{args.sketch}.npz"
            future = pool.submit(
                sketch_task,
                row,
                int(args.kmer),
                int(args.sketch),
                str(cache_path),
            )
            futures[future] = sample_id

        completed = 0
        for future in as_completed(futures):
            sample_id, cache_path, pair_count, r1_kmers, r2_kmers, cache_hit = future.result()
            results[sample_id] = {
                "sketch_cache": cache_path,
                "observed_pair_count": pair_count,
                "r1_kmer_count": r1_kmers,
                "r2_kmer_count": r2_kmers,
                "cache_hit": cache_hit,
            }
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(rows):
                print(
                    f"[SKETCH {completed}/560] {sample_id} "
                    f"({'cache' if cache_hit else 'computed'})",
                    flush=True,
                )

    output_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for row in rows:
        result = results[row["sample_id"]]
        output = {
            **row,
            **result,
            "kmer": int(args.kmer),
            "sketch_dimension": int(args.sketch),
        }
        output_rows.append(output)
        if result["observed_pair_count"] != parse_int(row["read_pairs"]):
            failures.append(
                {
                    "sample_id": row["sample_id"],
                    "observed_pair_count": result["observed_pair_count"],
                    "expected_pair_count": parse_int(row["read_pairs"]),
                }
            )

    write_tsv(outdir / "generated_sketch_manifest.tsv", output_rows)
    write_tsv(outdir / "generated_sketch_failures.tsv", failures)
    status = "PASS" if not failures and len(output_rows) == 560 else "FAIL"
    summary = {
        "status": status,
        "builder_version": SKETCH_BUILDER_VERSION,
        "generated_libraries": len(output_rows),
        "kmer": int(args.kmer),
        "sketch_dimension": int(args.sketch),
        "analysis_modes_available": ["r1", "paired"],
        "pair_count_failures": len(failures),
        "cache_hits": sum(bool(row["cache_hit"]) for row in output_rows),
        "computed": sum(not bool(row["cache_hit"]) for row in output_rows),
    }
    write_json(outdir / "generated_sketch_summary.json", summary)
    text = [
        "Finger millet generated-library sketch build",
        "=============================================",
        "",
        f"Status: {status}",
        f"Generated libraries: {len(output_rows)}",
        f"k-mer length: {args.kmer}",
        f"Sketch dimension: {args.sketch}",
        f"Cache hits: {summary['cache_hits']}",
        f"Newly computed: {summary['computed']}",
        f"Pair-count failures: {len(failures)}",
    ]
    (outdir / "GENERATED_SKETCH_SUMMARY.txt").write_text(
        "\n".join(text) + "\n", encoding="utf-8"
    )
    print("\n".join(text))
    marker = outdir / "GENERATED_SKETCH_PASS.txt"
    if status == "PASS":
        marker.write_text("PASS\n", encoding="utf-8")
    else:
        if marker.exists():
            marker.unlink()
        raise SystemExit(4)


if __name__ == "__main__":
    main()
