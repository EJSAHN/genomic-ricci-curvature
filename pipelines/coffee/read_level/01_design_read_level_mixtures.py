# -*- coding: utf-8 -*-
"""
Create deterministic paired-read mixture designs.

The output defines single-source controls and synthetic mixtures. It does not
read or modify FASTQ files.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from read_level_common import (
    ensure_dir,
    largest_remainder_counts,
    normalized_entropy,
    stable_seed,
    write_json,
)


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return token or "scenario"


def choose_balanced_parents(
    parents: List[str],
    n_parents: int,
    usage_reads: Dict[str, int],
    rng: np.random.Generator,
) -> List[str]:
    jitter = {sample: float(rng.random()) for sample in parents}
    ordered = sorted(parents, key=lambda sample: (usage_reads[sample], jitter[sample], sample))
    return ordered[:n_parents]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_manifest", required=True)
    parser.add_argument("--ratio_patterns", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--scenario_columns",
        default="include_primary,include_conservative",
        help="Comma-separated 0/1 columns in the sample manifest",
    )
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--mixtures_per_pattern", type=int, default=4)
    parser.add_argument("--read_pairs_per_sample", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42000)
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    manifest = pd.read_csv(args.sample_manifest, sep="\t", dtype=str).fillna("")
    patterns = pd.read_csv(args.ratio_patterns, sep="\t", dtype=str).fillna("")

    required_manifest = {"sample_id", "source_role"}
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise SystemExit(f"[ERR] Missing sample-manifest columns: {sorted(missing)}")

    required_patterns = {"pattern_id", "n_parents", "weights"}
    missing = required_patterns - set(patterns.columns)
    if missing:
        raise SystemExit(f"[ERR] Missing ratio-pattern columns: {sorted(missing)}")

    scenario_columns = [x.strip() for x in args.scenario_columns.split(",") if x.strip()]
    for column in scenario_columns:
        if column not in manifest.columns:
            raise SystemExit(f"[ERR] Scenario column not found in sample manifest: {column}")

    pattern_rows = []
    for row in patterns.itertuples(index=False):
        n_parents = int(getattr(row, "n_parents"))
        weights = [float(x) for x in str(getattr(row, "weights")).split(",") if x.strip()]
        if len(weights) != n_parents:
            raise SystemExit(
                f"[ERR] Pattern {getattr(row, 'pattern_id')} has n_parents={n_parents} "
                f"but {len(weights)} weights"
            )
        weights = (np.asarray(weights, dtype=float) / np.sum(weights)).tolist()
        pattern_rows.append(
            {
                "pattern_id": safe_token(str(getattr(row, "pattern_id"))),
                "n_parents": n_parents,
                "weights": weights,
            }
        )

    design_rows = []
    allocation_rows = []
    scenario_summary = []

    for scenario_index, scenario_column in enumerate(scenario_columns, start=1):
        selected = manifest[manifest[scenario_column].str.strip().str.lower().isin({"1", "true", "yes", "y"})]
        parents = sorted(selected["sample_id"].tolist())
        if len(parents) < max(x["n_parents"] for x in pattern_rows):
            raise SystemExit(f"[ERR] Too few parents for {scenario_column}: {len(parents)}")

        scenario = safe_token(scenario_column.replace("include_", ""))
        for replicate in range(1, args.replicates + 1):
            rng = np.random.default_rng(stable_seed(args.seed, scenario, replicate))
            usage_reads = {sample: 0 for sample in parents}

            # One depth-matched, single-source control per parent.
            for sample in parents:
                sample_id = f"CTRL_{scenario}_R{replicate:02d}_{sample}"
                design_rows.append(
                    {
                        "scenario": scenario,
                        "replicate": replicate,
                        "sample_id": sample_id,
                        "class_label": "single_source_control",
                        "pattern_id": "single_source",
                        "n_parents": 1,
                        "parents": sample,
                        "target_weights": "1.000000",
                        "target_counts": str(args.read_pairs_per_sample),
                        "target_entropy_norm": 0.0,
                        "target_minor_fraction": 0.0,
                        "read_pairs": args.read_pairs_per_sample,
                    }
                )
                allocation_rows.append(
                    {
                        "scenario": scenario,
                        "replicate": replicate,
                        "sample_id": sample_id,
                        "class_label": "single_source_control",
                        "pattern_id": "single_source",
                        "parent_id": sample,
                        "target_weight": 1.0,
                        "read_pairs": args.read_pairs_per_sample,
                    }
                )
                usage_reads[sample] += args.read_pairs_per_sample

            for pattern in pattern_rows:
                for instance in range(1, args.mixtures_per_pattern + 1):
                    chosen = choose_balanced_parents(
                        parents=parents,
                        n_parents=pattern["n_parents"],
                        usage_reads=usage_reads,
                        rng=rng,
                    )
                    permuted_weights = np.asarray(pattern["weights"], dtype=float)[
                        rng.permutation(pattern["n_parents"])
                    ]
                    counts = largest_remainder_counts(
                        permuted_weights.tolist(),
                        args.read_pairs_per_sample,
                    )
                    sample_id = (
                        f"MIX_{scenario}_R{replicate:02d}_"
                        f"{pattern['pattern_id']}_{instance:02d}"
                    )
                    for parent, count in zip(chosen, counts):
                        usage_reads[parent] += int(count)
                    design_rows.append(
                        {
                            "scenario": scenario,
                            "replicate": replicate,
                            "sample_id": sample_id,
                            "class_label": "synthetic_mixture",
                            "pattern_id": pattern["pattern_id"],
                            "n_parents": pattern["n_parents"],
                            "parents": ",".join(chosen),
                            "target_weights": ",".join(f"{x:.6f}" for x in permuted_weights),
                            "target_counts": ",".join(str(x) for x in counts),
                            "target_entropy_norm": normalized_entropy(permuted_weights),
                            "target_minor_fraction": float(np.min(permuted_weights)),
                            "read_pairs": args.read_pairs_per_sample,
                        }
                    )
                    for parent, weight, count in zip(chosen, permuted_weights, counts):
                        allocation_rows.append(
                            {
                                "scenario": scenario,
                                "replicate": replicate,
                                "sample_id": sample_id,
                                "class_label": "synthetic_mixture",
                                "pattern_id": pattern["pattern_id"],
                                "parent_id": parent,
                                "target_weight": float(weight),
                                "read_pairs": int(count),
                            }
                        )

        scenario_summary.append(
            {
                "scenario": scenario,
                "scenario_column": scenario_column,
                "n_reference_libraries": len(parents),
                "reference_libraries": parents,
                "replicates": args.replicates,
                "mixtures_per_replicate": len(pattern_rows) * args.mixtures_per_pattern,
                "controls_per_replicate": len(parents),
                "read_pairs_per_sample": args.read_pairs_per_sample,
            }
        )

    design_df = pd.DataFrame(design_rows).sort_values(
        ["scenario", "replicate", "class_label", "sample_id"]
    )
    allocation_df = pd.DataFrame(allocation_rows).sort_values(
        ["parent_id", "scenario", "replicate", "sample_id"]
    )

    design_path = outdir / "design_manifest.tsv"
    allocation_path = outdir / "read_allocations.tsv"
    design_df.to_csv(design_path, sep="\t", index=False)
    allocation_df.to_csv(allocation_path, sep="\t", index=False)

    payload = {
        "seed": args.seed,
        "replicates": args.replicates,
        "mixtures_per_pattern": args.mixtures_per_pattern,
        "read_pairs_per_sample": args.read_pairs_per_sample,
        "patterns": pattern_rows,
        "scenarios": scenario_summary,
        "n_generated_libraries": int(len(design_df)),
        "n_allocation_rows": int(len(allocation_df)),
    }
    write_json(outdir / "design_summary.json", payload)

    print(f"[DONE] Design manifest: {design_path}")
    print(f"[DONE] Read allocations: {allocation_path}")
    print(f"[DONE] Generated libraries planned: {len(design_df)}")
    for item in scenario_summary:
        print(
            f"[INFO] {item['scenario']}: {item['n_reference_libraries']} references, "
            f"{item['mixtures_per_replicate']} mixtures/replicate, "
            f"{item['replicates']} replicates"
        )


if __name__ == "__main__":
    main()
