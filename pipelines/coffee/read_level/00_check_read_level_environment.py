# -*- coding: utf-8 -*-
"""
Preflight checks for paired-read synthetic validation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import networkx
import numpy
import openpyxl
import pandas
import scipy
import sklearn

from read_level_common import discover_fastq_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_manifest", required=True)
    parser.add_argument("--source_fastq_root", required=True)
    parser.add_argument("--module_root", required=True)
    parser.add_argument("--minimum_free_gb", type=float, default=2.0)
    args = parser.parse_args()

    manifest = pandas.read_csv(args.sample_manifest, sep="\t", dtype=str).fillna("")
    required = manifest[
        manifest[["include_primary", "include_conservative"]]
        .apply(lambda col: col.str.lower().isin({"1", "true", "yes", "y"}))
        .any(axis=1)
    ]["sample_id"].tolist()
    pairs = discover_fastq_pairs(args.source_fastq_root, required)
    missing = [
        sample for sample in required
        if sample not in pairs or "1" not in pairs[sample] or "2" not in pairs[sample]
    ]
    free_gb = shutil.disk_usage(args.module_root).free / (1024**3)

    report = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "networkx": networkx.__version__,
        "openpyxl": openpyxl.__version__,
        "source_fastq_root": str(Path(args.source_fastq_root).resolve()),
        "required_source_libraries": len(required),
        "missing_paired_libraries": missing,
        "free_disk_gb": free_gb,
        "minimum_free_gb": args.minimum_free_gb,
        "status": "PASS" if not missing and free_gb >= args.minimum_free_gb else "FAIL",
    }
    out = Path(args.module_root) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for key, value in report.items():
        print(f"{key}: {value}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
