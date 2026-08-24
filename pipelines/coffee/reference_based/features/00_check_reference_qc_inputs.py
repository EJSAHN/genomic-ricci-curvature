# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from reference_qc_common import ensure_dir, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--read_level_root", required=True)
    ap.add_argument("--baseline_audit_json", required=True)
    ap.add_argument("--reference_config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--minimum_free_gb", type=float, default=10.0)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    project_root = Path(args.project_root)
    read_level_root = Path(args.read_level_root)
    baseline_audit = Path(args.baseline_audit_json)
    reference_config = Path(args.reference_config)

    checks: list[dict[str, object]] = []

    def check(name: str, ok: bool, detail: object) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    check("project root", project_root.is_dir(), str(project_root))
    check("read-level root", read_level_root.is_dir(), str(read_level_root))
    pass_file = read_level_root / "audit" / "READ_LEVEL_VALIDATION_PASS.txt"
    check("read-level validation audit", pass_file.is_file(), str(pass_file))
    truth = read_level_root / "generated" / "manifests" / "truth_manifest.tsv"
    generated = read_level_root / "generated" / "manifests" / "generated_fastq_manifest.tsv"
    check("truth manifest", truth.is_file(), str(truth))
    check("generated FASTQ manifest", generated.is_file(), str(generated))
    check("baseline reproduction audit", baseline_audit.is_file(), str(baseline_audit))
    check("reference configuration", reference_config.is_file(), str(reference_config))

    if baseline_audit.is_file():
        payload = json.loads(baseline_audit.read_text(encoding="utf-8-sig"))
        check("baseline audit status", payload.get("status") == "PASS", payload.get("status"))

    n_generated = 0
    missing_fastq: list[str] = []
    scenarios: list[str] = []
    if generated.is_file():
        df = pd.read_csv(generated, sep="\t")
        n_generated = len(df)
        scenarios = sorted(df["scenario"].astype(str).unique().tolist())
        for row in df.itertuples(index=False):
            for p in [str(row.r1_path), str(row.r2_path)]:
                if not Path(p).is_file():
                    missing_fastq.append(p)
        check("generated libraries", n_generated == 365, n_generated)
        check("generated FASTQ paths", len(missing_fastq) == 0, f"missing={len(missing_fastq)}")
        check("scenario set", scenarios == ["conservative", "primary"], scenarios)

    enabled_refs: list[str] = []
    if reference_config.is_file():
        cfg = pd.read_csv(reference_config, sep="\t")
        enabled = cfg[cfg["enabled"].astype(str).isin(["1", "true", "True", "yes", "YES"])]
        enabled_refs = enabled["reference_id"].astype(str).tolist()
        check("enabled reference count", len(enabled_refs) >= 1, enabled_refs)

    free_gb = shutil.disk_usage(project_root.anchor or str(project_root)).free / (1024**3)
    check("free disk space", free_gb >= args.minimum_free_gb, f"{free_gb:.2f} GB (minimum {args.minimum_free_gb:.2f})")

    report = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "project_root": str(project_root),
        "read_level_root": str(read_level_root),
        "n_generated_libraries": n_generated,
        "scenarios": scenarios,
        "enabled_references": enabled_refs,
        "free_disk_gb": free_gb,
        "minimum_free_gb": args.minimum_free_gb,
        "checks": checks,
        "status": "PASS" if all(bool(c["ok"]) for c in checks) else "FAIL",
    }
    write_json(report, outdir / "preflight.json")
    if report["status"] != "PASS":
        raise SystemExit(1)
    print("STATUS: PASS")


if __name__ == "__main__":
    main()
