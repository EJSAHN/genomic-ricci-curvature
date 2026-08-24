# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crossfit_common import ensure_dir, load_matrix, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_root", required=True)
    ap.add_argument("--read_level_root", required=True)
    ap.add_argument("--baseline_audit_json", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--expected_references", type=int, default=2)
    ap.add_argument("--expected_generated", type=int, default=185)
    ap.add_argument("--expected_real", type=int, default=18)
    ap.add_argument("--expected_replicates", type=int, default=5)
    args = ap.parse_args()

    reference_root = Path(args.reference_root)
    read_level_root = Path(args.read_level_root)
    outdir = ensure_dir(args.outdir)

    checks: list[dict[str, object]] = []

    def check(name: str, observed: object, expected: object, ok: bool) -> None:
        checks.append(
            {
                "name": name,
                "observed": observed,
                "expected": expected,
                "ok": bool(ok),
            }
        )
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: observed={observed}; expected={expected}")

    check("reference root", str(reference_root), "existing directory", reference_root.is_dir())
    check("read-level root", str(read_level_root), "existing directory", read_level_root.is_dir())

    baseline_path = Path(args.baseline_audit_json)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    check("baseline reproduction audit", baseline.get("status"), "PASS", baseline.get("status") == "PASS")

    reference_audit_path = reference_root / "audit" / "reference_qc_audit.json"
    check(
        "source reference-QC audit",
        str(reference_audit_path),
        "existing PASS audit",
        reference_audit_path.is_file(),
    )
    if reference_audit_path.is_file():
        reference_audit = json.loads(reference_audit_path.read_text(encoding="utf-8"))
        check(
            "source reference-QC audit status",
            reference_audit.get("status"),
            "PASS",
            reference_audit.get("status") == "PASS",
        )
        source_checks = {
            str(row.get("name")): row
            for row in reference_audit.get("checks", [])
        }
        for name in [
            "marker-discovery overlap with generated benchmark reads",
            "marker-discovery overlap with real-data scoring segment",
        ]:
            row = source_checks.get(name, {})
            check(
                name,
                row.get("observed"),
                0,
                bool(row.get("ok")) and row.get("observed") == 0,
            )

    read_level_pass = read_level_root / "audit" / "READ_LEVEL_VALIDATION_PASS.txt"
    check("paired-read validation audit", str(read_level_pass), "existing PASS marker", read_level_pass.is_file())

    manifest_path = reference_root / "reference" / "reference_manifest.tsv"
    check("reference manifest", str(manifest_path), "existing file", manifest_path.is_file())
    refs = pd.read_csv(manifest_path, sep="\t") if manifest_path.is_file() else pd.DataFrame()
    check(
        "reference count",
        int(len(refs)),
        int(args.expected_references),
        len(refs) == args.expected_references,
    )

    feature_rows: list[dict[str, object]] = []
    for row in refs.itertuples(index=False):
        ref_id = str(row.reference_id)
        ref_features = reference_root / "features" / ref_id
        try:
            Xg, dg, ids_g, sg = load_matrix(ref_features / "generated")
            Xr, dr, ids_r, sr = load_matrix(ref_features / "real")
            gmap = pd.read_csv(
                ref_features / "generated_mapping_and_marker_metrics.tsv", sep="\t"
            )
            rmap = pd.read_csv(
                ref_features / "real_mapping_and_marker_metrics.tsv", sep="\t"
            )
            check(
                f"{ref_id} generated sample count",
                int(len(sg)),
                int(args.expected_generated),
                len(sg) == args.expected_generated,
            )
            check(
                f"{ref_id} real sample count",
                int(len(sr)),
                int(args.expected_real),
                len(sr) == args.expected_real,
            )
            replicate_set = sorted(sg["replicate"].astype(int).unique().tolist())
            check(
                f"{ref_id} replicate set",
                replicate_set,
                list(range(1, args.expected_replicates + 1)),
                replicate_set == list(range(1, args.expected_replicates + 1)),
            )
            check(
                f"{ref_id} generated mapping rows",
                int(len(gmap)),
                int(args.expected_generated),
                len(gmap) == args.expected_generated,
            )
            check(
                f"{ref_id} real mapping rows",
                int(len(rmap)),
                int(args.expected_real),
                len(rmap) == args.expected_real,
            )
            check(
                f"{ref_id} generated finite matrix entries",
                int(np.isfinite(Xg).sum()),
                "> 0",
                int(np.isfinite(Xg).sum()) > 0,
            )
            feature_rows.append(
                {
                    "reference_id": ref_id,
                    "generated_rows": len(sg),
                    "real_rows": len(sr),
                    "markers": Xg.shape[1],
                    "generated_finite_entries": int(np.isfinite(Xg).sum()),
                    "real_finite_entries": int(np.isfinite(Xr).sum()),
                }
            )
        except Exception as exc:
            check(f"{ref_id} feature bundle", str(exc), "loadable", False)

    pd.DataFrame(feature_rows).to_csv(
        outdir / "crossfit_input_inventory.tsv", sep="\t", index=False
    )
    status = "PASS" if all(bool(x["ok"]) for x in checks) else "FAIL"
    payload = {"status": status, "checks": checks, "feature_inventory": feature_rows}
    write_json(payload, outdir / "crossfit_preflight.json")
    (outdir / "crossfit_preflight.txt").write_text(
        "\n".join(
            [f"STATUS: {status}"]
            + [
                f"[{'PASS' if x['ok'] else 'FAIL'}] {x['name']}: "
                f"observed={x['observed']}; expected={x['expected']}"
                for x in checks
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"STATUS: {status}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
