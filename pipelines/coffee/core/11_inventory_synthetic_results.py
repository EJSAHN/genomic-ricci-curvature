# -*- coding: utf-8 -*-
"""
11_inventory_synthetic_results.py
Scan a directory tree for synthetic-mixing workbooks and summarize the
metrics_summary sheet into a single inventory table.

Outputs:
- Synthetic_Result_Inventory.xlsx
- Synthetic_Result_Inventory.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search_root", required=True, help="Directory to scan recursively")
    ap.add_argument("--filename", default="Supplementary_Data_S4_SyntheticMixing.xlsx",
                    help="Workbook filename to inventory")
    ap.add_argument("--sheet", default="metrics_summary", help="Worksheet to read")
    ap.add_argument("--outdir", required=True, help="Output directory")
    args = ap.parse_args()

    search_root = Path(args.search_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    paths = sorted(search_root.rglob(args.filename))
    rows = []
    for path in paths:
        try:
            metrics = pd.read_excel(path, sheet_name=args.sheet)
            row0 = metrics.iloc[0].to_dict() if not metrics.empty else {}
            rows.append({
                "path": str(path),
                "roc_auc": row0.get("roc_auc_synthetic"),
                "reads_per_sample": row0.get("reads_per_sample"),
                "kmer": row0.get("kmer"),
                "sketch": row0.get("sketch"),
                "knn": row0.get("knn"),
                "orc_alpha": row0.get("orc_alpha"),
                "use_r2": row0.get("use_r2"),
                "seed": row0.get("seed"),
                "n_synthetic": row0.get("n_synthetic"),
                "n_individual": row0.get("n_individual"),
                "n_real_pools": row0.get("n_real_pools"),
            })
        except Exception as exc:
            rows.append({"path": str(path), "error": str(exc)})

    df = pd.DataFrame(rows)
    if not df.empty and "roc_auc" in df.columns:
        df = df.sort_values(["roc_auc", "path"], ascending=[False, True], na_position="last")

    xlsx = outdir / "Synthetic_Result_Inventory.xlsx"
    csv = outdir / "Synthetic_Result_Inventory.csv"
    df.to_excel(xlsx, index=False)
    df.to_csv(csv, index=False)

    print("[DONE]", xlsx)
    print("[DONE]", csv)
    print("[INFO] files inventoried:", len(df))


if __name__ == "__main__":
    main()
