from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

SOURCE_URLS = [
    (
        "springer_supplementary_file1",
        "https://media.springernature.com/original/springer-static/esm/"
        "art%3A10.1007%2Fs11104-024-07000-2/MediaObjects/"
        "11104_2024_7000_MOESM1_ESM.xlsx",
    ),
]

USER_AGENT = "genomic-ricci-curvature-external-validation/1.0"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_header(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in fields})
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path, refresh: bool = False) -> None:
    if path.exists() and path.stat().st_size > 5000 and not refresh:
        return
    ensure_dir(path.parent)
    last_error: Exception | None = None
    for attempt in range(1, 7):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                data = response.read()
            if len(data) < 5000:
                raise RuntimeError(f"Downloaded file is unexpectedly small: {len(data)} bytes")
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
            return
        except Exception as exc:
            last_error = exc
            if attempt < 6:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Unable to download authoritative accession workbook: {last_error}")


def row_score(values: Sequence[Any]) -> int:
    headers = [normalize_header(v) for v in values]
    score = 0
    if any(any(term in h for term in ("accession", "genotype", "entry", "germplasm")) for h in headers):
        score += 5
    if any(any(term in h for term in ("origin", "location", "region", "zone", "country", "population", "group")) for h in headers):
        score += 5
    if any(any(term in h for term in ("cultivar", "variety", "material", "status", "type")) for h in headers):
        score += 2
    score += min(4, sum(bool(h) for h in headers) // 3)
    return score


def parse_workbook(path: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    try:
        import openpyxl
    except Exception as exc:
        raise RuntimeError("openpyxl is required in gwas_env") from exc

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    all_records: List[Dict[str, str]] = []
    inventory: List[Dict[str, str]] = []

    for sheet in workbook.worksheets:
        raw_rows: List[List[str]] = []
        for row in sheet.iter_rows(values_only=True):
            values = ["" if value is None else str(value).strip() for value in row]
            if any(values):
                raw_rows.append(values)
        if not raw_rows:
            continue

        best_index = None
        best_score = -1
        for idx, row in enumerate(raw_rows[:40]):
            score = row_score(row)
            if score > best_score:
                best_score = score
                best_index = idx

        inventory.append(
            {
                "sheet": sheet.title,
                "nonempty_rows": str(len(raw_rows)),
                "candidate_header_row_1based": "" if best_index is None else str(best_index + 1),
                "header_score": str(best_score),
            }
        )

        if best_index is None or best_score < 8:
            continue

        headers_raw = raw_rows[best_index]
        headers: List[str] = []
        seen: Dict[str, int] = {}
        for col_idx, value in enumerate(headers_raw, start=1):
            header = normalize_header(value) or f"column_{col_idx}"
            seen[header] = seen.get(header, 0) + 1
            if seen[header] > 1:
                header = f"{header}_{seen[header]}"
            headers.append(header)

        empty_streak = 0
        for row_number, row in enumerate(raw_rows[best_index + 1 :], start=best_index + 2):
            padded = list(row) + [""] * max(0, len(headers) - len(row))
            values = padded[: len(headers)]
            if not any(v.strip() for v in values):
                empty_streak += 1
                if empty_streak >= 5:
                    break
                continue
            empty_streak = 0
            record = {headers[i]: values[i].strip() for i in range(len(headers))}
            record["_source_name"] = "springer_supplementary_file1"
            record["_workbook"] = path.name
            record["_sheet"] = sheet.title
            record["_row_number"] = str(row_number)
            record["_row_text"] = " | ".join(v.strip() for v in values if v.strip())
            all_records.append(record)

    return all_records, inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    outdir = ensure_dir(Path(args.outdir))
    cache_dir = ensure_dir(Path(args.cache_dir))

    source_name, source_url = SOURCE_URLS[0]
    workbook_path = cache_dir / "11104_2024_7000_MOESM1_ESM.xlsx"

    print("[FETCH] Authoritative accession/location workbook")
    print(f"        {source_name}")
    download(source_url, workbook_path, refresh=args.refresh)

    records, inventory = parse_workbook(workbook_path)
    if len(records) < 250:
        raise RuntimeError(
            f"Only {len(records)} candidate accession rows were parsed; expected approximately 288."
        )

    output_workbook = outdir / workbook_path.name
    if not output_workbook.exists() or sha256_file(output_workbook) != sha256_file(workbook_path):
        shutil_copy(workbook_path, output_workbook)

    write_tsv(outdir / "authoritative_accession_records.tsv", records)
    write_tsv(outdir / "authoritative_workbook_inventory.tsv", inventory)

    manifest = {
        "status": "COMPLETE",
        "source_name": source_name,
        "source_url": source_url,
        "source_article_doi": "10.1007/s11104-024-07000-2",
        "relationship_to_gbs_project": (
            "The open-access source article reports the same 288 finger millet genotypes "
            "and the same GBS dataset originally described under PRJNA791522."
        ),
        "workbook": str(output_workbook),
        "workbook_bytes": output_workbook.stat().st_size,
        "workbook_sha256": sha256_file(output_workbook),
        "parsed_record_count": len(records),
        "sheet_inventory_count": len(inventory),
    }
    write_json(outdir / "authoritative_source_manifest.json", manifest)

    print("")
    print("[DONE] Authoritative accession metadata retrieved and parsed")
    print(f"  Workbook: {output_workbook}")
    print(f"  Parsed candidate rows: {len(records)}")


def shutil_copy(source: Path, destination: Path) -> None:
    ensure_dir(destination.parent)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as src, tmp.open("wb") as dst:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    os.replace(tmp, destination)


if __name__ == "__main__":
    main()
