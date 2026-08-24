from __future__ import annotations

import argparse
import csv
import io
import os
import re
import tarfile
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from common import (
    ensure_dir,
    fetch_bytes,
    fetch_to_file,
    flatten_mapping,
    normalize_text,
    normalize_token,
    read_tsv,
    safe_extract_tar,
    split_semicolon,
    write_json,
    write_tsv,
)


PRIMARY_FIELDS = [
    "run_accession",
    "study_accession",
    "sample_accession",
    "experiment_accession",
    "sample_alias",
    "sample_title",
    "study_title",
    "scientific_name",
    "library_name",
    "library_strategy",
    "library_source",
    "library_selection",
    "library_layout",
    "instrument_platform",
    "instrument_model",
    "read_count",
    "base_count",
    "fastq_ftp",
    "fastq_md5",
    "fastq_bytes",
    "submitted_ftp",
    "submitted_md5",
    "submitted_bytes",
]

FALLBACK_FIELDS = [
    "run_accession",
    "study_accession",
    "sample_accession",
    "experiment_accession",
    "scientific_name",
    "library_strategy",
    "library_source",
    "library_selection",
    "library_layout",
    "instrument_platform",
    "instrument_model",
    "read_count",
    "base_count",
    "fastq_ftp",
    "fastq_md5",
    "fastq_bytes",
]


def fetch_ena_run_report(bioproject: str, output_path: Path, refresh: bool) -> List[Dict[str, str]]:
    if output_path.exists() and output_path.stat().st_size > 100 and not refresh:
        rows = read_tsv(output_path)
        if rows:
            return rows

    base_url = "https://www.ebi.ac.uk/ena/portal/api/filereport"
    last_error: Optional[Exception] = None
    for fields in (PRIMARY_FIELDS, FALLBACK_FIELDS):
        query = urllib.parse.urlencode(
            {
                "accession": bioproject,
                "result": "read_run",
                "fields": ",".join(fields),
                "format": "tsv",
                "download": "false",
            }
        )
        url = base_url + "?" + query
        try:
            payload = fetch_bytes(url, retries=8, timeout=180)
            text = payload.decode("utf-8-sig", errors="replace")
            if text.lower().startswith("error") or "\n" not in text:
                raise RuntimeError(text[:500])
            rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
            if not rows:
                raise RuntimeError("ENA run report returned no data rows")
            output_path.write_text(text, encoding="utf-8")
            return rows
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to retrieve ENA run report for {bioproject}: {last_error}")


def parse_ena_sample_xml(xml_path: Path) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    root = ET.parse(xml_path).getroot()
    sample = root.find(".//SAMPLE")
    if sample is None:
        sample = root
    metadata: Dict[str, str] = {}
    metadata["sample_accession"] = sample.attrib.get("accession", "")
    metadata["sample_alias_xml"] = sample.attrib.get("alias", "")

    title = sample.findtext("./TITLE") or sample.findtext(".//TITLE") or ""
    description = sample.findtext("./DESCRIPTION") or sample.findtext(".//DESCRIPTION") or ""
    taxon_id = sample.findtext(".//SAMPLE_NAME/TAXON_ID") or ""
    scientific_name = sample.findtext(".//SAMPLE_NAME/SCIENTIFIC_NAME") or ""
    common_name = sample.findtext(".//SAMPLE_NAME/COMMON_NAME") or ""
    metadata.update(
        {
            "sample_title_xml": title,
            "sample_description_xml": description,
            "taxon_id_xml": taxon_id,
            "scientific_name_xml": scientific_name,
            "common_name_xml": common_name,
        }
    )

    long_rows: List[Dict[str, str]] = []
    for attr in sample.findall(".//SAMPLE_ATTRIBUTE"):
        tag = (attr.findtext("./TAG") or "").strip()
        value = (attr.findtext("./VALUE") or "").strip()
        units = (attr.findtext("./UNITS") or "").strip()
        if not tag:
            continue
        normalized_tag = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
        if normalized_tag:
            existing = metadata.get(normalized_tag, "")
            metadata[normalized_tag] = value if not existing else f"{existing}; {value}"
        long_rows.append(
            {
                "sample_accession": metadata["sample_accession"],
                "tag": tag,
                "normalized_tag": normalized_tag,
                "value": value,
                "units": units,
            }
        )
    return metadata, long_rows


def fetch_sample_xmls(
    sample_accessions: Sequence[str],
    xml_dir: Path,
    refresh: bool,
    workers: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    ensure_dir(xml_dir)

    def worker(accession: str) -> Tuple[str, Optional[Dict[str, str]], List[Dict[str, str]], str]:
        path = xml_dir / f"{accession}.xml"
        try:
            if refresh or not path.exists() or path.stat().st_size < 100:
                url = f"https://www.ebi.ac.uk/ena/browser/api/xml/{urllib.parse.quote(accession)}"
                fetch_to_file(url, path, force=refresh, min_bytes=100)
            metadata, long_rows = parse_ena_sample_xml(path)
            return accession, metadata, long_rows, ""
        except Exception as exc:
            return accession, None, [], str(exc)

    wide: List[Dict[str, str]] = []
    long_rows: List[Dict[str, str]] = []
    failures: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(worker, acc): acc for acc in sample_accessions}
        completed = 0
        total = len(futures)
        for future in as_completed(futures):
            accession, metadata, attrs, error = future.result()
            completed += 1
            if metadata is not None:
                wide.append(metadata)
                long_rows.extend(attrs)
            else:
                failures.append({"sample_accession": accession, "error": error})
            if completed == 1 or completed % 25 == 0 or completed == total:
                print(f"[SAMPLE XML] {completed}/{total}")
    return wide, long_rows, failures


def retrieve_pmc_package(pmcid: str, cache_dir: Path, extract_dir: Path, refresh: bool) -> List[Path]:
    ensure_dir(cache_dir)
    ensure_dir(extract_dir)
    marker = extract_dir / ".complete"
    if marker.exists() and not refresh:
        return [p for p in extract_dir.rglob("*") if p.is_file() and p.name != ".complete"]

    oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={urllib.parse.quote(pmcid)}"
    payload = fetch_bytes(oa_url, retries=6, timeout=120)
    root = ET.fromstring(payload)
    link = None
    for candidate in root.findall(".//link"):
        fmt = candidate.attrib.get("format", "").lower()
        href = candidate.attrib.get("href", "")
        if fmt in {"tgz", "tar.gz"} or href.endswith((".tar.gz", ".tgz")):
            link = href
            break
    if not link:
        raise RuntimeError(f"PMC OA package link was not returned for {pmcid}")
    if link.startswith("ftp://"):
        link = "https://" + link[6:]
    tar_path = cache_dir / f"{pmcid}.tar.gz"
    fetch_to_file(link, tar_path, force=refresh, min_bytes=1000)
    if extract_dir.exists():
        for child in extract_dir.iterdir():
            if child.name != ".complete":
                if child.is_dir():
                    import shutil
                    shutil.rmtree(child)
                else:
                    child.unlink()
    safe_extract_tar(tar_path, extract_dir)
    marker.write_text("complete\n", encoding="utf-8")
    return [p for p in extract_dir.rglob("*") if p.is_file() and p.name != ".complete"]


def iter_xlsx_tables(path: Path) -> Iterable[Tuple[str, List[List[str]]]]:
    try:
        import openpyxl
    except Exception:
        return
    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception:
        return
    for sheet in workbook.worksheets:
        rows: List[List[str]] = []
        for row in sheet.iter_rows(values_only=True):
            values = ["" if value is None else str(value).strip() for value in row]
            if any(values):
                rows.append(values)
            if len(rows) >= 1000:
                break
        if rows:
            yield f"{path.name}::{sheet.title}", rows


def iter_delimited_tables(path: Path) -> Iterable[Tuple[str, List[List[str]]]]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    first = raw[:4096]
    delimiter = "\t" if first.count("\t") >= first.count(",") else ","
    rows = list(csv.reader(io.StringIO(raw), delimiter=delimiter))
    rows = [[cell.strip() for cell in row] for row in rows if any(cell.strip() for cell in row)]
    if rows:
        yield path.name, rows[:1000]


def iter_docx_tables(path: Path) -> Iterable[Tuple[str, List[List[str]]]]:
    try:
        with zipfile.ZipFile(path) as archive:
            document = ET.fromstring(archive.read("word/document.xml"))
    except Exception:
        return
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    table_index = 0
    for table in document.findall(".//w:tbl", ns):
        table_index += 1
        rows: List[List[str]] = []
        for tr in table.findall("./w:tr", ns):
            cells: List[str] = []
            for tc in tr.findall("./w:tc", ns):
                text = " ".join(t.text or "" for t in tc.findall(".//w:t", ns)).strip()
                cells.append(text)
            if any(cells):
                rows.append(cells)
        if rows:
            yield f"{path.name}::table{table_index}", rows


def detect_table_records(source: str, rows: List[List[str]]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    header_idx: Optional[int] = None
    header: List[str] = []
    accession_tokens = ("ACCESSION", "GENOTYPE", "SAMPLE", "ENTRY", "LINE", "EBI NUMBER")
    context_tokens = ("POP", "GROUP", "REGION", "LOCATION", "ORIGIN", "COUNTRY", "DISTRICT", "ZONE", "CULTIVAR")

    for idx, row in enumerate(rows[:30]):
        norm = [normalize_text(cell) for cell in row]
        has_accession = any(any(token in cell for token in accession_tokens) for cell in norm)
        has_context = any(any(token in cell for token in context_tokens) for cell in norm)
        if has_accession and has_context:
            header_idx = idx
            header = [cell.strip() or f"column_{i+1}" for i, cell in enumerate(row)]
            break
    if header_idx is None:
        return records

    normalized_headers = [re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_") for h in header]
    for row_number, row in enumerate(rows[header_idx + 1 :], start=header_idx + 2):
        padded = list(row) + [""] * max(0, len(header) - len(row))
        record = {normalized_headers[i]: padded[i].strip() for i in range(len(header))}
        record["_source"] = source
        record["_row_number"] = str(row_number)
        record["_row_text"] = " | ".join(cell.strip() for cell in padded if cell.strip())
        if record["_row_text"]:
            records.append(record)
    return records


def extract_supplement_records(files: Sequence[Path], output_inventory: Path, output_records: Path) -> List[Dict[str, str]]:
    inventory: List[Dict[str, str]] = []
    records: List[Dict[str, str]] = []
    for path in sorted(files):
        suffix = path.suffix.lower()
        inventory.append(
            {
                "file": str(path),
                "name": path.name,
                "suffix": suffix,
                "bytes": str(path.stat().st_size),
            }
        )
        table_iter: Iterable[Tuple[str, List[List[str]]]] = []
        if suffix == ".xlsx":
            table_iter = iter_xlsx_tables(path)
        elif suffix in {".csv", ".tsv", ".txt"}:
            table_iter = iter_delimited_tables(path)
        elif suffix == ".docx":
            table_iter = iter_docx_tables(path)
        for source, table_rows in table_iter:
            records.extend(detect_table_records(source, table_rows))
    write_tsv(output_inventory, inventory)
    write_tsv(output_records, records)
    return records


def consolidate_runs_and_samples(
    run_rows: List[Dict[str, str]],
    sample_wide: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    sample_index = {row.get("sample_accession", ""): row for row in sample_wide}
    consolidated_runs: List[Dict[str, str]] = []
    sample_runs: Dict[str, List[Dict[str, str]]] = {}
    for run in run_rows:
        sample_accession = run.get("sample_accession", "")
        sample_runs.setdefault(sample_accession, []).append(run)
        combined = dict(run)
        for key, value in sample_index.get(sample_accession, {}).items():
            if key not in combined or not combined.get(key):
                combined[key] = value
            else:
                combined[f"xml_{key}"] = value
        combined["metadata_text"] = flatten_mapping(combined)
        consolidated_runs.append(combined)

    consolidated_samples: List[Dict[str, str]] = []
    for sample_accession, runs in sorted(sample_runs.items()):
        combined: Dict[str, str] = dict(sample_index.get(sample_accession, {}))
        combined["sample_accession"] = sample_accession
        combined["run_accessions"] = ";".join(sorted(r.get("run_accession", "") for r in runs))
        combined["run_count"] = str(len(runs))
        for key in (
            "sample_alias",
            "sample_title",
            "scientific_name",
            "library_name",
            "library_strategy",
            "library_source",
            "library_layout",
        ):
            values = sorted({r.get(key, "") for r in runs if r.get(key, "")})
            combined[key] = ";".join(values)
        combined["metadata_text"] = flatten_mapping(combined)
        consolidated_samples.append(combined)
    return consolidated_runs, consolidated_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bioproject", default="PRJNA791522")
    parser.add_argument("--pmcid", default="PMC9090224")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    outdir = ensure_dir(Path(args.outdir))
    raw_dir = ensure_dir(outdir / "raw")
    cache_dir = ensure_dir(Path(args.cache_dir))

    run_report_path = raw_dir / "ena_run_report_raw.tsv"
    print(f"[FETCH] ENA run report: {args.bioproject}")
    run_rows = fetch_ena_run_report(args.bioproject, run_report_path, args.refresh)
    sample_accessions = sorted({row.get("sample_accession", "") for row in run_rows if row.get("sample_accession", "")})
    print(f"[INFO] Runs: {len(run_rows)}; unique samples: {len(sample_accessions)}")

    print("[FETCH] ENA sample XML metadata")
    sample_wide, sample_long, sample_failures = fetch_sample_xmls(
        sample_accessions,
        raw_dir / "sample_xml",
        args.refresh,
        args.workers,
    )
    write_tsv(outdir / "ena_sample_attributes_long.tsv", sample_long)
    write_tsv(outdir / "ena_sample_metadata_wide.tsv", sample_wide)
    write_tsv(outdir / "ena_sample_xml_failures.tsv", sample_failures)

    supplement_files: List[Path] = []
    supplement_error = ""
    try:
        print(f"[FETCH] PMC OA package: {args.pmcid}")
        supplement_files = retrieve_pmc_package(
            args.pmcid,
            cache_dir / "pmc",
            raw_dir / "pmc_oa_package",
            args.refresh,
        )
    except Exception as exc:
        supplement_error = str(exc)
        print(f"[WARN] PMC supplementary package unavailable: {exc}")

    supplement_records = extract_supplement_records(
        supplement_files,
        outdir / "supplement_file_inventory.tsv",
        outdir / "supplement_candidate_records.tsv",
    )

    consolidated_runs, consolidated_samples = consolidate_runs_and_samples(run_rows, sample_wide)
    write_tsv(outdir / "run_metadata_consolidated.tsv", consolidated_runs)
    write_tsv(outdir / "sample_metadata_consolidated.tsv", consolidated_samples)

    report = {
        "bioproject": args.bioproject,
        "pmcid": args.pmcid,
        "run_count": len(run_rows),
        "sample_count": len(sample_accessions),
        "sample_xml_success": len(sample_wide),
        "sample_xml_failures": len(sample_failures),
        "supplement_files": len(supplement_files),
        "supplement_candidate_records": len(supplement_records),
        "supplement_error": supplement_error,
        "status": "COMPLETE" if run_rows and len(sample_wide) == len(sample_accessions) else "PARTIAL",
    }
    write_json(outdir / "metadata_fetch_summary.json", report)
    print("")
    print("[DONE] Metadata retrieval complete")
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
