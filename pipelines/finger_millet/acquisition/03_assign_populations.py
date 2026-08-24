from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

EXPECTED_COUNTS = {
    "Pop-1": 105,
    "Pop-2": 31,
    "Pop-3": 55,
    "Pop-4": 28,
    "Pop-5": 9,
    "Pop-6": 14,
    "Pop-7": 46,
}

SAMPLE_ID_FIELDS = (
    "sample_accession",
    "sample_alias",
    "sample_alias_xml",
    "sample_title",
    "sample_title_xml",
    "library_name",
    "source_name",
    "isolate",
)

ACCESSION_FIELD_TERMS = (
    "accession",
    "genotype",
    "entry",
    "germplasm",
    "sample",
    "cultivar",
    "variety",
    "line",
    "name",
    "code",
    "id",
)

CONTEXT_FIELD_TERMS = (
    "population",
    "group",
    "origin",
    "location",
    "region",
    "zone",
    "district",
    "country",
    "material",
    "cultivar",
    "variety",
    "status",
    "type",
)


def normalize_token(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"[^A-Z0-9]+", "", text.upper())


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.upper().replace("–", "-").replace("—", "-")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def candidate_id_fields(row: Mapping[str, str]) -> List[str]:
    result: List[str] = []
    for field in row:
        if field.startswith("_"):
            continue
        lower = field.lower()
        if any(term in lower for term in ACCESSION_FIELD_TERMS):
            result.append(field)
    return result


def accession_keys_from_value(value: str, *, full_weight: int = 3) -> Dict[str, int]:
    keys: Dict[str, int] = {}
    text = normalize_text(value)
    token = normalize_token(value)

    if 4 <= len(token) <= 40 and token not in {
        "UNKNOWN", "NONE", "NA", "ETHIOPIA", "ZIMBABWE", "LANDRACE", "CULTIVAR"
    }:
        keys[f"FULL:{token}"] = full_weight

    for prefix, digits in re.findall(r"\b(EC|ECA|EBI|FM)[\s_-]*([0-9]{4,10})\b", text):
        canonical = f"EC{digits.lstrip('0') or '0'}"
        keys[f"ACC:{canonical}"] = max(keys.get(f"ACC:{canonical}", 0), 5)

    for digits in re.findall(r"\b([0-9]{5,10})\b", text):
        canonical = digits.lstrip("0") or "0"
        # Exclude likely years.
        if len(canonical) == 4 and 1900 <= int(canonical) <= 2100:
            continue
        keys[f"NUM:{canonical}"] = max(keys.get(f"NUM:{canonical}", 0), 2)

    # Retain cultivar/variety names that are not generic labels.
    if 4 <= len(token) <= 30 and not token.isdigit():
        generic = {
            "ACCESSION", "GENOTYPE", "ENTRY", "SAMPLE", "CULTIVAR", "VARIETY",
            "LANDRACE", "ETHIOPIA", "ZIMBABWE", "UNKNOWNLOCATION", "UNKNOWNORIGIN",
        }
        if token not in generic:
            keys[f"NAME:{token}"] = max(keys.get(f"NAME:{token}", 0), 3)

    return keys


def sample_keys(sample: Mapping[str, str]) -> Dict[str, int]:
    keys: Dict[str, int] = {}
    for field in SAMPLE_ID_FIELDS:
        value = sample.get(field, "")
        for key, weight in accession_keys_from_value(value, full_weight=4).items():
            keys[key] = max(keys.get(key, 0), weight)

    # Extract accession-like patterns from the full metadata without using the
    # entire metadata string as a direct name key.
    metadata_text = sample.get("metadata_text", "")
    for prefix, digits in re.findall(
        r"\b(EC|ECA|EBI|FM)[\s_-]*([0-9]{4,10})\b",
        normalize_text(metadata_text),
    ):
        canonical = f"EC{digits.lstrip('0') or '0'}"
        keys[f"ACC:{canonical}"] = max(keys.get(f"ACC:{canonical}", 0), 5)
    for digits in re.findall(r"\b([0-9]{5,10})\b", normalize_text(metadata_text)):
        canonical = digits.lstrip("0") or "0"
        keys[f"NUM:{canonical}"] = max(keys.get(f"NUM:{canonical}", 0), 1)
    return keys


def record_keys(record: Mapping[str, str]) -> Dict[str, int]:
    keys: Dict[str, int] = {}
    for field in candidate_id_fields(record):
        for key, weight in accession_keys_from_value(record.get(field, ""), full_weight=4).items():
            keys[key] = max(keys.get(key, 0), weight)
    return keys


def explicit_population(text: str) -> Optional[str]:
    values = sorted(
        {
            int(x)
            for x in re.findall(
                r"\bPOP(?:ULATION)?\s*[-_]?\s*([1-7])\b",
                normalize_text(text),
            )
        }
    )
    if len(values) == 1:
        return f"Pop-{values[0]}"
    return None


def classify_population(record: Mapping[str, str]) -> Tuple[Optional[str], str]:
    # Prefer an explicit population column.
    for field, value in record.items():
        lower = field.lower()
        if field.startswith("_"):
            continue
        if "population" in lower or re.fullmatch(r"pop(?:ulation)?_?group", lower):
            pop = explicit_population(value)
            if pop:
                return pop, f"explicit:{field}"

    context_parts = []
    for field, value in record.items():
        if field.startswith("_"):
            continue
        lower = field.lower()
        if any(term in lower for term in CONTEXT_FIELD_TERMS):
            context_parts.append(value)
    context_parts.append(record.get("_row_text", ""))
    text = " " + normalize_text(" | ".join(context_parts)) + " "

    pop = explicit_population(text)
    if pop:
        return pop, "explicit_population_in_context"

    if " ZIMBABWE " in text or " ZIMBABWEAN " in text:
        return "Pop-7", "country_zimbabwe"

    improved_terms = (
        " IMPROVED CULTIVAR ",
        " IMPROVED CULTIVARS ",
        " IMPROVED VARIETY ",
        " IMPROVED VARIETIES ",
        " RELEASED CULTIVAR ",
        " RELEASED VARIETY ",
        " CULTIVAR IMPROVED ",
        " IMPROVED ",
    )
    if any(term in text for term in improved_terms):
        return "Pop-6", "improved_cultivar"

    unknown_terms = (
        " UNKNOWN LOCATION ",
        " UNKNOWN ORIGIN ",
        " LOCATION UNKNOWN ",
        " ORIGIN UNKNOWN ",
        " UNSPECIFIED ",
        " NOT KNOWN ",
    )
    if any(term in text for term in unknown_terms):
        return "Pop-5", "unknown_sampling_location"

    groups = {
        "Pop-1": (
            " AGEW AWI ", " AWEG AWI ", " AWI ZONE ", " GOJAM ", " GOJJAM ",
            " BAHIR DAR ", " BAHR DAR ", " BAHRDAR ", " METEKEL ",
        ),
        "Pop-2": (
            " WESTERN TIGRAY ", " WEST TIGRAY ", " WESTERN TIGRAI ",
            " GONDER ", " GONDAR ",
        ),
        "Pop-3": (
            " WELLEGA ", " WOLLEGA ", " WELEGA ", " ILLUABABORA ",
            " ILLU ABABORA ", " ILLUBABOR ", " ILUBABOR ", " ILLUABABOR ",
        ),
        "Pop-4": (
            " CENTRAL TIGRAY ", " CENTRAL TIGRAI ", " EASTERN TIGRAY ",
            " EAST TIGRAY ", " SOUTHERN TIGRAY ", " SOUTH TIGRAY ",
            " NORTHERN WELLO ", " NORTH WELLO ", " NORTHERN WOLLO ",
            " NORTH WOLLO ",
        ),
    }
    matched: List[str] = []
    for population, terms in groups.items():
        if any(term in text for term in terms):
            matched.append(population)
    matched = sorted(set(matched))
    if len(matched) == 1:
        return matched[0], "published_geographic_group"
    if len(matched) > 1:
        return None, "ambiguous_geographic_groups:" + ",".join(matched)
    return None, "unresolved_population_context"


def load_overrides(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    rows = read_tsv(path)
    result: Dict[str, Dict[str, str]] = {}
    for row in rows:
        population = row.get("population", "").strip()
        if population not in EXPECTED_COUNTS:
            continue
        for field in ("sample_accession", "accession_name", "sample_alias"):
            token = normalize_token(row.get(field, ""))
            if token:
                result[token] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_metadata", required=True)
    parser.add_argument("--authoritative_records", required=True)
    parser.add_argument("--override_tsv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--expected_samples", type=int, default=288)
    args = parser.parse_args()

    samples = read_tsv(Path(args.sample_metadata))
    records = read_tsv(Path(args.authoritative_records))
    overrides = load_overrides(Path(args.override_tsv))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    record_key_maps = [record_keys(record) for record in records]
    key_to_records: Dict[str, List[int]] = defaultdict(list)
    for idx, key_map in enumerate(record_key_maps):
        for key in key_map:
            key_to_records[key].append(idx)

    assignments: List[Dict[str, str]] = []
    unresolved: List[Dict[str, str]] = []
    match_audit: List[Dict[str, str]] = []
    used_records: Counter[int] = Counter()

    for sample in samples:
        sample_accession = sample.get("sample_accession", "")
        skeys = sample_keys(sample)

        override = None
        for token in (
            normalize_token(sample_accession),
            normalize_token(sample.get("sample_alias", "")),
            normalize_token(sample.get("sample_alias_xml", "")),
        ):
            if token and token in overrides:
                override = overrides[token]
                break

        candidate_scores: Dict[int, int] = defaultdict(int)
        candidate_evidence: Dict[int, List[str]] = defaultdict(list)
        for key, sample_weight in skeys.items():
            for record_idx in key_to_records.get(key, []):
                record_weight = record_key_maps[record_idx].get(key, 0)
                score = sample_weight + record_weight
                candidate_scores[record_idx] += score
                candidate_evidence[record_idx].append(f"{key}:{score}")

        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: (-item[1], records[item[0]].get("_row_number", ""), item[0]),
        )

        matched_idx: Optional[int] = None
        match_status = ""
        if override is not None:
            override_population = override.get("population", "").strip()
            override_accession_name = override.get("accession_name", "").strip()
            override_token = normalize_token(override_accession_name)
            override_matches: List[int] = []
            if override_token:
                for record_idx, record in enumerate(records):
                    for field in candidate_id_fields(record):
                        if normalize_token(record.get(field, "")) == override_token:
                            override_matches.append(record_idx)
                            break
            override_matches = sorted(set(override_matches))

            if len(override_matches) != 1:
                population = None
                assignment_source = ""
                assignment_reason = (
                    f"override_authoritative_match_count={len(override_matches)};"
                    f"accession_name={override_accession_name}"
                )
                matched_record = {}
                match_status = "override_authoritative_match_failed"
            else:
                matched_idx = override_matches[0]
                matched_record = records[matched_idx]
                record_population, record_reason = classify_population(matched_record)
                if record_population != override_population:
                    population = None
                    assignment_source = ""
                    assignment_reason = (
                        f"override_population={override_population};"
                        f"authoritative_population={record_population};"
                        f"authoritative_reason={record_reason}"
                    )
                    match_status = "override_population_conflict"
                else:
                    population = override_population
                    assignment_source = "curated_archive_to_authoritative_crosswalk"
                    assignment_reason = override.get("note", "")
                    match_status = "manual_crosswalk_with_authoritative_record"
        else:
            if ranked:
                best_score = ranked[0][1]
                best = [idx for idx, score in ranked if score == best_score]
                if len(best) == 1:
                    matched_idx = best[0]
                    matched_record = records[matched_idx]
                    population, assignment_reason = classify_population(matched_record)
                    assignment_source = "authoritative_accession_workbook"
                    match_status = "unique_best_match"
                else:
                    matched_record = {}
                    population = None
                    assignment_source = ""
                    assignment_reason = "tied_best_accession_matches"
                    match_status = "ambiguous_match"
            else:
                matched_record = {}
                population = None
                assignment_source = ""
                assignment_reason = "no_accession_match"
                match_status = "no_match"

        if matched_idx is not None:
            used_records[matched_idx] += 1

        row = {
            "sample_accession": sample_accession,
            "sample_alias": sample.get("sample_alias", "") or sample.get("sample_alias_xml", ""),
            "sample_title": sample.get("sample_title", "") or sample.get("sample_title_xml", ""),
            "run_accessions": sample.get("run_accessions", ""),
            "run_count": sample.get("run_count", ""),
            "population": population or "",
            "assignment_source": assignment_source,
            "assignment_reason": assignment_reason,
            "match_status": match_status,
            "match_score": "" if matched_idx is None else str(candidate_scores[matched_idx]),
            "matched_record_index": "" if matched_idx is None else str(matched_idx + 1),
            "matched_workbook_sheet": matched_record.get("_sheet", ""),
            "matched_workbook_row": matched_record.get("_row_number", ""),
            "matched_record_text": matched_record.get("_row_text", ""),
            "match_evidence": "" if matched_idx is None else ";".join(sorted(candidate_evidence[matched_idx])),
            "override_accession_name": "" if override is None else override.get("accession_name", ""),
            "override_note": "" if override is None else override.get("note", ""),
        }
        assignments.append(row)
        match_audit.append(
            {
                "sample_accession": sample_accession,
                "sample_keys": ";".join(f"{k}:{v}" for k, v in sorted(skeys.items())),
                "candidate_count": str(len(candidate_scores)),
                "top_candidates": ";".join(
                    f"{idx+1}:{score}" for idx, score in ranked[:5]
                ),
                "final_match_status": match_status,
                "final_population": population or "",
            }
        )
        if not population:
            unresolved.append(row)

    counts = Counter(row["population"] for row in assignments if row["population"])
    duplicate_samples = len(assignments) - len({row["sample_accession"] for row in assignments})
    duplicate_record_use = {
        str(idx + 1): count for idx, count in used_records.items() if count > 1
    }
    all_counts_match = all(counts.get(pop, 0) == expected for pop, expected in EXPECTED_COUNTS.items())

    status = (
        "PASS"
        if len(assignments) == args.expected_samples
        and duplicate_samples == 0
        and not unresolved
        and not duplicate_record_use
        and all_counts_match
        else "FAIL"
    )

    write_tsv(outdir / "population_assignment.tsv", assignments)
    write_tsv(outdir / "authoritative_match_audit.tsv", match_audit)
    write_tsv(outdir / "unresolved_population_assignments.tsv", unresolved)
    write_tsv(
        outdir / "population_counts.tsv",
        [
            {
                "population": pop,
                "observed": str(counts.get(pop, 0)),
                "expected": str(expected),
                "match": str(counts.get(pop, 0) == expected),
            }
            for pop, expected in EXPECTED_COUNTS.items()
        ],
    )

    report = {
        "status": status,
        "sample_count": len(assignments),
        "expected_sample_count": args.expected_samples,
        "unresolved_count": len(unresolved),
        "duplicate_sample_accessions": duplicate_samples,
        "duplicate_authoritative_record_use": duplicate_record_use,
        "authoritative_record_count": len(records),
        "population_counts_observed": {pop: counts.get(pop, 0) for pop in EXPECTED_COUNTS},
        "population_counts_expected": EXPECTED_COUNTS,
        "all_population_counts_match": all_counts_match,
    }
    write_json(outdir / "metadata_audit.json", report)

    lines = [
        "Finger millet authoritative metadata audit",
        "==========================================",
        "",
        f"Status: {status}",
        f"Samples: {len(assignments)} (expected {args.expected_samples})",
        f"Authoritative workbook records: {len(records)}",
        f"Unresolved assignments: {len(unresolved)}",
        f"Duplicate workbook-record use: {len(duplicate_record_use)}",
        "",
        "Population counts:",
    ]
    for pop, expected in EXPECTED_COUNTS.items():
        lines.append(f"  {pop}: {counts.get(pop, 0)} (expected {expected})")
    lines.extend(
        [
            "",
            "Population assignment used archived sample identifiers linked to an",
            "open-access accession/location workbook for the same 288-genotype panel.",
            "No read depth, FASTQ size, TMS, PCA, mapping score, or downstream result",
            "was used to assign populations.",
        ]
    )
    (outdir / "metadata_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    marker = outdir / "METADATA_AUDIT_PASS.txt"
    if status == "PASS":
        marker.write_text("PASS\n", encoding="utf-8")
    else:
        if marker.exists():
            marker.unlink()
        print("")
        print("[STOP] Exact one-to-one population reconciliation was not achieved.")
        print(f"Review: {outdir / 'unresolved_population_assignments.tsv'}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
