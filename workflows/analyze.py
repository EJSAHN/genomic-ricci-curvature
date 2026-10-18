"""Explicit-path execution of the corrected coffee analyses.

Raw execution is opt-in. Existing analysis folders are read-only inputs. Completed
stages are reused only when their input identities and output checksums agree.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SOURCES, POOLS, RUNS, execute, load_module, read_rows, require, sha256, verify_metadata, write_json


def path_from_config(base: Path, value: str) -> Path:
    require(isinstance(value, str) and bool(value), "Missing configured path")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def write_tsv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(bool(records), "Cannot write an empty input table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def seal(folder: Path, identity: dict) -> None:
    files = {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob("*"))
             if p.is_file() and p.name != "COMPLETE.json" and "__pycache__" not in p.parts}
    require(files, "Cannot seal an empty analysis")
    write_json(folder / "COMPLETE.json", {"identity": identity, "files": files})


def reusable(folder: Path, identity: dict) -> bool:
    record = folder / "COMPLETE.json"
    if not record.exists():
        return False
    data = json.loads(record.read_text())
    require(data["identity"] == identity, f"Input configuration changed: {folder}")
    for name, checksum in data["files"].items():
        require((folder / name).is_file() and sha256(folder / name) == checksum, f"Completed output changed: {folder / name}")
    print(f"[REUSE] {folder.name}", flush=True)
    return True


def verify_raw(raw: Path) -> list[dict]:
    require(raw.is_dir(), f"Raw input folder not found: {raw}")
    expected = read_rows(ROOT / "metadata/coffee/raw_file_identity.tsv")
    found = []
    for row in expected:
        path = raw / row["filename"]
        require(path.is_file() and path.stat().st_size == int(row["bytes"]), f"Missing/wrong-size raw input: {path}")
        require(sha256(path) == row["sha256"], f"Raw input checksum differs: {path}")
        found.append({**row, "path": str(path)})
    return found


def verify_generated(folder: Path, target: Path, raw: Path) -> Path:
    """Relocate the recorded 365-library allocation and verify all paired files.

    Both historical scenarios are retained solely to reconstruct exactly the read
    exclusions used in marker discovery. Only the conservative 180 libraries are
    supplied to the corrected benchmark and comparator.
    """
    table = read_rows(ROOT / "examples/coffee/rare_input/generated_fastq_manifest.tsv")
    truth = read_rows(ROOT / "examples/coffee/rare_input/truth_manifest.tsv")
    truth_by_id = {r["sample_id"]: r for r in truth}
    source_summary = read_rows(ROOT / "examples/coffee/rare_input/source_sampling_summary.tsv")
    counts = {r["parent_id"]: int(r["read_pairs_available"]) for r in source_summary}
    used = {s: bytearray(n) for s, n in counts.items()}
    seen_counts = {s: 0 for s in counts}
    for i, row in enumerate(table, 1):
        name = row["sample_id"]
        expected_truth = truth_by_id[name]
        expected_sources = set(expected_truth["parents"].split(","))
        if row["scenario"] == "conservative":
            require(expected_sources <= set(SOURCES), "Pool-series source in corrected mixture")
        pair_files = []
        for mate in [1, 2]:
            path = folder / row["scenario"] / ("rep_%02d" % int(row["replicate"])) / f"{name}_{mate}.fastq.gz"
            require(path.is_file() and path.stat().st_size == int(row[f"r{mate}_size_bytes"]), f"Missing/wrong-size generated input: {path}")
            require(sha256(path) == row[f"r{mate}_sha256"], f"Generated checksum mismatch: {path}")
            row[f"r{mate}_path"] = str(path)
            pair_files.append(path)
        with gzip.open(pair_files[0], "rt", encoding="ascii") as a, gzip.open(pair_files[1], "rt", encoding="ascii") as b:
            observed = {s: 0 for s in expected_sources}
            for _ in range(int(row["read_pairs"])):
                r1 = [a.readline().rstrip("\r\n") for _ in range(4)]
                r2 = [b.readline().rstrip("\r\n") for _ in range(4)]
                require(r1[0].endswith("/1") and r2[0].endswith("/2") and r1[0][:-2] == r2[0][:-2], "Generated mate synchronization failed")
                for r in [r1, r2]:
                    require(r[2].startswith("+") and len(r[1]) == len(r[3]) > 0, "Malformed generated FASTQ record")
                match = re.fullmatch(r"@(.+)\|source=(SRR\d+)\|source_pair=(\d+)/1", r1[0])
                require(match is not None and match[1] == name, "Malformed source header")
                source, index = match[2], int(match[3])
                require(source in expected_sources and 0 <= index < counts[source], "Unexpected source/read index")
                require(not used[source][index], "Read reused across generated libraries")
                used[source][index] = 1
                seen_counts[source] += 1
                observed[source] += 1
            require(not a.readline() and not b.readline(), "Generated FASTQ has excess records")
            targets = dict(zip(expected_truth["parents"].split(","), map(int, expected_truth["target_counts"].split(","))))
            require(observed == targets, "Per-source allocation count differs")
        if i % 60 == 0:
            print(f"[INPUT] Paired-library verification {i}/365", flush=True)
    require(len(table) == 365 and sum(r["scenario"] == "conservative" for r in table) == 180, "Incomplete recorded allocation")
    for row in source_summary:
        s = row["parent_id"]
        require(seen_counts[s] == int(row["read_pairs_assigned"]), "Source read-assignment count differs")
        row["r1_path"] = str(raw / row["r1_path"])
        row["r2_path"] = str(raw / row["r2_path"])
    write_tsv(target / "generated_fastq_manifest.tsv", table)
    write_tsv(target / "source_sampling_summary.tsv", source_summary)
    write_tsv(target / "truth_manifest.tsv", truth)
    write_json(target / "PAIR_ALLOCATION_CHECK.json", {"generated_libraries": 365, "corrected_libraries": 180, "source_pair_reuse": 0, "per_source_pair_counts": seen_counts})
    return target / "generated_fastq_manifest.tsv"


def prepare_references(cfg: dict, base: Path, out: Path) -> Path:
    assets = read_rows(ROOT / "metadata/coffee/reference_asset_identity.tsv")
    executable = path_from_config(base, cfg["bowtie2_executable"])
    require(executable.is_file(), "Bowtie 2 executable is missing")
    import subprocess
    version = subprocess.run([str(executable), "--version"], capture_output=True, text=True, timeout=30, cwd=executable.parent)
    require(version.returncode == 0 and re.search(r"\b2\.5\.5\b", version.stdout + version.stderr), "Recorded analysis used Bowtie 2 2.5.5")
    launcher = executable
    if os.name == "nt":
        launcher = out / "bowtie2.cmd"
        wrapper = " --wrapper basic-0" if executable.name.lower().startswith("bowtie2-align") else ""
        launcher.write_text('@echo off\npushd "' + str(executable.parent) + '"\n"' + str(executable) + '"' + wrapper + ' %*\nset RC=%ERRORLEVEL%\npopd\nexit /b %RC%\n', newline="\r\n", encoding="utf-8")
    rows = []
    for row in read_rows(ROOT / "config/coffee/reference_based/reference_assemblies.tsv"):
        ref = row["reference_id"]
        setting = cfg["references"][ref]
        fasta = path_from_config(base, setting["fasta"])
        index = path_from_config(base, setting["index_prefix"])
        for asset in [r for r in assets if r["reference_id"] == ref]:
            path = fasta if asset["kind"] == "fasta" else Path(str(index) + asset["suffix"])
            require(path.is_file() and path.stat().st_size == int(asset["bytes"]), f"Reference/index missing: {path}")
            require(sha256(path) == asset["sha256"], f"Reference/index identity differs: {path}")
        rows.append({**row, "fasta_path": str(fasta), "fasta_size_bytes": fasta.stat().st_size, "fasta_sha256": sha256(fasta), "index_prefix": str(index), "align_launcher": str(launcher)})
    path = out / "reference_manifest.tsv"
    write_tsv(path, rows)
    return path


def main():
    from common import verify_design_contract
    verify_design_contract()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", choices=["calibration", "sensitivity", "restore-generated", "batch", "rare", "reference", "all"], required=True)
    parser.add_argument("--execute", action="store_true", help="Required for raw-read work; without this only the requested plan is printed")
    args = parser.parse_args()
    verify_metadata()
    path = args.config.resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    base = path.parent
    raw = path_from_config(base, cfg["raw_fastq_dir"])
    work = args.output.resolve()
    generated = path_from_config(base, cfg["generated_fastq_dir"]) if cfg.get("generated_fastq_dir") else work / "restored_generated/fastq"
    require(work != ROOT and not ROOT.is_relative_to(work), "Output cannot contain the repository")
    require(not work.is_relative_to(raw) and not raw.is_relative_to(work), "Output cannot overlap raw input")
    require(not cfg.get("generated_fastq_dir") or (not work.is_relative_to(generated) and not generated.is_relative_to(work)), "Output cannot overlap configured generated input")
    for relative in ["pipelines", "metadata", "config", "examples", "reporting", "tools", "tests", "workflows", "history"]:
        require(not work.is_relative_to(ROOT / relative), "Output overlaps distributed input")
    targets = ["calibration", "sensitivity", "batch", "rare", "reference"] if args.target == "all" else [args.target]
    plan = {"targets": targets, "raw_fastq_dir": str(raw), "generated_fastq_dir": str(generated), "output": str(work), "corrected_sources": SOURCES, "excluded_pool_series": POOLS, "raw_work_enabled": args.execute}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    work.mkdir(parents=True, exist_ok=True)
    identity = {"config": cfg, "raw_inputs": verify_raw(raw), "current_source_manifest_sha256": sha256(ROOT / "config/coffee/coffee_sample_manifest.tsv"), "code": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(list((ROOT / "pipelines/coffee").rglob("*.py")) + list((ROOT / "workflows").glob("*.py")))}}
    previous = work / "INPUT_IDENTITY.json"
    if previous.exists():
        require(json.loads(previous.read_text()) == identity, "Run inputs or code changed; use a new output directory")
    else:
        require(not any(work.iterdir()), "Nonempty output without a valid input identity")
        write_json(previous, identity)
    inputs = work / "inputs"
    inputs.mkdir(exist_ok=True)
    try:
        cal = ROOT / "pipelines/coffee/calibration"
        if "calibration" in targets:
            for historical, name in [(True, "calibration_R1_replay"), (False, "calibration_corrected")]:
                ids = POOLS[:-1] if historical else POOLS
                dest = work / name
                if not reusable(dest, identity):
                    execute(cal / "run_calibration.py", ["--fastq_dir", raw, "--outdir", dest, "--reads_per_sample", "6000", "--kmer", "17", "--sketch", "16384", "--n_synth", "40", "--knn", "4", "--alpha", ".5", "--seed", "42", "--known_pools", ",".join(ids)], work, name)
                    expected_dir = ROOT / "examples/coffee" / ("calibration_r1" if historical else "calibration")
                    for f in ["node_scores.csv", "synthetic_truth.csv", "js_distance.csv"]:
                        a, b = pd.read_csv(dest / f), pd.read_csv(expected_dir / f)
                        pd.testing.assert_frame_equal(a, b, check_exact=False, check_dtype=False, atol=2e-8, rtol=0)
                    seal(dest, identity)
        if "sensitivity" in targets:
            dest = work / "sensitivity_corrected"
            if not reusable(dest, identity):
                execute(cal / "run_sensitivity.py", ["--fastq_dir", raw, "--outdir", dest, "--calib_script", cal / "run_calibration.py", "--reads_list", "3000,6000,12000", "--kmer_list", "15,17,19,21", "--n_boot", "30", "--base_seed", "42", "--base_kmer", "17", "--base_reads", "6000", "--known_pools", ",".join(POOLS)], work, "sensitivity_corrected")
                from replay import metrics
                expected = read_rows(ROOT / "reporting/Sweep_Runs.tsv")
                for row in expected:
                    nodes = read_rows(dest / row["run_dir"] / "node_scores.csv", ",")
                    m = metrics([int(r["label"] == "synthetic_pool") for r in nodes], [r["mixture_score_ot"] for r in nodes])
                    require(abs(m["auc"] - float(row["roc_auc"])) < 2e-8 and abs(m["ap"] - float(row["avg_precision"])) < 2e-8, "Sensitivity result differs")
                seal(dest, identity)
        if "restore-generated" in targets or (any(t in targets for t in ["batch", "reference"]) and not generated.exists()):
            restore = work / "restored_generated"
            require(not cfg.get("generated_fastq_dir"), "Configured generated input is missing; do not replace it automatically")
            if not reusable(restore, identity):
                # Reconstruct the full historical reservoir layout before selecting the corrected 180.
                rd = ROOT / "pipelines/coffee/read_level"
                design = work / "historical_sampling_design"
                execute(rd / "01_design_read_level_mixtures.py", ["--sample_manifest", ROOT / "history/coffee_source_manifest_13.tsv", "--ratio_patterns", ROOT / "config/coffee/mixture_ratio_patterns.tsv", "--outdir", design, "--seed", "42000"], work, "restore_read_design")
                execute(rd / "02_build_read_level_fastqs.py", ["--sample_manifest", ROOT / "history/coffee_source_manifest_13.tsv", "--design_manifest", design / "design_manifest.tsv", "--read_allocations", design / "read_allocations.tsv", "--source_fastq_root", raw, "--outdir", restore, "--seed", "42000"], work, "restore_read_fastqs")
                verify_generated(restore / "fastq", inputs, raw)
                seal(restore, identity)
            generated = restore / "fastq"
        if any(t in targets for t in ["batch", "reference"]):
            generated_manifest = verify_generated(generated, inputs, raw)
        if "batch" in targets:
            # Preserve the historical allocation but fit only the corrected scenario.
            all_rows = read_rows(inputs / "generated_fastq_manifest.tsv")
            write_tsv(inputs / "corrected_generated_manifest.tsv", [r for r in all_rows if r["scenario"] == "conservative"])
            batch = work / "batch_corrected"
            if not reusable(batch, identity):
                execute(ROOT / "pipelines/coffee/read_level/03_analyze_read_level_validation.py", ["--generated_fastq_manifest", inputs / "corrected_generated_manifest.tsv", "--truth_manifest", inputs / "truth_manifest.tsv", "--outdir", batch, "--modes", "r1,paired", "--kmer", "17", "--sketch", "16384", "--knn", "4", "--alpha", ".5"], work, "batch_corrected")
                actual = pd.read_csv(batch / "node_scores_all.tsv", sep="\t")
                expected = pd.read_csv(ROOT / "reporting/C_Batch_Scores.tsv", sep="\t")
                expected = expected[expected.scenario == "conservative"]
                keys = ["sample_id", "analysis_mode"]
                actual = actual.sort_values(keys).reset_index(drop=True)
                expected = expected.sort_values(keys).reset_index(drop=True)
                require(len(actual) == 360 and actual[keys].equals(expected[keys]), "Batch identity/count mismatch")
                require(np.allclose(actual.tms, expected.tms, atol=2e-8, rtol=0), "Batch scores differ")
                seal(batch, identity)
        if "rare" in targets:
            from replay import rare_event
            dest = work / "rare_corrected"
            if not reusable(dest, identity):
                dest.mkdir(exist_ok=True)
                matrices = work / "batch_corrected" if (work / "batch_corrected/COMPLETE.json").exists() else ROOT / "examples/coffee/rare_input/analysis"
                rare_event(dest, matrices, ROOT / "examples/coffee/rare_input/truth_manifest.tsv", scenarios=("conservative",))
                seal(dest, identity)
        if "reference" in targets:
            refs = prepare_references(cfg, base, inputs)
            rb = ROOT / "pipelines/coffee/reference_based"
            support = load_module(rb / "reference_support.py", "reference_support")
            marker = work / "markers_corrected"
            if not reusable(marker, identity):
                execute(rb / "guarded_reference.py", ["markers", "--reference_manifest", refs, "--sample_manifest", ROOT / "config/coffee/coffee_sample_manifest.tsv", "--expected_source_count", "12", "--source_fastq_root", raw, "--generated_fastq_manifest", inputs / "generated_fastq_manifest.tsv", "--source_sampling_summary", inputs / "source_sampling_summary.tsv", "--outdir", marker, "--logs_dir", work / "logs/marker_alignments", "--source_pairs", "100000", "--reserve_real_skip_pairs", "100000", "--reserve_real_pairs", "50000", "--threads", "4", "--seed", "52001", "--min_mapq", "20", "--min_baseq", "20", "--min_alt_total", "4", "--min_alt_samples", "2", "--min_alt_in_one_sample", "4", "--max_markers", "50000"], work, "marker_discovery")
                audit = pd.read_csv(marker / "marker_discovery_independence.tsv", sep="\t")
                require(set(audit.sample_id) == set(SOURCES), "Marker-source set differs")
                for name in ["overlap_with_generated_pairs", "overlap_with_reserved_real_segment"]:
                    require(audit[name].eq(0).all(), "Marker discovery overlaps scored reads")
                for ref in ["ET39", "Cara1"]:
                    panel = pd.read_csv(marker / ref / "marker_panel.tsv", sep="\t")
                    for sources in panel.source_library_counts:
                        require({v.split(":")[0] for v in sources.split(";")} <= set(SOURCES), "Pool-series source in marker panel")
                    expected = pd.read_csv(ROOT / f"reporting/Markers_{ref}.tsv", sep="\t")
                    pd.testing.assert_frame_equal(panel, expected, check_dtype=False, check_exact=False, atol=1e-10, rtol=0)
                seal(marker, identity)
            features = work / "features_corrected"
            if not reusable(features, identity):
                execute(rb / "guarded_reference.py", ["features", "--reference_manifest", refs, "--marker_root", marker, "--generated_fastq_manifest", inputs / "generated_fastq_manifest.tsv", "--truth_manifest", inputs / "truth_manifest.tsv", "--sample_manifest", ROOT / "config/coffee/coffee_sample_manifest.tsv", "--source_fastq_root", raw, "--outdir", features, "--logs_dir", work / "logs/feature_alignments", "--scenario", "conservative", "--generated_pairs", "6000", "--real_pairs", "50000", "--real_skip_pairs", "100000", "--threads", "4", "--seed", "53001", "--min_mapq", "20", "--min_baseq", "20"], work, "feature_extraction")
                support.validate_feature_root(features, "conservative", 12)
                seal(features, identity)
            scoring = work / "reference_corrected"
            if not reusable(scoring, identity):
                from replay import reference
                scoring.mkdir(exist_ok=True)
                reference(scoring, features)
                seal(scoring, identity)
        write_json(work / "RUN_STATUS.json", {"result": "PASS", "targets": targets, "new_raw_alignment_completed": "reference" in targets, "reporting_sd_ddof": 1})
    except BaseException as exc:
        write_json(work / "RUN_STATUS.json", {"result": "STOPPED", "targets": targets, "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
