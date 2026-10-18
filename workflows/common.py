"""File identities, explicit workspaces, and numerical comparison helpers."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [f"SRR170376{i}" for i in list(range(10, 18)) + list(range(24, 28))]
POOLS = [f"SRR170376{i}" for i in range(18, 24)]
RUNS = [f"SRR170376{i}" for i in range(10, 28)]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path, delimiter: str = "\t") -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_module(path: Path, name: str):
    path = Path(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_workspace(path: Path) -> Path:
    path = Path(path).resolve()
    require(path != ROOT and not ROOT.is_relative_to(path), "Output cannot contain the repository.")
    for forbidden in [ROOT / n for n in ["examples", "metadata", "config", "reporting", "pipelines", "workflows", "tools", "tests", "history"]]:
        require(not path.is_relative_to(forbidden), "Output cannot overwrite a distributed input or code directory.")
    require(not path.exists() or not any(path.iterdir()), f"Use a new, empty output folder: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def execute(script: Path, arguments: list, work: Path, name: str, timeout: int = 21600) -> None:
    log = Path(work) / "logs" / (name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(script), *map(str, arguments)]
    environment = os.environ.copy()
    environment.update(PYTHONUTF8="1", PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    print(f"[RUN] {name}", flush=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(command) + "\n")
        handle.flush()
        try:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=environment, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Timed out: {name}; retain {log}") from None
    if result.returncode:
        print(log.read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
        raise RuntimeError(f"{name} failed ({result.returncode}); see {log}")
    print(f"[DONE] {name}: {time.monotonic() - started:.1f} seconds", flush=True)


def metrics(labels, scores) -> dict:
    """AUC by pair comparison and non-interpolated AP with grouped ties."""
    labels, scores = list(labels), list(scores)
    require(len(labels) == len(scores), "Label/score length mismatch")
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores)]
    require(len(pairs) > 0 and all(math.isfinite(s) for _, s in pairs), "Empty/nonfinite scores")
    pos = [s for y, s in pairs if y == 1]
    neg = [s for y, s in pairs if y == 0]
    require(len(pos) > 0 and len(neg) > 0 and len(pos) + len(neg) == len(pairs), "Need two binary classes")
    auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    groups = {}
    for y, s in pairs:
        groups.setdefault(s, []).append(y)
    total = true_pos = 0
    ap = 0.0
    for score in sorted(groups, reverse=True):
        values = groups[score]
        total += len(values)
        added = sum(values)
        true_pos += added
        ap += (added / len(pos)) * (true_pos / total)
    return {"auc": auc, "ap": ap, "n": len(pairs), "n_positive": len(pos), "prevalence": len(pos) / len(pairs)}


def close(a, b, tolerance=1e-10) -> bool:
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tolerance


def verify_metadata() -> dict:
    records = read_rows(ROOT / "metadata/coffee/SraRunTable.csv", ",")
    require(len(records) == 18 and {r["Run"] for r in records} == set(RUNS), "Run inventory differs")
    def cultivars(value):
        return {v.strip() for v in value.replace("\\,", ",").split(",") if v.strip()}
    individual = {r["Run"] for r in records if len(cultivars(r["Cultivar"])) == 1}
    require(individual == set(SOURCES), "Detailed metadata source identities differ")
    one_set = set().union(*(cultivars(r["Cultivar"]) for r in records if r["Run"] in SOURCES))
    require(len(one_set) == 12, "Expected twelve distinct cultivar entries")
    for r in records:
        if r["Run"] in POOLS:
            require(cultivars(r["Cultivar"]) == one_set and r["Library Name"].startswith("296_"), "Pooled-series fields differ")
    import re
    replicate_pairs = []
    for r in records:
        if r["Run"] in POOLS:
            match = re.search(r"Ligation replicate ([12]) of 2.*PCR replicate ([123]) of 3", r["replicate"])
            require(match is not None, "Pooled-series technical-replicate annotation missing")
            replicate_pairs.append((int(match[1]), int(match[2])))
    require(set(replicate_pairs) == {(a,b) for a in [1,2] for b in [1,2,3]}, "Technical-replicate layout differs")
    manifest = read_rows(ROOT / "config/coffee/coffee_sample_manifest.tsv")
    require(len(manifest) == 18 and {r["sample_id"] for r in manifest} == set(RUNS), "Current manifest run inventory differs")
    require({r["sample_id"] for r in manifest if r["include_primary"] == "1"} == set(SOURCES), "Current source manifest differs")
    require({r["sample_id"] for r in manifest if r["include_conservative"] == "1"} == set(SOURCES), "Current conservative flag differs")
    return {"runs": 18, "individual_cultivar_sources": SOURCES, "excluded_technical_series": POOLS,
            "interpretation": "Six technical libraries from one pooled series, not six independent biological pools."}


def verify_design_contract() -> dict:
    """The integrated runners reproduce a fixed design, not a parameter search."""
    data = json.loads((ROOT / "config/coffee/correction_design.json").read_text())
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    require(hashlib.sha256(canonical).hexdigest() == "bd670500556409a6dfe69aba504a130492ef8e9546a7c5aade22e40cb340cfeb",
            "Recorded correction design changed; do not treat a changed design as published reproduction")
    return data
