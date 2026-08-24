from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


USER_AGENT = "genomic-ricci-curvature-external-validation/1.0"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    if fieldnames is None:
        seen: List[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in fieldnames})
    os.replace(tmp, path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_token(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.upper().strip()
    return re.sub(r"[^A-Z0-9]+", "", text)


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.upper().replace("–", "-").replace("—", "-")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_semicolon(value: str) -> List[str]:
    return [x.strip() for x in (value or "").split(";") if x.strip()]


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def fetch_bytes(
    url: str,
    *,
    retries: int = 6,
    timeout: int = 120,
    headers: Optional[Mapping[str, str]] = None,
) -> bytes:
    request_headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        request_headers.update(headers)
    last_error: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Failed to retrieve {url}: {last_error}")


def fetch_to_file(url: str, path: Path, *, force: bool = False, min_bytes: int = 1) -> Path:
    if path.exists() and path.stat().st_size >= min_bytes and not force:
        return path
    data = fetch_bytes(url)
    if len(data) < min_bytes:
        raise RuntimeError(f"Downloaded content is unexpectedly small ({len(data)} bytes): {url}")
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def url_https_from_ena(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        return value.replace("http://", "https://", 1)
    if value.startswith("ftp://"):
        return "https://" + value[6:]
    return "https://" + value.lstrip("/")


def stable_selection_key(seed: int, population: str, sample_id: str) -> str:
    payload = f"{seed}|{population}|{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def disk_free_bytes(path: Path) -> int:
    return shutil.disk_usage(str(path)).free


def format_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.2f}"


def safe_extract_tar(tar_path: Path, destination: Path) -> None:
    import tarfile
    ensure_dir(destination)
    with tarfile.open(tar_path, "r:*") as archive:
        dest_resolved = destination.resolve()
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if not str(member_path).startswith(str(dest_resolved)):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        archive.extractall(destination)


def flatten_mapping(mapping: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for key in sorted(mapping):
        value = mapping[key]
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return " | ".join(parts)
