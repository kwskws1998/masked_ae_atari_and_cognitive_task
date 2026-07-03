"""Zenodo utilities for selective Atari-HEAD downloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request


ZENODO_RECORD_ID = "2603190"
ZENODO_API_TEMPLATE = "https://zenodo.org/api/records/{record_id}"
ZENODO_FILE_TEMPLATE = "https://zenodo.org/records/{record_id}/files/{filename}?download=1"


@dataclass(frozen=True)
class ZenodoFile:
    name: str
    size: int
    checksum: str | None
    download_url: str


def fetch_zenodo_manifest(record_id: str = ZENODO_RECORD_ID) -> list[ZenodoFile]:
    url = ZENODO_API_TEMPLATE.format(record_id=record_id)
    with urllib.request.urlopen(url) as response:
        payload = json.loads(response.read().decode("utf-8"))
    files = []
    for item in payload.get("files", []):
        name = item.get("key") or item.get("filename")
        if not name:
            continue
        links = item.get("links", {})
        download_url = links.get("self") or links.get("content") or ZENODO_FILE_TEMPLATE.format(
            record_id=record_id,
            filename=name,
        )
        files.append(
            ZenodoFile(
                name=name,
                size=int(item.get("size", 0)),
                checksum=item.get("checksum"),
                download_url=download_url,
            )
        )
    return files


def files_for_trial(files: list[ZenodoFile], trial: str | int) -> list[ZenodoFile]:
    prefix = f"{int(trial)}_"
    selected = [
        item
        for item in files
        if item.name.startswith(prefix) and item.name.endswith((".tar.bz2", ".txt"))
    ]
    return sorted(selected, key=lambda item: item.name)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_md5(checksum: str | None) -> str | None:
    if not checksum:
        return None
    if checksum.startswith("md5:"):
        return checksum.split(":", 1)[1]
    return checksum


def download_files(
    files: list[ZenodoFile],
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    verify_md5: bool = True,
) -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for item in files:
        target = out_path / item.name
        if target.exists() and not overwrite:
            downloaded.append(target)
            continue
        with urllib.request.urlopen(item.download_url) as response, target.open("wb") as f:
            shutil.copyfileobj(response, f)
        expected = _expected_md5(item.checksum)
        if verify_md5 and expected is not None:
            actual = _md5(target)
            if actual != expected:
                raise ValueError(f"MD5 mismatch for {target}: expected {expected}, got {actual}")
        downloaded.append(target)
    return downloaded
