"""Download Atari-HEAD Zenodo v4 files with size and MD5 verification."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request


ZENODO_API = "https://zenodo.org/api/records/{record_id}"
DEFAULT_RECORD_ID = "3451402"
DEFAULT_OUT_DIR = Path("data/atari_head_full/v4")
DEFAULT_HF_REPO = "skboy/atari-head-v4"


@dataclass(frozen=True)
class ZenodoFile:
    name: str
    size: int
    checksum: str | None
    url: str


def normalize_game_name(game: str) -> str:
    return game.strip().lower().replace("-", "_").replace(" ", "_")


def expected_md5(checksum: str | None) -> str | None:
    if not checksum:
        return None
    if checksum.startswith("md5:"):
        return checksum.split(":", 1)[1]
    return checksum


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_manifest(record_id: str) -> list[ZenodoFile]:
    with urllib.request.urlopen(ZENODO_API.format(record_id=record_id)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    files = []
    for item in payload.get("files", []):
        name = item["key"]
        files.append(
            ZenodoFile(
                name=name,
                size=int(item["size"]),
                checksum=item.get("checksum"),
                url=item["links"]["self"],
            )
        )
    return sorted(files, key=lambda item: item.name)


def write_manifest_tsv(files: list[ZenodoFile], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("name\tsize\tchecksum\turl\n")
        for item in files:
            handle.write(f"{item.name}\t{item.size}\t{item.checksum or ''}\t{item.url}\n")


def select_files(
    manifest: list[ZenodoFile],
    *,
    games: list[str] | None,
    filenames: list[str] | None,
    download_all: bool,
) -> list[ZenodoFile]:
    if sum(bool(value) for value in (games, filenames, download_all)) != 1:
        raise ValueError("choose exactly one of --games, --files, or --all")

    by_name = {item.name: item for item in manifest}
    if download_all:
        return manifest

    if filenames:
        wanted = set(filenames)
    else:
        wanted = {f"{normalize_game_name(game)}.zip" for game in games or []}

    missing = sorted(wanted - set(by_name))
    if missing:
        available = ", ".join(item.name.removesuffix(".zip") for item in manifest[:8])
        raise ValueError(f"missing files in Zenodo manifest: {missing}. Available examples: {available}")
    return [by_name[name] for name in sorted(wanted)]


def select_hf_include_patterns(
    *,
    games: list[str] | None,
    filenames: list[str] | None,
    download_all: bool,
) -> list[str]:
    if sum(bool(value) for value in (games, filenames, download_all)) != 1:
        raise ValueError("choose exactly one of --games, --files, or --all")
    if download_all:
        return ["*.zip"]
    if filenames:
        return sorted(filenames)
    return sorted(f"{normalize_game_name(game)}.zip" for game in games or [])


def ensure_hf_cli(login: bool) -> None:
    hf = shutil.which("hf")
    if hf is None:
        raise RuntimeError(
            "hf CLI is not installed. Install it with: python -m pip install 'huggingface_hub[cli]'"
        )
    if login:
        subprocess.run([hf, "auth", "login"], check=True)
        return
    auth = subprocess.run([hf, "auth", "whoami"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if auth.returncode != 0:
        raise RuntimeError("HF auth is missing. Run `hf auth login` first, or pass `--hf-login`.")


def download_from_hf(
    *,
    repo_id: str,
    include_patterns: list[str],
    out_dir: Path,
    max_workers: int,
    login: bool,
) -> None:
    ensure_hf_cli(login)
    hf = shutil.which("hf")
    if hf is None:
        raise RuntimeError("hf CLI disappeared after availability check")

    command = [
        hf,
        "download",
        repo_id,
        "--type",
        "dataset",
        "--local-dir",
        str(out_dir),
        "--max-workers",
        str(max_workers),
    ]
    for pattern in include_patterns:
        command.extend(["--include", pattern])
    print(" ".join(command))
    subprocess.run(command, check=True)


def should_skip(path: Path, item: ZenodoFile, verify_md5: bool) -> bool:
    if not path.exists() or path.stat().st_size != item.size:
        return False
    expected = expected_md5(item.checksum)
    if verify_md5 and expected is not None:
        actual = file_md5(path)
        if actual != expected:
            print(f"md5 mismatch, redownloading: {path.name}")
            path.unlink()
            return False
    print(f"skip existing: {path.name}")
    return True


def download_one(item: ZenodoFile, out_dir: Path, verify_md5: bool) -> Path:
    target = out_dir / item.name
    if should_skip(target, item, verify_md5):
        return target

    resume_at = target.stat().st_size if target.exists() else 0
    mode = "ab" if resume_at > 0 else "wb"
    request = urllib.request.Request(item.url)
    if resume_at > 0:
        request.add_header("Range", f"bytes={resume_at}-")
        print(f"resume {item.name}: {resume_at / (1024 ** 2):.1f} MiB / {item.size / (1024 ** 2):.1f} MiB")
    else:
        print(f"download {item.name}: {item.size / (1024 ** 2):.1f} MiB")

    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and target.exists() and target.stat().st_size == item.size:
            return target
        raise

    with response:
        if resume_at > 0 and response.status == 200:
            mode = "wb"
            resume_at = 0
        downloaded = resume_at
        last_report = time.monotonic()
        with target.open(mode + "") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 10:
                    pct = downloaded / item.size * 100 if item.size else 0.0
                    print(f"  {item.name}: {downloaded / (1024 ** 2):.1f} MiB ({pct:.1f}%)")
                    last_report = now

    actual_size = target.stat().st_size
    if actual_size != item.size:
        raise RuntimeError(f"size mismatch for {target}: expected {item.size}, got {actual_size}")
    expected = expected_md5(item.checksum)
    if verify_md5 and expected is not None:
        actual = file_md5(target)
        if actual != expected:
            raise RuntimeError(f"MD5 mismatch for {target}: expected {expected}, got {actual}")
    print(f"done {item.name}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["zenodo", "hf"], default="zenodo")
    parser.add_argument("--record-id", default=DEFAULT_RECORD_ID)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--games", nargs="*", help="Game names to download, e.g. breakout seaquest.")
    parser.add_argument("--files", nargs="*", help="Optional subset of Zenodo filenames.")
    parser.add_argument("--all", action="store_true", help="Download the full v4 archive.")
    parser.add_argument("--manifest-out", type=Path, help="Write a TSV manifest before downloading.")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-login", action="store_true", help="Run `hf auth login` before HF download.")
    parser.add_argument("--hf-max-workers", type=int, default=8)
    parser.add_argument("--no-md5", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.source == "hf":
        if args.manifest_out:
            raise SystemExit("--manifest-out is only available with --source zenodo")
        try:
            include_patterns = select_hf_include_patterns(
                games=args.games,
                filenames=args.files,
                download_all=args.all,
            )
            download_from_hf(
                repo_id=args.hf_repo,
                include_patterns=include_patterns,
                out_dir=args.out,
                max_workers=args.hf_max_workers,
                login=args.hf_login,
            )
        except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
            raise SystemExit(str(exc)) from exc
        return

    manifest = fetch_manifest(args.record_id)
    if args.manifest_out:
        write_manifest_tsv(manifest, args.manifest_out)
        print(args.manifest_out)
        if not (args.games or args.files or args.all):
            return

    try:
        selected = select_files(
            manifest,
            games=args.games,
            filenames=args.files,
            download_all=args.all,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    total = sum(item.size for item in selected)
    print(f"{len(selected)} files, {total / (1024 ** 3):.2f} GiB -> {args.out}")
    for item in selected:
        download_one(item, args.out, verify_md5=not args.no_md5)


if __name__ == "__main__":
    main()
