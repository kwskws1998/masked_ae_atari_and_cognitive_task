"""Selective Atari-HEAD downloader for the Zenodo record."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atari_gaze_cmae import download_files, fetch_zenodo_manifest, files_for_trial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--out", type=Path, required=True)

    trial_parser = subparsers.add_parser("trial")
    trial_parser.add_argument("--out", type=Path, required=True)
    trial_parser.add_argument("--trial", required=True)
    trial_parser.add_argument("--overwrite", action="store_true")
    trial_parser.add_argument("--no-md5", action="store_true")

    args = parser.parse_args()
    manifest = fetch_zenodo_manifest()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.command == "manifest":
        manifest_file = args.out / "zenodo_manifest.tsv"
        with manifest_file.open("w", encoding="utf-8") as f:
            f.write("name\tsize\tchecksum\turl\n")
            for item in manifest:
                f.write(f"{item.name}\t{item.size}\t{item.checksum or ''}\t{item.download_url}\n")
        print(manifest_file)
        return

    if args.command == "trial":
        selected = files_for_trial(manifest, args.trial)
        if not selected:
            raise SystemExit(f"no files found for trial prefix {args.trial}")
        downloaded = download_files(
            selected,
            args.out,
            overwrite=args.overwrite,
            verify_md5=not args.no_md5,
        )
        for path in downloaded:
            print(path)


if __name__ == "__main__":
    main()
