"""Unit tests for Atari-HEAD v4 downloader selection logic."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_atari_head_v4 import (
    ZenodoFile,
    normalize_game_name,
    select_files,
    select_hf_include_patterns,
)


def sample_manifest() -> list[ZenodoFile]:
    return [
        ZenodoFile("breakout.zip", 100, "md5:a", "https://example.com/breakout.zip"),
        ZenodoFile("seaquest.zip", 200, "md5:b", "https://example.com/seaquest.zip"),
        ZenodoFile("space_invaders.zip", 300, "md5:c", "https://example.com/space_invaders.zip"),
    ]


def test_normalize_game_name_accepts_common_spellings() -> None:
    assert normalize_game_name("Breakout") == "breakout"
    assert normalize_game_name("space invaders") == "space_invaders"
    assert normalize_game_name("Space-Invaders") == "space_invaders"


def test_select_files_by_game_names() -> None:
    selected = select_files(
        sample_manifest(),
        games=["Breakout", "Space Invaders"],
        filenames=None,
        download_all=False,
    )
    assert [item.name for item in selected] == ["breakout.zip", "space_invaders.zip"]


def test_select_files_requires_one_selection_mode() -> None:
    try:
        select_files(sample_manifest(), games=["breakout"], filenames=["breakout.zip"], download_all=False)
    except ValueError as exc:
        assert "choose exactly one" in str(exc)
    else:
        raise AssertionError("select_files should reject multiple selection modes")


def test_select_hf_include_patterns_by_game_names() -> None:
    patterns = select_hf_include_patterns(
        games=["Breakout", "Space Invaders"],
        filenames=None,
        download_all=False,
    )
    assert patterns == ["breakout.zip", "space_invaders.zip"]


if __name__ == "__main__":
    test_normalize_game_name_accepts_common_spellings()
    test_select_files_by_game_names()
    test_select_files_requires_one_selection_mode()
    test_select_hf_include_patterns_by_game_names()
