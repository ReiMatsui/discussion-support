"""das listen-soniox の argv 合成のテスト.

推奨構成（pyannote＋クラスタ名前付け）は文字起こし側 (das.asr.live) の
既定になったため、argv 合成は「既定から外れる指定」だけを運ぶ。
"""
from __future__ import annotations

from pathlib import Path

from das.cli._listen import _build_soniox_argv


def test_default_argv_is_empty_because_recommended_config_is_the_default() -> None:
    """無指定なら何も渡さない＝文字起こし側の既定（推奨構成）で動く."""
    assert _build_soniox_argv() == []


def test_max_speakers_maps_to_diarization_max_speakers() -> None:
    assert _build_soniox_argv(max_speakers=3) == [
        "--diarization-max-speakers",
        "3",
    ]


def test_wav_and_max_speakers() -> None:
    argv = _build_soniox_argv(max_speakers=3, wav=Path("x.wav"))
    assert argv == [
        "--wav",
        "x.wav",
        "--diarization-max-speakers",
        "3",
    ]


def test_soniox_args_appended_last_for_click_last_wins() -> None:
    """--soniox-args は末尾に付き、click の後勝ちで第一級オプションより優先。"""
    argv = _build_soniox_argv(
        wav=Path("m.wav"),
        soniox_args="--diarization none --no-agent",
    )
    assert argv == [
        "--wav",
        "m.wav",
        "--diarization",
        "none",
        "--no-agent",
    ]


def test_wav_path_with_spaces() -> None:
    argv = _build_soniox_argv(wav=Path("dir with space/x.wav"))
    assert argv == ["--wav", "dir with space/x.wav"]


def test_af_docs_forwarded() -> None:
    argv = _build_soniox_argv(af_docs=Path("data/docs"))
    assert argv == ["--docs", "data/docs"]
