"""das listen-soniox の引数合成 (_build_soniox_argv) の単体テスト。"""

from __future__ import annotations

from pathlib import Path

from das.cli._listen import _build_soniox_argv


def test_default_is_empty() -> None:
    """何も指定しなければ空 (既存挙動と同じ)。"""
    assert _build_soniox_argv() == []


def test_hybrid_expands_to_recommended_config() -> None:
    assert _build_soniox_argv(hybrid=True) == [
        "--diarization",
        "pyannote",
        "--vp-cluster-naming",
    ]


def test_max_speakers_maps_to_diarization_max_speakers() -> None:
    assert _build_soniox_argv(max_speakers=3) == [
        "--diarization-max-speakers",
        "3",
    ]


def test_hybrid_with_max_speakers_and_wav() -> None:
    argv = _build_soniox_argv(hybrid=True, max_speakers=3, wav=Path("x.wav"))
    assert argv == [
        "--diarization",
        "pyannote",
        "--vp-cluster-naming",
        "--wav",
        "x.wav",
        "--diarization-max-speakers",
        "3",
    ]


def test_explicit_options_without_hybrid() -> None:
    argv = _build_soniox_argv(diarization="assemblyai", vp_cluster_naming=True)
    assert argv == ["--diarization", "assemblyai", "--vp-cluster-naming"]


def test_vp_cluster_naming_not_duplicated_with_hybrid() -> None:
    argv = _build_soniox_argv(hybrid=True, vp_cluster_naming=True)
    assert argv.count("--vp-cluster-naming") == 1


def test_soniox_args_appended_last_for_click_last_wins() -> None:
    """--soniox-args は末尾に付き、click の後勝ちで第一級オプションより優先。"""
    argv = _build_soniox_argv(
        hybrid=True,
        soniox_args="--diarization assemblyai --stt speechmatics",
    )
    assert argv == [
        "--diarization",
        "pyannote",
        "--vp-cluster-naming",
        "--diarization",
        "assemblyai",
        "--stt",
        "speechmatics",
    ]


def test_wav_path_with_spaces() -> None:
    argv = _build_soniox_argv(wav=Path("/tmp/my meeting/rec 1.wav"))
    assert argv == ["--wav", "/tmp/my meeting/rec 1.wav"]


def test_soniox_args_shlex_handles_quoted_spaces() -> None:
    argv = _build_soniox_argv(soniox_args="--wav '/tmp/my meeting/rec 1.wav'")
    assert argv == ["--wav", "/tmp/my meeting/rec 1.wav"]
