"""ライブ起動の推奨デフォルトのテスト."""
from __future__ import annotations

from click.testing import CliRunner

from das.asr.live import main
from das.asr.live._bootstrap import LiveArgs


def test_live_args_defaults_match_recommended_live_setup() -> None:
    args = LiveArgs()

    assert args.stt == "soniox"
    assert args.model == "stt-rt-v5"
    assert args.agent is True
    assert args.setup is True
    assert args.proactivity == "standard"
    assert args.soniox_endpoint is True


def test_cli_help_documents_simplified_switches() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "--setup / --no-setup" in result.output
    assert "--agent / --no-agent" in result.output


def test_vp_cluster_naming_warning_for_non_pyannote_diarization() -> None:
    """--vp-cluster-naming は pyannote 以外の diarization では警告する（F6）.

    従来は assemblyai 併用時のみ警告し、--diarization none（既定）では黙って
    無効化されていた（2026-07-15 レビュー）。例: --hybrid --soniox-args
    "--diarization none" は後勝ちで diarization だけ none に上書きされ、
    ユーザーはハイブリッド構成のつもりのまま気づけない。
    """
    from das.asr.live._bootstrap import vp_cluster_naming_disabled_warning as warn

    assert warn("pyannote", True) is None            # 有効な構成は警告なし
    assert warn("none", False) is None               # 未指定なら警告なし
    for diar in ("none", "assemblyai"):
        msg = warn(diar, True)
        assert msg is not None
        assert "--vp-cluster-naming" in msg and "pyannote" in msg and "無効" in msg
