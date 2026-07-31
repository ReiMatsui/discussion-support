"""ライブ起動の推奨デフォルトのテスト."""
from __future__ import annotations

from click.testing import CliRunner

from das.asr.live import main
from das.asr.live._bootstrap import LiveArgs


def test_live_args_defaults_match_recommended_live_setup() -> None:
    args = LiveArgs()

    assert args.model == "stt-rt-v5"
    assert args.agent is True
    assert args.setup is True
    assert args.proactivity == "standard"
    # 推奨構成（pyannote＋クラスタ名前付け）が既定（2026-07-31）
    assert args.diarization == "pyannote"
    assert args.vp_cluster_naming is True


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
    assert warn("none", False) is None               # 両方無効なら警告なし
    msg = warn("none", True)
    assert msg is not None
    assert "話者分離なし" in msg and "無効" in msg   # 縮退の明示（既定Trueでも威圧的でない文言）


def test_live_args_has_docs_field_so_af_can_ingest():
    """AF ランタイムの文書取り込みが到達可能である（LiveArgs.docs が存在する）.

    run_session は `getattr(args, "docs", None)` で docs_dir を作っていたが、
    LiveArgs にこのフィールドが無く **常に None** だったため、
    AFRuntime.ingest_documents が一度も走らなかった（2026-07-25 監査。
    過去の merge_sim「本番から到達不能」と同型）。
    """
    import dataclasses

    from das.asr.live._bootstrap import LiveArgs
    names = {f.name for f in dataclasses.fields(LiveArgs)}
    assert "docs" in names
    assert LiveArgs(docs="data/docs").docs == "data/docs"


def test_live_cli_exposes_docs_option():
    """--docs が click のオプションとして存在し、LiveArgs へ渡る."""
    from das.asr.live import main
    names = {p.name for p in main.params}
    assert "docs" in names


def test_listen_forwards_docs_only_when_explicit():
    """das listen は --docs を明示したときだけ文字起こし側へ転送する."""
    from pathlib import Path

    from das.cli._listen import _build_soniox_argv
    assert "--docs" not in _build_soniox_argv()
    argv = _build_soniox_argv(af_docs=Path("data/docs"))
    assert argv[argv.index("--docs") + 1] == "data/docs"
