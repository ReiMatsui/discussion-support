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
    assert args.proactivity == "controlled"
    assert args.soniox_endpoint is True


def test_cli_help_documents_simplified_switches() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "--setup / --no-setup" in result.output
    assert "--agent / --no-agent" in result.output
