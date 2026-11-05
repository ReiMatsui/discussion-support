"""本体起動と目的別チェックの入口。実API・録音・pytest子プロセスは起動しない。"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from das.cli import app
from das.cli._checks import CheckProfile, check_targets


def test_help_exposes_main_and_study_and_checks():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert all(name in result.output for name in ("listen-soniox", "study", "check"))


def test_study_forwards_explicit_condition_without_graph_lane(monkeypatch):
    from das.asr import live

    seen = []
    monkeypatch.setattr(live, "main", lambda args, **kw: seen.append((args, kw)))
    result = CliRunner().invoke(app, ["study", "--condition", "none", "--no-open"])
    assert result.exit_code == 0, result.output
    assert seen == [([
        "--study-config", "study/facility.json", "--study-condition", "none", "--no-open",
    ], {"standalone_mode": False})]


def test_study_defaults_to_system_and_forwards_device(monkeypatch):
    from das.asr import live

    seen = []
    monkeypatch.setattr(live, "main", lambda args, **_: seen.extend(args))
    result = CliRunner().invoke(app, ["study", "--device", "test-device"])
    assert result.exit_code == 0, result.output
    assert seen[seen.index("--study-condition") + 1] == "system"
    assert seen[seen.index("--device") + 1] == "test-device"


def test_invalid_study_condition_never_starts_native(monkeypatch):
    from das.asr import live

    monkeypatch.setattr(live, "main", lambda *_a, **_kw: pytest.fail("must not start"))
    assert CliRunner().invoke(app, ["study", "--condition", "invalid"]).exit_code == 2


def test_check_defaults_to_quick_with_api_tests_excluded(monkeypatch):
    from das.cli import _checks

    seen = []
    monkeypatch.setattr(_checks.subprocess, "call", lambda cmd, **kw: seen.append((cmd, kw)) or 0)
    result = CliRunner().invoke(app, ["check"])
    assert result.exit_code == 0
    command, kwargs = seen[0]
    assert "tests/unit/test_smoke.py" in command
    assert command[command.index("-m", 3) + 1] == "not integration"
    assert kwargs["cwd"] == Path(__file__).resolve().parents[2]


def test_check_list_does_not_execute(monkeypatch):
    from das.cli import _checks

    monkeypatch.setattr(_checks.subprocess, "call", lambda *_a, **_kw: pytest.fail("list only"))
    result = CliRunner().invoke(app, ["check", "study", "--list"])
    assert result.exit_code == 0 and "tests/unit/study" in result.output


def test_check_propagates_failures_and_duration_option(monkeypatch):
    from das.cli import _checks

    seen = []
    monkeypatch.setattr(_checks.subprocess, "call", lambda cmd, **_: seen.append(cmd) or 1)
    result = CliRunner().invoke(app, ["check", "live", "--durations", "10"])
    assert result.exit_code == 1
    assert "--durations=10" in seen[0]


def test_check_core_has_no_live_or_study_subtrees():
    root = Path(__file__).resolve().parents[2]
    selected = check_targets(CheckProfile.core, root)
    assert selected
    assert all(Path(name).parent == Path("tests/unit") for name in selected)
    assert check_targets(CheckProfile.all, root) == ["tests"]
