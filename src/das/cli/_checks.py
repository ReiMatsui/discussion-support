"""開発者向けの確認コマンド。用途に応じて既存のテストを選択する。"""

from __future__ import annotations

import subprocess
import sys
from enum import StrEnum
from pathlib import Path

import typer

from das.cli import app


class CheckProfile(StrEnum):
    quick = "quick"
    study = "study"
    core = "core"
    live = "live"
    all = "all"


def check_targets(profile: CheckProfile, root: Path) -> list[str]:
    """選択は一か所で管理。新しいテストも領域のプロファイルへ自動で入る。"""
    if profile == CheckProfile.quick:
        return [
            "tests/unit/test_smoke.py", "tests/unit/test_cli_commands.py",
            "tests/unit/test_cost.py", "tests/unit/study/test_engine.py",
            "tests/unit/study/test_probe_usage.py",
        ]
    if profile == CheckProfile.study:
        return ["tests/unit/study"]
    if profile == CheckProfile.live:
        return ["tests/unit/live"]
    if profile == CheckProfile.core:
        return [str(p.relative_to(root)) for p in sorted((root / "tests/unit").glob("test_*.py"))]
    return ["tests"]


@app.command(name="check")
def check(
    profile: CheckProfile = typer.Argument(CheckProfile.quick, help="確認する範囲"),
    list_only: bool = typer.Option(False, "--list", help="対象だけ表示し、テストは実行しない"),
    durations: int = typer.Option(0, "--durations", min=0, help="遅いテストを上位N件表示"),
) -> None:
    """目的別に自動テストを実行する。実API向けのintegrationは常に除外。"""
    root = Path(__file__).resolve().parents[3]
    if not (root / "tests").is_dir():
        typer.echo("テストがある開発用チェックアウトで実行してください。", err=True)
        raise typer.Exit(2)
    targets = check_targets(profile, root)
    typer.echo(f"確認範囲: {profile.value}")
    if list_only:
        for target in targets:
            typer.echo(target)
        return
    command = [sys.executable, "-m", "pytest", *targets, "-m", "not integration", "-q"]
    if durations:
        command.append(f"--durations={durations}")
    raise typer.Exit(subprocess.call(command, cwd=root))
