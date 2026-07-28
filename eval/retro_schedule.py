#!/usr/bin/env python3
"""遡及訂正の「予定表」を比べる（何分ごとに貼り直すのが良いか）.

初回実装（2分・5分、以後5分ごと）は 85.2% で、上限に近い retro_5m（89.5%）に
届かなかった。原因は間隔で、10分の会話だと 300秒→600秒の間に一度も貼り直しが
入らない区間ができる。

**貼り直しは実質ただである**: `RetroAttributor.revise` は保存済みの声紋
（192次元）と席の参照の内積を取るだけで、埋め込みの計算は一切しない。
1200発話×3席でも内積3600回。したがって間隔を詰めない理由は計算量ではなく、
**表示が頻繁に書き換わること**（UX）だけになる。

ここでは席の参照の推移を1回だけ計算して保存し、予定表だけを差し替えて比べる
（埋め込みの計算をやり直さないので、条件を増やしても時間が増えない）。

使い方:
    uv run python eval/retro_schedule.py --prefix 2026-07-20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _pipeline as pipe  # noqa: E402
import decompose_attribution as dec  # noqa: E402

# (名前, 予定表, 以後の間隔秒)。間隔 0 は「発話ごと」
CONFIGS = (
    ("2分5分/5分ごと（初回実装）", (120.0, 300.0), 300.0),
    ("2分5分/2分ごと", (120.0, 300.0), 120.0),
    ("2分5分/1分ごと", (120.0, 300.0), 60.0),
    ("1分ごと", (60.0,), 60.0),
    ("発話ごと", (0.0,), 0.0),
)


def collect(run: str, vp) -> dict | None:
    """席の参照の推移を1回だけ計算する（再現の本体は `_pipeline`）."""
    return pipe.replay_seats(run, vp)


def evaluate(data, schedule, interval) -> tuple[float, float, float]:
    """予定表を1つ当てて (正解, 誤帰属, 未確定) を返す."""
    steps = data["steps"]
    final = pipe.apply_schedule(steps, schedule, interval)
    acc, wrong, uns, _n = pipe.score(
        [(f, st["code"]) for f, st in zip(final, steps, strict=True)])
    return acc, wrong, uns


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="2026-07-20")
    p.add_argument("--model", default="redimnet")
    p.add_argument("--split", type=int, default=5)
    args = p.parse_args(argv)
    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(model=args.model)
    runs = [r for r in sorted(dec.discover()) if r.startswith(args.prefix)]
    data = [d for d in (collect(x, vp) for x in runs) if d]
    if not data:
        raise SystemExit("# 測れるランが無い")

    def _report(subset, label):
        print(f"\n## {label}（{len(subset)}本）")
        print(f"{'予定表':<26}{'正解':>8}{'誤帰属':>9}{'未確定':>9}{'貼直し回数':>10}")
        for name, sched, interval in CONFIGS:
            vals = [evaluate(d, sched, interval) for d in subset]
            n = len(vals)
            acc, wrong, uns = (sum(v[i] for v in vals) / n for i in (0, 1, 2))
            # 貼り直しの回数（10分の会話での目安）
            times = 1 if interval <= 0 else int(600 / max(interval, 1))
            shown = "毎発話" if interval <= 0 else str(len(sched) + times)
            print(f"{name:<26}{acc:>8.1%}{wrong:>9.1%}{uns:>9.1%}{shown:>10}")

    _report(data, "全体")
    if 0 < args.split < len(data):
        _report(data[:args.split], "開発")
        _report(data[args.split:], "検証")
    print("\n読み方:")
    print("  貼り直しは保存済みの声紋との内積だけで、埋め込みの計算は要らない。")
    print("  したがって間隔を詰めない理由は計算量ではなく、表示が頻繁に")
    print("  書き換わること（UX）だけになる。")


if __name__ == "__main__":
    main()
