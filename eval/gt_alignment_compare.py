#!/usr/bin/env python3
"""正解の当て方（時間の重なり／文章の一致）で数字がどう変わるかを並べる.

**問い**: いまの 89.9% は「測りやすい発話だけ」の数字ではないか。

時間の重なりで正解を割り当てると、8割を一人が占める発話にしか正解が付かない。
3人の会話では長い発話ほど笑いや相づちが重なるので、この条件は**重なりの多い
場面＝難しい場面**を丸ごと落とす。落ちた側の成績が悪ければ、残りだけを測った
数字は実力より高く出る。

文章の一致で割り当てれば、重なっていても「この一文は誰のものか」は決まる。
両方で測って、採点できる範囲と数字の差を見る。

件数と文字数の両方で出すのは、体感との一致を確かめるため——議事録として
目に入るのは長い発言なので、件数だけだと短い相づちに数字が支配される。

使い方:
    uv run python eval/gt_alignment_compare.py --prefix 2026-07-20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import _pipeline as pipe  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._constants import UNSURE_SPEAKER  # noqa: E402


def measure(run: str, vp, align: str) -> dict | None:
    """1本を今日の実装で再生し、指定の当て方で採点する."""
    data = pipe.replay_seats(run, vp, align=align)
    if data is None:
        return None
    steps = data["steps"]
    final = pipe.apply_schedule(steps)
    pairs = [(f, st["code"]) for f, st in zip(final, steps, strict=True)]
    acc, wrong, uns, n = pipe.score(pairs)
    # 文字数で重み付け（長い発言の取り違えを重く見る）
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=UNSURE_SPEAKER)
    w = [len(str(st["utt"].get("_text") or "")) for st in steps]
    tot = sum(w) or 1
    wok = sum(x for (p, c), x in zip(pairs, w, strict=True) if m.get(p) == c)
    wun = sum(x for (p, _), x in zip(pairs, w, strict=True) if p == UNSURE_SPEAKER)
    # 全発話（相槌を含む）に対して、何件を採点できたか
    loaded = dec.load_run(run)
    total_utts = len(loaded[0]) if loaded else n
    return {"n": n, "total": total_utts, "acc": acc, "wrong": wrong, "uns": uns,
            "wacc": wok / tot, "wwrong": (tot - wok - wun) / tot, "wuns": wun / tot}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="2026-07-20")
    p.add_argument("--model", default="redimnet")
    args = p.parse_args(argv)
    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(model=args.model)
    runs = [r for r in sorted(dec.discover()) if r.startswith(args.prefix)]

    rows = []
    for run in runs:
        got = {a: measure(run, vp, a) for a in ("time", "text")}
        if got["time"] and got["text"]:
            rows.append((run, got))
    if not rows:
        raise SystemExit("# 測れるランが無い")

    print("\n## 件数で数えた場合")
    print(f"{'run':<18}{'採点できた割合':>14}{'正解':>8}{'誤帰属':>8}{'未確定':>8}"
          f"{'  |':>4}{'採点できた割合':>14}{'正解':>8}{'誤帰属':>8}{'未確定':>8}")
    print(f"{'':<18}{'—— 時間の重なり（従来） ——':>46}"
          f"{'':>4}{'—— 文章の一致 ——':>44}")
    for run, g in rows:
        t, x = g["time"], g["text"]
        print(f"{run:<18}{t['n'] / t['total']:>14.0%}{t['acc']:>8.1%}"
              f"{t['wrong']:>8.1%}{t['uns']:>8.1%}{'  |':>4}"
              f"{x['n'] / x['total']:>14.0%}{x['acc']:>8.1%}"
              f"{x['wrong']:>8.1%}{x['uns']:>8.1%}")
    n = len(rows)
    t = {k: sum(g["time"][k] for _, g in rows) / n
         for k in ("acc", "wrong", "uns")}
    x = {k: sum(g["text"][k] for _, g in rows) / n
         for k in ("acc", "wrong", "uns")}
    tc = sum(g["time"]["n"] / g["time"]["total"] for _, g in rows) / n
    xc = sum(g["text"]["n"] / g["text"]["total"] for _, g in rows) / n
    print(f"{'平均':<18}{tc:>14.0%}{t['acc']:>8.1%}{t['wrong']:>8.1%}"
          f"{t['uns']:>8.1%}{'  |':>4}{xc:>14.0%}{x['acc']:>8.1%}"
          f"{x['wrong']:>8.1%}{x['uns']:>8.1%}")

    print("\n## 文字数で重み付けした場合（文章の一致）")
    print(f"{'run':<18}{'正解':>8}{'誤帰属':>8}{'未確定':>8}")
    for run, g in rows:
        x = g["text"]
        print(f"{run:<18}{x['wacc']:>8.1%}{x['wwrong']:>8.1%}{x['wuns']:>8.1%}")
    w = {k: sum(g["text"][k] for _, g in rows) / n
         for k in ("wacc", "wwrong", "wuns")}
    print(f"{'平均':<18}{w['wacc']:>8.1%}{w['wwrong']:>8.1%}{w['wuns']:>8.1%}")

    print("\n読み方:")
    print("  「採点できた割合」が低いほど、測れていない発話が多い。落ちるのは")
    print("  重なりの多い＝難しい場面なので、そこを外した数字は高く出る。")
    print("  文字数で重み付けした数字は、議事録として読んだときの体感に近い")
    print("  （目に入るのは長い発言で、短い相づちは流し読みされるため）。")


if __name__ == "__main__":
    main()
