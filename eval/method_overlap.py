#!/usr/bin/env python3
"""本番と「単純方式」を発話ごとに突き合わせ、どちらが何を取っているかを見る.

**問い**: 単純方式（長い発話でまとめて短い発話を寄せるだけ。人数も pyannote も
声紋台帳も使わない）は文字ベース 86.4%、本番は 91.5%（§40）。この5ポイントの
差は「本番が全面的に上」なのか、それとも**別のところで勝っている**のか。

平均だけ見ると前者に見えるが、両方が同じ発話で正解していれば差は薄い上積み、
違う発話で正解していれば**組み合わせる余地**がある。発話ごとに突き合わせない
と区別できない。

本番側の結末は `eval/_error_anatomy.json`（`error_anatomy.py` が保存する）を
読む。単純方式は `roster_free.causal_predict` をそのまま呼ぶ——ここに書き写す
と、また2種類の「単純方式」ができる。

使い方:
    uv run python eval/error_anatomy.py --prefix 2026-07-20   # 先に本番側を保存
    uv run python eval/method_overlap.py --prefix 2026-07-20
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import _utt_embeddings as ue  # noqa: E402
import decompose_attribution as dec  # noqa: E402
import roster_free as rf  # noqa: E402

CACHE = ROOT / "eval" / "_error_anatomy.json"
CELLS = ("両方正解", "本番だけ正解", "単純だけ正解", "両方不正解")


def _band(dur_ms: int) -> str:
    return "~1秒" if dur_ms < 1000 else "1〜2秒" if dur_ms < 2000 else "2秒以上"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="2026-07-20")
    p.add_argument("--min-ms", type=int, default=2000,
                   help="まとめる材料にする発話の最短の長さ")
    args = p.parse_args(argv)
    if not CACHE.exists():
        raise SystemExit("# 先に eval/error_anatomy.py を走らせて本番側を保存する")
    prod = {(r["run"], int(r["ms"])): r
            for r in json.loads(CACHE.read_text(encoding="utf-8"))}

    tab, wtab = Counter(), Counter()
    byband: dict[str, Counter] = defaultdict(Counter)
    bykind: dict[str, Counter] = defaultdict(Counter)
    for run in [r for r in sorted(dec.discover()) if r.startswith(args.prefix)]:
        g = ue.load(run)
        if not g:
            continue
        pred = rf.causal_predict(g["emb"], g["dur_ms"], g["ms"],
                                 min_ms=args.min_ms)
        pairs = list(zip(rf.labels_of(pred, len(g["code"])), g["code"],
                         strict=True))
        _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure="?")
        for i, (pp, c) in enumerate(pairs):
            row = prod.get((run, int(g["ms"][i])))
            if row is None:
                continue
            s_ok, p_ok = m.get(pp) == c, row["outcome"] == "正解"
            cell = ("両方正解" if s_ok and p_ok else "本番だけ正解" if p_ok
                    else "単純だけ正解" if s_ok else "両方不正解")
            tab[cell] += 1
            wtab[cell] += int(g["chars"][i])
            byband[_band(int(g["dur_ms"][i]))][cell] += 1
            if cell in ("本番だけ正解", "単純だけ正解"):
                bykind[cell][row.get("kind") or "（なし）"] += 1

    n, w = sum(tab.values()), sum(wtab.values())
    if not n:
        raise SystemExit("# 突き合わせられる発話が無い")
    print(f"\n## 突き合わせ（{n}発話 / {w}文字）")
    for k in CELLS:
        print(f"  {k:<10}{tab[k]:>6}件 {tab[k] / n:>7.1%}"
              f"   {wtab[k]:>6}文字 {wtab[k] / w:>7.1%}")
    print(f"  {'どちらかが正解':<10}"
          f"{n - tab['両方不正解']:>6}件 {1 - tab['両方不正解'] / n:>7.1%}"
          f"   {w - wtab['両方不正解']:>6}文字 {1 - wtab['両方不正解'] / w:>7.1%}")

    print("\n## 発話長ごと（件数）")
    print(f"{'発話長':<9}{'件数':>6}{'両方':>7}{'本番だけ':>9}"
          f"{'単純だけ':>9}{'両方×':>8}")
    for band in ("~1秒", "1〜2秒", "2秒以上"):
        c = byband[band]
        t = sum(c.values())
        if not t:
            continue
        print(f"{band:<9}{t:>6}{c['両方正解'] / t:>7.0%}"
              f"{c['本番だけ正解'] / t:>9.0%}{c['単純だけ正解'] / t:>9.0%}"
              f"{c['両方不正解'] / t:>8.0%}")

    for cell in ("本番だけ正解", "単純だけ正解"):
        print(f"\n## {cell} の声紋判定の種別")
        for k, v in bykind[cell].most_common(8):
            print(f"  {k:<10}{v:>5}件")

    print("\n読み方: 「どちらかが正解」が両者の上限。ここが本番単独より高ければ、")
    print("  同じ場所を取り合っているのではなく別の場所を取っている＝組み合わせ")
    print("  る余地がある。種別の内訳が、その余地の正体を示す。")


if __name__ == "__main__":
    main()
