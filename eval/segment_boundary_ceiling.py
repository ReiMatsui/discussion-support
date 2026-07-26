#!/usr/bin/env python3
"""区切りの誤りが帰属精度の上限をどこまで下げているかを測る（§11 選択肢b の根拠）.

いまの帰属は **Soniox の区切り**（＋ラベル）を1単位として「この発話は誰か」を
決める。しかし Soniox の区切りが2人以上の発話にまたがっていると、**どの話者を
選んでも正解にならない**。帰属ロジックをいくら磨いてもここは取れない。

§11 選択肢b は帰属の単位を pyannote の話者区間へ移す構造変更で、狙いはこの
上限そのものを引き上げることにある。着手前に「上限がいくらか」を知りたい。

測り方（新規録音もAPIコストも不要）:

  システムの各発話 [ms, end] を GT のターン境界と突き合わせ、重なる GT 話者を
  時間で数える。重なりが2人以上に割れている発話は**区切りがまたいでいる**。
  その発話は現行の単位では最良でも「多数派の話者」しか取れないので、
  少数派ぶんの時間は原理的に失われる。

出力:

  またぎ率      またいでいる発話の割合（件数と、実発話時間に占める割合）
  取りこぼし時間 またいだ発話の少数派側の時間。**これが上限の目減り分**
  誤帰属との関係 いま誤帰属になっている発話のうち、またぎが原因のもの

使い方:
    uv run python eval/segment_boundary_ceiling.py --prefix 2026-07-20
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import decompose_attribution as dec  # noqa: E402


def gt_spans(conv: str) -> list[tuple[int, int, str]]:
    """参照（GT定義セッション）の (開始, 終了, 話者) を返す."""
    path = ROOT / "transcripts" / f"{conv}.turns.jsonl"
    if not path.exists():
        return []
    out = []
    for t in _gtlib.read_jsonl(path):
        a = int(t["ms"])
        b = int(t.get("end_ms") or a)
        sp = str(t.get("speaker", "")).strip()
        if b > a and sp:
            out.append((a, b, sp))
    out.sort()
    return out


def analyze(run: str) -> dict | None:
    import json
    gt_path = ROOT / "eval" / f"gt_{run}.json"
    if not gt_path.exists():
        return None
    conv = json.loads(gt_path.read_text(encoding="utf-8")).get("transplanted_from")
    if not conv:
        return None
    spans = gt_spans(conv)
    if not spans:
        return None
    loaded = dec.load_run(run)
    if loaded is None:
        return None
    utts, code_by_ms = loaded
    rows = [(u, code_by_ms.get(u["ms"])) for u in utts]
    rows = [(u, c) for u, c in rows if c in dec.GT_CODES
            and not dec._BACKCHANNEL_RE.match(str(u["_text"]).strip())]
    if not rows:
        return None
    _a, mapping = _gtlib.best_mapping(
        [(str(u["final_key"]), c) for u, c in rows], dec.GT_CODES,
        unsure=dec.UNSURE_SPEAKER)

    n = straddle = 0
    total_ms = lost_ms = 0
    wrong = wrong_straddle = 0
    for u, code in rows:
        a, b = int(u["ms"]), int(u.get("end") or u["ms"])
        if b <= a:
            continue
        by_sp: Counter = Counter()
        for x, y, sp in spans:
            ov = min(b, y) - max(a, x)
            if ov > 0:
                by_sp[sp] += ov
        if not by_sp:
            continue
        n += 1
        dur = sum(by_sp.values())
        total_ms += dur
        top = by_sp.most_common(1)[0][1]
        minority = dur - top
        # 少数派が1割を超えるなら「またいでいる」とみなす（端の数十msは除く）
        is_straddle = len(by_sp) > 1 and minority / dur > 0.10
        if is_straddle:
            straddle += 1
            lost_ms += minority
        final = str(u["final_key"])
        if final != dec.UNSURE_SPEAKER and mapping.get(final) != code:
            wrong += 1
            wrong_straddle += is_straddle
    if not n:
        return None
    return {"run": run, "n": n, "straddle": straddle,
            "lost_ratio": lost_ms / total_ms if total_ms else 0.0,
            "wrong": wrong, "wrong_straddle": wrong_straddle}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default=None)
    args = p.parse_args(argv)
    runs = [r for r in sorted(dec.discover())
            if not args.prefix or r.startswith(args.prefix)]
    out = [r for r in (analyze(x) for x in runs) if r]
    if not out:
        raise SystemExit("# 参照本文つきのラン（transplanted_from あり）が必要")
    print("# Soniox の区切りが2人以上にまたがっている割合と、その代償")
    print(f"{'run':<20}{'発話':>6}{'またぎ':>7}{'率':>7}"
          f"{'失う時間':>9}{'誤帰属':>7}{'うちまたぎ':>10}")
    for r in out:
        print(f"{r['run']:<20}{r['n']:>6}{r['straddle']:>7}"
              f"{r['straddle'] / r['n']:>7.0%}{r['lost_ratio']:>9.1%}"
              f"{r['wrong']:>7}{r['wrong_straddle']:>10}")
    n = sum(r["n"] for r in out)
    st = sum(r["straddle"] for r in out)
    w = sum(r["wrong"] for r in out)
    ws = sum(r["wrong_straddle"] for r in out)
    lost = sum(r["lost_ratio"] for r in out) / len(out)
    print(f"{'合計':<20}{n:>6}{st:>7}{st / n:>7.0%}{lost:>9.1%}"
          f"{w:>7}{ws:>10}")
    print("\n読み方:")
    print(f"  またぎ {st / n:.0%} の発話は、どの話者を選んでも一部が誤りになる。")
    print(f"  現在の誤帰属 {w} 件のうち {ws} 件"
          f"（{ws / w:.0%}）がまたぎのある発話で起きている。" if w else "")
    print("  区切りを話者区間に合わせれば、この分は帰属ロジックを変えずに減る。")


if __name__ == "__main__":
    main()
