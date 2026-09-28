#!/usr/bin/env python3
"""事前登録あり／なしの再生ランを、経過時間帯ごとに文字数重みで採点して並べる.

    uv run python eval/enroll_breakdown.py --stamp 20260928_160809 --minutes 4
    uv run python eval/enroll_breakdown.py --gt eval/gt_rehacq_pA4K_none.json \
        --none rehacq_pA4K_none --enroll rehacq_pA4K_enroll --minutes 10

前者は run_chiba_batch.py の一括ラン。後者は任意の 2 ラン（YouTube・ゼミ録音など、
同じ音声を 2 条件で流したもの）で、GT は eval/annotate.py で片方に付けたもの。

run_chiba_batch.py が作った transcripts/<会話>_m<分>_<none|enroll>_<日時>.turns.jsonl と
eval/gt_<会話>m<分>.json を読み、会話ごと・時間帯ごとに 正解／誤り／未確定 の
文字数割合を出す。相槌は除く。システムのラベルと正解話者は文字数で最適 1:1 対応。
最後に全会話を合算した表を出す（docs/research/enroll_eval_2026-09-28.md の元）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import _gtlib  # noqa: E402
from _gtlib import gt_timeline, read_jsonl  # noqa: E402

BC = _gtlib.load_backchannel_re()


def best_assignment(cnt: dict[str, Counter], gts: list[str]) -> dict[str, str]:
    """システムのラベルと正解話者の、文字数が最大になる 1:1 対応.

    ラベル数×正解話者の部分集合の DP（ラベル 10 個・話者 9 人でも一瞬）。
    permutations の総当たりは 3 人までしか現実的でないので、4 人以上の
    会議（ゼミ録音など）に備えて置き換えた。
    """
    labels = [s for s in cnt if sum(cnt[s].values()) > 0]
    g_n = len(gts)
    # dp[mask] = (最大文字数, 対応) で、ラベルを 1 つずつ見ていく
    dp: dict[int, tuple[int, dict[str, str]]] = {0: (0, {})}
    for s in labels:
        nxt = dict(dp)
        for mask, (v, m) in dp.items():
            for j in range(g_n):
                if mask & (1 << j):
                    continue
                gain = cnt[s][gts[j]]
                if gain <= 0:
                    continue
                nm = mask | (1 << j)
                cand = v + gain
                if cand > nxt.get(nm, (-1, {}))[0]:
                    nxt[nm] = (cand, {**m, s: gts[j]})
        dp = nxt
    return max(dp.values(), key=lambda x: x[0])[1]


def score(conv: str, cond: str, stamp: str, minutes: float, wins):
    gt_path = ROOT / "eval" / f"gt_{conv}m{minutes:g}.json"
    session = f"{conv}_m{minutes:g}_{cond}_{stamp}"
    return score_session(gt_path, session, wins)


def score_session(gt_path: Path, session: str, wins):
    """GT（annotate.py / prep_chiba の labels 形式）で、任意のセッションを時間帯別に採点する."""
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    gtt = read_jsonl(ROOT / "transcripts" / f"{gt['session']}.turns.jsonl")
    tl = gt_timeline(gtt, gt["labels"])
    gts = sorted(tl, key=lambda c: int(c[1:]))
    turns = read_jsonl(ROOT / "transcripts" / f"{session}.turns.jsonl")
    rows = []
    for t in turns:
        code = _gtlib.gt_code_by_timeline(t["ms"], t["end_ms"], tl)
        if code in gts and not BC.match(t.get("text", "").strip()):
            rows.append((t["ms"] / 60000, t["speaker"], code, len(t.get("text", ""))))
    cnt: dict[str, Counter] = defaultdict(Counter)
    for _, s, g, n in rows:
        if s != "未確定":
            cnt[s][g] += n
    best = best_assignment(cnt, gts)
    out = {}
    for a, b in wins:
        seg = [r for r in rows if a <= r[0] < b]
        n = sum(r[3] for r in seg)
        ok = sum(r[3] for r in seg if best.get(r[1]) == r[2])
        un = sum(r[3] for r in seg if r[1] == "未確定")
        out[(a, b)] = (ok, n - ok - un, un, n)
    return out


def pct(ok, wr, un, n) -> str:
    return f"{ok / n:.0%}/{wr / n:.0%}/{un / n:.0%}" if n else "-"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stamp", default=None, help="run_chiba_batch.py の日時（セッション名の末尾）")
    p.add_argument("--minutes", type=float, default=4.0)
    p.add_argument("--convs", default=",".join(f"chiba{i:02d}32" for i in range(1, 13)))
    p.add_argument("--gt", default=None, help="任意の 2 ラン用: GT の json（annotate.py で作ったもの）")
    p.add_argument("--none", default=None, help="任意の 2 ラン用: 登録なしのセッション名")
    p.add_argument("--enroll", default=None, help="任意の 2 ラン用: 登録ありのセッション名")
    a = p.parse_args()
    m = math.ceil(a.minutes)   # 11.3 分の録音は --minutes 11.3 で 0-12 分まで見る
    wins = [(i, i + 1) for i in range(m)] + [(0, m)]
    conds = ("none", "enroll")

    if a.gt:
        if not (a.none and a.enroll):
            sys.exit("--gt には --none と --enroll のセッション名が要ります")
        res = {"none": score_session(Path(a.gt), a.none, wins),
               "enroll": score_session(Path(a.gt), a.enroll, wins)}
        print("| 経過時間 | 条件 | 正解 | 誤り | 未確定 | 文字数 |")
        print("|---|---|---|---|---|---|")
        for w in wins:
            for cond in conds:
                ok, wr, un, n = res[cond][w]
                label = "登録なし" if cond == "none" else "登録あり"
                if n:
                    print(f"| {w[0]}-{w[1]} 分 | {label} | {ok / n:.1%} | {wr / n:.1%} | {un / n:.1%} | {n} |")
        print("\n（正解／誤り／未確定 の文字数割合。相槌除く）")
        return
    if not a.stamp:
        sys.exit("--stamp（一括ラン）か --gt --none --enroll（任意の 2 ラン）を指定してください")
    convs = [c for c in a.convs.split(",") if c]
    agg = {c: {w: [0, 0, 0, 0] for w in wins} for c in conds}

    print("| 会話 | 登録なし 0-1分 | 登録なし 0-%d分 | 登録あり 0-1分 | 登録あり 0-%d分 |" % (m, m))
    print("|---|---|---|---|---|")
    for conv in convs:
        cells = [conv]
        for cond in conds:
            sc = score(conv, cond, a.stamp, a.minutes, wins)
            for w, v in sc.items():
                for i in range(4):
                    agg[cond][w][i] += v[i]
            cells.append(pct(*sc[(0, 1)]))
            cells.append(pct(*sc[(0, m)]))
        print("| " + " | ".join(cells) + " |")
    print("\n（正解／誤り／未確定 の文字数割合。相槌除く）\n")
    print("| 経過時間 | 条件 | 正解 | 誤り | 未確定 | 文字数 |")
    print("|---|---|---|---|---|---|")
    for w in wins:
        for cond in conds:
            ok, wr, un, n = agg[cond][w]
            label = "登録なし" if cond == "none" else "登録あり"
            print(f"| {w[0]}-{w[1]} 分 | {label} | {ok / n:.1%} | {wr / n:.1%} | {un / n:.1%} | {n} |")


if __name__ == "__main__":
    main()
