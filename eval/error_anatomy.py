#!/usr/bin/env python3
"""いま残っている誤りを、あらゆる断面で切って「伸びしろ」を探す.

これまで見てきたのは「原因の層」（どの kind か、席の割当てか）だけだった。
ここでは**まだ見ていない断面**を切る:

  経過時間     セッション開始からの時刻。序盤に偏るなら、席の参照が育った後に
               **過去の発話を貼り直す**（遡及訂正）という手が効く。いまは一度
               決めた帰属を二度と見直していない
  発話長       短い発話に偏るなら、短発話専用の扱い（保留して後で決める等）
  重なり       重なり発話（diag の ``ov``）に偏るなら、重なりの扱いが本丸
  話者         特定の話者に偏るなら、その人の登録・音量・座席の問題
  またぎ       Sonioxの区切りが GT の話者境界をまたいでいる発話。ここは
               **どの答えでも一部が誤り**になるので、精度の上限そのもの

判定は本番と同じ順序で1回だけ流し、結果を JSON に落としてから切る
（埋め込みが重いので、断面を増やすたびに再計算しない）。

使い方:
    uv run python eval/error_anatomy.py --prefix 2026-07-20          # 計算して保存
    uv run python eval/error_anatomy.py --report                     # 保存済みを切る
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import _pipeline as pipe  # noqa: E402
import _textgt  # noqa: E402
import decompose_attribution as dec  # noqa: E402
import segment_boundary_ceiling as seg  # noqa: E402

from das.asr.live._constants import UNSURE_SPEAKER  # noqa: E402

CACHE = ROOT / "eval" / "_error_anatomy.json"


def compute(run: str, vp, align: str = "text") -> list[dict] | None:
    """本番と同じ順序で1ラン流し、発話ごとの結末と属性を返す.

    再生・遡及訂正・採点は `_pipeline` に一本化してある（書き写すとずれる）。
    `align` は正解の当て方で、既定は文章の一致——時間の重なりだと重なりの多い
    場面が丸ごと落ち、残った易しい側だけを見ることになる（§34）。
    """
    data = pipe.replay_seats(run, vp, align=align)
    if data is None:
        return None
    steps = data["steps"]
    final = pipe.apply_schedule(steps)
    pairs = [(f, st["code"]) for f, st in zip(final, steps, strict=True)]
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=UNSURE_SPEAKER)

    # またぎ判定のための GT 区間（Sonioxの区切りが話者境界をまたいでいるか）
    conv = _textgt.source_of(run)
    spans = seg.gt_spans(conv) if conv else []

    out = []
    for f, st in zip(final, steps, strict=True):
        u = st["utt"]
        a = int(u["ms"])
        b = int(u.get("end") or a)
        by_sp: Counter = Counter()
        for x, y, sp in spans:
            ov = min(b, y) - max(a, x)
            if ov > 0:
                by_sp[sp] += ov
        dur = sum(by_sp.values()) or max(0, b - a)
        top = by_sp.most_common(1)[0][1] if by_sp else dur
        out.append({
            "run": run, "ms": a, "dur_ms": max(0, b - a),
            "elapsed_s": st["elapsed"],
            "chars": len(str(u.get("_text") or "")),
            "ov": bool(u.get("ov")), "gt": st["code"],
            "outcome": ("未確定" if f == UNSURE_SPEAKER
                        else "正解" if m.get(f) == st["code"] else "誤帰属"),
            "straddle": bool(len(by_sp) > 1 and (dur - top) / dur > 0.10),
            "kind": u.get("kind"),
            "src": u.get("src"),
        })
    return out


def _slice(rows, key, label):
    """断面ごとに、件数と文字数の両方で内訳を出す.

    件数だけだと短い相づちに数字が支配され、文字数だけだと1回の長い取り違えが
    大きく見える。両方を並べて初めて「どこを直すと読み手に効くか」が分かる。
    """
    tab: dict = {}
    for r in rows:
        k = key(r)
        d = tab.setdefault(k, Counter())
        d[r["outcome"]] += 1
        d["n"] += 1
        d["w"] += r.get("chars", 0)
        d["w" + r["outcome"]] += r.get("chars", 0)
    print(f"\n## {label}")
    print(f"{'区分':<14}{'発話':>6}{'正解':>7}{'誤帰属':>7}{'未確定':>7}"
          f"{'   ':>3}{'文字':>7}{'正解':>7}{'誤帰属':>7}{'未確定':>7}")
    for k in sorted(tab):
        d = tab[k]
        n, w = d["n"], max(1, d["w"])
        print(f"{k!s:<14}{n:>6}{d['正解'] / n:>7.0%}"
              f"{d['誤帰属'] / n:>7.0%}{d['未確定'] / n:>7.0%}{'   ':>3}"
              f"{w:>7}{d['w正解'] / w:>7.0%}{d['w誤帰属'] / w:>7.0%}"
              f"{d['w未確定'] / w:>7.0%}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="2026-07-20")
    p.add_argument("--report", action="store_true", help="保存済みを読んで切るだけ")
    p.add_argument("--model", default="redimnet")
    p.add_argument("--align", default="text",
                   choices=("text", "time"),
                   help="正解の当て方（既定: 文章の一致。§34）")
    args = p.parse_args(argv)

    if args.report and CACHE.exists():
        rows = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        from das.asr.live._voice_profiles import VoiceProfiles
        vp = VoiceProfiles(model=args.model)
        runs = [r for r in sorted(dec.discover()) if r.startswith(args.prefix)]
        rows = [x for r in runs for x in (compute(r, vp, args.align) or [])]
        CACHE.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"# {len(rows)}件を {CACHE} に保存")

    n = len(rows)
    c = Counter(r["outcome"] for r in rows)
    w = sum(r.get("chars", 0) for r in rows) or 1
    wc = Counter()
    for r in rows:
        wc[r["outcome"]] += r.get("chars", 0)
    print(f"\n# 全体（{n}発話・{w}文字）")
    print(f"#   件数  正解 {c['正解'] / n:.1%} / 誤帰属 {c['誤帰属'] / n:.1%}"
          f" / 未確定 {c['未確定'] / n:.1%}")
    print(f"#   文字  正解 {wc['正解'] / w:.1%} / 誤帰属 {wc['誤帰属'] / w:.1%}"
          f" / 未確定 {wc['未確定'] / w:.1%}")

    def bucket_time(r):
        e = r["elapsed_s"]
        for lo in (60, 120, 300, 600):
            if e < lo:
                return f"~{lo // 60}分"
        return "10分以降"

    def bucket_dur(r):
        d = r["dur_ms"] / 1000
        for lo in (0.5, 1, 2, 4):
            if d < lo:
                return f"~{lo}秒"
        return "4秒以上"

    _slice(rows, bucket_time, "経過時間（席の参照が育つまでの序盤に偏るか）")
    _slice(rows, bucket_dur, "発話長")
    _slice(rows, lambda r: "重なり" if r["ov"] else "重なりなし", "重なり")
    _slice(rows, lambda r: r["gt"], "GT話者")
    _slice(rows, lambda r: "またぎ" if r["straddle"] else "またぎなし",
           "Sonioxの区切りが話者境界をまたいでいるか（精度の上限）")
    _slice(rows, lambda r: r.get("kind") or "（なし）", "声紋判定の種別")
    _slice(rows, lambda r: r.get("src") or "（なし）", "どの経路で決まったか")

    # 遡及訂正の見積り: 序盤の誤りが「後半の参照」で直るなら何pt返るか
    early = [r for r in rows if r["elapsed_s"] < 120]
    if early:
        bad = sum(1 for r in early if r["outcome"] != "正解")
        print(f"\n# 遡及訂正の上限: 開始2分以内の非正解は {bad}件"
              f"（全体の {bad / n:.1%}）。ここを全部直せば正解は最大 +{bad / n:.1%}")


if __name__ == "__main__":
    main()
