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
import cluster_merge_feasibility as feas  # noqa: E402
import decompose_attribution as dec  # noqa: E402
import segment_boundary_ceiling as seg  # noqa: E402

from das.asr.live._attribution import _VOICEPRINT_RELIABLE_KINDS  # noqa: E402
from das.asr.live._constants import SR, UNSURE_SPEAKER  # noqa: E402
from das.asr.live._recv_loop import _LABEL_ONLY_KINDS  # noqa: E402
from das.asr.live._seat_audio import SeatAudio  # noqa: E402

CACHE = ROOT / "eval" / "_error_anatomy.json"


def compute(run: str, vp) -> list[dict] | None:
    """本番と同じ順序で1ラン流し、発話ごとの結末と属性を返す."""
    loaded = dec.load_run(run)
    wav_path = ROOT / "transcripts" / f"{run}.wav"
    if loaded is None or not wav_path.exists():
        return None
    utts, code_by_ms = loaded
    rows = [(u, code_by_ms.get(u["ms"])) for u in utts]
    rows = [(u, c) for u, c in rows if c in dec.GT_CODES
            and not dec._BACKCHANNEL_RE.match(str(u["_text"]).strip())]
    if not rows or not any(u.get("final_key") is not None for u, _ in rows):
        return None
    rows.sort(key=lambda r: int(r[0]["ms"]))
    pcm = feas.read_wav(wav_path)

    # またぎ判定のための GT 区間
    gt_path = ROOT / "eval" / f"gt_{run}.json"
    conv = json.loads(gt_path.read_text(encoding="utf-8")).get("transplanted_from")
    spans = seg.gt_spans(conv) if conv else []

    seat = SeatAudio(vp)
    pick: dict[int, str] = {}
    for u, _c in rows:
        a, b = int(u["ms"]), int(u.get("end") or u["ms"])
        wav = pcm[int(a / 1000 * SR):int(b / 1000 * SR)]
        final = str(u["final_key"])
        kind = u.get("kind")
        if final != UNSURE_SPEAKER and kind == "蓄積中" and not dec.endorsed(u):
            final = UNSURE_SPEAKER
        if kind in _LABEL_ONLY_KINDS or (final == UNSURE_SPEAKER
                                         and str(u.get("key")) != UNSURE_SPEAKER):
            got = seat.nearest(wav)
            if got is not None:
                pick[int(u["ms"])] = got[0]
        elif final != UNSURE_SPEAKER and kind in _VOICEPRINT_RELIABLE_KINDS:
            seat.observe(final, wav)

    def _final(u):
        # 規則は eval/_pipeline.resolved_key に一本化（書き写すとずれる）
        return pipe.resolved_key(u, pick.get(int(u["ms"])))

    _a, m = _gtlib.best_mapping([(_final(u), c) for u, c in rows],
                                dec.GT_CODES, unsure=UNSURE_SPEAKER)
    t0 = int(rows[0][0]["ms"])
    out = []
    for u, code in rows:
        a, b = int(u["ms"]), int(u.get("end") or u["ms"])
        f = _final(u)
        by_sp: Counter = Counter()
        for x, y, sp in spans:
            ov = min(b, y) - max(a, x)
            if ov > 0:
                by_sp[sp] += ov
        dur = sum(by_sp.values()) or max(0, b - a)
        top = by_sp.most_common(1)[0][1] if by_sp else dur
        out.append({
            "run": run, "ms": a, "dur_ms": max(0, b - a),
            "elapsed_s": (a - t0) / 1000,
            "ov": bool(u.get("ov")), "gt": code,
            "outcome": ("未確定" if f == UNSURE_SPEAKER
                        else "正解" if m.get(f) == code else "誤帰属"),
            "straddle": bool(len(by_sp) > 1 and (dur - top) / dur > 0.10),
            "kind": u.get("kind"),
        })
    return out


def _slice(rows, key, label):
    tab: dict = {}
    for r in rows:
        k = key(r)
        d = tab.setdefault(k, Counter())
        d[r["outcome"]] += 1
        d["n"] += 1
    print(f"\n## {label}")
    print(f"{'区分':<16}{'発話':>6}{'正解':>8}{'誤帰属':>8}{'未確定':>8}")
    for k in sorted(tab):
        d = tab[k]
        n = d["n"]
        print(f"{k!s:<16}{n:>6}{d['正解'] / n:>8.0%}"
              f"{d['誤帰属'] / n:>8.0%}{d['未確定'] / n:>8.0%}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="2026-07-20")
    p.add_argument("--report", action="store_true", help="保存済みを読んで切るだけ")
    p.add_argument("--model", default="redimnet")
    args = p.parse_args(argv)

    if args.report and CACHE.exists():
        rows = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        from das.asr.live._voice_profiles import VoiceProfiles
        vp = VoiceProfiles(model=args.model)
        runs = [r for r in sorted(dec.discover()) if r.startswith(args.prefix)]
        rows = [x for r in runs for x in (compute(r, vp) or [])]
        CACHE.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"# {len(rows)}件を {CACHE} に保存")

    n = len(rows)
    c = Counter(r["outcome"] for r in rows)
    print(f"\n# 全体（{n}発話）  正解 {c['正解'] / n:.1%}"
          f" / 誤帰属 {c['誤帰属'] / n:.1%} / 未確定 {c['未確定'] / n:.1%}")

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

    # 遡及訂正の見積り: 序盤の誤りが「後半の参照」で直るなら何pt返るか
    early = [r for r in rows if r["elapsed_s"] < 120]
    if early:
        bad = sum(1 for r in early if r["outcome"] != "正解")
        print(f"\n# 遡及訂正の上限: 開始2分以内の非正解は {bad}件"
              f"（全体の {bad / n:.1%}）。ここを全部直せば正解は最大 +{bad / n:.1%}")


if __name__ == "__main__":
    main()
