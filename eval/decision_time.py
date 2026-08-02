"""介入時点(遡及訂正が掛かる前)の帰属精度を測る（§49.15）.

**問い**: 正本の成績(91.5%/86.8%)は遡及訂正込みの最終値だが、AI介入は
リアルタイムに判定するので**遡及前の帰属**で意思決定している。介入層が
実際に見ている精度はいくつか。

再生・採点は `_pipeline` の同じ実装を使い、遡及の予定表だけを空にする
（apply_schedule(schedule=(), interval=inf) = 各発話の flush 時点の判定）。
最終値との差が「議事録では直るが介入には間に合わない」量になる。
序盤の帯別も出す——介入の名指しをいつから解禁できるかの判断材料（§28）。

使い方（クラウドで可・APIキー不要）:
  uv run python eval/decision_time.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import _pipeline as pipe  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._constants import UNSURE_SPEAKER  # noqa: E402

GROUPS = (("校正9本", "2026-07-20"), ("持ち越し4本", "2026-07-16"))
BANDS = ((60, "0-1分"), (120, "1-2分"), (300, "2-5分"), (10**9, "5分以降"))


def outcomes(steps, final):
    pairs = [(f, st["code"]) for f, st in zip(final, steps, strict=True)]
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=UNSURE_SPEAKER)
    out = []
    for f, st in zip(final, steps, strict=True):
        chars = len(str(st["utt"].get("_text") or ""))
        oc = ("未確定" if f == UNSURE_SPEAKER
              else "正解" if m.get(f) == st["code"] else "誤帰属")
        out.append((oc, chars, st["elapsed"]))
    return out


def tally(rows):
    w = sum(c for _o, c, _e in rows) or 1
    wc = Counter()
    for o, c, _e in rows:
        wc[o] += c
    return wc["正解"] / w, wc["誤帰属"] / w, wc["未確定"] / w, w


def main() -> None:
    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(model="redimnet")
    for label, prefix in GROUPS:
        now_rows, full_rows = [], []
        for run in [r for r in sorted(dec.discover()) if r.startswith(prefix)]:
            data = pipe.replay_seats(run, vp, align="text")
            if data is None:
                continue
            steps = data["steps"]
            f_now = pipe.apply_impure_lowsim_guard(
                pipe.apply_schedule(steps, schedule=(),
                                    interval=float("inf")), steps)
            f_full = pipe.apply_impure_lowsim_guard(
                pipe.apply_schedule(steps), steps)
            now_rows += outcomes(steps, f_now)
            full_rows += outcomes(steps, f_full)
        if not now_rows:
            print(f"## {label}: データなし")
            continue
        g1, e1, u1, w = tally(now_rows)
        g2, e2, u2, _ = tally(full_rows)
        print(f"## {label}（{w}文字・文字ベース）")
        print(f"  介入時点(遡及前): 正解 {g1:.1%} / 誤帰属 {e1:.1%} / 未確定 {u1:.1%}")
        print(f"  最終(遡及後)    : 正解 {g2:.1%} / 誤帰属 {e2:.1%} / 未確定 {u2:.1%}")
        print(f"  差(遡及の寄与)  : 正解 {g2 - g1:+.1%}")
        print("  介入時点の帯別(経過時間):")
        lo = 0
        for lim, name in BANDS:
            band = [r for r in now_rows if lo <= r[2] < lim]
            lo = lim
            if band:
                g, e, u, bw = tally(band)
                print(f"    {name:7s}: 正解 {g:5.1%} / 誤帰属 {e:5.1%}"
                      f" / 未確定 {u:5.1%}（{bw}字）")
        print()


if __name__ == "__main__":
    main()
