"""ラベル不純×長発話×低確信の門番をフォールバック経路に足すと何が起きるかを測る.

背景（handoff §47）: 2026-07-30 の講義で、未登録の話者（プロファイルも席も
無い）が Soniox ラベルの写像で参加者Bへ吸われた。声紋層は6発話すべてで
「ラベル不純」を出しており、best sim も 0.37〜0.49 と低かった（本物のBの
長発話は 0.5〜0.77）。信号はあったのに受け皿が無かった。

候補ルール: **kind=ラベル不純 かつ chars>=30 かつ best sim<0.5 → 未確定**。
講義では未確定化7件中5件が実際の誤り（+907字回収 / 巻き添え ≤74字）。
このスクリプトは、同じルールを校正済みの実会話（2026-07-20 の9本）に
当てたときの損益を測る——未登録話者がいない場では、正しい発話を未確定へ
落とすコストしか無いはずで、それがどれだけ小さいかを確かめる。

結果（2026-07-30 実測）: 9本のラベル不純1377件のうち述語該当は**0件**。
どの条件でも成績は現行と完全一致（人数あり文字91.5%で正典と一致）。
登録済みの人しかいない場では、長い発話の best sim は 0.5 を割らない。
この結果を受けて本番に採用（規則9, _recv_loop 8b段）。

条件は2つ:
  - 人数あり（cap/seats=True・本番既定）: ラベル不純は席の音声で決め直される。
    ルールはその決め直しの**後に**掛かる（講義でも席の決め直し自体は誤った）
  - 人数なし（cap/seats=False・講義と同じ）: ラベル不純はラベル写像のまま。
    ルールはそこに掛かる

使い方: uv run python eval/impure_lowsim_guard.py [--prefix 2026-07-20]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import _gtlib
import _pipeline as pipe
import decompose_attribution as dec

from das.asr.live._attribution import impure_lowsim
from das.asr.live._constants import (
    IMPURE_LOWSIM_MAX_SIM,
    IMPURE_LOWSIM_MIN_CHARS,
    UNSURE_SPEAKER,
)


def guarded(u: dict) -> bool:
    """門番の述語（本番と共用の impure_lowsim。音声の再計算は不要）."""
    return impure_lowsim(u.get("kind"), u.get("chars"), u.get("sim"))


def measure(run: str, vp, *, cap: bool, seats: bool):
    data = pipe.replay_seats(run, vp, align="text", cap=cap, seats=seats)
    if data is None:
        return None
    steps = data["steps"]
    base = pipe.apply_schedule(steps)
    guard = [UNSURE_SPEAKER if guarded(st["utt"]) else f
             for f, st in zip(base, steps, strict=True)]
    out = {}
    outcomes = {}
    for name, final in (("現行", base), ("+門番", guard)):
        pairs = [(f, st["code"]) for f, st in zip(final, steps, strict=True)]
        _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=UNSURE_SPEAKER)
        rows = [{"chars": len(str(st["utt"].get("_text") or "")),
                 "outcome": ("未確定" if f == UNSURE_SPEAKER
                             else "正解" if m.get(f) == st["code"] else "誤帰属")}
                for f, st in zip(final, steps, strict=True)]
        out[name] = rows
        outcomes[name] = [r["outcome"] for r in rows]
    out["flips"] = [
        {"ms": st["ms"], "chars": int(st["utt"].get("chars") or 0),
         "sim": st["utt"].get("sim"), "was": was}
        for st, b, was in zip(steps, base, outcomes["現行"], strict=True)
        if guarded(st["utt"]) and b != UNSURE_SPEAKER]
    return out


def _rates(rows, weigh):
    tot = sum(weigh(r) for r in rows) or 1
    return tuple(sum(weigh(r) for r in rows if r["outcome"] == k) / tot
                 for k in ("正解", "誤帰属", "未確定"))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="2026-07-20")
    p.add_argument("--model", default="redimnet")
    args = p.parse_args(argv)

    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(model=args.model)
    runs = [r for r in sorted(dec.discover()) if r.startswith(args.prefix)]

    # 人数なしは §44 訂正後の条件（席の決め直しは人数に依存しないので残す）
    for cap, seats, cond in ((True, True, "人数あり"), (False, True, "人数なし")):
        got = {"現行": [], "+門番": []}
        flips = []
        for run in runs:
            r = measure(run, vp, cap=cap, seats=seats)
            if not r:
                continue
            for k in got:
                got[k].extend(r[k])
            flips.extend({**f, "run": run} for f in r["flips"])
        print(f"\n## {cond}（{len(runs)}本, ≥{IMPURE_LOWSIM_MIN_CHARS}字, "
              f"sim<{IMPURE_LOWSIM_MAX_SIM}）")
        print(f"{'条件':<8}{'件:正解':>9}{'誤帰属':>8}{'未確定':>8}"
              f"{'  文字:正解':>11}{'誤帰属':>8}{'未確定':>8}")
        for name, rows in got.items():
            c = _rates(rows, lambda r: 1)
            w = _rates(rows, lambda r: r["chars"])
            print(f"{name:<8}{c[0]:>9.1%}{c[1]:>8.1%}{c[2]:>8.1%}"
                  f"{w[0]:>11.1%}{w[1]:>8.1%}{w[2]:>8.1%}")
        print(f"門番が未確定へ倒した発話: {len(flips)}件 "
              f"(元の内訳: 正解 {sum(1 for f in flips if f['was'] == '正解')} / "
              f"誤帰属 {sum(1 for f in flips if f['was'] == '誤帰属')})")
        for f in flips:
            print(f"  {f['run']} ms={f['ms']} {f['chars']}字 sim={f['sim']}"
                  f" 元={f['was']}")


if __name__ == "__main__":
    main()
