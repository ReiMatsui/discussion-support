#!/usr/bin/env python3
"""参加人数を決めていないと精度がどこまで落ちるかを測る.

**なぜ要るのか**: いまの精度は「参加人数が分かっている」ことに寄りかかって
いる。席の割当て（§27）の正当化そのものが人数の情報から来ている——クラスタ
埋め込み同士の比較で同一人物と別人を分ける絶対しきい値は存在しない（別人でも
0.89 出る）ので「新しい人か既存の誰かか」という**開集合**の判定はできない。
できるのは「席を持つN人のうち誰か」という**閉集合**の割当てだけで、その N は
参加人数の設定から来ている。

したがって「人数を決めなかったら」の落ち幅は、実装の主張がどれだけ人数の
情報に依存しているかを直接示す。3条件を同じ測り方で並べる。

    決めている        いまの本番（統一席ルール＋席の割当て）
    上限だけ外す      席上限を掛けない。声紋が分裂したぶん参加者が増える。
                      席の割当ては残す（正当化は失われるが動きはする）
    人数を使わない    席の割当てもしない。人数の情報を一切使わない条件

**pyannote 側の人数は外せない**。`max_speakers` はクラスタリング自体に渡って
おり記録には焼き付いているので、そこまで含めた条件は録音の流し直し（API課金）
が要る。ここで測るのは**下流（こちらの実装）が人数にどれだけ依存しているか**
である。声紋層の自動登録の上限も同じ理由で外せないが、該当する発話は1788件中
8件なので結論を左右しない。

**話者数**の列は、出力に現れた別人扱いの数。正解は3人なので、ここが増える
こと自体が誤りである（7人に見える議事録は、たとえ主要な発話が当たっていても
読めない）。採点は最適1:1対応なので、4人目以降は必ず誤帰属になる。

使い方:
    uv run python eval/roster_dependency.py --prefix 2026-07-20
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

# (表示名, 席上限を掛けるか, 席の割当てを使うか)
CONDITIONS = [
    ("決めている（本番）", True, True),
    ("上限だけ外す", False, True),
    ("人数を使わない", False, False),
]
MIN_UTTS = 5    # 「人として見えている」とみなす最低発話数


def measure(run: str, vp, *, cap: bool, seats: bool) -> dict | None:
    data = pipe.replay_seats(run, vp, align="text", cap=cap, seats=seats)
    if data is None:
        return None
    steps = data["steps"]
    final = pipe.apply_schedule(steps)
    pairs = [(f, st["code"]) for f, st in zip(final, steps, strict=True)]
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=UNSURE_SPEAKER)
    rows = [{"chars": len(str(st["utt"].get("_text") or "")),
             "outcome": ("未確定" if f == UNSURE_SPEAKER
                         else "正解" if m.get(f) == st["code"] else "誤帰属")}
            for f, st in zip(final, steps, strict=True)]
    seen: dict[str, int] = {}
    for f in final:
        if f != UNSURE_SPEAKER:
            seen[f] = seen.get(f, 0) + 1
    return {"rows": rows, "keys": len(seen),
            "keys_visible": sum(1 for v in seen.values() if v >= MIN_UTTS)}


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

    got: dict[str, list[dict]] = {name: [] for name, _c, _s in CONDITIONS}
    for run in runs:
        for name, cap, seats in CONDITIONS:
            r = measure(run, vp, cap=cap, seats=seats)
            if r:
                got[name].append(r)
        print(f"# {run} 済み", flush=True)
    if not any(got.values()):
        raise SystemExit("# 測れるランが無い")

    print(f"\n## 参加人数の情報をどこまで使うか（{len(runs)}本・正解は3人）")
    print(f"{'条件':<20}{'件数':>6}{'正解':>8}{'誤帰属':>8}{'未確定':>8}"
          f"{'  ':>2}{'文字:正解':>9}{'誤帰属':>8}{'未確定':>8}"
          f"{'  ':>2}{'話者数':>7}{'うち5発話以上':>13}")
    for name, _c, _s in CONDITIONS:
        per = got[name]
        if not per:
            continue
        rows = [x for r in per for x in r["rows"]]
        c = _rates(rows, lambda r: 1)
        w = _rates(rows, lambda r: r["chars"])
        keys = sum(r["keys"] for r in per) / len(per)
        vis = sum(r["keys_visible"] for r in per) / len(per)
        print(f"{name:<20}{len(rows):>6}{c[0]:>8.1%}{c[1]:>8.1%}{c[2]:>8.1%}"
              f"{'  ':>2}{w[0]:>9.1%}{w[1]:>8.1%}{w[2]:>8.1%}"
              f"{'  ':>2}{keys:>7.1f}{vis:>13.1f}")

    print("\n読み方: 話者数は1本あたりの平均で、正解は3人。ここが増えると、")
    print("  主要な発話が当たっていても議事録としては読めなくなる。")
    print("  pyannote に渡す人数は記録に焼き付いているので外せない——ここで")
    print("  測っているのは**下流の実装**が人数にどれだけ寄りかかっているか。")


if __name__ == "__main__":
    main()
