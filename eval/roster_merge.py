#!/usr/bin/env python3
"""人数未設定のとき、上流のキーを「同じ人」で統合したらどうなるか（§40 の実装案）.

§40 で測ったのは「帰属をクラスタで置き換える」形（文字 86.4%）だった。ところが
それを本番に入れると、クラスタに名前を付けるために**6番目のキー名前空間**が
要る。`_speaker_keys.py` 冒頭が「5空間の統合は帰属の中核を6モジュール横断で
書き換える変更で、失敗すると話者の帰属が微妙にずれる」と警告している場所で、
触りたくない。

素直な形はこうである。人数未設定の問題は上流のキーが増えて（実測5.9個）
**まとまらない**ことなので、必要なのは新しいキーを作ることではなく
**どのキーが同じ人かを決めて統合すること**。統合は `SessionState.rekey` という
既存の単一入口があり、表示ラベルの詰め直しも席の参照の付け替えもそこから
伝播する。新しい名前空間も表示ロジックも要らない。

ただし**期待できる効果が違う**。置き換えは上流の判定を捨てるが、統合は残す
——上流が間違えた発話はそのまま間違いのままになる。代わりに、統合で席の参照が
綺麗になるぶん「席の音声で決め直す」経路が効く。どちらが勝つかは測らないと
分からない。それがこのスクリプトの目的。

やり方（因果的）:

  1. 2秒以上の発話の声紋を溜める（`_utt_embeddings` の保存を使う）
  2. 30秒ごとに、それまでの材料をまとめ直す（`roster_free.cluster`。人数は推定）
  3. 各まとまりの中で、上流のキーのうち**いちばん材料の多いもの**を代表に決め、
     残りをそれへ統合する（＝`rekey` を呼ぶのと同じこと）
  4. 以後の発話には、その時点の統合表を当ててから今日の規則を通す

使い方:
    uv run python eval/roster_merge.py --prefix 2026-07-20
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import _pipeline as pipe  # noqa: E402
import _utt_embeddings as ue  # noqa: E402
import decompose_attribution as dec  # noqa: E402
import roster_free as rf  # noqa: E402

from das.asr.live._constants import UNSURE_SPEAKER  # noqa: E402

MIN_MS = 2000        # 材料にする発話の最短の長さ（§40.2: 2秒以上で等誤り率6.1%）
EVERY_SEC = 30.0     # まとめ直す間隔
MIN_ANCHORS = 4      # これだけ材料が溜まるまでは統合しない


def merge_timeline(run: str) -> list[tuple[int, dict[str, str]]] | None:
    """(時刻, その時点の統合表) を時刻順に返す。統合表は 旧キー -> 代表キー.

    材料（長い発話）とその上流キーは記録から取る。上流キーが未確定の材料は
    代表になれないので除く。
    """
    g = ue.load(run)
    rows = pipe.gt_rows(run, align="text")
    if g is None or rows is None:
        return None
    key_by_ms = {int(u["ms"]): str(u.get("key")) for u, _c in rows}
    order = np.argsort(g["ms"])
    out: list[tuple[int, dict[str, str]]] = []
    next_at = 0.0
    t0 = float(g["ms"][order[0]])
    for i in order:
        ms = int(g["ms"][i])
        if (ms - t0) / 1000.0 < next_at:
            continue
        next_at = (ms - t0) / 1000.0 + EVERY_SEC
        sel = [j for j in order
               if g["ms"][j] <= ms and g["dur_ms"][j] >= MIN_MS
               and key_by_ms.get(int(g["ms"][j]), UNSURE_SPEAKER)
               != UNSURE_SPEAKER]
        if len(sel) < MIN_ANCHORS:
            continue
        lab = rf.cluster(g["emb"][sel])
        table: dict[str, str] = {}
        for c in np.unique(lab):
            keys = Counter(key_by_ms[int(g["ms"][j])]
                           for j, cc in zip(sel, lab, strict=True) if cc == c)
            rep = keys.most_common(1)[0][0]
            for k in keys:
                if k != rep:
                    table[k] = rep
        out.append((ms, table))
    return out


def _resolve(table: dict[str, str], key: str) -> str:
    """統合表を辿って代表キーにする（連鎖しても終端まで行く）."""
    seen = set()
    while key in table and key not in seen:
        seen.add(key)
        key = table[key]
    return key


def measure(run: str, vp, *, merge: bool) -> tuple | None:
    timeline = merge_timeline(run) if merge else []
    if merge and timeline is None:
        return None

    def rekey(ms: int, key: str) -> str:
        if key == UNSURE_SPEAKER:
            return key
        table: dict[str, str] = {}
        for at, t in timeline:      # その時刻までに決まっている統合を当てる
            if at > ms:
                break
            table = t
        return _resolve(table, key)

    data = pipe.replay_seats(run, vp, align="text", cap=False, seats=True,
                             rekey=rekey if merge else None)
    if data is None:
        return None
    steps = data["steps"]
    final = pipe.apply_schedule(steps)
    pairs = [(f, st["code"]) for f, st in zip(final, steps, strict=True)]
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=UNSURE_SPEAKER)
    w = [len(str(st["utt"].get("_text") or "")) for st in steps]
    n, tot = len(pairs), sum(w) or 1
    ok = sum(1 for f, c in pairs if m.get(f) == c)
    un = sum(1 for f, _ in pairs if f == UNSURE_SPEAKER)
    wok = sum(x for (f, c), x in zip(pairs, w, strict=True) if m.get(f) == c)
    wun = sum(x for (f, _), x in zip(pairs, w, strict=True)
              if f == UNSURE_SPEAKER)
    ident = len({f for f in final if f != UNSURE_SPEAKER})
    return (ok / n, (n - ok - un) / n, un / n,
            wok / tot, (tot - wok - wun) / tot, wun / tot, ident)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="2026-07-20")
    p.add_argument("--model", default="redimnet")
    args = p.parse_args(argv)
    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(model=args.model)
    runs = [r for r in sorted(dec.discover()) if r.startswith(args.prefix)]

    print(f"{'条件':<26}{'件数:正解':>10}{'誤帰属':>8}{'未確定':>8}"
          f"{'  ':>2}{'文字:正解':>10}{'誤帰属':>8}{'未確定':>8}"
          f"{'  ':>2}{'話者数':>7}", flush=True)
    for label, merge in (("人数なし・統合しない", False),
                         ("人数なし・キーを統合", True)):
        got = [v for v in (measure(r, vp, merge=merge) for r in runs) if v]
        if not got:
            continue
        n = len(got)
        a = [sum(x[i] for x in got) / n for i in range(7)]
        print(f"{label:<26}{a[0]:>10.1%}{a[1]:>8.1%}{a[2]:>8.1%}{'  ':>2}"
              f"{a[3]:>10.1%}{a[4]:>8.1%}{a[5]:>8.1%}{'  ':>2}{a[6]:>7.1f}",
              flush=True)

    print("\n比較: 人数あり・本番は 校正セット91.5% / 持ち越し86.1%（文字:正解）。")
    print("  §40 の「帰属をクラスタで置き換える」形は 86.4% / 82.0% だった。")
    print("  統合の形がこれに並ぶなら、新しい名前空間を作らずに済む。")


if __name__ == "__main__":
    main()
