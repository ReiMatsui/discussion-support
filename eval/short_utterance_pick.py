#!/usr/bin/env python3
"""1秒未満の席選びで「迷ったら黙る」が割に合うかを測る（誤帰属の8割がここ）.

内訳（§35）: 全誤帰属226件のうち184件（81%）が1秒未満で、そのうち179件は
「席の音声で決め直す」経路（ラベル不純・ラベル継続）を通っている。席の割当ては
閉集合の rank-1 で棄権が無いため、0.5秒の「ほう。」からでも必ず1席を選ぶ。

**問い**: そこに棄権を許すと割に合うか。

    rank1    いまの実装（1位を無条件に採る）
    margin   1位と2位の差が小さければ未確定に落とす
    floor    1位の類似そのものが低ければ未確定に落とす

棄権は誤帰属だけでなく正解も削る。したがって見るべきは「誤帰属が何pt減った
か」ではなく、**未確定に移った分のうち何割が誤帰属だったか**である。半々なら
ただの置き換えで、方針（誤帰属より未確定を優先）に照らしても旨みは薄い。

長さで条件を変える案も並べるのは、この割合が長さで大きく違うため——1秒以上
では既に誤帰属4.0%しか無く、そこに棄権を入れても正解を削るほうが大きい。

**文脈を足す案は却下済み**（§36）。同じラベル／同じ上流キーの過去の音声を
2秒または10秒だけ前に足して測ったところ、全条件で現行を大きく下回った
（1秒未満の誤帰属 25.4% → 39〜54%）。短い発話に足せば足した側が埋め込みを
支配し、直前の話者に引っ張られる。以前も同じ結論だったが、当時は短い発話が
分母から落ちる測り方だったので、正しい土俵で測り直した上での却下である。

GT は採点にしか使わない。新規録音も STT の再課金も不要。

使い方:
    uv run python eval/short_utterance_pick.py --prefix 2026-07-20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import _pipeline as pipe  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._constants import UNSURE_SPEAKER  # noqa: E402

SHORT_MS = 1000     # 「短い発話」の境目（§35 の内訳がここで割れる）


# ------------------------------------------------------ 寄せ先の選び方


def _sims(emb, refs: dict) -> list[tuple[float, str]]:
    if emb is None or len(refs) < 2:
        return []
    return sorted(((float(np.dot(emb, v)), k) for k, v in refs.items()),
                  reverse=True)


def _dur_ms(st: dict) -> int:
    u = st["utt"]
    return max(0, int(u.get("end") or u["ms"]) - int(u["ms"]))


def gate(*, margin: float = 0.0, floor: float = 0.0, short_only: bool = False):
    """条件を満たさなければ未確定に落とす選び方を作る.

    `short_only` のとき、長い発話は現行どおり無条件に1位を採る。棄権を
    入れる価値は長さで違うので、同じ棄権則を全体に掛けた場合と分けて測る。
    """
    def pick(emb, refs, st):
        s = _sims(emb, refs)
        if not s:
            return None
        if short_only and _dur_ms(st) >= SHORT_MS:
            return s[0][1]
        if s[0][0] < floor or s[0][0] - s[1][0] < margin:
            return UNSURE_SPEAKER
        return s[0][1]
    return pick


VARIANTS: list[tuple[str, object]] = [
    ("いまの実装", None),
    ("差 0.03未満は未確定", gate(margin=0.03)),
    ("差 0.05未満は未確定", gate(margin=0.05)),
    ("差 0.10未満は未確定", gate(margin=0.10)),
    ("類似 0.30未満は未確定", gate(floor=0.30)),
    ("短のみ 差 0.03", gate(margin=0.03, short_only=True)),
    ("短のみ 差 0.05", gate(margin=0.05, short_only=True)),
    ("短のみ 差 0.10", gate(margin=0.10, short_only=True)),
    ("短のみ 類似 0.30", gate(floor=0.30, short_only=True)),
    ("短のみ 類似 0.40", gate(floor=0.40, short_only=True)),
]


# ---------------------------------------------------------------- 採点


def outcomes(data: dict, pick) -> list[dict]:
    """1ランを指定の案で採点し、発話ごとの結末を返す（対応づけはラン単位）."""
    steps = data["steps"]
    final = pipe.apply_schedule(steps, pick=pick)
    pairs = [(f, st["code"]) for f, st in zip(final, steps, strict=True)]
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=UNSURE_SPEAKER)
    return [{"dur_ms": _dur_ms(st),
             "chars": len(str(st["utt"].get("_text") or "")),
             "outcome": ("未確定" if f == UNSURE_SPEAKER
                         else "正解" if m.get(f) == st["code"] else "誤帰属")}
            for f, st in zip(final, steps, strict=True)]


def _rates(rows: list[dict], weigh) -> tuple[float, float, float]:
    tot = sum(weigh(r) for r in rows) or 1
    return tuple(sum(weigh(r) for r in rows if r["outcome"] == k) / tot
                 for k in ("正解", "誤帰属", "未確定"))


def _table(label: str, by_variant: dict[str, list[dict]],
           base: dict[str, list[dict]] | None = None,
           keep=lambda r: True) -> None:
    """案ごとの成績と、現行から未確定へ移った分の内訳を並べる."""
    print(f"\n## {label}")
    print(f"{'案':<22}{'件数':>6}{'正解':>7}{'誤帰属':>7}{'未確定':>7}"
          f"{'  ':>2}{'文字':>7}{'正解':>7}{'誤帰属':>7}{'未確定':>7}"
          f"{'  ':>2}{'棄権の内訳':>12}")
    ref = (base or by_variant)["いまの実装"]
    for name, _p in VARIANTS:
        rows = [r for r in by_variant[name] if keep(r)]
        if not rows:
            continue
        cnt = _rates(rows, lambda r: 1)
        ch = _rates(rows, lambda r: r["chars"])
        w = sum(r["chars"] for r in rows)
        # 現行と比べ、未確定が増えた分の何割が誤帰属だったか
        old = [r for r in ref if keep(r)]
        d_uns = cnt[2] - _rates(old, lambda r: 1)[2]
        d_bad = _rates(old, lambda r: 1)[1] - cnt[1]
        trade = f"{d_bad / d_uns:>11.0%}" if d_uns > 0.001 else f"{'—':>12}"
        print(f"{name:<22}{len(rows):>6}{cnt[0]:>7.1%}{cnt[1]:>7.1%}"
              f"{cnt[2]:>7.1%}{'  ':>2}{w:>7}{ch[0]:>7.1%}{ch[1]:>7.1%}"
              f"{ch[2]:>7.1%}{'  ':>2}{trade}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="2026-07-20")
    p.add_argument("--model", default="redimnet")
    p.add_argument("--split", type=int, default=5,
                   help="開発/検証に分ける本数（0で分けない）")
    args = p.parse_args(argv)

    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(model=args.model)
    runs = [r for r in sorted(dec.discover()) if r.startswith(args.prefix)]

    per_run: list[dict] = []
    for run in runs:
        data = pipe.replay_seats(run, vp, align="text")
        if data is None:
            continue
        per_run.append({name: outcomes(data, pick) for name, pick in VARIANTS})
        print(f"# {run} 済み（{len(data['steps'])}発話）", flush=True)
    if not per_run:
        raise SystemExit("# 測れるランが無い")

    def _pool(subset):
        return {name: [r for g in subset for r in g[name]]
                for name, _p in VARIANTS}

    allr = _pool(per_run)
    _table(f"全体（{len(per_run)}本）", allr)
    _table("1秒未満だけ", allr, keep=lambda r: r["dur_ms"] < SHORT_MS)
    _table("1秒以上だけ（壊していないかの確認）", allr,
           keep=lambda r: r["dur_ms"] >= SHORT_MS)
    if 0 < args.split < len(per_run):
        _table(f"開発（{args.split}本）", _pool(per_run[:args.split]))
        _table(f"検証（{len(per_run) - args.split}本）",
               _pool(per_run[args.split:]))

    print("\n読み方: 右端の「棄権の内訳」は、現行より未確定が増えた分のうち")
    print("  誤帰属だった割合。50%なら正解と誤帰属を同じだけ捨てており、")
    print("  ただの置き換えである。高いほど、黙った甲斐があったことになる。")


if __name__ == "__main__":
    main()
