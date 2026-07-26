#!/usr/bin/env python3
"""席の音声による再判定を、いまより広い場面に広げたらどうなるかを測る.

いま（`5d81b34`+`57dffdd`）の適用範囲は「上流は決めていたのに constrain が
席上限で落とした発話」だけ。手持ちの記録で、次の2つの拡張を測る:

  A. 声紋層が「ラベル不純」で棄権した発話も席の音声で判定する
     未確定の主因がこれ。棄権の理由は人物プロファイルとの照合が弱かったこと
     で、席の実音声を参照にすると分離が良くなることは §27.4 で実測済み
  B. 「蓄積中」に §18.8 の裏付け門番を掛け、切られた分を席の音声で拾い直す
     門番単独は「誤帰属 -2.7pt と引き換えに未確定 +2.7pt」の純粋な交換だった
     ため §27.6 で保留にした。切った先を席の音声が拾えるなら純増になりうる

**席の参照は全条件で同一**（`observe` は高信頼4種かつ確定済みの発話だけを
取り込み、割当てた発話は取り込まない）。したがって1回のストリーミングで
参照と各発話の1位を記録しておけば、条件の比較は再計算なしでできる。

本番の `SeatAudio` をそのまま駆動する（シミュレーションではない）。

使い方:
    uv run python eval/seat_assign_extensions.py --prefix 2026-07-20
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
import cluster_merge_feasibility as feas  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._attribution import _VOICEPRINT_RELIABLE_KINDS  # noqa: E402
from das.asr.live._constants import SR  # noqa: E402
from das.asr.live._seat_audio import SeatAudio  # noqa: E402

# 条件比較のために1位を控えておく kind（参照は条件によらず同一なので、
# ここで広めに控えておけば再計算なしで条件を足せる）
_PICK_KINDS = {"蓄積中", "ラベル不純", "ラベル継続"}


def collect_run(run: str, vp) -> dict | None:
    """1ランを時系列に流し、各発話の「その時点での席の1位」を記録する."""
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
    seat = SeatAudio(vp)
    pick: dict[int, str] = {}
    for u, _c in rows:
        a, b = int(u["ms"]), int(u.get("end") or u["ms"])
        wav = pcm[int(a / 1000 * SR):int(b / 1000 * SR)]
        final = str(u["final_key"])
        kind = u.get("kind")
        # _recv_loop.flush と同じ順序・同じ条件で参照を育てる
        if final != dec.UNSURE_SPEAKER and kind in _VOICEPRINT_RELIABLE_KINDS:
            seat.observe(final, wav)
        # 条件比較のため、拡張の候補になりうる発話は全部1位を控えておく
        # （参照は条件によらず同一なので、これで再計算が要らなくなる）
        if final == dec.UNSURE_SPEAKER or kind in _PICK_KINDS:
            got = seat.nearest(wav)
            if got is not None:
                pick[int(u["ms"])] = got[0]
    return {"run": run, "rows": rows, "pick": pick}


def score(rows, final_of) -> tuple[float, float, float]:
    pairs = [(final_of(u), c) for u, c in rows]
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=dec.UNSURE_SPEAKER)
    n = len(pairs)
    good = sum(1 for f, c in pairs if m.get(f) == c)
    uns = sum(1 for f, _ in pairs if f == dec.UNSURE_SPEAKER)
    return good / n, (n - good - uns) / n, uns / n


def make_final(data: dict, *, extend_abstain: bool, gate_accum: bool,
               override_kinds: frozenset = frozenset()):
    """条件に応じた「最終キー」を返す関数を作る（本番の意味論に合わせる）."""
    pick = data["pick"]

    def _final(u: dict) -> str:
        cur = str(u["final_key"])
        key = str(u.get("key"))
        kind = u.get("kind")
        ms = int(u["ms"])
        # C/D: この kind の発話は上流のキーを信用せず、席の音声で決め直す。
        # ラベル不純は「STTラベルが複数人を混載している」と分かっている状態
        # なので、そのラベルに基づく上流のキーは構造的に当てにならない。
        if kind in override_kinds and ms in pick:
            return pick[ms]
        # B: 蓄積中に §18.8 の裏付け門番を掛ける（最終キーだけ差し替える設計）
        if (gate_accum and cur != dec.UNSURE_SPEAKER and kind == "蓄積中"
                and not dec.endorsed(u)):
            cur = dec.UNSURE_SPEAKER
        if cur != dec.UNSURE_SPEAKER:
            return cur
        # いまの適用範囲: 上流は決めていたのに席上限で落ちた
        if key != dec.UNSURE_SPEAKER:
            return pick.get(ms, cur)
        # A: 声紋層が「ラベル不純」で棄権した発話にも広げる
        if extend_abstain and kind == "ラベル不純":
            return pick.get(ms, cur)
        return cur

    return _final


def breakdown(data, *, extend_abstain: bool, gate_accum: bool,
              override_kinds: frozenset = frozenset()) -> Counter:
    """A+B 適用後に残る誤帰属を「どこで生まれたか」で分ける.

    seat_assign  席の音声で寄せたが外した（割当ての精度の問題）
    kind:<種別>   上流の帰属がそのまま誤り（声紋層のその kind の問題）
    """
    final_of = make_final(data, extend_abstain=extend_abstain,
                          gate_accum=gate_accum, override_kinds=override_kinds)
    rows = data["rows"]
    pick = data["pick"]
    _a, m = _gtlib.best_mapping([(final_of(u), c) for u, c in rows],
                                dec.GT_CODES, unsure=dec.UNSURE_SPEAKER)
    out: Counter = Counter()
    for u, code in rows:
        f = final_of(u)
        if f == dec.UNSURE_SPEAKER or m.get(f) == code:
            continue
        ms = int(u["ms"])
        # その発話の最終キーが席の割当て由来か、上流由来かを見分ける
        if f == pick.get(ms) and str(u["final_key"]) != f:
            out["seat_assign"] += 1
        else:
            out[f"kind:{u.get('kind')}"] += 1
    return out


_NONE = frozenset()
_IMPURE = frozenset({"ラベル不純"})
_IMPURE_CONT = frozenset({"ラベル不純", "ラベル継続"})

CONDS = [
    ("いま（席落ちのみ）", False, False, _NONE),
    ("A+B（main の現状）", True, True, _NONE),
    ("+C ラベル不純は上流を上書き", True, True, _IMPURE),
    ("+D さらにラベル継続も", True, True, _IMPURE_CONT),
]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default=None)
    p.add_argument("--model", default="redimnet")
    p.add_argument("--split", type=int, default=5,
                   help="先頭N本を開発・残りを検証として分けて出す")
    args = p.parse_args(argv)

    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(model=args.model)
    runs = [r for r in sorted(dec.discover())
            if not args.prefix or r.startswith(args.prefix)]
    data = [d for d in (collect_run(x, vp) for x in runs) if d]
    if not data:
        raise SystemExit("# final_key と .wav の揃うランが無い")

    def _report(subset, label):
        print(f"\n## {label}（{len(subset)}本）")
        print(f"{'条件':<26}{'正解':>8}{'誤帰属':>9}{'未確定':>9}")
        base = None
        for name, ext, gate, ov in CONDS:
            vals = [score(d["rows"], make_final(d, extend_abstain=ext,
                                                gate_accum=gate,
                                                override_kinds=ov))
                    for d in subset]
            n = len(vals)
            acc, wrong, uns = (sum(v[i] for v in vals) / n for i in (0, 1, 2))
            mark = ""
            if base is None:
                base = (acc, wrong, uns)
            else:
                mark = (f"   正解{(acc - base[0]) * 100:+.1f}"
                        f" 誤{(wrong - base[1]) * 100:+.1f}"
                        f" 未{(uns - base[2]) * 100:+.1f}")
            print(f"{name:<26}{acc:>8.1%}{wrong:>9.1%}{uns:>9.1%}{mark}")

    for label, ov in (("A+B", _NONE), ("A+B+C", _IMPURE)):
        tot: Counter = Counter()
        for d in data:
            tot.update(breakdown(d, extend_abstain=True, gate_accum=True,
                                 override_kinds=ov))
        n_wrong = sum(tot.values())
        print(f"\n## {label} 適用後に残る誤帰属の内訳（{n_wrong}件）")
        for k, v in tot.most_common():
            print(f"  {k:<24}{v:>5}  {v / n_wrong:>5.0%}")

    _report(data, "全体")
    if 0 < args.split < len(data):
        _report(data[:args.split], "開発")
        _report(data[args.split:], "検証")


if __name__ == "__main__":
    main()
