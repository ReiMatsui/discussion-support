#!/usr/bin/env python3
"""分裂クラスタを声紋で正しい席へ寄せられるかを測る（名寄せの実現可能性）.

`decompose_attribution.py --merge-ceiling` は「分裂クラスタを多数派話者の席へ
完全に統合できたら +9.1pt / 未確定 -11.8pt」という**上限**を出す。上限は GT を
使った値なので、そのままでは投資判断に使えない。実際の名寄せは声紋で決めるため、
問うべきは「**声紋で寄せ先を当てられるのか**」である。本スクリプトはそれを測る。

やること（新規録音もAPIコストも不要。ローカルの ReDimNet と既存の .wav のみ）:

  1. GT付きランの diag から、席を得たキー（GTコードが付いたキー）と、
     constrain で落ちた `@diar:N` クラスタを取り出す
  2. それぞれの発話区間の音声を .wav から切り出し、連結して埋め込む
     （連結の上限は `_CONCAT_SEC`。無制限に連結すると1本の埋め込みが
      「その人の平均」に寄って分離が落ちるため。phase0 で同じ轍を踏んだ）
  3. 落ちたクラスタごとに、席持ちの中で最も似ている相手を1位として選ぶ
  4. その1位が「そのクラスタの多数派GT話者の席」と一致するかを数える

読み方:

  1位正解率   声紋で寄せ先を当てられる割合。これが高ければ名寄せは実装可能で、
              上限（+9.1pt）のかなりの部分が取れる。低ければ声紋では無理で、
              別の手（区切りの整列・DOA・話者数の事前指定）が要る
  sim / 余裕   1位の類似度と2位との差。既定のクラスタ確定しきい値
              PYANNOTE_CLUSTER_CONFIRM_MIN_SIM と並べて出す。1位が正しいのに
              sim がしきい値未満なら、**問題は照合能力ではなくしきい値**
              （データが揃ったので、ここで初めて閾値の議論ができる）

使い方:
    uv run python eval/cluster_merge_feasibility.py --prefix 2026-07-20
"""
from __future__ import annotations

import argparse
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._constants import (  # noqa: E402
    PYANNOTE_CLUSTER_CONFIRM_MIN_SIM,
    PYANNOTE_CLUSTER_NAMING_MIN_SEC,
    SR,
)

# 連結の上限（秒）。長くするほど埋め込みが平均に寄って分離が落ちる
# （eval/phase0_dual_ledger.py で無制限連結が同一人物/別人の差を潰した）。
_CONCAT_SEC = 30.0


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getnchannels() != 1 or w.getframerate() != SR:
            raise SystemExit(f"# {path} は mono/{SR}Hz ではない")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def concat(pcm: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """区間を連結する（上限 `_CONCAT_SEC`。長い方から取らず、時間順に頭から）."""
    out, budget = [], int(_CONCAT_SEC * SR)
    for a_ms, b_ms in spans:
        a, b = int(a_ms / 1000 * SR), int(b_ms / 1000 * SR)
        seg = pcm[max(0, a):max(0, b)]
        if seg.size == 0:
            continue
        out.append(seg[:budget])
        budget -= min(seg.size, budget)
        if budget <= 0:
            break
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def collect(run: str) -> dict | None:
    """(席持ちキー→区間, 落ちたクラスタ→(区間, 多数派GT)) を集める."""
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
    _a, seat_map = _gtlib.best_mapping(
        [(str(u["final_key"]), c) for u, c in rows], dec.GT_CODES,
        unsure=dec.UNSURE_SPEAKER)
    seat_of_code = {v: k for k, v in seat_map.items() if v}

    seats: dict[str, list] = {}
    frags: dict[str, list] = {}
    frag_gt: dict[str, Counter] = {}
    for u, code in rows:
        span = (int(u["ms"]), int(u.get("end") or u["ms"]))
        if span[1] <= span[0]:
            continue
        final = str(u["final_key"])
        if final != dec.UNSURE_SPEAKER and seat_map.get(final):
            seats.setdefault(final, []).append(span)
        elif final == dec.UNSURE_SPEAKER and str(u.get("key")) != dec.UNSURE_SPEAKER:
            k = str(u["key"])
            frags.setdefault(k, []).append(span)
            frag_gt.setdefault(k, Counter())[code] += 1
    if not seats or not frags:
        return None
    return {"run": run, "pcm": read_wav(wav_path), "seats": seats,
            "frags": frags, "frag_gt": frag_gt, "seat_of_code": seat_of_code}


def score_with_merge(run: str, decided: dict[str, str]) -> dict:
    """声紋で決めた寄せ先を適用したときの成績（GT は採点にしか使わない）.

    `decompose_attribution --merge-ceiling` は多数派GTへ寄せる**上限**だが、
    ここでは寄せ先を声紋だけで決め、しきい値を満たしたクラスタのみ統合する。
    したがって実装したときに実際に得られる値の推定になる。
    """
    loaded = dec.load_run(run)
    utts, code_by_ms = loaded          # collect が通った run のみ渡される
    rows = [(u, code_by_ms.get(u["ms"])) for u in utts]
    rows = [(u, c) for u, c in rows if c in dec.GT_CODES
            and not dec._BACKCHANNEL_RE.match(str(u["_text"]).strip())]

    def _score(final_of) -> dict:
        pairs = [(final_of(u), c) for u, c in rows]
        _a, mapping = _gtlib.best_mapping(pairs, dec.GT_CODES,
                                          unsure=dec.UNSURE_SPEAKER)
        n = len(pairs)
        good = sum(1 for f, c in pairs if mapping.get(f) == c)
        uns = sum(1 for f, _ in pairs if f == dec.UNSURE_SPEAKER)
        return {"acc": good / n, "unsure": uns / n,
                "wrong": (n - good - uns) / n, "n": n}

    def _cur(u: dict) -> str:
        return str(u["final_key"])

    def _merged(u: dict) -> str:
        cur = _cur(u)
        if cur != dec.UNSURE_SPEAKER:
            return cur
        return decided.get(str(u.get("key")), cur)

    return {"run": run, "before": _score(_cur), "after": _score(_merged)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default=None, help="対象ランの接頭辞で絞る")
    p.add_argument("--model", default="redimnet")
    p.add_argument("--min-sim", type=float,
                   default=PYANNOTE_CLUSTER_CONFIRM_MIN_SIM,
                   help="統合に要求する類似度の下限。負の値で「下限なし＝"
                        "席持ちの中の1位に必ず寄せる」（閉集合の割当て）")
    args = p.parse_args(argv)

    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(model=args.model)

    runs = [r for r in sorted(dec.discover())
            if not args.prefix or r.startswith(args.prefix)]
    print("# 分裂クラスタを声紋で正しい席へ寄せられるか"
          f"（連結上限{_CONCAT_SEC:.0f}秒・{args.model}）")
    print(f"{'run':<20}{'分裂':<10}{'件数':>5}{'総秒':>7}{'多数派GT':>9}"
          f"{'声紋1位':>10}{'sim':>7}{'2位差':>7}{'判定':>6}")
    ok = tot = 0
    ok_utt = tot_utt = 0
    below = short = short_utt = 0
    diff_sims: list[float] = []   # 別人との類似度（2位）。§15.12 との整合確認用
    applied: list[dict] = []      # しきい値を適用したときの実際の成績
    for run in runs:
        c = collect(run)
        if c is None:
            continue
        decided: dict[str, str] = {}   # 分裂キー -> 寄せ先（しきい値を満たした分）
        embs = {}
        for k, spans in c["seats"].items():
            e = vp.embed_audio(concat(c["pcm"], spans))
            if e is not None:
                embs[k] = e
        if len(embs) < 2:
            continue
        for k, spans in sorted(c["frags"].items()):
            e = vp.embed_audio(concat(c["pcm"], spans))
            if e is None:
                continue
            ranked = sorted(((float(np.dot(e, v)), kk) for kk, v in embs.items()),
                            reverse=True)
            sim, top = ranked[0]
            second = ranked[1][0] if len(ranked) > 1 else 0.0
            if len(ranked) > 1:
                # 2位は定義上「別人」。ここが確定線を超えるなら、絶対しきい値
                # では同一人物と別人を分けられない（§15.12 の実測と同じ結論）
                diff_sims.append(second)
            gt_code = c["frag_gt"][k].most_common(1)[0][0]
            want = c["seat_of_code"].get(gt_code)
            hit = top == want
            n = len(spans)
            tot += 1
            ok += hit
            tot_utt += n
            ok_utt += n if hit else 0
            below += hit and sim < PYANNOTE_CLUSTER_CONFIRM_MIN_SIM
            sec = sum(b - a for a, b in spans) / 1000
            # min_sec 未満のクラスタは live では照合の対象にすらならない。
            # 「照合したが確信に届かなかった」のか「そもそも試していない」のかを
            # 分けないと、しきい値の話と蓄積条件の話が混ざる
            short += sec < PYANNOTE_CLUSTER_NAMING_MIN_SEC
            short_utt += n if sec < PYANNOTE_CLUSTER_NAMING_MIN_SEC else 0
            if sim >= args.min_sim:
                decided[k] = top
            print(f"{run:<20}{k:<10}{n:>5}{sec:>7.1f}{gt_code:>9}"
                  f"{want or '?'!s:>10}{sim:>7.2f}{sim - second:>7.2f}"
                  f"{'○' if hit else '×':>6}")
        applied.append(score_with_merge(run, decided))
    if not tot:
        raise SystemExit("# 測れるランが無かった")
    print(f"\n1位正解率  クラスタ単位 {ok}/{tot} = {ok / tot:.0%}"
          f" / 発話重みつき {ok_utt}/{tot_utt} = {ok_utt / tot_utt:.0%}")
    print(f"うち sim がクラスタ確定しきい値"
          f"({PYANNOTE_CLUSTER_CONFIRM_MIN_SIM}) 未満だったもの: {below} 件")
    print(f"照合の蓄積下限({PYANNOTE_CLUSTER_NAMING_MIN_SEC}秒)に届かないクラスタ: "
          f"{short}/{tot} 個（発話 {short_utt}/{tot_utt} 件）"
          " — live では照合を試みてすらいない")
    if applied:
        n = len(applied)
        m = {f"{w}_{k}": sum(a[w][k] for a in applied) / n
             for w in ("before", "after") for k in ("acc", "wrong", "unsure")}
        how = ("下限なし＝席持ちの中の1位に必ず寄せる"
               if args.min_sim < 0 else f"しきい値{args.min_sim}以上のみ統合")
        print(f"\n# {how}（声紋だけで寄せ先を決める。上限ではない）")
        for a in applied:
            b, af = a["before"], a["after"]
            print(f"{a['run']:<20}{b['acc']:>8.1%}→{af['acc']:<8.1%}"
                  f"{b['wrong']:>9.1%}→{af['wrong']:<9.1%}"
                  f"{b['unsure']:>9.1%}→{af['unsure']:<9.1%}")
        print(f"{'平均':<20}{m['before_acc']:>8.1%}→{m['after_acc']:<8.1%}"
              f"{m['before_wrong']:>9.1%}→{m['after_wrong']:<9.1%}"
              f"{m['before_unsure']:>9.1%}→{m['after_unsure']:<9.1%}")
        print(f"  正解 {(m['after_acc'] - m['before_acc']) * 100:+.1f}pt"
              f" / 誤帰属 {(m['after_wrong'] - m['before_wrong']) * 100:+.1f}pt"
              f" / 未確定 {(m['after_unsure'] - m['before_unsure']) * 100:+.1f}pt")
    if diff_sims:
        diff_sims.sort(reverse=True)
        print(f"\n# 別人との類似度（各分裂クラスタの2位。n={len(diff_sims)}）")
        print("  最大 {:.2f} / 上位5件 {}".format(
            diff_sims[0], " ".join(f"{x:.2f}" for x in diff_sims[:5])))
        print(f"  うち確定しきい値({PYANNOTE_CLUSTER_CONFIRM_MIN_SIM})以上: "
              f"{sum(1 for x in diff_sims if x >= PYANNOTE_CLUSTER_CONFIRM_MIN_SIM)}件"
              " — 絶対しきい値では同一人物と別人を分けられない（§15.12 の再確認）")
    print("\n読み方:")
    print("  1位が正しいのに sim がしきい値未満なら、名寄せを妨げているのは")
    print("  照合能力ではなく**しきい値**。逆に1位自体が外れているなら、")
    print("  声紋では寄せられないので別の手（区切りの整列・DOA 等）が要る")


if __name__ == "__main__":
    main()
