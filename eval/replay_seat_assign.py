#!/usr/bin/env python3
"""記録から**本番の `SeatAudio`** を駆動して席落ち割当てを検証する.

新規録音が要らない理由: この機能は台帳（`diarization_speaker_keys` /
`_cluster_naming._confirmed`）に何も書かず、判定は1発話で閉じる。したがって
下流への影響が無く、必要な入力はすべて既存の記録から復元できる:

  上流キー        diag の ``key``（constrain 前）
  constrain の結果 diag の ``final_key``
  音声            transcripts/<run>.wav の該当区間
  席の参照        確定した発話の音声（時系列に前から流すので因果的）

`eval/replay_live_attribution.py` が新形式の記録を要求するのは、クラスタ層の
入力（diarization の窓）がメモリ上にしか無いからだった（§23）。こちらは
その窓を必要としない。

**シミュレーションではなく本番クラスを呼ぶ**: `SeatAudio.observe` /
`SeatAudio.nearest` を `_recv_loop.flush` と同じ順序・同じ条件で呼ぶ。
定数（参照秒数・成熟下限）も `_constants` の値をそのまま使う。

使い方:
    uv run python eval/replay_seat_assign.py --prefix 2026-07-20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import cluster_merge_feasibility as feas  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._constants import SR  # noqa: E402
from das.asr.live._seat_audio import SeatAudio  # noqa: E402


def replay(run: str, vp) -> dict | None:
    """1ランを時系列に流し、本番 `SeatAudio` の判断を適用した成績を返す."""
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

    seat = SeatAudio(vp)          # 既定値＝本番と同じ参照秒数・成熟下限
    picks: dict[int, str] = {}
    n_pick = 0
    for u, _code in rows:
        a, b = int(u["ms"]), int(u.get("end") or u["ms"])
        wav = pcm[int(a / 1000 * SR):int(b / 1000 * SR)]
        final = str(u["final_key"])
        # _recv_loop.flush と同じ順序・同じ条件
        if final != dec.UNSURE_SPEAKER:
            seat.observe(final, wav)
        elif str(u.get("key")) != dec.UNSURE_SPEAKER:
            picked = seat.nearest(wav)
            if picked is not None:
                picks[int(u["ms"])] = picked[0]
                n_pick += 1

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

    def _new(u: dict) -> str:
        cur = _cur(u)
        return picks.get(int(u["ms"]), cur) if cur == dec.UNSURE_SPEAKER else cur

    # 割当てた発話のうち GT と一致した数（寄せ先の当たる率）
    _a, mapping = _gtlib.best_mapping([(_new(u), c) for u, c in rows],
                                      dec.GT_CODES, unsure=dec.UNSURE_SPEAKER)
    hit = sum(1 for u, c in rows
              if int(u["ms"]) in picks and mapping.get(picks[int(u["ms"])]) == c)
    return {"run": run, "before": _score(_cur), "after": _score(_new),
            "picks": n_pick, "hit": hit}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default=None)
    p.add_argument("--model", default="redimnet")
    args = p.parse_args(argv)

    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(model=args.model)

    runs = [r for r in sorted(dec.discover())
            if not args.prefix or r.startswith(args.prefix)]
    out = [r for r in (replay(x, vp) for x in runs) if r]
    if not out:
        raise SystemExit("# final_key と .wav の揃うランが無い")
    print("# 本番 SeatAudio を記録から駆動（新規録音なし・因果的）")
    print(f"{'run':<20}{'割当':>5}{'的中':>5}{'正解(前→後)':>18}"
          f"{'誤帰属(前→後)':>20}{'未確定(前→後)':>20}")
    for r in out:
        b, a = r["before"], r["after"]
        print(f"{r['run']:<20}{r['picks']:>5}{r['hit']:>5}"
              f"{b['acc']:>8.1%}→{a['acc']:<8.1%}"
              f"{b['wrong']:>9.1%}→{a['wrong']:<9.1%}"
              f"{b['unsure']:>9.1%}→{a['unsure']:<9.1%}")
    n = len(out)
    m = {f"{w}_{k}": sum(x[w][k] for x in out) / n
         for w in ("before", "after") for k in ("acc", "wrong", "unsure")}
    picks = sum(x["picks"] for x in out)
    hits = sum(x["hit"] for x in out)
    print(f"{'平均':<20}{picks:>5}{hits:>5}"
          f"{m['before_acc']:>8.1%}→{m['after_acc']:<8.1%}"
          f"{m['before_wrong']:>9.1%}→{m['after_wrong']:<9.1%}"
          f"{m['before_unsure']:>9.1%}→{m['after_unsure']:<9.1%}")
    print(f"\n  正解 {(m['after_acc'] - m['before_acc']) * 100:+.1f}pt"
          f" / 誤帰属 {(m['after_wrong'] - m['before_wrong']) * 100:+.1f}pt"
          f" / 未確定 {(m['after_unsure'] - m['before_unsure']) * 100:+.1f}pt")
    if picks:
        print(f"  寄せ先の的中: {hits}/{picks} = {hits / picks:.0%}")


if __name__ == "__main__":
    main()
