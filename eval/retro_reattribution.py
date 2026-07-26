#!/usr/bin/env python3
"""序盤に決めた帰属を、席の参照が育ってから貼り直したら何が返るか（遡及訂正）.

`eval/error_anatomy.py` で誤りが**セッション序盤に極端に偏る**ことが分かった:

    開始0-1分 正解29% / 1-2分 69% / 2-5分 77% / 5-10分 90%

10分後には90%出ている。システムは収束していて、悪いのは立ち上がりだけ。
にもかかわらず、いまは**一度決めた帰属を二度と見直していない**。

ここで測るのは「席の参照が育った後に、序盤の発話を決め直したら何pt返るか」。

  実装可能性: 発話ごとの埋め込みは192次元しかないので、序盤の分を保持して
  おいて参照が育ってから比べ直せる（音声の保持は要らない）。過去レコードの
  書き換え機構も既にある（`SessionState.rekey`）。

条件:

  now        いまの実装（決めたら見直さない）
  retro_2m   開始2分までの発話を、2分時点の参照で決め直す
  retro_5m   同、5分時点の参照で決め直す
  retro_end  同、セッション終了時点の参照で決め直す（上限。ライブでは
             「最後に一括で直す」に相当し、途中の表示は直らない）

**貼り直す対象**: 「席の音声で決めた発話」と「未確定のまま残った発話」。
声紋層が高信頼で決めた発話（声紋一致・補正・自動登録・合流）は触らない
——そちらは席の平均より強いという判断は変えていない（§27.12）。

使い方:
    uv run python eval/retro_reattribution.py --prefix 2026-07-20
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
import cluster_merge_feasibility as feas  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._attribution import _VOICEPRINT_RELIABLE_KINDS  # noqa: E402
from das.asr.live._constants import SR, UNSURE_SPEAKER  # noqa: E402
from das.asr.live._recv_loop import _LABEL_ONLY_KINDS  # noqa: E402
from das.asr.live._seat_audio import RetroAttributor, SeatAudio  # noqa: E402

CUTS = (120.0, 300.0, float("inf"))       # 遡及訂正を行う時刻（秒）
NAMES = ("now", "retro_2m", "retro_5m", "retro_end", "prod")


def run_one(run: str, vp) -> dict | None:
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
    t0 = int(rows[0][0]["ms"])

    seat = SeatAudio(vp)
    live_pick: dict[int, str] = {}      # その場の判定
    emb_of: dict[int, np.ndarray] = {}  # 発話の埋め込み（遡及訂正の材料）
    snap: dict[float, dict] = {}        # 時刻 -> そのときの席の埋め込み

    def _snapshot(elapsed):
        for cut in CUTS:
            if cut not in snap and elapsed >= cut:
                snap[cut] = dict(seat._embeddings)

    for u, _c in rows:
        a, b = int(u["ms"]), int(u.get("end") or u["ms"])
        elapsed = (a - t0) / 1000
        _snapshot(elapsed)
        wav = pcm[int(a / 1000 * SR):int(b / 1000 * SR)]
        final = str(u["final_key"])
        kind = u.get("kind")
        if final != UNSURE_SPEAKER and kind == "蓄積中" and not dec.endorsed(u):
            final = UNSURE_SPEAKER
        if kind in _LABEL_ONLY_KINDS or (final == UNSURE_SPEAKER
                                         and str(u.get("key")) != UNSURE_SPEAKER):
            got = seat.nearest(wav)
            if got is not None:
                live_pick[a] = got[0]
            e = vp.embed_audio(wav)
            if e is not None:
                emb_of[a] = e
        elif final != UNSURE_SPEAKER and kind in _VOICEPRINT_RELIABLE_KINDS:
            seat.observe(final, wav)
        else:
            # 未確定のまま残った発話も、遡及訂正の対象にする
            e = vp.embed_audio(wav)
            if e is not None:
                emb_of[a] = e
    snap[float("inf")] = dict(seat._embeddings)

    def _base(u):
        """遡及訂正なしの最終キー（いまの実装）."""
        cur = str(u["final_key"])
        kind = u.get("kind")
        ms = int(u["ms"])
        if cur != UNSURE_SPEAKER and kind == "蓄積中" and not dec.endorsed(u):
            cur = UNSURE_SPEAKER
        if kind in _LABEL_ONLY_KINDS and ms in live_pick:
            return live_pick[ms]
        if cur != UNSURE_SPEAKER:
            return cur
        return live_pick.get(ms, cur)

    def _retro(u, cut):
        """cut 時点の参照で決め直す（対象は席由来の判定と未確定のみ）."""
        cur = _base(u)
        ms = int(u["ms"])
        kind = u.get("kind")
        if (kind in _VOICEPRINT_RELIABLE_KINDS
                and cur != UNSURE_SPEAKER and ms not in live_pick):
            return cur     # 声紋層が高信頼で決めた分は触らない
        if (ms - t0) / 1000 >= cut:
            return cur     # cut より後の発話は既にその参照で決まっている
        refs = snap.get(cut) or {}
        e = emb_of.get(ms)
        if e is None or len(refs) < 2:
            return cur
        return max(((float(np.dot(e, v)), k) for k, v in refs.items()))[1]

    def _score(fn):
        pairs = [(fn(u), c) for u, c in rows]
        _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=UNSURE_SPEAKER)
        n = len(pairs)
        good = sum(1 for f, c in pairs if m.get(f) == c)
        uns = sum(1 for f, _ in pairs if f == UNSURE_SPEAKER)
        return good / n, (n - good - uns) / n, uns / n

    out = {"now": _score(_base)}
    for name, cut in zip(NAMES[1:1 + len(CUTS)], CUTS, strict=True):
        out[name] = _score(lambda u, c=cut: _retro(u, c))
    out["prod"] = _score_production(rows, pcm, vp, t0)
    return out


def _score_production(rows, pcm, vp, t0):
    """**本番の RetroAttributor / SeatAudio をそのまま駆動**した成績.

    `_recv_loop.flush` と同じ順序で呼ぶ（声紋は1回だけ計算して判定と控えに
    使い回し、時刻が来たら `due` → `revise`）。予定表も `_constants` の値。
    実装が測定値（retro_5m 前後）を再現するかの確認になる。
    """
    seat = SeatAudio(vp)
    retro = RetroAttributor(seat)
    final: dict[int, str] = {}
    for u, _c in rows:
        a, b = int(u["ms"]), int(u.get("end") or u["ms"])
        wav = pcm[int(a / 1000 * SR):int(b / 1000 * SR)]
        cur = str(u["final_key"])
        kind = u.get("kind")
        if cur != UNSURE_SPEAKER and kind == "蓄積中" and not dec.endorsed(u):
            cur = UNSURE_SPEAKER
        revisable = (kind in _LABEL_ONLY_KINDS
                     or (cur == UNSURE_SPEAKER
                         and str(u.get("key")) != UNSURE_SPEAKER))
        if revisable:
            emb = seat.embed(wav)
            retro.remember(a, emb)
            got = seat.nearest_from(emb) if emb is not None else None
            final[a] = got[0] if got is not None else cur
        else:
            if cur != UNSURE_SPEAKER and kind in _VOICEPRINT_RELIABLE_KINDS:
                seat.observe(cur, wav)
            final[a] = cur
        if retro.due((a - t0) / 1000.0):
            for ms, key in retro.revise().items():
                # apply_retro_attribution と同じ保護: 席由来か未確定だけ直す
                if ms in final and (final[ms] == UNSURE_SPEAKER
                                    or ms in retro._embeddings):
                    final[ms] = key
    pairs = [(final[int(u["ms"])], c) for u, c in rows if int(u["ms"]) in final]
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=UNSURE_SPEAKER)
    n = len(pairs)
    good = sum(1 for f, c in pairs if m.get(f) == c)
    uns = sum(1 for f, _ in pairs if f == UNSURE_SPEAKER)
    return good / n, (n - good - uns) / n, uns / n


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="2026-07-20")
    p.add_argument("--model", default="redimnet")
    p.add_argument("--split", type=int, default=5)
    args = p.parse_args(argv)
    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(model=args.model)
    runs = [r for r in sorted(dec.discover()) if r.startswith(args.prefix)]
    out = [r for r in (run_one(x, vp) for x in runs) if r]
    if not out:
        raise SystemExit("# 測れるランが無い")

    def _report(subset, label):
        print(f"\n## {label}（{len(subset)}本）")
        print(f"{'条件':<12}{'正解':>8}{'誤帰属':>9}{'未確定':>9}")
        for name in NAMES:
            n = len(subset)
            acc, wrong, uns = (sum(d[name][i] for d in subset) / n
                               for i in (0, 1, 2))
            print(f"{name:<12}{acc:>8.1%}{wrong:>9.1%}{uns:>9.1%}")

    _report(out, "全体")
    if 0 < args.split < len(out):
        _report(out[:args.split], "開発")
        _report(out[args.split:], "検証")
    print("\n読み方:")
    print("  retro_2m/5m は「その時刻に一度だけ過去を貼り直す」に相当し、")
    print("  ライブでも実現できる（発話ごとの埋め込み192次元を持つだけ）。")
    print("  retro_end は終了時に一括で直す上限で、途中の表示は直らない。")


if __name__ == "__main__":
    main()
