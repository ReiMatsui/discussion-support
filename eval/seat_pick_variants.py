#!/usr/bin/env python3
"""席の「寄せ先の選び方」を比べる（残る誤帰属の51%がここ）.

§27.12 の時点で残る誤帰属182件の内訳は seat_assign 51% / ラベル継続 20% /
ラベル不純 13% で、**過半が席の割当て自体の誤り**になった。上流の帰属
ロジックではなく、寄せ先の選び方を改善する番。

いまの `SeatAudio` は席あたり**1本**の埋め込みを持つ（先頭30秒を連結して
1回embed、以後凍結）。連結して平均化すると、その人の声の幅（早口・小声・
笑い混じり）が1点に潰れる。ここで比べるのは:

  single   いまの実装（30秒連結を1本）
  multi    高信頼の発話ごとに埋め込みを持ち、**最大類似**で選ぶ
           （平均に潰さず「どれかに似ていれば良い」）
  top2     同上だが上位2本の平均で選ぶ（1本の外れ値に引きずられにくい）
  asnorm   multi に score normalization を掛ける。各席の類似度から
           「その席が他の発話一般とどれくらい似るか」を引き、席ごとの
           当たりやすさの偏り（声の大きい人が何にでも似る）を打ち消す

いずれも参照は高信頼4種の発話だけから作る（§27.9）。GT は採点にしか
使わない。新規録音もAPIコストも不要。

使い方:
    uv run python eval/seat_pick_variants.py --prefix 2026-07-20
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
from das.asr.live._constants import (  # noqa: E402
    SEAT_AUDIO_MIN_REF_SEC,
    SEAT_AUDIO_REF_SEC,
    SR,
)

# 根拠がSTTラベルしか無い kind（§27.12 と同じ集合）
_LABEL_ONLY = {"ラベル不純", "ラベル継続"}
_MAX_REFS = 12          # multi 系で席あたりに保持する埋め込みの本数


def _norm(v):
    n = float(np.linalg.norm(v))
    return None if n == 0 or not np.isfinite(n) else v / n


def stream(run: str, vp) -> dict | None:
    """1ランを時系列に流し、各方式が使う材料を因果的に作る."""
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

    # 参照の材料（因果的に育てる）
    concat_buf: dict[str, list] = {}      # single 用（30秒で凍結）
    single: dict[str, np.ndarray] = {}
    frozen: set[str] = set()
    refs: dict[str, list[np.ndarray]] = {}   # multi 用（発話ごと）
    secs: dict[str, float] = {}
    picks: dict[int, dict[str, str]] = {}    # ms -> 方式 -> 選んだ席

    for u, _c in rows:
        a, b = int(u["ms"]), int(u.get("end") or u["ms"])
        wav = pcm[int(a / 1000 * SR):int(b / 1000 * SR)]
        if wav.size == 0:
            continue
        final = str(u["final_key"])
        kind = u.get("kind")
        needs_pick = (final == dec.UNSURE_SPEAKER or kind in _LABEL_ONLY)
        if needs_pick:
            emb = vp.embed_audio(wav)
            usable = [k for k, s in secs.items() if s >= SEAT_AUDIO_MIN_REF_SEC]
            if emb is not None and len(usable) >= 2:
                picks[a] = _decide(emb, usable, single, refs)
        elif final != dec.UNSURE_SPEAKER and kind in _VOICEPRINT_RELIABLE_KINDS:
            # single: 30秒まで連結して凍結（いまの実装）
            if final not in frozen:
                buf = concat_buf.setdefault(final, [])
                buf.append(wav)
                total = sum(x.size for x in buf)
                e = vp.embed_audio(np.concatenate(buf) if len(buf) > 1 else buf[0])
                if e is not None:
                    single[final] = e
                secs[final] = total / SR
                if total >= SEAT_AUDIO_REF_SEC * SR:
                    frozen.add(final)
                    concat_buf.pop(final, None)
            # multi: 発話ごとに1本持つ（古い方から捨てる）
            e1 = vp.embed_audio(wav)
            if e1 is not None:
                r = refs.setdefault(final, [])
                r.append(e1)
                del r[:-_MAX_REFS]
    return {"run": run, "rows": rows, "picks": picks}


def _decide(emb, usable, single, refs) -> dict[str, str]:
    """各方式の選択を返す（同じ埋め込み・同じ候補集合で比較する）."""
    out = {}
    # single: 連結1本との類似
    s1 = {k: float(np.dot(emb, single[k])) for k in usable if k in single}
    if s1:
        out["single"] = max(s1, key=s1.get)
    # multi: 発話ごとの埋め込みとの最大類似
    per = {k: [float(np.dot(emb, r)) for r in refs.get(k, [])] for k in usable}
    per = {k: v for k, v in per.items() if v}
    if per:
        mx = {k: max(v) for k, v in per.items()}
        out["multi"] = max(mx, key=mx.get)
        t2 = {k: sum(sorted(v, reverse=True)[:2]) / min(2, len(v))
              for k, v in per.items()}
        out["top2"] = max(t2, key=t2.get)
        # asnorm: 席ごとの当たりやすさの偏りを引く（その席の参照同士の
        # 平均類似を基準にする簡易版）
        adj = {}
        for k, v in per.items():
            others = [x for kk, vv in per.items() if kk != k for x in vv]
            base = sum(others) / len(others) if others else 0.0
            adj[k] = max(v) - base
        out["asnorm"] = max(adj, key=adj.get)
    return out


def evaluate(data, method: str) -> tuple[float, float, float]:
    rows = data["rows"]
    picks = data["picks"]

    def _final(u):
        cur = str(u["final_key"])
        kind = u.get("kind")
        got = picks.get(int(u["ms"]), {}).get(method)
        if kind in _LABEL_ONLY and got:
            return got
        if cur != dec.UNSURE_SPEAKER:
            return cur
        if str(u.get("key")) != dec.UNSURE_SPEAKER and got:
            return got
        return cur

    pairs = [(_final(u), c) for u, c in rows]
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=dec.UNSURE_SPEAKER)
    n = len(pairs)
    good = sum(1 for f, c in pairs if m.get(f) == c)
    uns = sum(1 for f, _ in pairs if f == dec.UNSURE_SPEAKER)
    return good / n, (n - good - uns) / n, uns / n


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default=None)
    p.add_argument("--model", default="redimnet")
    p.add_argument("--split", type=int, default=5)
    args = p.parse_args(argv)

    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(model=args.model)
    runs = [r for r in sorted(dec.discover())
            if not args.prefix or r.startswith(args.prefix)]
    data = [d for d in (stream(x, vp) for x in runs) if d]
    if not data:
        raise SystemExit("# 測れるランが無い")

    def _report(subset, label):
        print(f"\n## {label}（{len(subset)}本）")
        print(f"{'寄せ先の選び方':<14}{'正解':>8}{'誤帰属':>9}{'未確定':>9}")
        for meth in ("single", "multi", "top2", "asnorm"):
            vals = [evaluate(d, meth) for d in subset]
            n = len(vals)
            acc, wrong, uns = (sum(v[i] for v in vals) / n for i in (0, 1, 2))
            print(f"{meth:<14}{acc:>8.1%}{wrong:>9.1%}{uns:>9.1%}")

    _report(data, "全体")
    if 0 < args.split < len(data):
        _report(data[:args.split], "開発")
        _report(data[args.split:], "検証")


if __name__ == "__main__":
    main()
