#!/usr/bin/env python3
"""席と比べる「問い合わせ音声」に文脈を足すと当たるようになるか.

いまは1発話の音声だけを席の参照と比べている。以前の測定では、複数発話を
まとめたクラスタ単位の音声だと的中94%、1発話だと89%だった（§27.3/§27.8）。
音声が長いほど埋め込みが安定するのだから当然で、まとめられるならまとめたい。

**ただし「何でまとめるか」が問題**である。まとめる基準が汚れていれば、
別人の声が問い合わせ音声に混ざり、誤りが伝播する。ここで比べるのはその点:

  own      いまの実装（その発話の音声だけ）
  label    同じ Soniox ラベルの**過去の**音声を足す
           ラベルは複数人を混載しうる（それが「ラベル不純」の意味）ので、
           まさにその汚染が効くかどうかを見る
  key      同じ上流キー（@diar:N 等）の**過去の**音声を足す
           pyannote のクラスタは同一人物の束ねとして78%一貫（設計8.4節）
  label_w  label と同じだが、直近30秒以内に限る（古い対応の持ち越しを断つ）

**遅延は増えない**。いずれも過去の音声しか使わないので、発話が確定した
時点でそのまま判定できる（まとまるのを待たない）。

GT は採点にしか使わない。新規録音もAPIコストも不要。

使い方:
    uv run python eval/seat_query_context.py --prefix 2026-07-20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _pipeline as pipe  # noqa: E402
import cluster_merge_feasibility as feas  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._attribution import _VOICEPRINT_RELIABLE_KINDS  # noqa: E402
from das.asr.live._constants import (  # noqa: E402
    SEAT_AUDIO_MIN_REF_SEC,
    SEAT_AUDIO_REF_SEC,
    SR,
    UNSURE_SPEAKER,
)

_LABEL_ONLY = {"ラベル不純", "ラベル継続"}
_CTX_SEC = 10.0        # 問い合わせに足す文脈の上限（秒）
_CTX_WINDOW_MS = 30_000    # label_w が遡る時間の上限
METHODS = ("own", "label", "key", "label_w")


def _tail(chunks: list[np.ndarray], sec: float) -> np.ndarray:
    """直近 sec 秒ぶんを新しい方から取り、時間順に戻して連結する."""
    budget = int(sec * SR)
    out = []
    for a in reversed(chunks):
        if budget <= 0:
            break
        out.append(a[-budget:] if a.size > budget else a)
        budget -= min(a.size, budget)
    return np.concatenate(out[::-1]) if out else np.zeros(0, dtype=np.float32)


def stream(run: str, vp) -> dict | None:
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

    # 席の参照（本番と同じ: 高信頼4種のみ・30秒で凍結）
    buf: dict[str, list] = {}
    seat_emb: dict[str, np.ndarray] = {}
    secs: dict[str, float] = {}
    frozen: set[str] = set()
    # 文脈の材料（過去の音声。ラベル別・キー別）
    by_label: dict[str, list[np.ndarray]] = {}
    by_label_ms: dict[str, list[int]] = {}
    by_key: dict[str, list[np.ndarray]] = {}
    picks: dict[int, dict[str, str]] = {}

    for u, _c in rows:
        a, b = int(u["ms"]), int(u.get("end") or u["ms"])
        wav = pcm[int(a / 1000 * SR):int(b / 1000 * SR)]
        if wav.size == 0:
            continue
        final = str(u["final_key"])
        key = str(u.get("key"))
        kind = u.get("kind")
        lab = str(u.get("label"))
        if final == UNSURE_SPEAKER or kind in _LABEL_ONLY:
            usable = {k: v for k, v in seat_emb.items()
                      if secs.get(k, 0.0) >= SEAT_AUDIO_MIN_REF_SEC}
            if len(usable) >= 2:
                got = {}
                for meth in METHODS:
                    q = _query(meth, wav, lab, key, a, by_label, by_label_ms,
                               by_key)
                    e = vp.embed_audio(q)
                    if e is not None:
                        got[meth] = max(
                            ((float(np.dot(e, v)), k) for k, v in usable.items())
                        )[1]
                if got:
                    picks[a] = got
        elif kind in _VOICEPRINT_RELIABLE_KINDS:
            if final not in frozen:
                bb = buf.setdefault(final, [])
                bb.append(wav)
                total = sum(x.size for x in bb)
                e = vp.embed_audio(np.concatenate(bb) if len(bb) > 1 else bb[0])
                if e is not None:
                    seat_emb[final] = e
                secs[final] = total / SR
                if total >= SEAT_AUDIO_REF_SEC * SR:
                    frozen.add(final)
                    buf.pop(final, None)
        # 文脈は「この発話より前」だけを使うので、記録は判定の後
        by_label.setdefault(lab, []).append(wav)
        by_label_ms.setdefault(lab, []).append(a)
        del by_label[lab][:-40], by_label_ms[lab][:-40]
        if key != UNSURE_SPEAKER:
            by_key.setdefault(key, []).append(wav)
            del by_key[key][:-40]
    return {"run": run, "rows": rows, "picks": picks}


def _query(meth, wav, lab, key, ms, by_label, by_label_ms, by_key):
    if meth == "own":
        return wav
    if meth == "key":
        past = by_key.get(key, []) if key != UNSURE_SPEAKER else []
    elif meth == "label_w":
        mss = by_label_ms.get(lab, [])
        chunks = by_label.get(lab, [])
        past = [c for c, m in zip(chunks, mss, strict=False)
                if ms - m <= _CTX_WINDOW_MS]
    else:
        past = by_label.get(lab, [])
    if not past:
        return wav
    return np.concatenate([_tail(past, _CTX_SEC), wav])


def evaluate(data, meth: str) -> tuple[float, float, float]:
    rows, picks = data["rows"], data["picks"]

    def _final(u):
        # 規則は eval/_pipeline.resolved_key に一本化。
        # 以前ここだけ「蓄積中の門番」を通しておらず、同じ
        # 「今日の実装」を名乗る数字が2種類あった。
        return pipe.resolved_key(u, picks.get(int(u["ms"]), {}).get(meth))


    pairs = [(_final(u), c) for u, c in rows]
    return pipe.score(pairs)[:3]


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
        print(f"{'問い合わせ音声':<12}{'正解':>8}{'誤帰属':>9}{'未確定':>9}")
        for meth in METHODS:
            vals = [evaluate(d, meth) for d in subset]
            n = len(vals)
            acc, wrong, uns = (sum(v[i] for v in vals) / n for i in (0, 1, 2))
            print(f"{meth:<12}{acc:>8.1%}{wrong:>9.1%}{uns:>9.1%}")

    _report(data, "全体")
    if 0 < args.split < len(data):
        _report(data[:args.split], "開発")
        _report(data[args.split:], "検証")


if __name__ == "__main__":
    main()
