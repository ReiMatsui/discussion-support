"""採点対象の発話すべてについて声紋を取り、ディスクに保存する.

**なぜ分けるのか**: 人数を使わない道（自分で人をまとめる）を調べるには、
発話ごとの声紋が要る。本番の再生は「決め直す対象」の分しか埋め込みを計算
しないので足りない。一方この計算は b5 で1件0.5秒かかり、9本で15分ほど。
案を試すたびに払うのは無駄なので、一度取ってファイルに置く。

正解（GTコード）と発話長も一緒に保存する。クラスタリングの評価に要るのと、
「短い発話ほど当たらない」という既知の傾向を切り分けるため。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _pipeline as pipe  # noqa: E402

from das.asr.live._constants import SR  # noqa: E402
from das.asr.live._seat_audio import SeatAudio, seat_embedder  # noqa: E402

CACHE_DIR = ROOT / "eval" / "_emb"


def load(run: str, vp=None, *, align: str = "text") -> dict | None:
    """(声紋, GTコード, 発話長ms, 開始ms) を返す。無ければ計算して保存する."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{run}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {"emb": z["emb"], "code": [str(x) for x in z["code"]],
                "dur_ms": z["dur_ms"], "ms": z["ms"], "chars": z["chars"]}
    if vp is None:
        return None
    rows = pipe.gt_rows(run, align=align)
    wav_path = ROOT / "transcripts" / f"{run}.wav"
    if rows is None or not wav_path.exists():
        return None
    pcm = pipe.read_wav(wav_path)
    seat = SeatAudio(vp, embedder=seat_embedder(vp))
    embs, codes, durs, mss, chars = [], [], [], [], []
    for u, code in rows:
        a, b = int(u["ms"]), int(u.get("end") or u["ms"])
        emb = seat.embed(pcm[int(a / 1000 * SR):int(b / 1000 * SR)])
        if emb is None:
            continue
        embs.append(emb)
        codes.append(code)
        durs.append(max(0, b - a))
        mss.append(a)
        chars.append(len(str(u.get("_text") or "")))
    if not embs:
        return None
    got = {"emb": np.array(embs), "code": codes,
           "dur_ms": np.array(durs), "ms": np.array(mss),
           "chars": np.array(chars)}
    np.savez_compressed(path, emb=got["emb"], code=np.array(codes),
                        dur_ms=got["dur_ms"], ms=got["ms"], chars=got["chars"])
    return got
