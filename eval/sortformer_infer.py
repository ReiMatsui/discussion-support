#!/usr/bin/env python3
"""Streaming Sortformer を録音に流し、話者区間を JSON に落とす（NeMo venv 用）.

このリポジトリの通常の依存では動かない（NeMo が要る）。専用の venv から
直接実行する:

    /root/nemo-venv/bin/python eval/sortformer_infer.py --prefix 2026-07-20

出力は `eval/_sortformer/<run>.json` に `[[開始ms, 終了ms, 話者], ...]`。
採点は `eval/sortformer_compare.py`（通常の venv）が行う——推論と採点を
分けるのは、重い依存を採点側に持ち込まないため。

モデルは `scripts/sortformer_worker.py` と同じ
`nvidia/diar_streaming_sortformer_4spk-v2.1`（ライブ経路と揃える）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "_sortformer"
MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"


def to_segments(preds, frame_sec: float, min_sec: float = 0.1):
    """フレーム単位の話者確率を (開始ms, 終了ms, 話者) の区間に畳む.

    しきい値0.5で各話者の活性を取り、連続するフレームをつなぐ。重なりは
    そのまま複数区間として残す（採点側が「最も重なった話者」を選ぶ）。
    """
    import numpy as np
    a = np.asarray(preds)
    if a.ndim != 2:
        return []
    segs = []
    for spk in range(a.shape[1]):
        on = a[:, spk] >= 0.5
        i = 0
        while i < len(on):
            if not on[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            start, end = i * frame_sec, (j + 1) * frame_sec
            if end - start >= min_sec:
                segs.append([int(start * 1000), int(end * 1000),
                             f"SPEAKER_{spk:02d}"])
            i = j + 1
    segs.sort()
    return segs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="2026-07-20")
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()

    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.from_pretrained(args.model,
                                                    map_location="cpu")
    model.eval()

    OUT.mkdir(parents=True, exist_ok=True)
    wavs = sorted((ROOT / "transcripts").glob(f"{args.prefix}*.wav"))
    if not wavs:
        raise SystemExit(f"# {args.prefix}*.wav が transcripts に無い")
    for w in wavs:
        run = w.stem
        dst = OUT / f"{run}.json"
        if dst.exists():
            print(f"# skip {run}（既にある）", flush=True)
            continue
        print(f"# 推論 {run} ...", flush=True)
        preds = model.diarize(audio=[str(w)], batch_size=1)
        # NeMo の戻りはバージョンで形が違う。フレーム確率か区間文字列のどちらか。
        item = preds[0] if isinstance(preds, list) else preds
        if isinstance(item, list) and item and isinstance(item[0], str):
            # "start end speaker" 形式（RTTM風）
            segs = []
            for line in item:
                parts = line.split()
                if len(parts) >= 3:
                    segs.append([int(float(parts[0]) * 1000),
                                 int(float(parts[1]) * 1000), parts[2]])
        else:
            segs = to_segments(item, frame_sec=0.08)
        dst.write_text(json.dumps(segs), encoding="utf-8")
        print(f"#   区間 {len(segs)} 件 -> {dst.name}", flush=True)


if __name__ == "__main__":
    main()
