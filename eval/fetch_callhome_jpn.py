#!/usr/bin/env python3
"""CallHome日本語（実電話会話・話者ラベル付き）を取得して評価用に整形する.

データ源: HuggingFace Fhrozen/CABankSakuraCHJP（TalkBank CABank CallHome jpn の
ミラー、認証不要）。実際の電話会話120本、CHAT転記に発話ごとの
ミリ秒タイムスタンプ（例: `120200_123350`）と話者ラベル（A/B）が付いている。
引用義務: TalkBank ルールに従い CABank Japanese CallHome Corpus
(doi:10.21415/T5H59V) を引用すること。

注意: 電話品質（8kHz μ-law・2話者）。声紋モデルは16kHz広帯域で訓練されて
いるため、これは声紋にとって不利な条件での下限評価になる。

使い方:
    uv run python eval/fetch_callhome_jpn.py 0696          # 1本取得
    uv run python eval/fetch_callhome_jpn.py 0696 0743 ... # 複数
出力:
    data/callhome/<id>.wav      16kHz mono PCM（--wav にそのまま使える）
    data/callhome/<id>.gt.json  正解タイムライン（eval_speaker_gt.py 互換）

その後:
    uv run das listen-soniox --skip-docs --max-speakers 2 --wav data/callhome/0696.wav
    uv run python eval/eval_speaker_gt.py data/callhome/0696.gt.json <セッション名>
"""
from __future__ import annotations

import audioop  # noqa: Python3.13で削除予定だが本プロジェクトは3.12
import json
import re
import struct
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np

BASE = ("https://huggingface.co/datasets/Fhrozen/CABankSakuraCHJP/"
        "resolve/main/dummy")
OUT = Path(__file__).resolve().parent.parent / "data" / "callhome"
SR = 16000

# CHAT転記の発話行: "*A:\t本文 ... \x15120200_123350\x15"
# （タイムスタンプは制御文字 0x15 で囲まれる。本文は複数行に折り返しあり）
_TS = re.compile(r"\x15?(\d+)_(\d+)\x15?\s*$")


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as r:
        return r.read()


def convert_wav(raw: bytes, out_path: Path) -> float:
    """μ-law/PCM の RIFF を 16kHz mono PCM16 に変換する（依存なし）."""
    # RIFFを手でパース（μ-law format=7 は wave モジュールが読めない）
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", "RIFF/WAVEではない"
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    assert fmt and data is not None, "fmt/dataチャンクが見つからない"
    tag, n_ch, rate = fmt[0], fmt[1], fmt[2]
    if tag == 7:      # μ-law
        pcm = audioop.ulaw2lin(data, 2)
    elif tag == 6:    # A-law
        pcm = audioop.alaw2lin(data, 2)
    elif tag == 1:    # PCM16
        pcm = data
    else:
        raise SystemExit(f"未対応のwav形式: format={tag}")
    y = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if n_ch > 1:
        y = y.reshape(-1, n_ch).mean(axis=1)
    if rate != SR:
        n_out = int(round(len(y) * SR / rate))
        y = np.interp(np.linspace(0, len(y) - 1, n_out),
                      np.arange(len(y)), y).astype(np.float32)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    return len(y) / SR


def parse_cha(text: str) -> list[dict]:
    """CHAT転記から (speaker, start_ms, end_ms) のタイムラインを抜き出す."""
    timeline = []
    cur_speaker = None
    buf = ""
    for line in text.splitlines():
        if line.startswith("*"):          # 新しい発話行
            cur_speaker = line[1:line.index(":")]
            buf = line[line.index(":") + 1:].strip()
        elif line.startswith("\t") and cur_speaker:   # 折り返し
            buf += " " + line.strip()
        else:
            cur_speaker = None
            continue
        m = _TS.search(buf)
        if m and cur_speaker:
            timeline.append({"speaker": cur_speaker,
                             "start_ms": int(m.group(1)),
                             "end_ms": int(m.group(2))})
            buf = ""
    return timeline


def main(ids: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for cid in ids:
        print(f"= {cid}: ダウンロード中…", flush=True)
        cha = fetch(f"{BASE}/cha/{cid}.cha").decode("utf-8", errors="replace")
        raw = fetch(f"{BASE}/wav/{cid}.wav")
        dur = convert_wav(raw, OUT / f"{cid}.wav")
        tl = parse_cha(cha)
        speakers = sorted({t["speaker"] for t in tl})
        gt = {
            "session": cid,
            "source": "CABank Japanese CallHome Corpus (doi:10.21415/T5H59V)",
            "kind": "timeline",           # 発話区間ラベルでなく時間タイムライン
            "speakers": speakers,
            "timeline": tl,
        }
        (OUT / f"{cid}.gt.json").write_text(
            json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
        total = sum(t["end_ms"] - t["start_ms"] for t in tl) / 1000
        print(f"  音声 {dur/60:.1f}分 / 発話 {len(tl)}区間 / 話者 {speakers} / "
              f"ラベル済み {total/60:.1f}分 → data/callhome/{cid}.*")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
