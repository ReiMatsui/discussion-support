#!/usr/bin/env python3
"""討論台本(turns.jsonl) → 複数ボイスTTS → 1本の討論wav を生成する実験ハーネス.

話者を集めずに音声込みE2E実験を回すためのツール。台本が正解ラベルなので、
話者帰属・介入の双方を採点できる。

使い方:
  uv run python scripts/make_debate_wav.py tests/fixtures/cafeteria_transcript.jsonl \
      -o data/debate.wav
  # 再生せずに擬似ライブ投入:
  uv run das listen-soniox --skip-docs --soniox-args "--wav data/debate.wav --no-polish"

- TTS: OpenAI tts-1 (OPENAI_API_KEY を使用)。話者名→ボイスは出現順に
  alloy/echo/fable/onyx/nova/shimmer を割当（日本語可）。
- 発話間に0.6〜1.2秒のランダム間隔を挿入（自然なターンテイキングの模擬）。
- 出力: 16kHz mono wav（live_sonioxの--wavにそのまま使える）。
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import wave
from pathlib import Path

VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
SR = 16000


def tts(client, text: str, voice: str) -> bytes:
    """1発話ぶんのTTS → 16kHz mono PCM(s16le)を返す."""
    res = client.audio.speech.create(
        model="tts-1", voice=voice, input=text, response_format="wav"
    )
    with wave.open(io.BytesIO(res.content)) as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    import numpy as np

    x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    if sr != SR:  # 簡易リサンプル
        idx = (np.arange(int(len(x) * SR / sr)) * sr / SR).astype(int)
        x = x[np.clip(idx, 0, len(x) - 1)]
    return x.astype("<i2").tobytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("script", type=Path, help="turns.jsonl (turn_id/speaker/text)")
    ap.add_argument("-o", "--out", type=Path, default=Path("data/debate.wav"))
    ap.add_argument("--gap", type=float, default=0.9, help="発話間の平均間隔(秒)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv()   # .env の OPENAI_API_KEY を読む
    from openai import OpenAI

    try:
        client = OpenAI()
    except Exception:
        sys.exit("OPENAI_API_KEY が見つかりません（.env か環境変数に設定してください）")
    rng = random.Random(args.seed)

    turns = [json.loads(line) for line in args.script.read_text(encoding="utf-8").splitlines() if line.strip()]
    voice_of: dict[str, str] = {}
    pcm = bytearray()
    answer_key = []
    t_cursor = 0.0
    for t in turns:
        spk, text = str(t["speaker"]), t["text"]
        if spk not in voice_of:
            voice_of[spk] = VOICES[len(voice_of) % len(VOICES)]
        print(f"[tts] {spk}({voice_of[spk]}): {text[:40]}...", file=sys.stderr)
        audio = tts(client, text, voice_of[spk])
        gap = max(0.2, rng.gauss(args.gap, 0.25))
        pcm += b"\x00" * int(SR * gap) * 2
        t_cursor += gap
        answer_key.append({"start_s": round(t_cursor, 2), "speaker": spk, "text": text})
        pcm += audio
        t_cursor += len(audio) / 2 / SR

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(pcm))
    key_path = args.out.with_suffix(".answer.json")
    key_path.write_text(json.dumps(
        {"voices": voice_of, "turns": answer_key}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"出力: {args.out} ({t_cursor/60:.1f}分) / 正解ラベル: {key_path}")


if __name__ == "__main__":
    main()
