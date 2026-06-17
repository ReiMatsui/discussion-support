#!/usr/bin/env python3
"""重なり量を制御した正解付きテスト音声(3条件)を生成する.

条件:
  A_clean   : きれいなターンテイキング(間隔0.8s)
  B_mild    : 発話の20%が直前の発話に0.3〜1.0秒食い込む(相槌的な重なり)
  C_heavy   : 40%が食い込み + 完全同時発話(2人が丸ごと同時)を3箇所挿入

出力: data/overlap_test/<条件>.wav と <条件>.answer.json
  answer.json: {"turns": [{start_s, end_s, speaker, text, overlapped}]}

使い方:
  uv run python scripts/make_overlap_testset.py            # 内蔵台本(4人討論)
  uv run python scripts/make_overlap_testset.py --script 任意.turns.jsonl

採点まで:
  uv run das listen-soniox を使わず、文字起こし層単体で:
  cd 任意 && uv run python -m das.asr.soniox_live --wav data/overlap_test/A_clean.wav --no-open --no-polish
  uv run python scripts/score_overlap_test.py data/overlap_test/A_clean.answer.json transcripts/<その時刻>.turns.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import wave
from pathlib import Path

SR = 16000
VOICES = ["alloy", "echo", "nova", "onyx", "shimmer", "fable"]

# 内蔵台本: 4人・対立構造あり・相槌入り(介入も発火しやすい)
SCRIPT = [
    ("青木", "今日は新製品の価格設定について決めたいと思います。私は思い切って低価格で市場を取りに行くべきだと思います。"),
    ("石田", "低価格は危険だと思いますよ。一度下げた価格は上げられないし、ブランド価値が毀損されます。"),
    ("青木", "でも競合は既に2割安い価格で出してきています。様子を見ている余裕はありません。"),
    ("内田", "なるほど。"),
    ("内田", "原価構造から見ると、2割下げると粗利がほぼ消えます。量で取り返すには販売数が3倍必要です。"),
    ("江口", "3倍は現実的じゃないですね。"),
    ("石田", "そうそう、だから価格じゃなくて機能で差別化すべきなんです。"),
    ("青木", "機能差別化は開発に半年かかります。その間に市場を取られたら意味がないでしょう。"),
    ("内田", "うん。"),
    ("江口", "ちょっと整理すると、論点は速度を取るか利益率を取るかですよね。"),
    ("石田", "私は利益率派です。安売りで取ったシェアは安売りでしか守れません。"),
    ("内田", "実はサブスクリプション型にすれば初期価格を下げつつ生涯収益は守れるという試算があります。"),
    ("青木", "それは面白いですね。初期障壁を下げられるなら賛成です。"),
    ("江口", "サブスクは解約率次第では赤字になりませんか。"),
    ("内田", "解約率が月5%を超えると赤字です。業界平均は7%なので楽観はできません。"),
    ("石田", "ほら、やっぱりリスクが高い。"),
    ("青木", "でも何もしないリスクの方が大きいと思いますけどね。"),
    ("江口", "では解約率を下げる施策とセットなら、サブスク案を検討する価値はありそうですね。"),
]


def tts_openai(client, text: str, voice: str) -> bytes:
    res = client.audio.speech.create(model="tts-1", voice=voice, input=text,
                                     response_format="wav")
    import numpy as np
    with wave.open(io.BytesIO(res.content)) as w:
        sr, nch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    if sr != SR:
        idx = (np.arange(int(len(x) * SR / sr)) * sr / SR).astype(int)
        x = x[np.clip(idx, 0, len(x) - 1)]
    return x.astype("<i2").tobytes()


def assemble(clips: list[tuple[str, str, bytes]], *, overlap_p: float,
             n_simul: int, seed: int) -> tuple[bytes, list[dict]]:
    """発話クリップ列をタイムラインに配置し、(PCM, 正解turns) を返す."""
    import numpy as np
    rng = random.Random(seed)
    events = []   # (start_sample, samples(np.int32), speaker, text, overlapped)
    cursor = 0
    simul_idx = set()
    if n_simul:
        cand = [i for i in range(1, len(clips)) if len(clips[i][2]) > SR]  # 1秒以上
        simul_idx = set(rng.sample(cand, min(n_simul, len(cand))))
    prev_end = 0
    for i, (spk, text, pcm) in enumerate(clips):
        x = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
        if i in simul_idx and events:
            start = events[-1][0]          # 直前の発話と丸ごと同時
            ov = True
        elif i > 0 and rng.random() < overlap_p:
            ov_len = int(SR * rng.uniform(0.3, 1.0))
            start = max(0, prev_end - ov_len)   # 食い込み
            ov = True
        else:
            start = prev_end + int(SR * max(0.25, rng.gauss(0.8, 0.2)))
            ov = False
        events.append((start, x, spk, text, ov))
        prev_end = max(prev_end, start + len(x))
        cursor = max(cursor, start + len(x))
    mix = np.zeros(cursor + SR, dtype=np.int32)
    answer = []
    for start, x, spk, text, ov in events:
        mix[start:start + len(x)] += x
        answer.append({"start_s": round(start / SR, 2), "end_s": round((start + len(x)) / SR, 2),
                       "speaker": spk, "text": text, "overlapped": ov})
    mix = np.clip(mix * 0.8, -32767, 32767).astype("<i2")
    return mix.tobytes(), answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", type=Path, default=None, help="turns.jsonl (省略時は内蔵台本)")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("data/overlap_test"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.script:
        turns = [(str(j["speaker"]), j["text"]) for j in
                 (json.loads(l) for l in args.script.read_text(encoding="utf-8").splitlines() if l.strip())]
    else:
        turns = SCRIPT

    from dotenv import load_dotenv
    load_dotenv()
    from openai import OpenAI
    try:
        client = OpenAI(timeout=30.0, max_retries=3)   # ハング対策
    except Exception:
        sys.exit("OPENAI_API_KEY が見つかりません（.env か環境変数に設定してください）")

    cache_dir = args.outdir / ".tts_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    voice_of: dict[str, str] = {}
    clips = []
    import hashlib
    for spk, text in turns:
        if spk not in voice_of:
            voice_of[spk] = VOICES[len(voice_of) % len(VOICES)]
        key = hashlib.md5(f"{voice_of[spk]}|{text}".encode()).hexdigest()[:16]
        cpath = cache_dir / f"{key}.pcm"
        if cpath.exists():
            print(f"[tts] {spk}({voice_of[spk]}): (キャッシュ) {text[:30]}…", file=sys.stderr)
            pcm = cpath.read_bytes()
        else:
            print(f"[tts] {spk}({voice_of[spk]}): {text[:36]}…", file=sys.stderr)
            pcm = tts_openai(client, text, voice_of[spk])
            cpath.write_bytes(pcm)
        clips.append((spk, text, pcm))

    conds = {"A_clean": dict(overlap_p=0.0, n_simul=0),
             "B_mild": dict(overlap_p=0.2, n_simul=0),
             "C_heavy": dict(overlap_p=0.4, n_simul=3)}
    args.outdir.mkdir(parents=True, exist_ok=True)
    for ci, (name, kw) in enumerate(conds.items()):
        # 条件ごとに乱数系列を変え、食い込み数が期待値の半分未満なら引き直す
        expect = kw["overlap_p"] * (len(clips) - 1)
        for retry in range(20):
            pcm, answer = assemble(clips, seed=args.seed + ci * 1000 + retry, **kw)
            n_ov = sum(1 for t in answer if t["overlapped"])
            if n_ov >= max(kw["n_simul"], int(expect / 2)) or kw["overlap_p"] == 0:
                break
        wav_path = args.outdir / f"{name}.wav"
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(pcm)
        (args.outdir / f"{name}.answer.json").write_text(
            json.dumps({"voices": voice_of, "turns": answer}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        dur = len(pcm) / 2 / SR
        n_ov = sum(1 for t in answer if t["overlapped"])
        print(f"{name}: {wav_path} ({dur/60:.1f}分, 重なり{n_ov}/{len(answer)}発話)")


if __name__ == "__main__":
    main()
