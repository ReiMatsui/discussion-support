#!/usr/bin/env python3
"""GPT-Live-1 の疎通と挙動を5分で確かめる（組み込み前の確認）.

確かめること:
  1. 接続して session.started が返るか、その所要時間
  2. 指示を送ると日本語で話すか、指示から最初の音声までの遅延、文字起こしの内容
  3. （--mic N）室内の音声を N 秒送ったとき、指示なしに勝手に話し出さないか
  4. 終了時の usage（課金の確認）

使い方（Mac。.env の OPENAI_API_KEY を使う）:
    uv run python scripts/live_smoke.py
    uv run python scripts/live_smoke.py --mic 30          # マイクを30秒聞かせる
    uv run python scripts/live_smoke.py --voice cedar
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--voice", default=None)
    ap.add_argument("--mic", type=float, default=0.0, help="マイク音声を送る秒数（0で送らない）")
    ap.add_argument("--text", default="参加者に、今日の議題を一言で確認する短い問いかけをしてください。")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        sys.exit("OPENAI_API_KEY が未設定です（.env）")

    from das.asr.live.agents._live import LIVE_DEFAULT_VOICE, LiveAgent

    said: list[tuple[float, str]] = []
    t0 = time.monotonic()
    agent = LiveAgent(api_key=key, voice=args.voice or LIVE_DEFAULT_VOICE,
                      listen="always" if args.mic else "never")
    agent.debug_events = True
    agent.on_ai_utterance = lambda text: said.append((time.monotonic() - t0, text))
    agent.on_speech_start = lambda: print(f"  [{time.monotonic() - t0:5.1f}s] 音声開始", flush=True)

    print("1) 接続", flush=True)
    agent.connect()
    if not agent._connected or agent._conn_error:
        sys.exit(f"接続できません: {agent._conn_error}")
    print(f"  session.started まで {time.monotonic() - t0:.1f}s")

    print("2) 指示を送って話させる", flush=True)
    agent.feed("Aさん", "今日は新製品の名前を決める会議ですね。")
    agent.feed("Bさん", "はい。候補は三つあります。")
    t1 = time.monotonic()
    agent.trigger(manual_request={"request": args.text})
    deadline = t1 + 25
    last_print = 0.0
    while time.monotonic() < deadline and (agent._responding or agent.ai_speaking):
        time.sleep(0.1)
        if time.monotonic() - last_print >= 2:
            last_print = time.monotonic()
            gap = (time.monotonic() - agent._last_audio_at) if agent._last_audio_at else -1
            print(f"  [{time.monotonic() - t0:5.1f}s] 状態 responding={agent._responding} "
                  f"speaking={agent.ai_speaking} 再生待ち={agent._audio_q.qsize()} "
                  f"最終delta {gap:.1f}s前 音声{agent._audio_bytes_this_turn // 48}ms "
                  f"文字{len(agent._ai_text_buf)}字", flush=True)
    if agent._responding or agent.ai_speaking:
        print("  25秒経っても発話が終わっていません（終端判定が効いていない）")
    if agent._last_speak_latency_ms is not None:
        print(f"  指示→最初の音声 {agent._last_speak_latency_ms:.0f} ms")
    else:
        print("  25秒以内に音声が来ませんでした")
    for at, text in said:
        print(f"  [{at:5.1f}s] 発話: {text}")

    if args.mic:
        print(f"3) マイクを {args.mic:.0f} 秒聞かせる（普通に会話してください。勝手に話し出すかを見る）",
              flush=True)
        import numpy as np
        import sounddevice as sd
        n_before = len(said)

        def cb(indata, frames, t, status):
            pcm = (np.clip(indata[:, 0], -1, 1) * 32767).astype("<i2").tobytes()
            agent.feed_audio(pcm)

        with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                            callback=cb, blocksize=1600):
            time.sleep(args.mic)
        time.sleep(1.5)
        spontaneous = said[n_before:]
        if spontaneous:
            print(f"  指示なしの発話 {len(spontaneous)} 回:")
            for at, text in spontaneous:
                print(f"    [{at:5.1f}s] {text}")
        else:
            print("  指示なしの発話はありませんでした")

    print("4) 終了", flush=True)
    agent.close()
    time.sleep(0.5)


if __name__ == "__main__":
    main()
