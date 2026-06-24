"""シミュレーション用ディスカッション（録音済み音声の再生）."""
from __future__ import annotations

import contextlib
import queue
import re
import threading
import time
from typing import ClassVar

import numpy as np

from .._constants import SR
from ._realtime import RealtimeAgent


class DiscussionSimulator:
    """Chat APIで対話テキストを生成し、TTS APIで話者別音声を出力する.

    ファシリテーターの介入テスト用。生成された音声は既存のASRパイプラインに流れ、
    声紋で話者分離される。ファシリテーターの介入テキストを受け取ると、
    次のターンで反応が変化する。

    使い方:
      sim = DiscussionSimulator(api_key, topic="AIツール導入の是非")
      sim.start(audio_q, stop_event)  # バックグラウンドで音声生成開始
      sim.inject_facilitator("少し視点を変えてみましょう")  # 介入を注入
      sim.shutdown()
    """

    # 話者 → TTSボイス（ファシリテーターのalloyと被らないよう選定）
    SPEAKERS: ClassVar[dict] = {
        "松井": "echo",
        "田中": "nova",
        "佐藤": "onyx",
    }
    DEFAULT_PAUSE = 1.6   # 発話間の無音（秒）。話者分離の切れ目を作るため長めに

    _SYSTEM_PROMPT = """\
あなたは3人の会議参加者（松井、田中、佐藤）の議論を生成するシミュレーターです。

ルール:
- 1回の応答で **1人の発言だけ** を生成してください（複数人の発言を含めない）
- 必ず1行で、改行や2人目の「話者名:」を入れないでください
- フォーマット: 「話者名: 発言内容」（例: 松井: コストの問題を議論しましょう）
- 話者を自然に交代させてください
- 各発言は1〜3文、自然な会話の長さにしてください
- ファシリテーターから介入があった場合、それに反応して議論の方向を変えてください
- 議論は日本語で行ってください"""

    def __init__(self, api_key: str, topic: str, scenario: str | None = None):
        self.api_key = api_key
        self.topic = topic
        self.scenario = scenario
        self._stop = threading.Event()
        self._audio_q: queue.Queue | None = None
        self._facilitator_q: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._history: list[dict] = []
        self._play_out = None  # スピーカー再生用OutputStream
        self._agent_ref: RealtimeAgent | None = None  # ファシリテーター待機用

    def start(self, audio_q: queue.Queue, stop: threading.Event,
              play_audio: bool = False):
        """バックグラウンドで議論音声の生成を開始する."""
        self._audio_q = audio_q
        self._stop = stop
        self._play_audio = play_audio
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def inject_facilitator(self, text: str):
        """ファシリテーターの介入テキストを議論に注入する."""
        self._facilitator_q.put(text)

    def shutdown(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self):
        """メインループ: ターンごとにテキスト生成→TTS音声化→パイプライン送出."""
        import openai
        client = openai.OpenAI(api_key=self.api_key)

        # スピーカー再生のセットアップ
        if self._play_audio:
            try:
                import sounddevice as sd
                self._play_out = sd.OutputStream(
                    samplerate=SR, channels=1, dtype="float32")
                self._play_out.start()
            except Exception as e:
                print(f"# Simulator: スピーカー再生の初期化失敗: {e}", flush=True)
                self._play_out = None

        # シナリオ固有の初期コンテキスト
        scenario_hint = ""
        if self.scenario:
            hints = {
                "stalled": "議論を意図的に停滞させてください。同じ論点（特にコスト）を繰り返し、新しい視点が出ないようにしてください。",
                "biased": "全員が提案に賛成する偏った議論にしてください。リスクや反対意見が一切出ないようにしてください。",
                "derailed": "最初の2-3発話後に雑談に脱線してください。本来の議題から完全に離れてください。",
                "consensus_needed": "参加者間で意見が対立する議論にしてください。合意に至らないまま堂々巡りしてください。",
                "healthy": "建設的で生産的な議論にしてください。参加者が互いの意見を尊重し、論点を深掘りしてください。",
                "imbalanced": "松井と田中が活発に議論し、佐藤はほとんど発言しないようにしてください。佐藤の登場は時々の短い相槌程度にとどめ、発話量に明確な偏りを作ってください（ファシリテーターの声かけテスト用）。",
            }
            scenario_hint = hints.get(self.scenario, "")

        self._history = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content":
             f"議題: {self.topic}\n{scenario_hint}\n\n議論を始めてください。"},
        ]

        # 冒頭の無音
        self._send_silence(1.5)
        print(f"# Simulator: 議論開始 — 議題「{self.topic}」", flush=True)

        turn = 0
        while not self._stop.is_set():
            # ファシリテーター介入があればコンテキストに注入
            facilitator_msgs = []
            while True:
                try:
                    msg = self._facilitator_q.get_nowait()
                    facilitator_msgs.append(msg)
                except queue.Empty:
                    break
            if facilitator_msgs:
                for msg in facilitator_msgs:
                    self._history.append({
                        "role": "user",
                        "content": f"[ファシリテーターからの介入]: {msg}\n\n"
                                   "この介入を受けて、次の参加者の反応を生成してください。"
                    })
                    print("# Simulator: ファシリテーター介入を受信 → 反応生成",
                          flush=True)

            # Chat APIでテキスト生成
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=self._history,
                    max_tokens=200,
                    temperature=0.9,
                )
                text = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"# Simulator: Chat API エラー: {e}", flush=True)
                time.sleep(2)
                continue

            # 「話者名: 発言」をパース
            speaker, utterance = self._parse_turn(text)
            if not speaker or not utterance:
                # パース失敗 → 再試行
                self._history.append({"role": "assistant", "content": text})
                self._history.append({"role": "user", "content":
                    "フォーマットが正しくありません。「話者名: 発言」の形式で1人の発言だけ生成してください。"})
                continue

            self._history.append({"role": "assistant", "content": text})
            # 会話履歴が長くなりすぎないよう制限
            if len(self._history) > 40:
                self._history = self._history[:2] + self._history[-30:]

            # 次のターンを促す
            self._history.append({"role": "user", "content": "次の参加者の発言を生成してください。"})

            turn += 1
            voice = self.SPEAKERS.get(speaker, "shimmer")
            print(f"# Simulator [{turn}] {speaker}({voice}): {utterance[:50]}...",
                  flush=True)

            # ファシリテーターが話している間は待機（被り防止）
            self._wait_for_facilitator()

            # TTS → PCM → パイプライン
            pcm = self._tts_to_pcm(client, utterance, voice)
            # TTS生成中にファシリテーターが話し始めた可能性 → 送信直前に再度待機
            self._wait_for_facilitator()
            if pcm and not self._stop.is_set():
                self._send_pcm(pcm)
                self._send_silence(self.DEFAULT_PAUSE)

        if self._play_out:
            self._play_out.stop()
            self._play_out.close()
        # senderスレッドに終端を通知
        if self._audio_q is not None:
            self._audio_q.put(None)
        print("# Simulator: 終了", flush=True)

    def _wait_for_facilitator(self):
        """ファシリテーターが話している/応答生成中の間は待機（被り防止）."""
        while not self._stop.is_set() and self._agent_ref is not None and (
                self._agent_ref.ai_speaking or self._agent_ref._responding):
            time.sleep(0.1)

    def _parse_turn(self, text: str) -> tuple[str | None, str | None]:
        """「話者名: 発言」をパースする.

        モデルが複数話者の行をまとめて返すことがあるため、最初の1人の発言だけを
        採用する（DOTALLで全部を1人に取り込むと、別々の声に分離できなくなる）。
        """
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # 「松井: テキスト」or 「松井：テキスト」（発言は次の話者行直前まで）
            m = re.match(r"^([^\s:：]+)\s*[:：]\s*(.+)$", line)
            if m and m.group(1) in self.SPEAKERS:
                return m.group(1), m.group(2).strip()
            # 最初の非空行が話者形式でなければパース失敗扱い
            return None, None
        return None, None

    def _tts_to_pcm(self, client, text: str, voice: str) -> bytes:
        """TTS APIで音声生成し、16kHz 16bit mono PCMを返す."""
        try:
            resp = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                response_format="pcm",  # 24kHz 16bit mono PCM
            )
            pcm_24k = resp.content
        except Exception as e:
            print(f"# Simulator: TTS エラー: {e}", flush=True)
            return b""

        # 24kHz → 16kHz リサンプル
        samples_24k = np.frombuffer(pcm_24k, dtype="<i2").astype(np.float32)
        n_out = int(len(samples_24k) * 16000 / 24000)
        if n_out < 2:
            return b""
        indices = np.linspace(0, len(samples_24k) - 1, n_out)
        idx_floor = indices.astype(int)
        idx_ceil = np.minimum(idx_floor + 1, len(samples_24k) - 1)
        frac = indices - idx_floor
        samples_16k = samples_24k[idx_floor] * (1 - frac) + samples_24k[idx_ceil] * frac
        return np.clip(samples_16k, -32768, 32767).astype("<i2").tobytes()

    def _send_pcm(self, pcm: bytes):
        """PCMをチャンクに分割してaudio_qとスピーカーに送出."""
        step_bytes = int(SR * 0.12) * 2  # 120ms分のバイト数
        for off in range(0, len(pcm), step_bytes):
            if self._stop.is_set():
                return
            chunk = pcm[off:off + step_bytes]
            # パイプラインに送出（Soniox ASRへ）
            self._audio_q.put(chunk)
            # スピーカー再生
            if self._play_out:
                samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
                with contextlib.suppress(Exception):
                    self._play_out.write(samples.reshape(-1, 1))
            else:
                time.sleep(0.12)  # 再生なしの場合はリアルタイムペースを維持

    def _send_silence(self, duration: float):
        """無音をパイプラインに送出."""
        n_samples = int(SR * duration)
        silence = b"\x00\x00" * n_samples
        step_bytes = int(SR * 0.12) * 2
        for off in range(0, len(silence), step_bytes):
            if self._stop.is_set():
                return
            self._audio_q.put(silence[off:off + step_bytes])
            if self._play_out:
                z = np.zeros(min(step_bytes // 2, n_samples - off // 2), dtype=np.float32)
                with contextlib.suppress(Exception):
                    self._play_out.write(z.reshape(-1, 1))
            else:
                time.sleep(0.12)
