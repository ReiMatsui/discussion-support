"""1マイク + Soniox リアルタイム議事録ツール（日本語・話者分離内蔵）.

【das統合版】speaker-attribution リポジトリ v1.0 からの移植（上流は凍結、以後はこちらが正）。
das連携フック: モジュール変数 ON_UTTERANCE に callable(speaker:str, text:str) を設定すると、
確定発話ごとに呼ばれる（das listen-soniox がオーケストレータへ流すのに使用）。

1本のマイクで「誰が・何を」をライブ取得する本線ツール。
Sonioxのストリーミング(WebSocket)に音声を流し、speaker付きトークンが返るので、
多マイク・ゲート・同期なしで who-said-what が出る。

機能:
  - 話者ごとに色分けしたライブ表示（確定前のテキストは薄く表示）
  - Markdown議事録 + HTML を transcripts/ に自動保存（発話確定ごと＝クラッシュ安全）
  - HTMLはブラウザ自動オープン、ライブ中2秒ごと自動更新（--no-openで無効）
  - 声紋プロファイル方式の話者特定（登録不要で自動補正）。判定は2経路のみ:
      ① 即時判定: 声紋が強一致した発話はその場で人物確定（入れ替わりも補正）
      ② それ以外は3発話バッファ: 一貫した3発話を束ねて「既存人物に合流 or 新規人物N」
      しきい値は2層: モデル別既定値 → 人物別しきい値(本人の一致sim中央値-0.12、
      新声の巻き取り防止。厳しくする方向にのみ働く)。
      不変条件: 一度確定した人物キーは書き換えない（遡及置換は 話者N→人物N の昇格のみ）。
      「1=松井」で実名化、実名のみ voices.json に永続化 → 次回から自動で実名表示。
  - 終了時に清書: 録音全体を非同期APIで再処理し、全文脈の話者分離＋声紋実名対応の
    最終版(日時.final.md/.html)を自動生成（高速応酬でのRT分離崩れへの対策。--polishで有効化）
  - 「fix 2=1」「fix 人物2=人物1」で誤った話者の統合（過去の発言も修正）
  - 診断ログ(日時.diag.jsonl): 発話ごとの判定根拠を常時記録（問題解析用）

準備(Mac):
  uv add websockets sounddevice
  export SONIOX_API_KEY=...   # https://console.soniox.com で取得

使い方:
  uv run python offshelf/live_soniox.py            # 実マイクでライブ
  uv run python offshelf/live_soniox.py --wav offshelf/ami_raw/mic0.wav  # ファイル擬似ライブ
  実行中: 「1=松井」Enter で話者登録 / Ctrl+C で終了（保存先を表示）
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import queue
import re
import sys
import threading
import time
import numpy as np

from das.asr._constants import (
    _AGENT_CONV_SILENCE,
    _AGENT_DEBATE_SILENCE,
    _AGENT_SILENCE,
    _AGENT_TRIGGER,
    AGENT_SPEAKER,
    AGENT_VOICES,
    _BACKCHANNEL_RE,
    CLEAR_LINE,
    DIM,
    HTML_PALETTE,
    HTML_TMPL,
    _INTERRUPT_MIN_CHARS,
    OPENAI_API,
    PALETTE,
    RESET,
    SM_WS_URL,
    SR,
    _TOPIC_PROMPT,
    WS_URL,
)
from das.asr._conversation_partner import ConversationPartner
from das.asr._discussion_simulator import DiscussionSimulator
from das.asr._realtime_agent import RealtimeAgent
from das.asr._voice_profiles import VoiceProfiles

ON_UTTERANCE = None   # das連携: 確定発話ごとに (話者表示名, テキスト) で呼ばれる
_SYS_HOOK = None      # main()実行中のみ登録される(add_sys+saveへの橋)


def post_system(text: str) -> None:
    """das連携: ライブ議事録のタイムラインにシステム行(💡介入など)を外部から追加する."""
    if _SYS_HOOK is not None:
        _SYS_HOOK(text)


def sm_to_res(msg: dict, lang: str = "ja") -> dict:
    """SpeechmaticsのRTメッセージをSoniox互換のトークン列に翻訳する.

    供給源を差し替えるだけで、声紋層・表示・保存・清書は無変更で動く。
    話者ラベル: S1→"1"(表示は話者1)、不明UUはそのまま。
    """
    m = msg.get("message")
    if m == "Error":
        return {"error_code": msg.get("type"), "error_message": msg.get("reason")}
    if m == "EndOfTranscript":
        return {"finished": True, "tokens": []}
    if m == "EndOfUtterance":
        return {"tokens": [{"text": "<end>", "is_final": True}]}
    if m in ("AddTranscript", "AddPartialTranscript"):
        final = m == "AddTranscript"
        toks = []
        for r in msg.get("results", []):
            alts = r.get("alternatives") or []
            content = alts[0].get("content", "") if alts else ""
            if not content:
                continue
            spk = (alts[0].get("speaker") or "UU")
            if spk.startswith("S") and spk[1:].isdigit():
                spk = spk[1:]
            if (lang not in ("ja", "zh", "cmn", "yue") and toks
                    and r.get("type") == "word"):
                content = " " + content   # 分かち書き言語は語間スペースを補う
            toks.append({"text": content, "speaker": spk,
                         "start_ms": int(r["start_time"] * 1000),
                         "end_ms": int(r["end_time"] * 1000),
                         "is_final": final})
        return {"tokens": toks}
    return {"tokens": []}   # RecognitionStarted / AudioAdded / Info / Warning 等は無視

def load_env(path: str = ".env") -> None:
    """プロジェクト直下の .env からAPIキー等を読み込む（既に設定済みの環境変数を優先）.

    形式: KEY=VALUE の行（#始まりはコメント）。依存なしの最小実装。
    """
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def fmt_ts(ms: int | None) -> str:
    if ms is None:
        return "--:--"
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------- 論点抽出（非同期LLM処理） ----------


def _extract_topics(utterances: list[dict], existing: list[str],
                    api_key: str, model: str) -> list[dict]:
    """OpenAI APIで新論点を抽出する（同期呼び出し、バックグラウンドスレッド用）."""
    if not utterances or not api_key:
        return []
    utt_text = "\n".join(f"- {u['speaker']}: {u['text']}" for u in utterances)
    ex_text = "\n".join(f"- {t}" for t in existing) if existing else "（まだなし）"
    prompt = _TOPIC_PROMPT.format(existing=ex_text, utterances=utt_text)
    # GPT-5系/o系はtemperature指定不可、max_tokensはmax_completion_tokensに改名
    name = model.lower()
    is_new = name.startswith(("gpt-5", "o1", "o3", "o4"))
    params: dict = {"model": model,
                    "messages": [{"role": "user", "content": prompt}]}
    if not is_new:
        params["temperature"] = 0.3
        params["max_tokens"] = 512
    else:
        params["max_completion_tokens"] = 512
    body = json.dumps(params).encode()
    import urllib.request
    req = urllib.request.Request(OPENAI_API, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
        text = resp["choices"][0]["message"]["content"].strip()
        # JSON配列を抽出（前後にmarkdownコードブロックがある場合も対応）
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception:
        return []


# ---------- 清書（会議後の非同期再処理） ----------
# RTの話者分離は速い応酬で崩れる(実測: 高速応酬区間で1ラベルに併合)。非同期APIは
# 全文脈を見られるため分離精度が大幅に高い(公式)。終了時に録音全体を再処理し、
# async話者を声紋プロファイルで実名に対応づけて「清書版」議事録を作る。

API_BASE = "https://api.soniox.com"


def _wav_bytes(pcm: bytes) -> bytes:
    import struct
    n = len(pcm)
    return (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16) +
            b"data" + struct.pack("<I", n) + pcm)


def _api(api_key: str, method: str, path: str, body=None, ctype=None, timeout=120):
    import urllib.request
    req = urllib.request.Request(API_BASE + path, data=body, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    if ctype:
        req.add_header("Content-Type", ctype)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def _group_tokens(tokens: list[dict]) -> list[tuple]:
    """async結果のトークン列を (start_ms, end_ms, 話者, テキスト) の発話列へ."""
    utts = []
    cur = None   # [start, end, spk, text]
    for tk in tokens:
        text = tk.get("text") or ""
        if not text or text == "<end>":
            continue
        spk = tk.get("speaker")
        if cur is None or spk != cur[2]:
            if cur and cur[3].strip():
                utts.append(tuple(cur))
            cur = [tk.get("start_ms"), tk.get("end_ms"), spk, ""]
        if tk.get("end_ms") is not None:
            cur[1] = tk["end_ms"]
        cur[3] += text
    if cur and cur[3].strip():
        utts.append(tuple(cur))
    return utts


def _map_speakers(utts: list[tuple], pcm: bytes, tracker) -> dict:
    """async話者ID → 表示キー（人物との1対1割当）.

    各async話者の長い発話の声紋平均をプロファイルと照合し、類似の高いペアから
    貪欲に1対1で割り当てる。1対1にしないと、同一再生チェーン等で複数のasync話者が
    同じ人物に畳まれ、清書の話者数がライブより減る事故が起きる（2026-06-12実測）。
    """
    mapping = {}
    if tracker is None:
        return mapping
    by_spk: dict = {}
    for s, e, spk, _ in utts:
        if s is None or e is None or spk is None:
            continue
        by_spk.setdefault(str(spk), []).append((e - s, s, e))
    # アクティブなプロファイルのみ対象（セッション中に使ったもの＋自動登録）
    active = {k: v for k, v in tracker.profiles.items() if k in tracker._active_keys}
    pairs = []   # (sim, async話者, 人物)
    for spk, segs in by_spk.items():
        segs = [x for x in sorted(segs, reverse=True) if x[0] >= 1200][:6]
        embs = []
        for _, s, e in segs:
            wav = np.frombuffer(pcm[s * 32: e * 32], dtype="<i2").astype(np.float32) / 32768.0
            emb = tracker._embed(wav)
            if emb is not None:
                embs.append(emb)
        if embs:
            prof = np.mean(embs, axis=0)
            prof = prof / np.linalg.norm(prof)
            for n, v in active.items():
                sim = float(np.dot(v, prof))
                if sim >= tracker.dedupe:
                    pairs.append((sim, spk, n))
    used_spk, used_person = set(), set()
    for sim, spk, n in sorted(pairs, reverse=True):
        if spk in used_spk or n in used_person:
            continue
        mapping[spk] = n
        used_spk.add(spk)
        used_person.add(n)
    return mapping


def polish(api_key: str, pcm: bytes, lang: str, tracker, log=print) -> list[dict]:
    """録音全体を非同期APIで再処理し、清書版のrecordsを返す."""
    log("# 清書: 音声をアップロード中…")
    import uuid
    b = "----spkattr" + uuid.uuid4().hex
    body = ((f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"meeting.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
            + _wav_bytes(pcm) + f"\r\n--{b}--\r\n".encode())
    file_id = _api(api_key, "POST", "/v1/files", body,
                   f"multipart/form-data; boundary={b}", timeout=600)["id"]
    tid = None
    try:
        cfg = {"model": "stt-async-v4", "language_hints": [lang],
               "enable_speaker_diarization": True, "file_id": file_id}
        tid = _api(api_key, "POST", "/v1/transcriptions",
                   json.dumps(cfg).encode(), "application/json")["id"]
        log("# 清書: 再処理を待っています…")
        t0 = time.time()
        while True:
            st = _api(api_key, "GET", f"/v1/transcriptions/{tid}")
            if st["status"] == "completed":
                break
            if st["status"] == "error":
                raise RuntimeError(st.get("error_message", "unknown"))
            if time.time() - t0 > 600:
                raise TimeoutError("非同期処理が10分以内に完了しませんでした")
            time.sleep(2)
        tokens = _api(api_key, "GET", f"/v1/transcriptions/{tid}/transcript")["tokens"]
    finally:   # 後始末（失敗しても続行）
        try:
            if tid:
                _api(api_key, "DELETE", f"/v1/transcriptions/{tid}")
            _api(api_key, "DELETE", f"/v1/files/{file_id}")
        except Exception:
            pass
    utts = _group_tokens(tokens)
    log(f"# 清書: {len(utts)}発話を取得、話者を声紋で照合中…")
    mapping = _map_speakers(utts, pcm, tracker)
    return [{"ms": s, "speaker": mapping.get(str(spk), "#" + str(spk)), "text": tx.strip()}
            for s, e, spk, tx in utts]


def _print_line(text: str):
    """ターミナルの現在行をクリアして1行出力."""
    sys.stdout.write(CLEAR_LINE + text + "\n")
    sys.stdout.flush()


class _UIHandler:
    """UIサーバー用HTTPハンドラ（トップレベル定義）.

    BaseHTTPRequestHandlerのサブクラスを動的に生成するファクトリ。
    クロージャ変数の代わりにクラス変数 _state でSessionStateを参照する。

    使い方:
        handler_cls = _UIHandler.create(state)
        httpd = HTTPServer(("127.0.0.1", port), handler_cls)
    """

    @staticmethod
    def create(state: "SessionState") -> type:
        """state を束縛した BaseHTTPRequestHandler サブクラスを返す."""
        from http.server import BaseHTTPRequestHandler

        class Handler(BaseHTTPRequestHandler):
            _state = state

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/?"):
                    try:
                        with open(self._state.html_path, "rb") as f:
                            content = f.read()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(content)
                    except FileNotFoundError:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.end_headers()
                        self.wfile.write("<p>準備中…</p>".encode())
                else:
                    self.send_error(404)

            def do_POST(self):
                s = self._state
                if self.path == "/rename":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    label = str(body.get("label", ""))
                    name = str(body.get("name", ""))
                    if not label or not name:
                        self._json(400, {"error": "label と name を指定してください"})
                        return
                    if s.tracker is not None:
                        old = s.tracker.enroll(label, name)
                        if old is None:
                            self._json(400, {"error": f"話者{label}の音声がまだ足りません"})
                            return
                        s.rekey(old, name)
                        s.add_sys(None, f"「{name}」の声を登録（次回の会議から自動表示）")
                        s.save()
                        _print_line(f"# {name} の声を登録しました（UIから）")
                    else:
                        with s.state_lock:
                            s.names["#" + label] = name
                        s.save()
                        _print_line(f"# 話者{label} → {name}（UIから）")
                    self._json(200, {"ok": True, "name": name})
                elif self.path == "/activate":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    name = str(body.get("name", ""))
                    active = bool(body.get("active", True))
                    if not name:
                        self._json(400, {"error": "name を指定してください"})
                        return
                    if s.tracker is None:
                        self._json(400, {"error": "声紋照合が無効です"})
                        return
                    if active:
                        merged = s.tracker.activate(name)
                        if merged is not None:
                            s.rekey(merged, name)
                            s.add_sys(None, f"「{name}」を有効化（{merged}と統合）")
                            _print_line(f"# {name} を有効化（{merged}と統合、UIから）")
                        else:
                            _print_line(f"# {name} を有効化（UIから）")
                        s.save()
                    else:
                        s.tracker.deactivate(name)
                        _print_line(f"# {name} を無効化（UIから）")
                        s.save()
                    self._json(200, {"ok": True, "name": name, "active": active})
                elif self.path == "/agent":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    if s.agent is None:
                        self._json(400, {"error": "AIエージェントが無効です（--agent で起動してください）"})
                        return
                    mode = body.get("mode")
                    voice = body.get("voice")
                    trigger_n = body.get("trigger_n")
                    if trigger_n is not None:
                        trigger_n = int(trigger_n)
                    s.agent.apply_config(mode=mode, voice=voice, trigger_n=trigger_n)
                    _print_line(f"# AI Agent 設定変更: mode={s.agent.mode} voice={s.agent.voice}"
                                f" trigger={s.agent.trigger_n}（UIから）")
                    s.save()
                    self._json(200, {"ok": True, "mode": s.agent.mode,
                                     "voice": s.agent.voice, "trigger_n": s.agent.trigger_n})
                else:
                    self.send_error(404)

            def _json(self, code, data):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

            def log_message(self, format, *args):
                pass

        return Handler


def _run_topic_worker(state: "SessionState", oai_key: str, oai_model: str):
    """論点抽出のバックグラウンドワーカー（モジュールレベル関数）."""
    while not state.stop.is_set():
        time.sleep(3)
        if not oai_key:
            continue
        with state.state_lock:
            talk_rs = [r for r in state.records if "speaker" in r and r.get("text")]
        n = len(talk_rs)
        if n - state.topic_cursor < state._TOPIC_TRIGGER:
            continue
        window = talk_rs[max(0, n - state._TOPIC_WINDOW):]
        utts = [{"speaker": state.disp_name(r["speaker"]), "text": r["text"]} for r in window]
        with state.topics_lock:
            existing = [t["topic"] for t in state.topics]
        new_topics = _extract_topics(utts, existing, oai_key, oai_model)
        if new_topics:
            ms = window[-1].get("ms")
            with state.topics_lock:
                for t in new_topics:
                    if isinstance(t, dict) and "topic" in t:
                        state.topics.append({"topic": t["topic"],
                                             "speaker": t.get("speaker", "?"),
                                             "ms": ms})
            state.save()
            for t in new_topics:
                if isinstance(t, dict) and "topic" in t:
                    _print_line(f"# 💡論点: {t['topic']}（{t.get('speaker', '?')}）")
        state.topic_cursor = n


def _on_agent_text_factory(state: "SessionState"):
    """ファシリテーター発言コールバックを生成."""
    def _on_agent_text(text: str):
        with state.state_lock:
            state.records.append({"ms": None, "end_ms": None,
                                  "speaker": AGENT_SPEAKER, "text": text.strip()})
            state.color_of(AGENT_SPEAKER)
        if ON_UTTERANCE is not None:
            try:
                ON_UTTERANCE("ファシリテーター", text.strip())
            except Exception:
                pass
        _print_line(f"\x1b[96m[ファシリテーター]\x1b[0m: {text.strip()}")
        state.save()
    return _on_agent_text


def _connect_agent(state: "SessionState", on_text):
    """ファシリテーターAgentのコールバック設定・接続・ワーカー起動."""
    agent = state.agent
    partner = state.partner
    simulator = state.simulator
    if simulator is not None:
        def _on_agent_with_sim(text: str):
            on_text(text)
            if "介入不要" not in text:
                simulator.inject_facilitator(text)
        agent.on_ai_utterance = _on_agent_with_sim
    elif partner is not None:
        def _on_agent_with_partner(text: str):
            on_text(text)
            if "介入不要" not in text and partner._connected:
                partner.interrupt()
                partner.inject_context("ファシリテーター", text)
        agent.on_ai_utterance = _on_agent_with_partner
        def _on_facilitator_speech_start():
            if partner._connected and (partner.ai_speaking or partner._responding):
                partner.interrupt()
        agent.on_speech_start = _on_facilitator_speech_start
    else:
        agent.on_ai_utterance = on_text
    agent.connect()
    threading.Thread(target=_run_agent_worker, args=(state,),
                     daemon=True).start()
    print(f"# AI Agent: mode={agent.mode} voice={agent.voice}"
          f" trigger={agent.trigger_n}（ブラウザから変更可能）", flush=True)


def _on_partner_text_factory(state: "SessionState"):
    """Partner発言コールバックを生成."""
    def _on_partner_text(text: str):
        with state.state_lock:
            state.records.append({"ms": None, "end_ms": None,
                                  "speaker": "パートナー", "text": text.strip()})
            state.color_of("パートナー")
        _print_line(f"\x1b[93mパートナー\x1b[0m: {text.strip()}")
        state.save()
        state._last_utt_time[0] = time.monotonic()
        if state.agent is not None:
            state.agent.feed("パートナー", text, trigger_count=False)
    return _on_partner_text


def _run_agent_worker(state: "SessionState"):
    """バックグラウンドでAI応答のトリガーを管理（ターンテイキング）.

    自然な会話のフロア交代を模倣:
      - 人間のターン: 発話を即座にfeed、沈黙で譲渡 → AIがtrigger
      - AIのターン: 応答を再生。人間の実質的な発話で自動interrupt
      - AIターン終了: フロアを人間に返す（沈黙タイマーをリセット）
    """
    agent = state.agent
    partner = state.partner
    _last_utt_time = state._last_utt_time
    _was_in_echo = state._was_in_echo
    _diag_tick = 0
    while not state.stop.is_set():
        time.sleep(0.5)
        _diag_tick += 1
        if agent is None or not agent._connected or not agent.enabled:
            if _diag_tick % 20 == 0:
                print(f"# [diag] _agent_worker skip: agent={agent is not None}"
                      f" conn={agent._connected if agent else '?'}"
                      f" enabled={agent.enabled if agent else '?'}", flush=True)
            continue
        with state.state_lock:
            _skip = {AGENT_SPEAKER, "パートナー"}
            talk_rs = [r for r in state.records
                       if "speaker" in r and r.get("text")
                       and r.get("speaker") not in _skip]
        n = len(talk_rs)
        if n > state.agent_cursor:
            _last_utt_time[0] = time.monotonic()
            new_texts = [r.get("text", "") for r in talk_rs[state.agent_cursor:]]
            for r in talk_rs[state.agent_cursor:]:
                agent.feed(state.disp_name(r.get("speaker", "")), r.get("text", ""))
            state.agent_cursor = n
            # --- 自動割り込み ---
            _human_spoke = any(len(t.strip()) > _INTERRUPT_MIN_CHARS
                               for t in new_texts)
            if _human_spoke and agent.ai_speaking:
                agent.interrupt()
            if partner is not None and (partner.ai_speaking or partner._responding):
                _real_utterances = [t.strip() for t in new_texts
                                    if t.strip()
                                    and not _BACKCHANNEL_RE.match(t.strip())]
                if _real_utterances:
                    partner.interrupt()
                    for i, utt in enumerate(_real_utterances):
                        is_last = (i == len(_real_utterances) - 1)
                        partner.inject_context(
                            "人間", utt,
                            request_response=is_last)
        # --- ファシリテーター優先 ---
        if (partner is not None
                and (partner.ai_speaking or partner._responding)
                and agent is not None
                and agent.ai_speaking):
            partner.interrupt()
        # エコーウィンドウ中はtriggerしない
        if agent is not None and agent.in_echo_window:
            _was_in_echo[0] = True
            continue
        # Partnerが発話中はtriggerしない
        if partner is not None and (partner.ai_speaking or partner._responding):
            _was_in_echo[0] = True
            continue
        # --- フロア返却 ---
        if _was_in_echo[0]:
            _was_in_echo[0] = False
            _last_utt_time[0] = time.monotonic()
        # モード別トリガー判定
        if _diag_tick % 20 == 0:
            _elapsed = time.monotonic() - _last_utt_time[0]
            print(f"# [diag] agent: mode={agent.mode} pending={agent.pending_count}"
                  f" trigger_n={agent.trigger_n} responding={agent._responding}"
                  f" silence={_elapsed:.1f}s echo={agent.in_echo_window}"
                  f" partner_talk={partner.ai_speaking if partner else '?'}", flush=True)
        if agent.mode == "conversation":
            if (agent.pending_count > 0
                    and time.monotonic() - _last_utt_time[0] > _AGENT_CONV_SILENCE):
                agent.trigger()
        else:
            _silence_thresh = (_AGENT_DEBATE_SILENCE if partner is not None
                               else _AGENT_SILENCE)
            if agent.pending_count >= agent.trigger_n:
                print(f"# [diag] TRIGGER by count: {agent.pending_count}>={agent.trigger_n}", flush=True)
                agent.trigger()
            elif (agent.pending_count > 0
                  and time.monotonic() - _last_utt_time[0] > _silence_thresh):
                print(f"# [diag] TRIGGER by silence: {time.monotonic() - _last_utt_time[0]:.1f}s > {_silence_thresh}s", flush=True)
                agent.trigger()


def _run_stdin_commands(state: "SessionState"):
    """標準入力からの話者リネーム・統合コマンドを処理."""
    while not state.stop.is_set():
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        mfix = re.match(r"^\s*fix\s+(\S+)\s*=\s*(\S+)\s*$", line)
        m = re.match(r"^\s*(\S+?)\s*=\s*(.+?)\s*$", line)
        if mfix:
            src, dst = state.key_of(mfix.group(1)), state.key_of(mfix.group(2))
            if state.tracker is not None:
                state.tracker.remap(src, dst)
            state.rekey(src, dst)
            state.add_sys(None, f"{state.disp_name(src)} を {state.disp_name(dst)} に統合（手動fix）")
            state.save()
            _print_line(f"# {state.disp_name(src)} を {state.disp_name(dst)} に統合しました（過去の発言も修正済み）")
        elif m:
            label, name = m.group(1), m.group(2)
            if state.tracker is not None:
                old = state.tracker.enroll(label, name)
                if old is None:
                    _print_line(f"# 話者{label}の音声がまだ足りません（1秒以上話してから再実行）")
                    continue
                state.rekey(old, name)
                state.add_sys(None, f"「{name}」の声を登録（次回の会議から自動表示）")
                state.save()
                _print_line(f"# {name} の声を登録しました（過去の発言も置換、次回の会議から自動表示）")
            else:
                with state.state_lock:
                    state.names["#" + label] = name
                state.save()
                _print_line(f"# 話者{label} → {name}（過去の発言も置換済み）")
        elif line.strip():
            _print_line("# コマンド: 「1=松井」(声を登録) / 「fix 2=1」「fix 人物2=人物1」(統合) / Ctrl+Cで終了")


def _run_from_mic(state: "SessionState", device):
    """マイクからPCMを読み取り audio_q に送信."""
    import sounddevice as sd
    partner = state.partner
    agent = state.agent

    def cb(indata, frames, t, status):
        pcm = (np.clip(indata[:, 0], -1, 1) * 32767).astype("<i2").tobytes()
        state.audio_q.put(pcm)
        if (partner is not None and partner._connected
                and not partner.in_echo_window
                and not (agent is not None and agent.in_echo_window)):
            partner.feed_audio(pcm)
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        device=device, callback=cb, blocksize=int(SR * 0.1)):
        while not state.stop.is_set():
            time.sleep(0.1)
    state.audio_q.put(None)


def _run_from_wav(state: "SessionState", args):
    """WAVファイルを擬似ライブで送信する.

    Reactive WAV: agentが発話中はWAV再生・ASR送信を一時停止し、
    介入終了後に自動再開する。
    """
    import librosa
    agent = state.agent
    y, _ = librosa.load(args.wav, sr=SR)
    step = int(SR * 0.12)
    out = mic = None
    if args.play or args.join:
        import sounddevice as sd
        out = sd.OutputStream(samplerate=SR, channels=1, dtype="float32")
        out.start()
    if args.join:
        import sounddevice as sd
        mic = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=step)
        mic.start()
    i = 0
    _wav_paused = False
    while not state.stop.is_set():
        if agent is not None and (agent.ai_speaking or agent._responding):
            if not _wav_paused:
                _wav_paused = True
                print("# WAV: AI介入中 — 再生を一時停止", flush=True)
            time.sleep(0.05)
            continue
        if _wav_paused:
            _wav_paused = False
            print("# WAV: 再生を再開", flush=True)
        chunk = np.clip(y[i:i + step], -1, 1).astype("float32") if i < len(y) else \
            np.zeros(0, dtype="float32")
        if len(chunk) < step:
            chunk = np.pad(chunk, (0, step - len(chunk)))
        i += step
        if i - step >= len(y) and mic is None:
            break
        if mic is not None:
            mdata, _ = mic.read(step)
            mix = np.clip(chunk + mdata[:, 0], -1, 1)
        else:
            mix = chunk
        state.audio_q.put((mix * 32767).astype("<i2").tobytes())
        if out is not None:
            out.write(chunk.reshape(-1, 1))
            if mic is None:
                continue
        if mic is None and out is None:
            time.sleep(0.12)
    for s in (out, mic):
        if s is not None:
            s.stop()
            s.close()
    state.audio_q.put(None)


def _run_sender(state: "SessionState", ws, stt_type: str):
    """audio_qからPCMを読みWebSocketに送信 + PCMバッファ/ファイル書き出し."""
    seq = 0
    while True:
        pcm = state.audio_q.get()
        if pcm is None:
            if stt_type == "speechmatics":
                ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": seq}))
            else:
                ws.send("")
            break
        with state.buf_lock:
            state.pcm_buf.extend(pcm)
            state.pcm_total_bytes += len(pcm)
            if len(state.pcm_buf) > state._PCM_KEEP_BYTES + SR * 2 * 10:
                trim = len(state.pcm_buf) - state._PCM_KEEP_BYTES
                del state.pcm_buf[:trim]
                state.pcm_buf_offset += trim
        if state.pcm_file is not None:
            try:
                state.pcm_file.write(pcm)
                state.pcm_file.flush()
            except OSError:
                pass
        ws.send(pcm)
        seq += 1


def _cleanup(state: "SessionState", args, api_key: str,
             tracker, wav_path: str, out_path: str, html_path: str):
    """セッション終了時のリソース解放・ファイル保存."""
    globals()["_SYS_HOOK"] = None
    state.stop.set()
    if state.partner is not None:
        state.partner.close()
    if state.simulator is not None:
        state.simulator.shutdown()
    if state.agent is not None:
        state.agent.close()
    state.save(live=False)
    if tracker is not None:
        _print_line(f"# レイテンシ統計: {tracker.stats()}")
    _print_line(f"# 議事録を保存しました: {out_path} / {html_path}")
    # WAVファイルのヘッダを更新して正規のWAVにする
    if state.pcm_file is not None:
        try:
            import struct as _struct
            state.pcm_file.flush()
            data_size = state.pcm_total_bytes
            state.pcm_file.seek(4)
            state.pcm_file.write(_struct.pack("<I", 36 + data_size))
            state.pcm_file.seek(40)
            state.pcm_file.write(_struct.pack("<I", data_size))
            state.pcm_file.close()
            if state.pcm_total_bytes > SR * 2 * 10:
                _print_line(f"# 録音を保存しました: {wav_path}")
            else:
                os.remove(wav_path)
        except OSError as e:
            _print_line(f"# 録音保存に失敗: {e}")
        state.pcm_file = None
    # 清書
    if args.polish and not api_key and state.pcm_total_bytes > SR * 2 * 10:
        _print_line("# 清書はスキップ（SONIOX_API_KEY未設定。清書はSoniox非同期APIを使用）")
    if args.polish and api_key and state.pcm_total_bytes > SR * 2 * 10:
        try:
            with open(wav_path, "rb") as f:
                wav_data = f.read()
            recs = polish(api_key, wav_data[44:], args.lang, tracker, log=_print_line)
            fmd = os.path.splitext(out_path)[0] + ".final.md"
            fht = os.path.splitext(out_path)[0] + ".final.html"
            state.write_md(recs, fmd)
            state.write_html(live=False, recs=recs, path=fht, status="清書（非同期再処理済み）")
            state.write_turns(recs, os.path.splitext(out_path)[0] + ".final.turns.jsonl")
            _print_line(f"# 清書版を保存しました: {fmd} / {fht}")
            if not args.no_open:
                import webbrowser
                webbrowser.open("file://" + os.path.abspath(fht))
        except KeyboardInterrupt:
            _print_line("# 清書をスキップしました")
        except Exception as e:
            _print_line(f"# 清書に失敗しました: {type(e).__name__}: {e}")


class _RecvLoop:
    """WebSocketメッセージ受信ループ + flush処理.

    STTからのトークンストリームを処理し、発話を確定(flush)してrecordsに追加する。
    main()内のnonlocal変数群をインスタンス変数で置き換え。
    """

    _FLUSH_TIMEOUT = 30.0     # トークンが来なくなってからの強制flush（秒）
    _FLUSH_SOFT_CHARS = 500   # この文字数を超えたら文の切れ目でflush
    _FLUSH_HARD_CHARS = 1000  # この文字数を超えたら問答無用で強制flush

    def __init__(self, state: "SessionState", args):
        self.state = state
        self.args = args
        self.cur_speaker = None
        self.cur_text = ""
        self.cur_ms: int | None = None
        self.cur_end: int | None = None
        self.cur_last_token_time: float = time.monotonic()
        self.recent_segs: list[tuple] = []

    def overlaps_other(self, start, end, label) -> bool:
        if start is None or end is None:
            return False
        return any(l != label and min(e, end) - max(s, start) > 0
                   for s, e, l in self.recent_segs)

    def flush(self):
        s = self.state
        if not self.cur_text.strip():
            self.cur_text = ""
            self.cur_ms = None
            self.cur_end = None
            self.cur_last_token_time = time.monotonic()
            return
        label = str(self.cur_speaker)
        tracker = s.tracker
        agent = s.agent
        partner = s.partner
        if tracker is not None:
            if self.cur_ms is not None and self.cur_end is not None and self.cur_end > self.cur_ms:
                with s.buf_lock:
                    abs_start = self.cur_ms * 32
                    abs_end = self.cur_end * 32
                    rel_start = max(abs_start - s.pcm_buf_offset, 0)
                    rel_end = max(abs_end - s.pcm_buf_offset, 0)
                    seg = bytes(s.pcm_buf[rel_start: rel_end])
                wav = np.frombuffer(seg, dtype="<i2").astype(np.float32) / 32768.0
            else:
                wav = np.zeros(0, dtype=np.float32)
            sp_id = tracker.classify(wav, self.cur_speaker,
                                     overlapped=self.overlaps_other(self.cur_ms, self.cur_end, label))
            # --- 声紋ベースのAIエコー除去 ---
            if (sp_id is not None
                    and sp_id.startswith("__") and sp_id.endswith("__")):
                if self.args.vp_debug:
                    _print_line(f"# AI声紋エコー除去: sp={sp_id}"
                                f" ({self.cur_text.strip()[:40]}...)")
                self.cur_text = ""
                self.cur_ms = None
                self.cur_end = None
                return
            d = tracker.last
            rec_extra = {}
            if d and d["kind"] == "補正":
                note = (f"声紋でラベル{d['label']}の取り違えを修正"
                        f"（類似{d['sim']:.2f}、放置なら{s.disp_name(d['prev'])}の発言になっていた）")
                rec_extra = {"vp": "補正", "note": note}
                _print_line(f"# ⚡補正: {note}")
            elif d and d["kind"] == "自動登録":
                if d["rename"]:
                    s.rekey(*d["rename"])
                s.add_sys(self.cur_ms, f"この声を「{d['name']}」として追跡開始"
                                       f"（実名にするには {d['label']}=名前）")
                _print_line(f"# この声を「{d['name']}」として追跡します"
                            f"（実名にするには {d['label']}=名前 と入力）")
            elif d and d["kind"] == "合流":
                if d["rename"]:
                    s.rekey(*d["rename"])
                if self.args.vp_debug:
                    _print_line(f"# 合流: ラベル{d['label']}→{d['name']}")
            elif self.args.vp_debug and d:
                extra = f" 類似{d['sim']:.2f}({d['name']})" if "sim" in d else ""
                _print_line(f"# vp判定[{d['kind']}]{extra}")
        else:
            sp_id = "#" + str(self.cur_speaker)
            rec_extra = {}
        # --- テキスト類似度エコー判定（安全網） ---
        for _src_name, _src in [("agent", agent), ("partner", partner)]:
            if _src is None:
                continue
            if _src_name == "agent" and not _src.in_echo_window:
                continue
            sim = _src._best_similarity(self.cur_text)
            if sim > 0.35:
                if self.args.vp_debug:
                    _print_line(f"# テキスト安全網エコー除去({_src_name})"
                                f" sim={sim:.2f}: sp={sp_id}"
                                f" ({self.cur_text.strip()[:40]}...)")
                self.cur_text = ""
                self.cur_ms = None
                self.cur_end = None
                return
        if self.cur_ms is not None and self.cur_end is not None:
            self.recent_segs.append((self.cur_ms, self.cur_end, label))
            del self.recent_segs[:-12]
        if tracker is not None and tracker.last is not None:
            try:
                with open(s.diag_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ms": self.cur_ms, "end": self.cur_end, "label": label,
                                        "key": sp_id, **tracker.last},
                                       ensure_ascii=False, default=str) + "\n")
            except OSError:
                pass
        with s.state_lock:
            s.records.append({"ms": self.cur_ms, "end_ms": self.cur_end,
                              "speaker": sp_id, "text": self.cur_text.strip(),
                              **rec_extra})
            c = s.color_of(sp_id)
        if ON_UTTERANCE is not None:
            try:
                ON_UTTERANCE(s.disp_name(sp_id), self.cur_text.strip())
            except Exception:
                pass
        _print_line(f"{c}[{fmt_ts(self.cur_ms)}] {s.disp_name(sp_id)}{RESET}: {self.cur_text.strip()}")
        s.save()
        self.cur_text = ""
        self.cur_ms = None
        self.cur_end = None
        self.cur_last_token_time = time.monotonic()

    def run(self, ws):
        """WebSocket受信ループのメイン."""
        args = self.args
        try:
            while True:
                res = json.loads(ws.recv())
                if args.stt == "speechmatics":
                    res = sm_to_res(res, args.lang)
                if res.get("error_code") is not None:
                    _print_line(f"# エラー: {res['error_code']} - {res.get('error_message')}")
                    break
                partial = ""
                partial_sp = self.cur_speaker
                for token in res.get("tokens", []):
                    text = token.get("text") or ""
                    if text == "<end>":
                        self.flush()
                        continue
                    if not text:
                        continue
                    if token.get("is_final"):
                        sp = token.get("speaker")
                        if sp != self.cur_speaker:
                            self.flush()
                            self.cur_speaker = sp
                        if self.cur_ms is None:
                            self.cur_ms = token.get("start_ms")
                        if token.get("end_ms") is not None:
                            self.cur_end = token["end_ms"]
                        self.cur_text += text
                        self.cur_last_token_time = time.monotonic()
                    else:
                        partial += text
                        partial_sp = token.get("speaker") or partial_sp
                # --- 強制flush ---
                if self.cur_text:
                    clen = len(self.cur_text)
                    if (time.monotonic() - self.cur_last_token_time > self._FLUSH_TIMEOUT
                            or clen > self._FLUSH_HARD_CHARS):
                        self.flush()
                    elif clen > self._FLUSH_SOFT_CHARS and self.cur_text.rstrip()[-1:] in "。？！.?!\n":
                        self.flush()
                self.state.show_partial(partial_sp if partial else self.cur_speaker,
                                        self.cur_text + partial)
                if res.get("finished"):
                    self.flush()
                    _print_line("# 終了")
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.flush()


class SessionState:
    """main()内の共有状態を集約するコンテナ.

    巨大だった main() のクロージャ変数をインスタンス属性に集約し、
    ヘルパーメソッドとして外部からアクセス可能にする。
    """

    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------
    def __init__(self, *, args, started, out_path, html_path, diag_path,
                 turns_path, wav_path, tracker=None, serve=True):
        self.args = args
        self.started = started
        self.out_path = out_path
        self.html_path = html_path
        self.diag_path = diag_path
        self.turns_path = turns_path
        self.wav_path = wav_path
        self._serve = serve

        # 発話記録
        self.names: dict[str, str] = {}
        self.colors: dict[str, str] = {}
        self.records: list[dict] = []
        self.state_lock = threading.Lock()

        # 声紋
        self.tracker: VoiceProfiles | None = tracker

        # AI
        self.agent: RealtimeAgent | None = None
        self.partner: ConversationPartner | None = None
        self.simulator: DiscussionSimulator | None = None

        # 論点
        self.topics: list[dict] = []
        self.topics_lock = threading.Lock()
        self.topic_cursor = 0
        self._TOPIC_WINDOW = 10
        self._TOPIC_TRIGGER = 5

        # PCMバッファ
        self.pcm_buf = bytearray()
        self.pcm_buf_offset = 0
        self.pcm_total_bytes = 0
        self._PCM_KEEP_BYTES = SR * 2 * 120
        self.buf_lock = threading.Lock()
        self.pcm_file = None  # IO[bytes] | None

        # 制御
        self.stop = threading.Event()
        self.audio_q: "queue.Queue[bytes | None]" = queue.Queue()

        # エージェントワーカー状態
        self._last_utt_time = [time.monotonic()]
        self._was_in_echo = [False]
        self.agent_cursor = 0

    # ------------------------------------------------------------------
    # 表示ヘルパー
    # ------------------------------------------------------------------
    def disp_name(self, key) -> str:
        key = str(key)
        if key in self.names:
            return self.names[key]
        return f"話者{key[1:]}" if key.startswith("#") else key

    def key_for_label(self, sp) -> str:
        sp = str(sp)
        if self.tracker is not None and sp in self.tracker.sp_map:
            return self.tracker.sp_map[sp]
        return "#" + sp

    def color_of(self, key) -> str:
        key = str(key)
        if key not in self.colors:
            self.colors[key] = PALETTE[len(self.colors) % len(PALETTE)]
        return self.colors[key]

    def rekey(self, old: str, new: str):
        """表示キーの付け替え: recordsと色を一括移行."""
        with self.state_lock:
            for r in self.records:
                if r.get("speaker") == old:
                    r["speaker"] = new
            if old in self.colors:
                self.colors.setdefault(new, self.colors.pop(old))

    def add_sys(self, ms, text: str):
        """システムイベントを議事録のタイムラインに残す."""
        with self.state_lock:
            self.records.append({"ms": ms, "sys": text})

    def key_of(self, tok: str) -> str:
        """コマンド引数を表示キーへ: 人物名はそのまま、数字はそのラベルの現在の表示先."""
        if self.tracker is not None:
            if tok in self.tracker.profiles:
                return tok
            if tok in self.tracker.sp_map:
                return self.tracker.sp_map[tok]
        return "#" + tok

    def show_partial(self, sp, text: str):
        if not text.strip():
            sys.stdout.write(CLEAR_LINE)
        else:
            cols = os.get_terminal_size().columns if sys.stdout.isatty() else 120
            line = f"{self.disp_name(self.key_for_label(sp))}: {text.strip()}"
            sys.stdout.write(CLEAR_LINE + DIM + line[-(cols - 2):] + RESET)
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # 出力
    # ------------------------------------------------------------------
    def write_md(self, recs=None, path=None):
        with self.state_lock:
            rs = self.records if recs is None else recs
            speakers = list(dict.fromkeys(r["speaker"] for r in rs if "speaker" in r))
            lines = [
                f"# 議事録 {self.started.strftime('%Y-%m-%d %H:%M')}",
                "",
                "話者: " + (", ".join(self.disp_name(s) for s in speakers) or "（未検出）"),
                "",
            ]
            for r in rs:
                if "sys" in r:
                    lines.append(f"> [{fmt_ts(r['ms'])}] {r['sys']}")
                    continue
                mark = " ⚡" if r.get("vp") == "補正" else ""
                lines.append(f"- **[{fmt_ts(r['ms'])}] {self.disp_name(r['speaker'])}{mark}**: {r['text']}")
            dst = path or self.out_path
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(tmp, dst)

    def write_html(self, live: bool = True, recs=None, path=None, status=None):
        import html as _html
        with self.state_lock:
            rs = self.records if recs is None else recs
            parts = []
            for r in rs:
                if "sys" in r:
                    parts.append(f'<div class="sys">⚙ {_html.escape(r["sys"])}</div>')
                    continue
                sp = str(r["speaker"])
                self.color_of(sp)
                idx = list(self.colors).index(sp)
                c = HTML_PALETTE[idx % len(HTML_PALETTE)]
                badge = ""
                if r.get("vp") == "補正":
                    note = _html.escape(r.get("note", ""))
                    badge = f'<span class="badge" title="{note}">⚡声紋補正</span>'
                parts.append(
                    f'<div class="u"><span class="ts">{fmt_ts(r["ms"])}</span>'
                    f'<span class="who" style="color:{c}">{_html.escape(self.disp_name(sp))}</span>'
                    f'{_html.escape(r["text"])}{badge}</div>'
                )
            speakers = list(dict.fromkeys(r["speaker"] for r in rs if "speaker" in r))
            sp_tags = []
            for s in speakers:
                dn = _html.escape(self.disp_name(s))
                idx_s = list(self.colors).index(s) if s in self.colors else 0
                c = HTML_PALETTE[idx_s % len(HTML_PALETTE)]
                is_renameable = self._serve and self.tracker is not None and not s.startswith("#")
                if is_renameable:
                    lbl = s
                    for _l, _k in self.tracker.sp_map.items():
                        if _k == s:
                            lbl = _l
                            break
                    is_anon = re.match(r"^人物\d+$", s)
                    ph = "名前" if is_anon else "新しい名前"
                    sp_tags.append(
                        f'<div class="speaker-tag">'
                        f'<div class="speaker-name"><span class="dot" style="background:{c}"></span>{dn}</div>'
                        f'<div class="rename-row">'
                        f'<input class="rename-input" placeholder="{ph}" data-label="{_html.escape(lbl)}">'
                        f'<button class="rename-btn" onclick="rename(this)">登録</button>'
                        f'</div></div>')
                else:
                    sp_tags.append(
                        f'<div class="speaker-tag">'
                        f'<div class="speaker-name"><span class="dot" style="background:{c}"></span>{dn}</div>'
                        f'</div>')
            if sp_tags:
                speaker_panel = ('<div class="sidebar"><p class="sidebar-title">この会議の話者</p>'
                                 '<div class="speaker-panel">' + ''.join(sp_tags) + '</div></div>')
            else:
                speaker_panel = ''
            profile_panel = ''
            if self._serve and self.tracker is not None:
                all_names = self.tracker.all_profile_names()
                if all_names:
                    active_names = set(self.tracker.active_profile_names())
                    items = []
                    for n in all_names:
                        cls = 'profile-item active' if n in active_names else 'profile-item'
                        items.append(
                            f'<div class="{cls}" data-name="{_html.escape(n)}" '
                            f'onclick="toggleProfile(this)">'
                            f'<span class="profile-toggle"></span>'
                            f'{_html.escape(n)}</div>')
                    profile_panel = ('<div class="profile-section">'
                                     '<p class="sidebar-title">プロファイル</p>'
                                     + ''.join(items) + '</div>')
            stats_panel = ''
            talk_rs = [r for r in rs if "speaker" in r and r.get("text")]
            if talk_rs:
                sp_dur: dict[str, float] = {}
                sp_chars: dict[str, int] = {}
                sp_turns: dict[str, int] = {}
                for r in talk_rs:
                    s = r["speaker"]
                    ms, end = r.get("ms"), r.get("end_ms")
                    dur = (end - ms) / 1000.0 if ms is not None and end is not None and end > ms else 0.0
                    sp_dur[s] = sp_dur.get(s, 0.0) + dur
                    sp_chars[s] = sp_chars.get(s, 0) + len(r["text"])
                    sp_turns[s] = sp_turns.get(s, 0) + 1
                total_dur = sum(sp_dur.values()) or 1.0
                total_chars = sum(sp_chars.values()) or 1
                total_turns = sum(sp_turns.values()) or 1
                ranked = sorted(sp_dur.keys(), key=lambda s: sp_dur[s], reverse=True)

                def _bar_rows(data, total, unit=""):
                    rows = []
                    for s in ranked:
                        v = data.get(s, 0)
                        pct = v / total * 100 if total else 0
                        idx_s = list(self.colors).index(s) if s in self.colors else 0
                        c = HTML_PALETTE[idx_s % len(HTML_PALETTE)]
                        dn = _html.escape(self.disp_name(s))
                        short = dn[:2] if len(dn) > 3 else dn
                        rows.append(
                            f'<div class="stats-row">'
                            f'<span class="stats-name" title="{dn}">{short}</span>'
                            f'<div class="stats-bar-bg">'
                            f'<div class="stats-bar" style="width:{pct:.0f}%;background:{c}"></div>'
                            f'</div>'
                            f'<span class="stats-pct">{pct:.0f}%</span>'
                            f'</div>')
                    return ''.join(rows)

                groups = []
                if total_dur > 0.5:
                    groups.append(f'<div class="stats-group">'
                                  f'<div class="stats-label">発話時間</div>'
                                  + _bar_rows(sp_dur, total_dur) + '</div>')
                groups.append(f'<div class="stats-group">'
                              f'<div class="stats-label">文字数</div>'
                              + _bar_rows(sp_chars, total_chars) + '</div>')
                groups.append(f'<div class="stats-group">'
                              f'<div class="stats-label">発話回数</div>'
                              + _bar_rows(sp_turns, total_turns) + '</div>')
                stats_panel = ('<div class="stats-section">'
                               '<p class="sidebar-title">発言量</p>'
                               + ''.join(groups) + '</div>')
            topics_panel = ''
            with self.topics_lock:
                if self.topics:
                    items = []
                    for t in self.topics:
                        tt = _html.escape(t.get("topic", ""))
                        ts = _html.escape(t.get("speaker", ""))
                        items.append(f'<div class="topic-item">'
                                     f'<div class="topic-text">{tt}</div>'
                                     f'<div class="topic-by">{ts}</div></div>')
                    topics_panel = ('<div class="topics-section">'
                                   '<p class="sidebar-title">論点</p>'
                                   + ''.join(items) + '</div>')
            agent_panel = ''
            if self.agent is not None:
                cur_mode = self.agent.mode
                if self.agent._connected:
                    conn = '接続中'
                elif self.agent._conn_error:
                    conn = f'エラー: {_html.escape(self.agent._conn_error)}'
                else:
                    conn = '未接続'
                mode_btns = []
                for m, lbl in [("off", "OFF"), ("facilitator", "進行役"),
                               ("conversation", "会話")]:
                    cls = "agent-mode-btn active" if m == cur_mode else "agent-mode-btn"
                    mode_btns.append(f'<button class="{cls}" data-mode="{m}" '
                                     f'onclick="setAgentMode(this)">{lbl}</button>')
                voice_opts = []
                for v in AGENT_VOICES:
                    sel = 'selected' if v == self.agent.voice else ''
                    voice_opts.append(f'<option value="{v}" {sel}>{v}</option>')
                trigger_val = self.agent.trigger_n
                agent_panel = (
                    f'<div class="agent-section" data-mode="{cur_mode}">'
                    f'<div class="agent-header">'
                    f'<span class="agent-label">🤖 AI Agent</span>'
                    f'<span class="agent-conn">{conn}</span>'
                    f'</div>'
                    f'<div class="agent-modes">{"".join(mode_btns)}</div>'
                    f'<div class="agent-opts">'
                    f'<label class="agent-opt-label">声'
                    f'<select class="agent-select" onchange="setAgentVoice(this)">'
                    f'{"".join(voice_opts)}</select></label>'
                    f'<label class="agent-opt-label agent-trigger-row">'
                    f'間隔 <input type="number" class="agent-num" value="{trigger_val}" '
                    f'min="1" max="50" onchange="setAgentTrigger(this)">発話'
                    f'</label>'
                    f'</div></div>')
            doc = HTML_TMPL.format(
                refresh='<meta http-equiv="refresh" content="2">' if live else "",
                title=self.started.strftime("%Y-%m-%d %H:%M"),
                status=status or ('<span class="live">● ライブ（2秒ごと自動更新）</span>'
                                  if live else "終了"),
                speaker_panel=speaker_panel,
                profile_panel=profile_panel,
                stats_panel=stats_panel,
                topics_panel=topics_panel,
                agent_panel=agent_panel,
                body="\n".join(parts) or '<p class="meta">（まだ発話なし）</p>',
            )
            dst = path or self.html_path
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(doc)
            os.replace(tmp, dst)

    def write_turns(self, recs=None, path=None):
        """discussion-support(das)のUtteranceスキーマでJSONL出力."""
        with self.state_lock:
            rs = self.records if recs is None else recs
            lines = []
            tid = 0
            for r in rs:
                if "speaker" not in r or not r.get("text"):
                    continue
                tid += 1
                lines.append(json.dumps({"turn_id": tid, "speaker": self.disp_name(r["speaker"]),
                                         "text": r["text"], "ms": r.get("ms"),
                                         "end_ms": r.get("end_ms")},
                                        ensure_ascii=False))
            dst = path or self.turns_path
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            os.replace(tmp, dst)

    def save(self, live: bool = True):
        self.write_md()
        self.write_html(live)
        self.write_turns()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--model", default="stt-rt-v4")
    ap.add_argument("--wav", default=None, help="指定で実マイクの代わりにファイル擬似ライブ")
    ap.add_argument("--play", action="store_true",
                    help="--wav使用時、注入と同時にスピーカーからも再生する（観戦用）")
    ap.add_argument("--join", action="store_true",
                    help="--wav使用時、再生しつつ自分のマイクも混ぜて参加する（イヤホン推奨。"
                         "wav終了後もマイクは生き続けるのでCtrl+Cで終了）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="保存先mdファイル（省略時 transcripts/日時.md）")
    ap.add_argument("--no-open", action="store_true", help="ブラウザを自動で開かない")
    ap.add_argument("--no-vp", action="store_true", help="声紋照合を無効化（Sonioxのラベルをそのまま使う）")
    ap.add_argument("--voices", default="voices.json", help="声紋プロファイルの保存先(既定 voices.json)")
    ap.add_argument("--vp-model", default="redimnet", choices=["redimnet", "ecapa", "resemblyzer"],
                    help="声紋モデル(既定redimnet=2024年世代、実測の分離・通し精度とも最良。"
                         "読み込み失敗時は ecapa → resemblyzer へ自動フォールバック)")
    ap.add_argument("--vp-match", type=float, default=None,
                    help="即時判定のしきい値。省略時はモデル別の既定値"
                         "(redimnet 0.42 / ecapa 0.35 / resemblyzer 0.75)")
    ap.add_argument("--vp-no-auto", action="store_true",
                    help="未知の声の自動登録（匿名「人物N」）を無効化")
    ap.add_argument("--vp-debug", action="store_true", help="発話ごとの声紋判定の内訳を表示")
    ap.add_argument("--polish", action="store_true",
                    help="終了時に清書を行う（非同期APIでの全体再処理。デフォルトオフ）")
    ap.add_argument("--no-polish", action="store_true",
                    help="(後方互換用、現在はデフォルトでオフ)")
    ap.add_argument("--stt", default="soniox", choices=["soniox", "speechmatics"],
                    help="リアルタイムSTTの供給源。speechmaticsは要 SPEECHMATICS_API_KEY"
                         "（話者分離の評判が良い代替。声紋層など他の機能は不変）")
    ap.add_argument("--port", type=int, default=8231,
                    help="UIサーバーのポート番号（ブラウザからの話者リネームに必要。0で無効）")
    ap.add_argument("--agent", action="store_true",
                    help="AIエージェント（ファシリテーター）を有効化。OPENAI_API_KEYが必要。"
                         "Realtime API v2 WebSocketで会議に参加する")
    ap.add_argument("--agent-voice", default="shimmer",
                    help="AIエージェントの声（alloy/ash/ballad/coral/echo/sage/shimmer/verse）")
    ap.add_argument("--agent-trigger", type=int, default=_AGENT_TRIGGER,
                    help=f"AIの応答を検討する発話間隔（既定{_AGENT_TRIGGER}）")
    ap.add_argument("--simulate", metavar="TOPIC",
                    help="AI議論シミュレーション。Chat API+TTSで複数話者の議論を自動生成し、"
                         "ファシリテーターが介入する。--agentと組み合わせて使用。"
                         "例: --simulate 'AIツール導入の是非'")
    ap.add_argument("--sim-scenario", default=None,
                    choices=["stalled", "biased", "derailed", "consensus_needed", "healthy"],
                    help="シミュレーションの議論パターン")
    ap.add_argument("--debate", metavar="TOPIC",
                    help="AI会話相手と議論。Realtime APIで音声対話し、"
                         "ファシリテーターが介入する。--agentと組み合わせて使用。"
                         "例: --debate 'AIツール導入の是非'")
    ap.add_argument("--debate-voice", default="echo",
                    help="会話相手の声（既定echo。ファシリテーターのalloyと被らないこと）")
    args = ap.parse_args(argv)
    _serve = args.port > 0

    load_env()   # .env からAPIキーを読み込み（export済みの値が優先）
    if args.wav and not os.path.exists(args.wav):
        raise SystemExit(f"音声ファイルがありません: {args.wav}\n"
                         "（テスト音声は scripts/make_overlap_testset.py 等で先に生成してください）")

    api_key = os.environ.get("SONIOX_API_KEY")
    sm_key = os.environ.get("SPEECHMATICS_API_KEY")
    if args.stt == "speechmatics":
        if not sm_key:
            raise SystemExit("環境変数 SPEECHMATICS_API_KEY を設定してください"
                             "（https://portal.speechmatics.com/settings/api-keys）")
    elif not api_key:
        raise SystemExit("環境変数 SONIOX_API_KEY を設定してください（https://console.soniox.com）")

    try:
        from websockets.sync.client import connect
    except ImportError:
        raise SystemExit("uv add websockets を実行してください")

    if args.stt == "speechmatics":
        ws_url = SM_WS_URL
        ws_headers = {"Authorization": f"Bearer {sm_key}"}
        start_msg = {
            "message": "StartRecognition",
            "audio_format": {"type": "raw", "encoding": "pcm_s16le", "sample_rate": SR},
            "transcription_config": {
                "language": args.lang,
                "operating_point": "enhanced",
                "diarization": "speaker",
                "enable_partials": True,
                "max_delay": 1.2,
                "conversation_config": {"end_of_utterance_silence_trigger": 0.8},
            },
        }
    else:
        ws_url = WS_URL
        ws_headers = None
        start_msg = {
            "api_key": api_key,
            "model": args.model,
            "language_hints": [args.lang],
            "enable_speaker_diarization": True,
            "enable_endpoint_detection": True,
            "audio_format": "pcm_s16le",
            "sample_rate": SR,
            "num_channels": 1,
        }

    started = datetime.datetime.now()
    if args.out:
        out_path = args.out
    else:
        os.makedirs("transcripts", exist_ok=True)
        out_path = os.path.join("transcripts", started.strftime("%Y-%m-%d_%H%M") + ".md")
    html_path = os.path.splitext(out_path)[0] + ".html"
    diag_path = os.path.splitext(out_path)[0] + ".diag.jsonl"   # 発話ごとの判定根拠(劣化解析用)
    turns_path = os.path.splitext(out_path)[0] + ".turns.jsonl"  # das(議論支援)連携用

    # --- 声紋モデル読み込み ---
    tracker: VoiceProfiles | None = None
    if not args.no_vp:
        print("# 声紋モデルを読み込み中…", flush=True)
        for model in dict.fromkeys([args.vp_model, "ecapa", "resemblyzer"]):
            try:
                tracker = VoiceProfiles(path=args.voices, thresh=args.vp_match,
                                        auto=not args.vp_no_auto, model=model)
                if model != args.vp_model:
                    print(f"# 注意: {args.vp_model} を読み込めなかったため {model} で動作します"
                          f"（依存: uv add speechbrain torchaudio / redimnetは初回ネット接続必要）",
                          flush=True)
                print(f"# 声紋モデル: {model}", flush=True)
                break
            except Exception as e:   # 依存欠如(ImportError)もDL失敗等も次の候補へ
                print(f"#   {model}: 読み込み失敗 ({type(e).__name__})", flush=True)
                continue
        if tracker is None:
            print("# 警告: 声紋照合がOFFです！ 依存が未導入のため人物の確定・補正は行われません。", flush=True)
            print("#   有効化するには: uv add speechbrain torchaudio  →  再起動", flush=True)
        elif tracker.profiles:
            print(f"# 声紋プロファイル: {', '.join(tracker.profiles)}（{args.voices}）", flush=True)
        else:
            print(f"# 声紋プロファイル: なし。未知の声は「人物N」として自動追跡、"
                  f"「1=松井」で実名化すると次回から自動表示（{args.voices}）", flush=True)

    # --- SessionState: 共有状態の一括管理 ---
    wav_path = os.path.splitext(out_path)[0] + ".wav"
    state = SessionState(args=args, started=started, out_path=out_path,
                         html_path=html_path, diag_path=diag_path,
                         turns_path=turns_path, wav_path=wav_path,
                         tracker=tracker, serve=_serve)

    # --- AIエージェント ---
    _agent_oai_key = os.environ.get("OPENAI_API_KEY", "")
    if args.agent:
        if not _agent_oai_key:
            print("# AI Agent: OPENAI_API_KEY が未設定です。--agent は無効になります。", flush=True)
        else:
            state.agent = RealtimeAgent(api_key=_agent_oai_key, voice=args.agent_voice,
                                        mode="facilitator", trigger_n=args.agent_trigger)
            if tracker is not None:
                state.agent.set_tracker(tracker)

    # --- WAVストリーミング書き出し（クラッシュ時もファイルが残る） ---
    try:
        state.pcm_file = open(wav_path, "wb")
        import struct as _struct
        state.pcm_file.write(b"RIFF" + _struct.pack("<I", 0) + b"WAVEfmt " +
                              _struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16) +
                              b"data" + _struct.pack("<I", 0))
        state.pcm_file.flush()
    except OSError as e:
        print(f"# 警告: 録音ファイルを開けません: {e}", flush=True)
        state.pcm_file = None

    global _SYS_HOOK

    def _sys_hook(text: str) -> None:
        state.add_sys(None, text)
        state.save()
    _SYS_HOOK = _sys_hook

    # --- 論点抽出 ---
    _oai_key = os.environ.get("OPENAI_API_KEY", "")
    _oai_model = os.environ.get("OPENAI_MODEL_FAST", "gpt-5-mini")

    # --- AIエージェント: コールバック ---
    _on_agent_text = _on_agent_text_factory(state)

    # --- UIサーバー（ブラウザからの話者リネーム用）---
    _httpd = None
    if _serve:
        from http.server import HTTPServer
        try:
            _httpd = HTTPServer(("127.0.0.1", args.port), _UIHandler.create(state))
            threading.Thread(target=_httpd.serve_forever, daemon=True).start()
        except OSError as e:
            print(f"# 警告: UIサーバーをポート{args.port}で起動できません ({e})", flush=True)
            _serve = False
            state._serve = False


    if args.simulate and args.debate:
        raise SystemExit("--simulate と --debate は同時に使えません")

    # --- DiscussionSimulator ---
    if args.simulate:
        if not _oai_key:
            raise SystemExit("--simulate には OPENAI_API_KEY が必要です")
        if not args.agent:
            print("# ヒント: --agent を付けるとファシリテーターが介入します", flush=True)
        if args.agent and args.agent_voice in DiscussionSimulator.SPEAKERS.values():
            print(f"# 警告: --agent-voice={args.agent_voice} はSimulator話者と重複しています。"
                  f"声紋分離に影響する可能性があります。alloy を推奨します。", flush=True)
        state.simulator = DiscussionSimulator(
            api_key=_oai_key, topic=args.simulate,
            scenario=args.sim_scenario)
    # --- ConversationPartner（--debate モード）---
    if args.debate:
        if not _oai_key:
            raise SystemExit("--debate には OPENAI_API_KEY が必要です")
        if not args.agent:
            print("# ヒント: --agent を付けるとファシリテーターが介入します", flush=True)
        if args.agent and args.debate_voice == args.agent_voice:
            print(f"# 警告: --debate-voice と --agent-voice が同じ ({args.debate_voice})。"
                  f"声紋分離に影響します。", flush=True)
        state.partner = ConversationPartner(
            api_key=_oai_key, voice=args.debate_voice, topic=args.debate)
        if tracker is not None:
            state.partner.set_tracker(tracker)

    print(f"# {args.stt} に接続中…", flush=True)
    with connect(ws_url, additional_headers=ws_headers) as ws:
        ws.send(json.dumps(start_msg))
        # 音声ソース選択: simulate > wav > mic
        if state.simulator is not None:
            if state.agent is not None:
                state.simulator._agent_ref = state.agent
            state.simulator.start(state.audio_q, state.stop, play_audio=True)
            print(f"# Simulator: 議論を自動生成中（議題: {args.simulate}）", flush=True)
        else:
            if args.wav:
                threading.Thread(target=_run_from_wav, args=(state, args),
                                 daemon=True).start()
            else:
                threading.Thread(target=_run_from_mic, args=(state, args.device),
                                 daemon=True).start()
        threading.Thread(target=_run_stdin_commands, args=(state,),
                         daemon=True).start()
        if _oai_key:
            threading.Thread(target=_run_topic_worker,
                            args=(state, _oai_key, _oai_model), daemon=True).start()
            print("# 論点抽出: 有効（5発話ごとにLLMで分析）", flush=True)
        else:
            print("# 論点抽出: 無効（OPENAI_API_KEYが未設定）", flush=True)
        if state.agent is not None:
            _connect_agent(state, _on_agent_text)
        if state.partner is not None:
            state.partner.on_ai_utterance = _on_partner_text_factory(state)
            state.partner.connect()
            print(f"# Partner: voice={state.partner.voice} topic={state.partner.topic}",
                  flush=True)

        threading.Thread(target=_run_sender, args=(state, ws, args.stt),
                         daemon=True).start()

        state.save()
        print("# 開始。話してください（「1=松井」で声を登録 / Ctrl+Cで終了）", flush=True)
        print(f"# 保存先: {out_path}", flush=True)
        print(f"# ブラウザ表示: open {html_path}（ライブ中は2秒ごと自動更新）\n", flush=True)
        if not args.no_open:
            import webbrowser
            if _serve:
                webbrowser.open(f"http://127.0.0.1:{args.port}/")
            else:
                webbrowser.open("file://" + os.path.abspath(html_path))

        recv = _RecvLoop(state, args)
        try:
            recv.run(ws)
        finally:
            _cleanup(state, args, api_key, tracker, wav_path, out_path, html_path)


if __name__ == "__main__":
    main()
