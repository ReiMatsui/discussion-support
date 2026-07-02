"""UIサーバー用HTTPハンドラ + ターミナル出力."""
from __future__ import annotations

import json
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._session_state import SessionState

from ._constants import _ENROLL_MIN_VOICED_SEC, CLEAR_LINE, SR


def _print_line(text: str):
    """ターミナルの現在行をクリアして1行出力."""
    sys.stdout.write(CLEAR_LINE + text + "\n")
    sys.stdout.flush()


def _voiced_seconds(wav: Any, *, frame_sec: float = 0.02,
                    rms_thresh: float = 0.01) -> float:
    """無音を除いた実効音声長（秒）を返す（P2-5 の事前登録品質ゲート用）.

    20msフレームごとのRMSが閾値を超えたフレームだけを「有声」として数える。
    末尾N秒を無検査で平均する登録が、間や無音で低品質な声紋になるのを防ぐ。
    """
    import numpy as np

    if wav is None or len(wav) == 0:
        return 0.0
    frame = int(SR * frame_sec)
    if frame <= 0:
        return float(len(wav) / SR)
    n = (len(wav) // frame) * frame
    if n == 0:
        return 0.0
    frames = np.asarray(wav[:n], dtype=np.float32).reshape(-1, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    return float(int((rms > rms_thresh).sum()) * frame_sec)


class _UIHandler:
    """UIサーバー用HTTPハンドラ（トップレベル定義）.

    BaseHTTPRequestHandlerのサブクラスを動的に生成するファクトリ。
    クロージャ変数の代わりにクラス変数 _state でSessionStateを参照する。

    使い方:
        handler_cls = _UIHandler.create(state)
        httpd = HTTPServer(("127.0.0.1", port), handler_cls)
    """

    @staticmethod
    def create(state: SessionState) -> type:
        """state を束縛した BaseHTTPRequestHandler サブクラスを返す."""
        from http.server import BaseHTTPRequestHandler

        class Handler(BaseHTTPRequestHandler):
            _state = state

            def do_GET(self):
                if self.path == "/api/state":
                    self._json(200, self._state.api_snapshot())
                elif self.path == "/api/stream":
                    self._stream()
                elif self.path == "/" or self.path.startswith("/?"):
                    self._serve_html()
                else:
                    self.send_error(404)

            def _stream(self):
                """SSE: 状態が変わるたびに最新スナップショットを配信する（F2）.

                rev の変化を見て差分配信。無変化時はハートビートで接続を保つ。
                stop でセッションが終了したら end イベントを送って閉じる。
                """
                s = self._state
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                last_rev = -1
                last_partial = None
                try:
                    while not s.stop.is_set():
                        rev = s.rev
                        partial = s.partial_text  # 認識途中経過も変化を見る（課題①）
                        if rev != last_rev or partial != last_partial:
                            payload = json.dumps(s.api_snapshot(), ensure_ascii=False)
                            self.wfile.write(f"data: {payload}\n\n".encode())
                            last_rev = rev
                            last_partial = partial
                        else:
                            self.wfile.write(b": ping\n\n")  # ハートビート
                        self.wfile.flush()
                        time.sleep(1.0)
                    self.wfile.write(b"event: end\ndata: {}\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # クライアント切断は正常
                except OSError:
                    pass

            def _serve_html(self):
                # サーバー配信時は新SPA（_webapp.INDEX_HTML）を返す。
                # 生成済みの議事録HTML(html_path)は file:// 表示用に別途残る。
                from das.asr.live._webapp import INDEX_HTML
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(INDEX_HTML.encode("utf-8"))

            def do_POST(self):
                s = self._state
                if self.path == "/api/stop":
                    _print_line("# セッションを停止します（UIから）")
                    if s.request_stop is not None:
                        s.request_stop()
                    else:
                        s.stop.set()
                    self._json(200, {"ok": True, "running": False})
                elif self.path == "/api/reset":
                    _print_line("# 新しい会議に切り替えます（UIから）")
                    if s.request_reset is not None:
                        # STT接続ごと作り直す（実処理はメインスレッド）
                        s.request_reset()
                        self._json(200, {"ok": True, "resetting": True})
                    else:
                        # 接続なし（テスト等）は状態クリアのみ
                        self._json(200, s.reset_for_new_meeting())
                elif self.path == "/api/start":
                    s.waiting_to_start = False
                    s.start_requested.set()
                    s.rev += 1
                    s.save()
                    _print_line("# 会議を開始します（UIから）")
                    self._json(200, {"ok": True, "waiting": False})
                elif self.path == "/api/mode":
                    from das.asr.live._workers import set_session_mode
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    result = set_session_mode(s, str(body.get("mode", "")))
                    self._json(200 if result.get("ok") else 400, result)
                elif self.path == "/api/topic":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    result = s.set_agenda(str(body.get("topic", "")))
                    _print_line(f"# 議題を設定（UIから）: {result['agenda']}")
                    self._json(200, result)
                elif self.path == "/api/intervention":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    result = {"ok": True}
                    if "enabled" in body:
                        result = s.set_intervention_enabled(bool(body.get("enabled")))
                        if not result.get("ok"):
                            self._json(400, result)
                            return
                    proactivity = body.get("proactivity")
                    if proactivity is not None:
                        result = s.set_proactivity(str(proactivity))
                        if not result.get("ok"):
                            self._json(400, result)
                            return
                    trigger_n = body.get("trigger_n")
                    if trigger_n is not None:
                        trigger_n = int(trigger_n)
                        if trigger_n < 1 or trigger_n > 50:
                            self._json(400, {"ok": False, "error": "発話数は1〜50で指定してください"})
                            return
                        if s.agent is not None:
                            s.agent.apply_config(trigger_n=trigger_n)
                    s.rev += 1
                    s.save()
                    _print_line(
                        f"# 介入設定を更新（UIから）: enabled={s.intervention_enabled}"
                        f" proactivity={s.proactivity_name}"
                        f" trigger={getattr(s.agent, 'trigger_n', None)}"
                    )
                    self._json(200, {
                        "ok": True,
                        "enabled": s.intervention_enabled,
                        "proactivity": s.proactivity_name,
                        "trigger_n": getattr(s.agent, "trigger_n", None),
                    })
                elif self.path == "/api/facilitator/call":
                    # 手動呼び出し（Phase1）: キューに積むだけ。発話は worker + Controller
                    # 経路が採否する（直接 agent.trigger() しない）。
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length)) if length else {}
                    result = s.queue_manual_facilitator_call(
                        str(body.get("request", "")))
                    if not result.get("ok"):
                        self._json(400, result)
                        return
                    _print_line("# ファシリテーター手動呼び出し（UIから）: "
                                f"{result.get('request') or '直近の議論整理'}")
                    self._json(200, result)
                elif self.path == "/api/diarization":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    raw = body.get("max_speakers")
                    max_speakers = None if raw in (None, "", "auto") else int(raw)
                    result = s.set_diarization_max_speakers(max_speakers)
                    if result.get("ok"):
                        label = max_speakers if max_speakers is not None else "未指定"
                        s.add_sys(None, f"想定話者数を更新: {label}（新しい会議/再接続で確実に反映）")
                        s.save()
                        _print_line(f"# 想定話者数を更新（UIから）: {label}")
                    self._json(200 if result.get("ok") else 400, result)
                elif self.path == "/rename":
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
                            reason = ((getattr(s.tracker, "last", None) or {})
                                      .get("reason"))
                            error = ("同じ名前の声紋が既に登録されています"
                                     if reason == "duplicate_name"
                                     else "この参加者の音声がまだ足りません")
                            self._json(400, {"error": error})
                            return
                        s.rekey(old, name)
                        msg = f"「{name}」の声を登録しました（以後の新しい発話から照合）"
                        s.add_sys(None, msg)
                        s.save()
                        _print_line(f"# {name} の声を登録しました（UIから）")
                    else:
                        s.set_display_name("#" + label, name)
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
                elif self.path == "/api/enroll":
                    # 会議前の事前登録: 直近の音声(その人が単独で喋った分)で声紋を作る
                    import numpy as np
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    name = str(body.get("name", "")).strip()
                    seconds = float(body.get("seconds", 20))
                    if not name:
                        self._json(400, {"error": "name を指定してください"})
                        return
                    if s.tracker is None:
                        self._json(400, {"error": "声紋照合が無効です（--no-vp 指定中など）"})
                        return
                    # P2-5: AI発話中/エコー窓中の登録は、回り込んだAI音声を混ぜて
                    # 声紋を汚すため拒否する。
                    _ai_busy = (
                        (s.agent is not None
                         and (s.agent.ai_speaking or s.agent.in_echo_window))
                        or (s.partner is not None
                            and (s.partner.ai_speaking or s.partner.in_echo_window))
                    )
                    if _ai_busy:
                        self._json(400, {"error": "AIの発話が終わってからもう一度お願いします"})
                        return
                    nbytes = max(int(seconds * SR * 2), 0)
                    with s.buf_lock:
                        seg = bytes(s.pcm_buf[-nbytes:]) if nbytes else bytes(s.pcm_buf)
                    if len(seg) < SR * 2 * 2:   # 2秒未満
                        self._json(400, {"error": "登録中は、その人が1人で5秒ほど話してください"})
                        return
                    wav = np.frombuffer(seg, dtype="<i2").astype(np.float32) / 32768.0
                    # P2-5: 無音を除いた実効音声長が下限未満なら reject（末尾N秒を
                    # 無検査で平均すると、間や無音だけで低品質な声紋ができるため）。
                    if _voiced_seconds(wav) < _ENROLL_MIN_VOICED_SEC:
                        self._json(400, {"error": "声が短すぎます。その人が1人で"
                                         "5秒ほど続けて話してから登録してください"})
                        return
                    if not s.tracker.enroll_from_audio(name, wav):
                        self._json(400, {"error": "声紋の計算に失敗しました"})
                        return
                    actual_seconds = round(len(seg) / (SR * 2), 1)
                    msg = f"「{name}」の声を事前登録しました（以後の新しい発話から照合）"
                    s.add_sys(None, msg)
                    s.save()
                    _print_line(f"# {name} を事前登録（{actual_seconds:.0f}秒、UIから）")
                    self._json(200, {"ok": True, "name": name,
                                     "seconds": actual_seconds,
                                     "active": True,
                                     "applies_from": "future_utterances",
                                     "message": msg})
                elif self.path == "/api/roster":
                    # 名簿の確定/解除: 確定すると自動登録オフ（登録済みだけと照合し増殖を止める）
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    locked = bool(body.get("locked", True))
                    if s.tracker is None:
                        self._json(400, {"error": "声紋照合が無効です"})
                        return
                    s.tracker.auto = not locked
                    s.add_sys(None, "名簿を確定しました（登録済みの人だけで進めます）"
                              if locked else "名簿の確定を解除しました（自動登録ON）")
                    s.save()
                    _print_line(f"# 名簿{'確定' if locked else '解除'}（UIから）")
                    self._json(200, {"ok": True, "locked": locked})
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

            def _json(self, code: int, data: dict) -> None:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

            def log_message(self, format, *args):
                pass

        return Handler
