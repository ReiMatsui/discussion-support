"""UIサーバー用HTTPハンドラ + ターミナル出力."""
from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._session_state import SessionState

from ._constants import CLEAR_LINE


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
    def create(state: SessionState) -> type:
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
