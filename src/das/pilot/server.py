"""localhost専用の試行GUI。POSTは同じ画面のトークンを必須とする。"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .recording import MAX_CHUNK_BYTES
from .session import PilotSession


def make_server(session: PilotSession, port: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, body):
            payload = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if not self.valid_host():
                return
            if self.path == "/":
                html = Path(__file__).with_name("panel.html").read_text().replace("__TOKEN__", session.token)
                payload = html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif self.path == "/api/status":
                self.respond(200, session.view())
            elif self.path == "/api/native":
                self.respond(200, session.native_view())
            elif self.path.startswith("/video/"):
                try:
                    path, mime = session.completed_video(self.path.removeprefix("/video/"))
                except ValueError:
                    self.respond(404, {"error": "動画がありません"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(path.stat().st_size))
                self.end_headers()
                try:
                    with path.open("rb") as video:
                        while chunk := video.read(1024 * 1024):
                            self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.respond(404, {"error": "ページがありません"})

        def do_POST(self):
            if not self.valid_host():
                return
            if self.headers.get("X-Pilot-Token") != session.token:
                self.respond(403, {"error": "操作画面を開き直してください"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 <= length <= MAX_CHUNK_BYTES:
                    raise ValueError("送信サイズの上限です")
                raw = self.rfile.read(length)
                if self.path.startswith("/api/video/"):
                    parts = self.path.split("/")
                    if len(parts) != 5:
                        raise ValueError("録画先が不正です")
                    session.append_video(parts[3], int(parts[4]), raw)
                    self.respond(200, {"ok": True})
                    return
                body = json.loads(raw or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("操作が不正です")
                if self.path == "/api/start":
                    result = session.start(**{k: body.get(k) for k in ("scenario", "condition", "mime", "consent")})
                elif self.path == "/api/video-finish":
                    session.finish_video(body.get("id"), int(body.get("chunks", -1)))
                    result = {"ok": True}
                elif self.path == "/api/stop":
                    result = session.stop()
                elif self.path == "/api/check-model":
                    result = session.check_model()
                elif self.path == "/api/note":
                    session.note(body.get("kind"), str(body.get("text", "")))
                    result = {"ok": True}
                else:
                    self.respond(404, {"error": "操作がありません"})
                    return
                self.respond(200, result)
            except (ValueError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
            except Exception as exc:
                session.event("operation_error", error=type(exc).__name__)
                self.respond(500, {"error": "操作に失敗しました。保存先のイベント記録を確認してください。"})

        def valid_host(self):
            if self.headers.get("Host") not in {
                f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}",
            }:
                self.respond(403, {"error": "localhostの操作画面を使用してください"})
                return False
            return True

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(session: PilotSession, port: int, open_browser=True):
    server = make_server(session, port)
    stopped = threading.Event()

    def watchdog():
        while not stopped.wait(1):
            view = session.view()
            active = view["active"]
            if active and active["elapsed_seconds"] > active["max_seconds"] + 20:
                try:
                    session.stop("safety_timeout")
                except Exception as exc:
                    session.event("timeout_stop_error", error=type(exc).__name__)

    threading.Thread(target=watchdog, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"試行の操作画面: {url}\n保存先: {session.output}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        session.stop("launcher_closed")
        server.server_close()
