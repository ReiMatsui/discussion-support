"""UIサーバー用HTTPハンドラ + ターミナル出力."""
from __future__ import annotations

import json
import sys
import time
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ._session_state import SessionState

from http.server import BaseHTTPRequestHandler

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


class _BadRequestError(ValueError):
    """UI リクエストの本文が不正（400 で返す）."""


class _UIRequestHandler(BaseHTTPRequestHandler):
    """UI の HTTP ハンドラ本体（`_UIHandler.create` が state を束ねて使う）.

    かつては `create` の中に入れ子で定義され、クロージャで state を捕まえて
    いた。360行がまるごと1関数の中にあり、個々のエンドポイントを読むにも
    関数全体をたどる必要があった。state はクラス変数で受け取る形に変え、
    ここをトップレベルに出してある。
    """

    _state: SessionState

    # POST の宛先表。個々の処理は _post_* にある（1エンドポイント1関数）。
    _POST_ROUTES: ClassVar[dict[str, str]] = {
        "/api/stop": "_post_stop",
        "/api/reset": "_post_reset",
        "/api/start": "_post_start",
        "/api/mode": "_post_mode",
        "/api/topic": "_post_topic",
        "/api/intervention": "_post_intervention",
        "/api/facilitator/call": "_post_facilitator_call",
        "/api/diarization": "_post_diarization",
        "/rename": "_post_rename",
        "/activate": "_post_activate",
        "/api/forget": "_post_forget",
        "/api/enroll": "_post_enroll",
        "/api/roster": "_post_roster",
        "/agent": "_post_agent",
    }

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
        last_backlog_bucket = 0
        try:
            while not s.stop.is_set():
                rev = s.rev
                partial = s.partial_text  # 認識途中経過も変化を見る（課題①）
                # 送信バックログは rev を上げずに変わるため、5秒刻みの段が
                # 変わったら配信する（遅延警告の表示・解除・更新のため）。
                backlog_bucket = s.send_backlog_ms // 5000
                if (rev != last_rev or partial != last_partial
                        or backlog_bucket != last_backlog_bucket):
                    payload = json.dumps(s.api_snapshot(), ensure_ascii=False)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    last_rev = rev
                    last_partial = partial
                    last_backlog_bucket = backlog_bucket
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
        """POST の入口。ここで例外を JSON エラーに翻訳する.

        従来 do_POST は本体を直に持ち、10箇所の json.loads が無防備だった
        （うち9箇所は Content-Length: 0 すら見ていない）。不正な本文で
        未捕捉例外→500＋トレースバックになり、UI には構造化エラーが
        返らなかった（2026-07-25 監査）。ルーティング本体は
        _dispatch_post に移し、翻訳はここ1箇所で行う。
        """
        try:
            self._dispatch_post()
        except _BadRequestError as e:
            self._json(400, {"ok": False, "error": str(e)})
        except Exception as e:   # UIを落とさず原因を返す
            _print_line(f"# UI: リクエスト処理でエラー {self.path}: {e!r}")
            self._json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    def _read_json(self) -> dict:
        """リクエスト本文を JSON dict として読む（不正なら _BadRequestError）.

        本文なし（Content-Length 無し/0）は空dictとして扱う——従来
        /api/facilitator/call だけがこの扱いで、他は json.loads(b"") で
        例外になっていた。JSON でない・dict でない本文は 400 にする。
        """
        raw_len = self.headers.get("Content-Length", 0)
        try:
            length = int(raw_len)
        except (TypeError, ValueError) as e:
            raise _BadRequestError("Content-Length が数値ではありません") from e
        if length <= 0:
            return {}
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise _BadRequestError(f"本文が JSON として読めません: {e}") from e
        if not isinstance(body, dict):
            raise _BadRequestError("本文は JSON オブジェクトである必要があります")
        return body

    def _dispatch_post(self):
        """POST を宛先表から該当の _post_* へ回す（未知のパスは404）."""
        method = self._POST_ROUTES.get(self.path)
        if method is None:
            self.send_error(404)
            return
        getattr(self, method)(self._state)

    def _post_stop(self, s) -> None:
        _print_line("# セッションを停止します（UIから）")
        if s.request_stop is not None:
            s.request_stop()
        else:
            s.stop.set()
        self._json(200, {"ok": True, "running": False})

    def _post_reset(self, s) -> None:
        _print_line("# 新しい会議に切り替えます（UIから）")
        if s.request_reset is not None:
            # STT接続ごと作り直す（実処理はメインスレッド）
            s.request_reset()
            self._json(200, {"ok": True, "resetting": True})
        else:
            # 接続なし（テスト等）は状態クリアのみ
            self._json(200, s.reset_for_new_meeting())

    def _post_start(self, s) -> None:
        s.waiting_to_start = False
        s.start_requested.set()
        s.rev += 1
        s.save()
        _print_line("# 会議を開始します（UIから）")
        self._json(200, {"ok": True, "waiting": False})

    def _post_mode(self, s) -> None:
        from das.asr.live._workers import set_session_mode
        body = self._read_json()
        result = set_session_mode(s, str(body.get("mode", "")))
        self._json(200 if result.get("ok") else 400, result)

    def _post_topic(self, s) -> None:
        body = self._read_json()
        result = s.set_agenda(str(body.get("topic", "")))
        _print_line(f"# 議題を設定（UIから）: {result['agenda']}")
        self._json(200, result)

    def _post_intervention(self, s) -> None:
        body = self._read_json()
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
            if s.agent is None:
                # 従来は 200 OK を返しつつ黙って無反映だった
                # （2026-07-25 監査 C: 「効いた顔をする設定」の同族）。
                self._json(400, {"ok": False, "error":
                                 "AIファシリテーターが未初期化のため発話数を設定できません"})
                return
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

    def _post_facilitator_call(self, s) -> None:
        # 手動呼び出し（Phase1）: キューに積むだけ。発話は worker + Controller
        # 経路が採否する（直接 agent.trigger() しない）。
        body = self._read_json()
        result = s.queue_manual_facilitator_call(
            str(body.get("request", "")))
        if not result.get("ok"):
            self._json(400, result)
            return
        _print_line("# ファシリテーター手動呼び出し（UIから）: "
                    f"{result.get('request') or '直近の議論整理'}")
        self._json(200, result)

    def _post_diarization(self, s) -> None:
        body = self._read_json()
        raw = body.get("max_speakers")
        max_speakers = None if raw in (None, "", "auto") else int(raw)
        result = s.set_diarization_max_speakers(max_speakers)
        if result.get("ok"):
            label = max_speakers if max_speakers is not None else "未指定"
            # 帰属側(constrain)は即時反映。旧文言「新しい会議/再接続で
            # 確実に反映」は即時性が伝わらず、ms=None のため更新時刻も
            # 残らず、「上限が古いまま会議が進んだ」事後切り分けが
            # できなかった（2026-07-25 実セッションの反省）。
            s.add_sys(getattr(s, "elapsed_ms", lambda: None)(),
                      f"想定話者数を更新: {label}"
                      "（帰属判定へ即時反映。STT/話者分離側は新しい会議で反映）")
            s.save()
            _print_line(f"# 想定話者数を更新（UIから）: {label}")
        self._json(200 if result.get("ok") else 400, result)

    def _post_rename(self, s) -> None:
        body = self._read_json()
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
            s.add_sys(s.elapsed_ms(), msg)
            s.save()
            _print_line(f"# {name} の声を登録しました（UIから）")
        else:
            s.set_display_name("#" + label, name)
            s.save()
            _print_line(f"# 話者{label} → {name}（UIから）")
        self._json(200, {"ok": True, "name": name})

    def _post_activate(self, s) -> None:
        body = self._read_json()
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
                s.add_sys(s.elapsed_ms(), f"「{name}」を有効化（{merged}と統合）")
                _print_line(f"# {name} を有効化（{merged}と統合、UIから）")
            else:
                _print_line(f"# {name} を有効化（UIから）")
            s.save()
        else:
            s.tracker.deactivate(name)
            _print_line(f"# {name} を無効化（UIから）")
            s.save()
        self._json(200, {"ok": True, "name": name, "active": active})

    def _post_forget(self, s) -> None:
        # 登録した声を完全に削除する（voices.json からも消す）。
        # 無効化(/activate)はセッション内で照合を止めるだけで、
        # 次の会議ではまた有効になる。名前の聞き間違いで作られた
        # プロファイルが溜まり続けるので、消す手段を用意する
        # （実会話で24人まで増えていた。handoff §28.9）。
        body = self._read_json()
        name = str(body.get("name", "")).strip()
        if not name:
            self._json(400, {"error": "name を指定してください"})
            return
        if s.tracker is None:
            self._json(400, {"error": "声紋照合が無効です"})
            return
        if not s.tracker.forget(name):
            self._json(404, {"error": f"「{name}」は登録されていません"})
            return
        s.add_sys(s.elapsed_ms(),
                  f"「{name}」の登録した声を削除しました")
        s.save()
        _print_line(f"# {name} の声紋を削除しました（UIから）")
        self._json(200, {"ok": True, "name": name})

    def _post_enroll(self, s) -> None:
        # 会議前の事前登録: 直近の音声(その人が単独で喋った分)で声紋を作る
        import numpy as np
        body = self._read_json()
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
        s.add_sys(s.elapsed_ms(), msg)
        s.save()
        _print_line(f"# {name} を事前登録（{actual_seconds:.0f}秒、UIから）")
        self._json(200, {"ok": True, "name": name,
                         "seconds": actual_seconds,
                         "active": True,
                         "applies_from": "future_utterances",
                         "message": msg})

    def _post_roster(self, s) -> None:
        # 名簿の確定/解除: 確定すると自動登録オフ（登録済みだけと照合し増殖を止める）
        body = self._read_json()
        locked = bool(body.get("locked", True))
        if s.tracker is None:
            self._json(400, {"error": "声紋照合が無効です"})
            return
        s.tracker.auto = not locked
        s.add_sys(s.elapsed_ms(), "名簿を確定しました（登録済みの人だけで進めます）"
                  if locked else "名簿の確定を解除しました（自動登録ON）")
        s.save()
        _print_line(f"# 名簿{'確定' if locked else '解除'}（UIから）")
        self._json(200, {"ok": True, "locked": locked})

    def _post_agent(self, s) -> None:
        body = self._read_json()
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

    def _json(self, code: int, data: dict) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def log_message(self, format, *args):
        pass


class _UIHandler:
    """UIサーバー用HTTPハンドラ（state を束ねたサブクラスを作る）.

    使い方:
        handler_cls = _UIHandler.create(state)
        httpd = HTTPServer(("127.0.0.1", port), handler_cls)
    """

    @staticmethod
    def create(state: SessionState) -> type:
        """state を束縛した BaseHTTPRequestHandler サブクラスを返す."""
        return type("Handler", (_UIRequestHandler,), {"_state": state})

